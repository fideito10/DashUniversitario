import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Nuevas importaciones para Machine Learning
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

dash.register_page(__name__, path='/team/analysis')

def load_data():
    """Carga los datos desde Google Sheets."""
    sheet_id = "1kajUuZwL9l1suipRNy7t2g_SGJUcDVh6MeTu66yK-Z4"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df = pd.read_csv(url)
        print(f"Datos cargados exitosamente. Forma: {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None

def create_injury_model():
    """Crea el modelo predictivo de lesiones."""
    df = load_data()
    if df is None:
        return None, None
    
    # Verificar columnas necesarias
    required_columns = ['Player Name', 'Distance (km)', 'Sprint Distance (m)']
    if not all(col in df.columns for col in required_columns):
        print(f"Columnas faltantes. Disponibles: {df.columns.tolist()}")
        return None, None
    
    # Limpiar y preparar datos
    df_model = df.copy()
    df_model = df_model.dropna(subset=required_columns)
    df_model['Distance (km)'] = pd.to_numeric(df_model['Distance (km)'], errors='coerce')
    df_model['Sprint Distance (m)'] = pd.to_numeric(df_model['Sprint Distance (m)'], errors='coerce')
    df_model = df_model.dropna(subset=['Distance (km)', 'Sprint Distance (m)'])
    
    # Crear variables derivadas
    df_model['Sprint_Distance_km'] = df_model['Sprint Distance (m)'] / 1000
    df_model['Sprint_Ratio'] = df_model['Sprint_Distance_km'] / df_model['Distance (km)']
    df_model['Total_Load'] = df_model['Distance (km)'] + (df_model['Sprint_Distance_km'] * 2)
    
    # Crear variable objetivo simulada (en caso real usarías datos históricos)
    np.random.seed(42)
    risk_score = (
        (df_model['Distance (km)'] - df_model['Distance (km)'].mean()) / df_model['Distance (km)'].std() +
        (df_model['Sprint_Ratio'] - df_model['Sprint_Ratio'].mean()) / df_model['Sprint_Ratio'].std() +
        (df_model['Total_Load'] - df_model['Total_Load'].mean()) / df_model['Total_Load'].std()
    )
    injury_probability = 1 / (1 + np.exp(-risk_score))
    df_model['Injury_Risk'] = (injury_probability > 0.6).astype(int)
    
    return df_model, train_models(df_model)

def train_models(df_model):
    """Entrena los modelos de predicción."""
    feature_columns = ['Distance (km)', 'Sprint Distance (m)', 'Sprint_Distance_km', 
                      'Sprint_Ratio', 'Total_Load']
    
    X = df_model[feature_columns]
    y = df_model['Injury_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Entrenar modelos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'scaler': scaler if name == 'Logistic Regression' else None,
            'accuracy': accuracy
        }
    
    return results

def predict_injury_by_player_name(player_name, model_results=None, df_model=None):
    """Predice el riesgo de lesión para un jugador específico por su nombre."""
    # Si no se proporcionan los modelos, entrenarlos
    if model_results is None or df_model is None:
        df_model, model_results = create_injury_model()
        if df_model is None or model_results is None:
            return None
    
    # Buscar jugador
    player_data = df_model[df_model['Player Name'].str.contains(player_name, case=False, na=False)]
    
    if player_data.empty:
        return None
    
    if len(player_data) > 1:
        player_data = player_data.iloc[0:1]
    
    # Obtener datos del jugador
    player = player_data.iloc[0]
    distance_km = player['Distance (km)']
    sprint_distance_m = player['Sprint Distance (m)']
    sprint_distance_km = sprint_distance_m / 1000
    sprint_ratio = sprint_distance_km / distance_km if distance_km > 0 else 0
    total_load = distance_km + (sprint_distance_km * 2)
    
    # Crear características para predicción
    features = np.array([[distance_km, sprint_distance_m, sprint_distance_km, 
                         sprint_ratio, total_load]])
    
    # Predicciones con manejo de errores
    try:
        rf_model = model_results['Random Forest']['model']
        rf_risk = rf_model.predict_proba(features)[0][1]
        
        lr_model = model_results['Logistic Regression']['model']
        scaler = model_results['Logistic Regression']['scaler']
        features_scaled = scaler.transform(features)
        lr_risk = lr_model.predict_proba(features_scaled)[0][1]
        
        avg_risk = (rf_risk + lr_risk) / 2
    except Exception as e:
        print(f"Error en predicción: {e}")
        return None
    
    # Determinar nivel de riesgo
    if avg_risk < 0.3:
        risk_level = "BAJO"
        risk_color = "success"
        risk_emoji = "🟢"
    elif avg_risk < 0.6:
        risk_level = "MEDIO"
        risk_color = "warning"
        risk_emoji = "🟡"
    else:
        risk_level = "ALTO"
        risk_color = "danger"
        risk_emoji = "🔴"
    
    return {
        'player_name': player['Player Name'],
        'distance_km': distance_km,
        'sprint_distance_m': sprint_distance_m,
        'sprint_ratio': sprint_ratio,
        'total_load': total_load,
        'rf_risk': rf_risk,
        'lr_risk': lr_risk,
        'avg_risk': avg_risk,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_emoji': risk_emoji
    }

# Layout de la página
layout = html.Div([
    # Encabezado con gradiente y diseño mejorado
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-brain me-3", style={'color': '#ff6b6b'}),
                            "Sistema de Lesiones Calculado con IA"
                        ], className="text-center mb-3", style={
                            'color': 'white',
                            'fontWeight': 'bold',
                            'fontSize': '3rem',
                            'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                            'marginBottom': '1rem'
                        }),
                        html.P([
                            "Predicción inteligente del riesgo de lesiones deportivas utilizando Machine Learning"
                        ], className="text-center", style={
                            'color': 'rgba(255,255,255,0.9)',
                            'fontSize': '1.2rem',
                            'marginBottom': '0'
                        })
                    ], style={
                        'padding': '40px 0',
                        'position': 'relative',
                        'zIndex': '2'
                    })
                ])
            ])
        ], fluid=True),
        # Elementos decorativos
        html.Div([
            html.I(className="fas fa-heartbeat", style={
                'position': 'absolute',
                'top': '20px',
                'right': '50px',
                'fontSize': '2rem',
                'color': 'rgba(255,255,255,0.1)',
                'animation': 'pulse 2s infinite'
            }),
            html.I(className="fas fa-chart-line", style={
                'position': 'absolute',
                'bottom': '20px',
                'left': '50px',
                'fontSize': '1.5rem',
                'color': 'rgba(255,255,255,0.1)',
                'animation': 'float 3s ease-in-out infinite'
            })
        ])
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'position': 'relative',
        'overflow': 'hidden',
        'marginBottom': '30px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'
    }),
    
    dbc.Container([
    
    # Tarjeta principal de predicción con diseño mejorado
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="fas fa-user-injured me-2", style={'color': '#ff6b6b'}),
                        "Predictor de Riesgo de Lesiones"
                    ], className="mb-0", style={'color': 'white'})
                ], style={
                    'background': 'linear-gradient(90deg, #2c3e50 0%, #34495e 100%)',
                    'border': 'none',
                    'borderRadius': '10px 10px 0 0'
                }),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Selecciona un Jugador:", className="fw-bold mb-2", 
                                     style={'color': 'white', 'fontSize': '1.1rem'}),
                            dcc.Dropdown(
                                id='player-dropdown',
                                placeholder="🔍 Buscar jugador...",
                                style={
                                    'marginBottom': '20px',
                                    'fontSize': '1rem'
                                },
                                className="custom-dropdown"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-search me-2"), "Analizar Riesgo"],
                                id="analyze-button",
                                color="primary",
                                size="lg",
                                className="w-100 custom-button",
                                style={
                                    'background': 'linear-gradient(45deg, #667eea, #764ba2)',
                                    'border': 'none',
                                    'borderRadius': '10px',
                                    'fontWeight': 'bold',
                                    'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                    'transition': 'all 0.3s ease'
                                }
                            )
                        ], md=4),
                        dbc.Col([
                            html.Div(id="prediction-results")
                        ], md=8)
                    ])
                ], style={'padding': '30px'})
            ], className="shadow-lg custom-card", style={
                'borderRadius': '15px',
                'border': 'none',
                'background': '#2c3e50'
            })
        ])
    ], className="mb-4"),
    
    # Área para mostrar estadísticas adicionales
    dbc.Row([
        dbc.Col([
            html.Div(id="additional-stats")
        ])
    ])
    ], fluid=True)
], style={
    'minHeight': '100vh',
    'background': 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
    'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
})

# Callback para cargar jugadores en el dropdown
@callback(
    Output('player-dropdown', 'options'),
    Input('player-dropdown', 'id')
)
def load_players(_):
    df = load_data()
    if df is not None and 'Player Name' in df.columns:
        players = df['Player Name'].dropna().unique()
        return [{'label': player, 'value': player} for player in sorted(players)]
    return []

# Callback para hacer la predicción
@callback(
    [Output('prediction-results', 'children'),
     Output('additional-stats', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('player-dropdown', 'value')],
    prevent_initial_call=True
)
def predict_injury(n_clicks, selected_player):
    if n_clicks is None or selected_player is None:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2", style={'fontSize': '1.2rem'}),
                "Selecciona un jugador y presiona 'Analizar Riesgo' para ver la predicción"
            ], color="info", className="fade-in", style={
                'borderRadius': '10px',
                'border': 'none',
                'background': 'linear-gradient(90deg, #34495e, #2c3e50)',
                'color': 'white',
                'fontWeight': '500'
            })
        ]), html.Div()
    
    # Realizar predicción con manejo de errores
    try:
        result = predict_injury_by_player_name(selected_player)
        
        if result is None:
            return html.Div([
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2", style={'fontSize': '1.2rem'}),
                    f"No se pudo encontrar información para el jugador: {selected_player}"
                ], color="warning", className="fade-in", style={
                    'borderRadius': '10px',
                    'border': 'none',
                    'background': 'linear-gradient(90deg, #f39c12, #e67e22)',
                    'color': 'white',
                    'fontWeight': '500'
                })
            ]), html.Div()
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-times-circle me-2", style={'fontSize': '1.2rem'}),
                f"Error al procesar la predicción: {str(e)}"
            ], color="danger", className="fade-in", style={
                'borderRadius': '10px',
                'border': 'none',
                'background': 'linear-gradient(90deg, #e74c3c, #c0392b)',
                'color': 'white',
                'fontWeight': '500'
            })
        ]), html.Div()
    
    # Crear tarjetas de resultados con diseño mejorado
    prediction_card = dbc.Card([
        dbc.CardHeader([
            html.H5([
                result['risk_emoji'],
                f" {result['player_name']}"
            ], className="mb-0", style={
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '1.4rem'
            })
        ], style={
            'background': 'linear-gradient(90deg, #2c3e50 0%, #34495e 100%)',
            'border': 'none',
            'borderRadius': '15px 15px 0 0'
        }),
        dbc.CardBody([
            # Nivel de riesgo principal con diseño mejorado
            dbc.Alert([
                html.Div([
                    html.H3([
                        html.I(className="fas fa-heartbeat me-3", style={'fontSize': '2rem'}),
                        f"Riesgo: {result['risk_level']}"
                    ], className="mb-3", style={'fontWeight': 'bold'}),
                    html.H2(f"{result['avg_risk']:.1%}", className="mb-0", style={
                        'fontSize': '3rem',
                        'fontWeight': 'bold',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.1)'
                    })
                ])
            ], color=result['risk_color'], className="text-center mb-4", style={
                'borderRadius': '15px',
                'border': 'none',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
                'background': f'linear-gradient(135deg, var(--bs-{result["risk_color"]}-bg-subtle), var(--bs-{result["risk_color"]}))'
            }),
            
            # Métricas detalladas con iconos y colores
            html.H6("📊 Métricas de Rendimiento", className="mb-3", style={
                'color': 'white',
                'fontWeight': 'bold',
                'borderBottom': '2px solid #34495e',
                'paddingBottom': '10px'
            }),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-route", style={'fontSize': '1.5rem', 'color': '#3498db', 'marginBottom': '10px'}),
                            html.Small("Distancia Total", className="text-muted d-block", style={'color': 'rgba(255,255,255,0.7)'}),
                            html.H5(f"{result['distance_km']:.2f} km", style={'color': 'white', 'fontWeight': 'bold'})
                        ], className="text-center")
                    ], className="h-100", style={'borderRadius': '10px', 'border': '1px solid #34495e', 'background': '#34495e'})
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-bolt", style={'fontSize': '1.5rem', 'color': '#f39c12', 'marginBottom': '10px'}),
                            html.Small("Distancia Sprint", className="text-muted d-block", style={'color': 'rgba(255,255,255,0.7)'}),
                            html.H5(f"{result['sprint_distance_m']:.0f} m", style={'color': 'white', 'fontWeight': 'bold'})
                        ], className="text-center")
                    ], className="h-100", style={'borderRadius': '10px', 'border': '1px solid #34495e', 'background': '#34495e'})
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-tachometer-alt", style={'fontSize': '1.5rem', 'color': '#e74c3c', 'marginBottom': '10px'}),
                            html.Small("Ratio Sprint", className="text-muted d-block", style={'color': 'rgba(255,255,255,0.7)'}),
                            html.H5(f"{result['sprint_ratio']:.3f}", style={'color': 'white', 'fontWeight': 'bold'})
                        ], className="text-center")
                    ], className="h-100", style={'borderRadius': '10px', 'border': '1px solid #34495e', 'background': '#34495e'})
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-weight-hanging", style={'fontSize': '1.5rem', 'color': '#9b59b6', 'marginBottom': '10px'}),
                            html.Small("Carga Total", className="text-muted d-block", style={'color': 'rgba(255,255,255,0.7)'}),
                            html.H5(f"{result['total_load']:.2f}", style={'color': 'white', 'fontWeight': 'bold'})
                        ], className="text-center")
                    ], className="h-100", style={'borderRadius': '10px', 'border': '1px solid #34495e', 'background': '#34495e'})
                ], md=3)
            ], className="mb-4"),
            
            # Predicciones de modelos con diseño mejorado
            html.Hr(style={'border': '2px solid #34495e', 'margin': '30px 0'}),
            html.H6("🤖 Predicciones por Modelo de IA:", className="mb-3", style={
                'color': 'white',
                'fontWeight': 'bold'
            }),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-tree", style={'fontSize': '1.3rem', 'color': '#27ae60', 'marginBottom': '10px'}),
                            html.Small("Random Forest", className="text-muted d-block", style={'color': 'rgba(255,255,255,0.7)'}),
                            html.H5(f"{result['rf_risk']:.1%}", style={'color': '#27ae60', 'fontWeight': 'bold'})
                        ], className="text-center")
                    ], style={'borderRadius': '10px', 'background': 'linear-gradient(135deg, #2c3e50, #34495e)'})
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-chart-line", style={'fontSize': '1.3rem', 'color': '#8e44ad', 'marginBottom': '10px'}),
                            html.Small("Regresión Logística", className="text-muted d-block", style={'color': 'rgba(255,255,255,0.7)'}),
                            html.H5(f"{result['lr_risk']:.1%}", style={'color': '#8e44ad', 'fontWeight': 'bold'})
                        ], className="text-center")
                    ], style={'borderRadius': '10px', 'background': 'linear-gradient(135deg, #2c3e50, #34495e)'})
                ], md=6)
            ])
        ], style={'padding': '30px'})
    ], className="mb-4 fade-in", style={
        'borderRadius': '15px',
        'border': 'none',
        'boxShadow': '0 8px 25px rgba(0,0,0,0.1)',
        'background': '#2c3e50'
    })
    
    # Recomendaciones con diseño mejorado
    if result['risk_level'] == "BAJO":
        recommendations = [
            "✅ El jugador puede continuar con la carga actual",
            "✅ Mantener monitoreo regular",
            "✅ Excelente estado físico detectado"
        ]
        rec_color = "success"
        rec_icon = "fas fa-check-circle"
        rec_gradient = "linear-gradient(135deg, #27ae60, #2ecc71)"
    elif result['risk_level'] == "MEDIO":
        recommendations = [
            "⚠️ Considerar reducir la intensidad de sprint",
            "⚠️ Incrementar tiempo de recuperación",
            "⚠️ Monitoreo más frecuente",
            "⚠️ Evaluación de técnica de carrera"
        ]
        rec_color = "warning"
        rec_icon = "fas fa-exclamation-triangle"
        rec_gradient = "linear-gradient(135deg, #f39c12, #e67e22)"
    else:
        recommendations = [
            "🚨 Reducir inmediatamente la carga de entrenamiento",
            "🚨 Evaluación médica recomendada",
            "🚨 Considerar días de descanso adicionales",
            "🚨 Implementar protocolo de prevención"
        ]
        rec_color = "danger"
        rec_icon = "fas fa-exclamation-circle"
        rec_gradient = "linear-gradient(135deg, #e74c3c, #c0392b)"
    
    recommendations_card = dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className=f"{rec_icon} me-3", style={'fontSize': '1.3rem'}),
                "Recomendaciones Inteligentes"
            ], className="mb-0", style={
                'color': 'white',
                'fontWeight': 'bold'
            })
        ], style={
            'background': rec_gradient,
            'border': 'none',
            'borderRadius': '15px 15px 0 0'
        }),
        dbc.CardBody([
            html.Div([
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.I(className="fas fa-arrow-right me-2", style={'color': f'var(--bs-{rec_color})'}),
                        rec
                    ], style={
                        'border': 'none',
                        'background': 'transparent',
                        'padding': '15px 0',
                        'fontSize': '1.1rem',
                        'fontWeight': '500',
                        'color': 'white'
                    }) for rec in recommendations
                ], flush=True)
            ])
        ], style={'padding': '25px'})
    ], className="fade-in", style={
        'borderRadius': '15px',
        'border': 'none',
        'boxShadow': '0 6px 20px rgba(0,0,0,0.1)',
        'background': '#2c3e50'
    })
    
    return prediction_card, recommendations_card