import dash
from dash import html, dcc, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output

dash.register_page(__name__, path='/stats/general')

def load_data():
    try:
        sheet_id = "1kajUuZwL9l1suipRNy7t2g_SGJUcDVh6MeTu66yK-Z4"
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Leer directamente como DataFrame
        df = pd.read_csv(url)
        return df
    except Exception as e:
        return None

# Función para obtener los tops por categoría
def get_top_players(df, metric, n=5, ascending=False):
    """
    Obtiene el top de jugadores según una métrica específica
    
    Parámetros:
    - df: DataFrame con los datos
    - metric: Columna a utilizar para el ranking
    - n: Número de jugadores a mostrar (top N)
    - ascending: Si es True ordena de menor a mayor, si es False de mayor a menor
    """
    if df is None or df.empty:
        return None
    
    # Agrupar por Player Name y obtener la media de la métrica
    try:
        player_stats = df.groupby('Player Name')[metric].agg(['mean', 'count']).reset_index()
        # Filtrar jugadores con al menos 3 sesiones registradas
        player_stats = player_stats[player_stats['count'] >= 3]
        # Ordenar según la métrica
        player_stats = player_stats.sort_values('mean', ascending=ascending).head(n)
        return player_stats
    except Exception as e:
        print(f"Error al procesar top jugadores: {e}")
        return None

# Función para crear gráfico de barras
def create_player_bar_chart(player_stats, metric_name, color_scale):
    if player_stats is None or player_stats.empty:
        return {}
    
    # Crear figura
    fig = go.Figure()
    
    # Determinar el máximo para calcular el gradiente de color
    max_value = player_stats['mean'].max()
    
    # Añadir barras para cada jugador con un gradiente de color
    for i, (idx, row) in enumerate(player_stats.iterrows()):
        # Calcular el color basado en el valor relativo (más alto = más oscuro)
        color_intensity = (row['mean'] / max_value) * 0.8 + 0.2  # Rango de 0.2 a 1.0
        color = color_scale[i % len(color_scale)]  # Alternar colores
        
        fig.add_trace(go.Bar(
            x=[row['Player Name']],
            y=[row['mean']],
            name=row['Player Name'],
            marker_color=color,
            text=f"{row['mean']:.2f}",
            textposition='outside',
            hovertemplate=f"<b>{row['Player Name']}</b><br>{metric_name}: %{{y:.2f}}<br>Sesiones: {row['count']}<extra></extra>"
        ))
    
    # Actualizar diseño
    fig.update_layout(
        title=f"Top Jugadores - {metric_name}",
        title_font=dict(size=16, color='white'),
        xaxis=dict(
            title="Jugador",
            tickangle=-45,
            tickfont=dict(size=12, color='white'),
            title_font=dict(size=14, color='white')
        ),
        yaxis=dict(
            title=metric_name,
            gridcolor='rgba(255, 255, 255, 0.2)',
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        height=400,
        margin=dict(l=40, r=40, t=60, b=80),
        showlegend=False
    )
    
    return fig

# Layout principal
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Competencia Entre Jugadores", className="text-center my-3", 
                   style={"fontWeight": "800", "letterSpacing": "1px", "color": "#ffffff", 
                          "textShadow": "2px 2px 4px rgba(0,0,0,0.5)", "padding": "10px 0"}),
            html.Hr(className="my-2", style={"borderColor": "rgba(255, 255, 255, 0.3)"})
        ], width=12)
    ]),
    
    # Selector de categoría y cantidad de jugadores
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filtros de Ranking", className="text-white font-weight-bold", 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),
                dbc.CardBody([
                    html.Label("Seleccione la categoría:", className="text-white mb-2"),
                    dcc.Dropdown(
                        id='categoria-dropdown',
                        options=[
                            {'label': 'Distancia Recorrida (km)', 'value': 'Distance (km)'},
                            {'label': 'Sprint Distance (m)', 'value': 'Sprint Distance (m)'},
                            {'label': 'Aceleraciones', 'value': 'Accelerations Zone Count: > 4 m/s/s'},
                            {'label': 'Desaceleraciones', 'value': 'Deceleration Zone Count: > 4 m/s/s'},
                            {'label': 'Velocidad Máxima (km/h)', 'value': 'Max Speed (km/h)'}
                        ],
                        value='Distance (km)',
                        clearable=False,
                        className="mb-3",
                        style={"backgroundColor": "#343a40", "color": "white"}
                    ),
                    html.Label("Número de jugadores a mostrar:", className="text-white mb-2"),
                    dcc.Slider(
                        id='n-jugadores-slider',
                        min=3,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(3, 11)},
                        className="mb-3"
                    ),
                    html.Div(id="n-jugadores-output", className="text-white mt-2")
                ])
            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=12)
    ]),
    
    # Gráfico de top jugadores
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Top Jugadores", className="mb-0 text-white", style={"fontWeight": "600"}), 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),
                dbc.CardBody([
                    dcc.Graph(id="grafico-top-jugadores"),
                    html.Div(id="mensaje-error-jugadores", className="alert alert-danger mt-3", style={"display": "none"})
                ])
            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=12)
    ]),
    
    # Tabla de mejoras semanales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Mejora Semanal", className="mb-0 text-white", style={"fontWeight": "600"}), 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),
                dbc.CardBody([
                    html.P("Jugadores con mayor progreso en las últimas 4 semanas:", 
                          className="text-white mb-3"),
                    html.Div(id="tabla-mejoras")
                ])
            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=12)
    ]),
    
    # Medallas y reconocimientos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Medallas y Reconocimientos", className="mb-0 text-white", style={"fontWeight": "600"}), 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),
                dbc.CardBody([
                    html.Div(id="medallas-reconocimientos")
                ])
            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=12)
    ])
], fluid=True, style={"backgroundColor": "#6c757d", "minHeight": "100vh"})

# Callbacks
@callback(
    [Output('grafico-top-jugadores', 'figure'),
     Output('mensaje-error-jugadores', 'children'),
     Output('mensaje-error-jugadores', 'style'),
     Output('n-jugadores-output', 'children'),
     Output('tabla-mejoras', 'children'),
     Output('medallas-reconocimientos', 'children')],
    [Input('categoria-dropdown', 'value'),
     Input('n-jugadores-slider', 'value')]
)
def actualizar_top_jugadores(categoria, n_jugadores):
    """
    Callback para actualizar el gráfico de top jugadores
    """
    # Mostrar el valor del slider
    slider_output = f"Mostrando top {n_jugadores} jugadores"
    
    # Cargar los datos
    df = load_data()
    if df is None:
        empty_fig = {}
        error_msg = "Error al cargar los datos. Verifique la conexión."
        error_style = {"display": "block"}
        return empty_fig, error_msg, error_style, slider_output, [], []
    
    # Nombres amigables para las categorías
    category_names = {
        'Distance (km)': 'Distancia Recorrida (km)',
        'Sprint Distance (m)': 'Sprint Distance (m)',
        'Accelerations Zone Count: > 4 m/s/s': 'Aceleraciones de Alta Intensidad',
        'Deceleration Zone Count: > 4 m/s/s': 'Desaceleraciones de Alta Intensidad',
        'Max Speed (km/h)': 'Velocidad Máxima (km/h)'
    }
    
    # Colores para el gráfico según la categoría
    color_schemes = {
        'Distance (km)': ['#1a1a2e', '#16213e', '#0f3460', '#152238', '#2c394b'],
        'Sprint Distance (m)': ['#800000', '#8B0000', '#A52A2A', '#B22222', '#DC143C'],
        'Accelerations Zone Count: > 4 m/s/s': ['#006400', '#2e8b57', '#228b22', '#3cb371', '#66cdaa'],
        'Deceleration Zone Count: > 4 m/s/s': ['#4b0082', '#483d8b', '#6a5acd', '#7b68ee', '#9370db'],
        'Max Speed (km/h)': ['#ff8c00', '#ffa500', '#ff4500', '#ff6347', '#ff7f50']
    }
    
    # Obtener top jugadores para la categoría seleccionada
    top_players = get_top_players(df, categoria, n_jugadores)
    
    # Verificar si hay datos
    if top_players is None or top_players.empty:
        empty_fig = {}
        error_msg = f"No se encontraron suficientes datos para la categoría: {category_names.get(categoria, categoria)}"
        error_style = {"display": "block"}
        return empty_fig, error_msg, error_style, slider_output, [], []
    
    # Crear gráfico de barras
    fig = create_player_bar_chart(top_players, category_names.get(categoria, categoria), color_schemes.get(categoria, ['#1a1a2e', '#16213e']))
    
    # Crear tabla de mejoras semanales (simulada para este ejemplo)
    tabla_mejoras = generar_tabla_mejoras(df, categoria)
    
    # Crear medallas y reconocimientos
    medallas = generar_medallas_reconocimientos(df)
    
    return fig, "", {"display": "none"}, slider_output, tabla_mejoras, medallas

def generar_tabla_mejoras(df, categoria):
    """
    Genera una tabla con las mejoras semanales de los jugadores
    """
    # En un caso real, calcularíamos las mejoras semanales
    # Para este ejemplo, generamos datos ilustrativos
    
    # Nombres amigables para las categorías
    category_names = {
        'Distance (km)': 'Distancia',
        'Sprint Distance (m)': 'Sprint',
        'Accelerations Zone Count: > 4 m/s/s': 'Aceleraciones',
        'Deceleration Zone Count: > 4 m/s/s': 'Desaceleraciones',
        'Max Speed (km/h)': 'Velocidad'
    }
    
    category_short = category_names.get(categoria, 'Métrica')
    
    # Lista de jugadores con mejoras (simulado)
    mejoras = [
        {"jugador": "Duclock Juan", "mejora": "+15%", "comentario": f"Mayor incremento en {category_short} en las últimas 4 semanas"},
        {"jugador": "Vergel Alejo", "mejora": "+12%", "comentario": f"Consistente mejora en {category_short} semana tras semana"},
        {"jugador": "RiveraCano Blas", "mejora": "+10%", "comentario": "Destacado en entrenamientos recientes"}
    ]
    
    # Crear la tabla
    tabla = dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Jugador", style={"color": "white"}), 
                    html.Th("Mejora", style={"color": "white"}), 
                    html.Th("Comentario", style={"color": "white"})
                ], style={"backgroundColor": "#343a40"})
            ),
            html.Tbody([
                html.Tr([
                    html.Td(m["jugador"], style={"color": "white"}),
                    html.Td(m["mejora"], style={"color": "#2ecc71", "fontWeight": "bold"}),
                    html.Td(m["comentario"], style={"color": "white"})
                ], style={"backgroundColor": "rgba(0,0,0,0.2)"}) 
                for m in mejoras
            ])
        ],
        bordered=False,
        dark=True,
        hover=True,
        responsive=True,
        striped=True,
        className="mt-3"
    )
    
    return tabla

def generar_medallas_reconocimientos(df):
    """
    Genera tarjetas con medallas y reconocimientos para los jugadores destacados
    """
    # En un caso real, esto se basaría en cálculos de datos
    # Para este ejemplo, creamos reconocimientos ilustrativos
    
    reconocimientos = [
        {
            "jugador": "Espinosa Fede",
            "titulo": "Maratón del Rugby",
            "descripcion": "Mayor distancia total acumulada en el mes",
            "color": "#1a1a2e"
        },
        {
            "jugador": "Urrejola Fabio",
            "titulo": "Velocista Imparable",
            "descripcion": "Mayor velocidad máxima registrada (32.5 km/h)",
            "color": "#800000"
        },
        {
            "jugador": "Wolcan Juani",
            "titulo": "Motor de Arranque",
            "descripcion": "Mayor número de aceleraciones de alta intensidad",
            "color": "#006400"
        },
        {
            "jugador": "Damioli Bautista",
            "titulo": "Freno Maestro",
            "descripcion": "Excelente control en desaceleraciones bruscas",
            "color": "#4b0082"
        }
    ]
    
    # Crear tarjetas para cada reconocimiento
    tarjetas = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H5(r["titulo"], className="mb-0 text-center text-white", 
                           style={"fontWeight": "600", "letterSpacing": "0.5px"}),
                    style={"backgroundColor": r["color"]}
                ),
                dbc.CardBody([
                    html.H6(r["jugador"], className="text-center mb-2", 
                           style={"fontWeight": "bold", "color": "white"}),
                    html.P(r["descripcion"], className="text-center text-white")
                ], style={"backgroundColor": "rgba(0,0,0,0.3)"})
            ], className="mb-3", style={"boxShadow": "0 4px 8px rgba(0,0,0,0.3)"})
        ], width=3) for r in reconocimientos
    ])
    
    return tarjetas