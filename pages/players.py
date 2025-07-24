import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path

# Registrar esta página en la aplicación Dash
dash.register_page(__name__, path='/stats/players')

# Función para cargar datos desde Google Sheets
def load_data():
    """
    Carga los datos desde una hoja de Google Sheets.
    
    Returns:
        DataFrame: Datos de los jugadores o None si hay error
    """
    sheet_id = "1kajUuZwL9l1suipRNy7t2g_SGJUcDVh6MeTu66yK-Z4"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        # Leer directamente como DataFrame
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None

# FUNCIONES DE ANÁLISIS
def obtener_valores_maximos_por_jugador(dataframe, columnas, jugador=None, excluir_aceleraciones=True):
    """
    Obtiene los valores máximos históricos por jugador para las columnas especificadas.
    
    Parámetros:
    - dataframe: DataFrame con los datos de Catapult
    - columnas: Lista de nombres de columnas para obtener sus valores máximos
    - jugador: (Opcional) Nombre del jugador específico. Si es None, devuelve para todos los jugadores
    - excluir_aceleraciones: (Opcional) Si es True, excluye columnas de aceleraciones y desaceleraciones
    
    Retorna:
    - DataFrame con los valores máximos de cada columna para cada jugador
    """
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()
    
    # Verificar que 'Player Name' está en las columnas
    if 'Player Name' not in dataframe.columns:
        print("Error: No se encuentra la columna 'Player Name' en el DataFrame")
        return pd.DataFrame()
    
    # Filtrar aceleraciones y desaceleraciones si se solicita
    if excluir_aceleraciones:
        columnas = [col for col in columnas if 'cceleration' not in col and 'eceleration' not in col]
    
    # Verificar que todas las columnas solicitadas existen
    columnas_validas = [col for col in columnas if col in dataframe.columns]
    if len(columnas_validas) < len(columnas):
        columnas_faltantes = set(columnas) - set(columnas_validas)
        print(f"Advertencia: No se encontraron estas columnas: {columnas_faltantes}")
    
    # Filtrar por jugador específico si se proporciona
    if jugador:
        df_filtrado = dataframe[dataframe['Player Name'] == jugador].copy()
        if df_filtrado.empty:
            print(f"No se encontraron datos para el jugador: {jugador}")
            return pd.DataFrame()
    else:
        df_filtrado = dataframe.copy()
    
    # Convertir columnas numéricas (manejo de posibles formatos de texto)
    for col in columnas_validas:
        if df_filtrado[col].dtype == 'object':
            # Reemplazar comas por puntos si es necesario
            df_filtrado[col] = df_filtrado[col].str.replace(',', '.', regex=True)
        # Convertir a numérico con manejo de errores
        df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')
    
    # Agrupar por jugador y obtener máximos por columna
    resultado = df_filtrado.groupby('Player Name')[columnas_validas].max().reset_index()
    
    # Renombrar columnas para mayor claridad
    for col in columnas_validas:
        resultado = resultado.rename(columns={col: f"Máximo {col}"})
    
    return resultado

def crear_gauge_carga_aguda(df, jugador_seleccionado=None):
    """
    Crea un gráfico gauge (tacómetro) para mostrar la carga aguda de un jugador.
    
    Parámetros:
    - df: DataFrame con los datos de Catapult
    - jugador_seleccionado: Nombre del jugador (opcional). Si no se proporciona, 
                          se usará el primer jugador disponible.
    
    Retorna:
    - fig: Figura de Plotly con el gráfico gauge
    - datos_procesados: DataFrame con los datos procesados del jugador
    """
    # Si no se proporciona un jugador, usar el primero disponible
    if jugador_seleccionado is None:
        if 'Player Name' in df.columns and not df.empty:
            jugador_seleccionado = df['Player Name'].unique()[0]
        else:
            return None, None
    
    # 1. Filtrar datos del jugador
    datos_jugador = df[df['Player Name'] == jugador_seleccionado].copy()
    
    if datos_jugador.empty:
        print(f"No se encontraron datos para el jugador {jugador_seleccionado}")
        return None, None
    
    print(f"Se encontraron {len(datos_jugador)} registros para {jugador_seleccionado}")
    
    # 2. Preparar datos (convertir fechas y columnas numéricas)
    datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'], errors='coerce')
    
    # Columnas numéricas que necesitamos
    columnas_numericas = [
        'Distance (km)', 
        'Sprint Distance (m)', 
        'Top Speed (m/s)',
        'Accelerations Zone Count: 2 - 3 m/s/s',
        'Accelerations Zone Count: 3 - 4 m/s/s',
        'Accelerations Zone Count: > 4 m/s/s',
        'Deceleration Zone Count: 2 - 3 m/s/s',
        'Deceleration Zone Count: 3 - 4 m/s/s',
        'Deceleration Zone Count: > 4 m/s/s'
    ]
    
    # Convertir columnas a formato numérico
    for col in columnas_numericas:
        if col in datos_jugador.columns:
            # Reemplazar comas por puntos si es string
            if datos_jugador[col].dtype == 'object':
                datos_jugador[col] = datos_jugador[col].str.replace(',', '.', regex=True)
            # Convertir a numérico
            datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce').fillna(0)
    
    # 3. Calcular carga diaria
    pesos = {
        'Distance (km)': 1.0,
        'Sprint Distance (m)': 0.02,
        'Top Speed (m/s)': 10.0,
        'Accelerations Zone Count: 2 - 3 m/s/s': 1.5,
        'Accelerations Zone Count: 3 - 4 m/s/s': 2.0,
        'Accelerations Zone Count: > 4 m/s/s': 2.5,
        'Deceleration Zone Count: 2 - 3 m/s/s': 1.5,
        'Deceleration Zone Count: 3 - 4 m/s/s': 2.0,
        'Deceleration Zone Count: > 4 m/s/s': 2.5
    }
    
    # Calcular carga diaria aplicando ponderaciones
    datos_jugador['Carga_Diaria'] = 0
    for var, peso in pesos.items():
        if var in datos_jugador.columns:
            datos_jugador['Carga_Diaria'] += datos_jugador[var] * peso
    
    # 4. Obtener últimas 7 sesiones
    datos_ultimos = datos_jugador.sort_values('Date', ascending=False).head(7)
    datos_ultimos = datos_ultimos.sort_values('Date')  # Ordenar cronológicamente para visualización
    
    # 5. Crear gauge chart (tacómetro) con Plotly
    # Calcular promedio de carga de las últimas 7 sesiones para el gauge
    carga_promedio = datos_ultimos['Carga_Diaria'].mean()
    
    # Determinar el valor máximo para la escala del gauge (200% del promedio o un valor fijo)
    max_gauge = max(carga_promedio * 2, datos_ultimos['Carga_Diaria'].max() * 1.2)
    
    # Crear figura con gauge chart
    fig = go.Figure()
    
    # Añadir el indicador gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=datos_ultimos['Carga_Diaria'].iloc[-1],  # Valor más reciente
        title={'text': "", 'font': {'color': 'white', 'size': 16}},  # Título simplificado
        number={'font': {'color': 'white', 'size': 36, 'family': 'Arial, sans-serif'}},
        delta={'reference': datos_ultimos['Carga_Diaria'].mean(), 'relative': True, 'valueformat': '.1%', 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, max_gauge], 'tickwidth': 1, 'tickcolor': "#ffffff", 'tickfont': {'color': 'white'}},
            'bar': {'color': "darkblue"},
            'bgcolor': "rgba(0,0,0,0.2)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.5)",
            'steps': [
                {'range': [0, max_gauge*0.3], 'color': "lightgreen"},
                {'range': [max_gauge*0.3, max_gauge*0.7], 'color': "gold"},
                {'range': [max_gauge*0.7, max_gauge], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': carga_promedio * 1.5  # Umbral en 150% del promedio
            }
        }
    ))

    # Actualizar el layout sin título
    fig.update_layout(
        height=400,  # Ajustado a 400px para igualar con la carga crónica
        font=dict(size=16, color='white'),
        margin=dict(l=40, r=40, t=40, b=40),  # Reducir margen superior
        paper_bgcolor='rgba(0,0,0,0.1)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )
    
    return fig, datos_jugador

def calcular_carga_cronica(datos_jugador, ventana_cronica=21):
    """
    Calcula la carga crónica de un jugador basada en una ventana de tiempo.
    
    Parámetros:
    - datos_jugador: DataFrame con los datos del jugador
    - ventana_cronica: Número de días/sesiones para calcular la carga crónica
    
    Retorna:
    - carga_cronica: Valor numérico de la carga crónica
    - datos_jugador: DataFrame con los datos procesados
    """
    # Verificar que tenemos datos válidos
    if datos_jugador is None or datos_jugador.empty:
        print("No hay datos para calcular carga crónica")
        return 0, pd.DataFrame()
        
    # Ponderación subjetiva para cada variable (mismos pesos que la carga aguda)
    pesos = {
        'Distance (km)': 1.0,
        'Sprint Distance (m)': 0.02,
        'Top Speed (m/s)': 10.0,
        'Accelerations Zone Count: 2 - 3 m/s/s': 1.5,
        'Accelerations Zone Count: 3 - 4 m/s/s': 2.0,
        'Accelerations Zone Count: > 4 m/s/s': 2.5,
        'Deceleration Zone Count: 2 - 3 m/s/s': 1.5,
        'Deceleration Zone Count: 3 - 4 m/s/s': 2.0,
        'Deceleration Zone Count: > 4 m/s/s': 2.5
    }

    # Asegurarse de que las fechas estén en formato datetime
    datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'])

    # Ordenar por fecha y aplicar la fórmula de carga diaria si no existe ya
    datos_jugador = datos_jugador.sort_values('Date').copy()
    
    # Verificar si ya existe la columna Carga_Diaria
    if 'Carga_Diaria' not in datos_jugador.columns:
        datos_jugador['Carga_Diaria'] = 0
        for var, peso in pesos.items():
            if var in datos_jugador.columns:
                # Verificar si los datos son numéricos y convertirlos si no lo son
                if datos_jugador[var].dtype == 'object':
                    try:
                        # Reemplazar comas por puntos si es necesario (formato europeo)
                        datos_jugador[var] = datos_jugador[var].str.replace(',', '.', regex=False)
                        datos_jugador[var] = pd.to_numeric(datos_jugador[var], errors='coerce')
                    except Exception as e:
                        print(f"Error al convertir {var}: {e}")
                        # Si hay error, usar 0
                        datos_jugador[var] = 0
                
                # Asegurarse que no hay valores NaN
                datos_temp = datos_jugador[var].fillna(0)
                # Multiplicar por el peso
                try:
                    datos_jugador['Carga_Diaria'] = datos_jugador['Carga_Diaria'] + (datos_temp * peso)
                except Exception as e:
                    print(f"Error al calcular carga para {var}: {e}")
                    # Ignorar esta variable si hay error

    # Calcular la carga crónica como media de los últimos 'ventana_cronica' días
    if 'Carga_Diaria' not in datos_jugador.columns or datos_jugador.empty:
        # Si no existe la columna o no hay datos, devolver 0
        carga_cronica = 0
    elif len(datos_jugador) >= ventana_cronica:
        # Tomar los últimos registros según la ventana
        datos_cronicos = datos_jugador.tail(ventana_cronica)
        
        # Verificar si los datos son numéricos
        if datos_cronicos['Carga_Diaria'].dtype == 'object':
            try:
                datos_cronicos['Carga_Diaria'] = pd.to_numeric(datos_cronicos['Carga_Diaria'], errors='coerce')
            except Exception as e:
                print(f"Error al convertir Carga_Diaria a numérico: {e}")
        
        # Calcular la media de la carga diaria, manejando valores nulos
        carga_cronica = datos_cronicos['Carga_Diaria'].fillna(0).mean()
    else:
        # Si no hay suficientes datos, usar todos los disponibles
        # Asegurarse de que sean numéricos
        if datos_jugador['Carga_Diaria'].dtype == 'object':
            try:
                datos_jugador['Carga_Diaria'] = pd.to_numeric(datos_jugador['Carga_Diaria'], errors='coerce')
            except Exception as e:
                print(f"Error al convertir Carga_Diaria a numérico: {e}")
        
        carga_cronica = datos_jugador['Carga_Diaria'].fillna(0).mean()
    
    # Verificar que la carga crónica sea un número válido
    if pd.isna(carga_cronica) or not isinstance(carga_cronica, (int, float)):
        print("La carga crónica calculada no es un número válido, retornando 0")
        carga_cronica = 0
    
    return carga_cronica, datos_jugador

# Función para cargar la foto del jugador
# Función para cargar la foto del jugador
def get_player_photo(player_name):
    """
    Intenta encontrar una foto del jugador en la carpeta de imágenes.
    Si no se encuentra, devuelve una imagen predeterminada.
    
    Parámetros:
    - player_name: Nombre del jugador
    
    Retorna:
    - path: Ruta a la imagen del jugador o a la imagen predeterminada
    """
    # Normalizar el nombre del jugador para coincidir con el formato de archivo
    # Convertir a minúsculas y reemplazar espacios por guiones bajos
    normalized_name = player_name.lower().replace(' ', '_')
    # Versión alternativa con mayúsculas iniciales
    title_case_name = '_'.join(word.capitalize() for word in player_name.split())
    
    # Ruta base a la carpeta de imágenes
    local_base_path = Path('assets/player_imagenes')
    
    # Lista de posibles formatos de nombre
    name_variations = [
        normalized_name,                 # todo minúsculas: cachaza_pato
        title_case_name,                 # mayúsculas iniciales: Cachaza_Pato
        player_name.replace(' ', '_')    # formato original con guiones: Cachaza_Pato o cachaza_pato
    ]
    
    # Verificación adicional: buscar en el directorio todas las imágenes
    try:
        # Listar todos los archivos en el directorio para depuración
        if local_base_path.exists():
            print(f"Archivos en {local_base_path}:")
            for file in local_base_path.glob('*'):
                print(f"  - {file.name}")
        else:
            print(f"La carpeta {local_base_path} no existe")
            # Intenta crear la carpeta si no existe
            os.makedirs(local_base_path, exist_ok=True)
            print(f"Carpeta {local_base_path} creada")
    except Exception as e:
        print(f"Error al listar archivos: {e}")
    
    # Comprobar diferentes combinaciones de nombre y formato de archivo localmente primero
    for name_var in name_variations:
        for ext in ['.jpg', '.jpeg', '.png']:
            # Verificamos primero si el archivo existe localmente
            local_img_path = local_base_path / f"{name_var}{ext}"
            if local_img_path.exists():
                # Dash sirve archivos de la carpeta assets directamente con el prefijo '/assets/'
                # pero nosotros ya incluimos 'assets/' en la ruta, así que lo quitamos
                return f"/assets/player_imagenes/{name_var}{ext}"
    
    # Si no se encuentra la imagen del jugador, devolver una imagen predeterminada
    default_img = local_base_path / "default_player.jpg"
    if default_img.exists():
        return "/assets/player_imagenes/default_player.jpg"
    else:
        # Si no existe ni siquiera la imagen predeterminada, usar otro recurso
        return "/assets/logo.png"
    
    
# Cargar los datos al iniciar la página
df_players = load_data()

# Verificar si tenemos datos y crear la lista de jugadores
if df_players is not None:
    # Obtener la lista de jugadores si existe la columna adecuada
    players_list = []
    if 'Player Name' in df_players.columns:
        players_list = sorted(df_players['Player Name'].unique())

# Definición del layout de la página
layout = dbc.Container([
    # Título principal
    html.H1("Dashboard de Jugadores", className="text-center mb-4", 
           style={"fontWeight": "800", "letterSpacing": "1px", "color": "#ffffff", 
                  "textShadow": "2px 2px 4px rgba(0,0,0,0.5)", "padding": "10px 0"}),
    
    # Primera fila: Tres columnas debajo del título
    dbc.Row([
        # Columna 1: Selector de jugador (sin título, solo dropdown)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # Selector dropdown de jugadores
                    dcc.Dropdown(
                        id='player-selector',
                        options=[{'label': player, 'value': player} for player in (players_list if 'players_list' in locals() else [])],
                        value=players_list[0] if 'players_list' in locals() and players_list else None,
                        clearable=False,
                        style={"marginBottom": "15px"}
                    ),
                    # Contenedor para estadísticas básicas del jugador
                    html.Div(id="player-stats-container", className="mt-2")
                ])
            ], style={"height": "100%", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=4),
        
        # Columna 2: Valores máximos históricos (sin título)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id="player-max-values")
                ])
            ], style={"height": "100%", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=4),
        
        # Columna 3: Foto del jugador (sin título)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id="player-photo", className="text-center", style={"height": "100%", "display": "flex", "alignItems": "center", "justifyContent": "center"})
                ])
            ], style={"height": "100%", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=4)
    ], className="mb-4 g-3"),
    
    # Segunda fila: Contenido principal
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Rendimiento de Jugadores", className="card-title mb-4",
                           style={"textAlign": "center", "fontWeight": "700", "color": "#ffffff",
                                 "textShadow": "1px 1px 3px rgba(0,0,0,0.3)"}),
                    
                    # Área para mostrar la carga (aguda y crónica)
                    dbc.Row([
                        dbc.Col([
                            html.H5("Carga Aguda (últimos 7 días)", className="text-center",
                                   style={"fontWeight": "600", "marginBottom": "15px"}),
                            html.Div([
                                dcc.Graph(id="player-acute-load-graph")
                            ], style={"height": "400px"})
                        ], width=6),
                        dbc.Col([
                            html.H5("Carga Crónica (últimos 21 días)", className="text-center",
                                   style={"fontWeight": "600", "marginBottom": "15px"}),
                            html.Div(id="player-chronic-load", className="mt-3 performance-chart", style={"height": "400px"})
                        ], width=6),
                    ], className="mt-4"),
                    
                    # Área para gráficos de distancia
                    html.H5("Evolución de Distancia", className="text-center mt-4",
                           style={"fontWeight": "600", "marginBottom": "15px", "marginTop": "30px"}),
                    dcc.Graph(id="player-distance-graph", className="performance-chart"),
                    
                    # Área para gráfico de sprint distance
                    html.H5("Evolución de Sprint Distance", className="text-center mt-4",
                           style={"fontWeight": "600", "marginBottom": "15px", "marginTop": "30px"}),
                    dcc.Graph(id="player-sprint-graph", className="performance-chart"),
                    
                    # Área para gráfico de aceleraciones y desaceleraciones
                    html.H5("Evolución de Aceleraciones y Desaceleraciones", className="text-center mt-4",
                           style={"fontWeight": "600", "marginBottom": "15px", "marginTop": "30px"}),
                    dcc.Graph(id="player-acceleration-graph", className="performance-chart"),
                ])
            ], style={"boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ])
    ])
], fluid=True, className="player-dashboard-container py-4", style={"backgroundColor": "#6c757d", "minHeight": "100vh"})

# Callback para actualizar las estadísticas del jugador
@callback(
    Output("player-stats-container", "children"),
    Input("player-selector", "value")
)
def update_player_stats(selected_player):
    """
    Actualiza las estadísticas básicas del jugador seleccionado
    """
    if not selected_player or df_players is None:
        return html.P("Seleccione un jugador para ver sus estadísticas.")
    
    # Filtrar datos para el jugador seleccionado
    player_data = df_players[df_players['Player Name'] == selected_player]
    
    if player_data.empty:
        return html.P(f"No se encontraron datos para {selected_player}")
    
    # Crear un resumen de estadísticas
    stats_card = dbc.Card([
        dbc.CardBody([
            html.H5(f"Estadísticas de {selected_player}", 
                   style={"textAlign": "center", "fontWeight": "700", "marginBottom": "15px", 
                          "borderBottom": "1px solid rgba(255,255,255,0.2)", "paddingBottom": "10px"}),
            html.Div([
                html.P([html.I(className="fas fa-calendar-check me-2"), f"Sesiones: {len(player_data)}"], 
                      style={"fontWeight": "600", "marginBottom": "8px"}),
                # Calculamos algunas estadísticas adicionales si están disponibles
                html.P([html.I(className="fas fa-running me-2"), 
                       f"Distancia promedio: {player_data['Distance km'].mean():.2f} km"], 
                      style={"fontWeight": "600", "marginBottom": "8px"}) if 'Distance km' in player_data.columns else None,
                html.P([html.I(className="fas fa-bolt me-2"), 
                       f"Velocidad promedio: {player_data['Vel Max'].mean():.2f} km/h"], 
                      style={"fontWeight": "600", "marginBottom": "8px"}) if 'Vel Max' in player_data.columns else None,
            ], className="ps-2")
        ], className="p-3")
    ], style={"backgroundColor": "#495057", "borderRadius": "8px", "boxShadow": "inset 0 0 10px rgba(0,0,0,0.2)"})
    
    return stats_card

# Callback para actualizar los valores máximos por jugador
@callback(
    Output("player-max-values", "children"),
    Input("player-selector", "value")
)
def update_max_values(selected_player):
    """
    Actualiza y muestra los valores máximos históricos del jugador seleccionado
    con visualización limpia y números grandes
    """
    if not selected_player or df_players is None:
        return html.P("Seleccione un jugador para ver sus valores máximos.")
    
    # Columnas para obtener valores máximos
    columnas = [
        'Distance (km)', 
        'Sprint Distance (m)', 
        'Top Speed (m/s)'
    ]
    
    # Obtener valores máximos para el jugador seleccionado
    max_values_df = obtener_valores_maximos_por_jugador(df_players, columnas, jugador=selected_player)
    
    if max_values_df.empty:
        return html.P(f"No se pudieron calcular valores máximos para {selected_player}")
    
    # Títulos más amigables para las métricas
    column_mapping = {
        'Máximo Distance (km)': 'Max Distance KM',
        'Máximo Sprint Distance (m)': 'Dist Sprint Mts',
        'Máximo Top Speed (m/s)': 'Vel Max'
    }
    
    # Colores para cada métrica
    colors = {
        'Max Distance km': "#2b3135",  # Azul
        'Max Sprint': "#3F3F3E",       # Rojo
        'Vel Max': "#323230"           # Verde
    }
    
    # Crear contenedor para las visualizaciones
    metrics_container = []
    
    # Para cada métrica, crear una visualización con número grande y título a la derecha
    for orig_col, new_col in column_mapping.items():
        if orig_col in max_values_df.columns:
            value = max_values_df.iloc[0][orig_col]
            if isinstance(value, (int, float)):
                value = round(value, 2)
            
            # Crear una fila con el valor y el título
            metric_row = dbc.Row([
                # Número grande a la izquierda
                dbc.Col(
                    html.Div(
                        f"{value}",
                        className="player-metric-value",
                        style={
                            "fontSize": "38px", 
                            "fontWeight": "800",
                            "color": colors.get(new_col, "#fff"),
                            "textAlign": "right",
                            "paddingRight": "15px",
                            "textShadow": "2px 2px 4px rgba(0,0,0,0.3)"
                        }
                    ),
                    width=6
                ),
                # Título a la derecha
                dbc.Col(
                    html.Div([
                        html.Div(
                            new_col, 
                            style={
                                "fontSize": "16px", 
                                "fontWeight": "600",
                                "color": "#ffffff",
                                "textAlign": "left",
                                "paddingTop": "12px",
                                "letterSpacing": "0.5px"
                            }
                        ),
                    ]),
                    width=6
                )
            ], className="mb-4 align-items-center")
            
            # Añadir línea separadora excepto después del último elemento
            metrics_container.append(metric_row)
            if orig_col != list(column_mapping.keys())[-1]:
                metrics_container.append(html.Hr(style={"margin": "10px 0", "border": "0", "height": "1px", "background-image": "linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.5), rgba(255,255,255,0))"}))
    
    # Devolver el contenedor con todas las métricas
    if metrics_container:
        return html.Div(metrics_container, className="p-3 player-stats-highlight", style={
            "backgroundColor": "#495057", 
            "borderRadius": "12px", 
            "boxShadow": "inset 0 0 10px rgba(0,0,0,0.2)",
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center"
        })
    else:
        return html.P("No se pudieron generar las visualizaciones")

# Callback para actualizar el gráfico de carga aguda
@callback(
    Output("player-acute-load-graph", "figure"),
    Input("player-selector", "value")
)
def update_acute_load(selected_player):
    """
    Crea y actualiza el gráfico tipo gauge para mostrar la carga aguda 
    (últimos 7 días) del jugador seleccionado
    """
    if not selected_player or df_players is None:
        return go.Figure()
    
    # Crear gráfico de gauge para la carga aguda
    fig, _ = crear_gauge_carga_aguda(df_players, selected_player)
    
    if fig is None:
        # Si no se pudo crear el gráfico, devolver una figura vacía
        fig = go.Figure()
        fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
    
    return fig

# Callback para actualizar el valor de carga crónica
@callback(
    Output("player-chronic-load", "children"),
    Input("player-selector", "value")
)
def update_chronic_load(selected_player):
    """
    Calcula y muestra la carga crónica (promedio de los últimos 21 días)
    del jugador seleccionado usando un medidor tipo gauge
    """
    if not selected_player or df_players is None:
        return html.P("Seleccione un jugador para ver su carga crónica.")
    
    # Filtrar datos para el jugador seleccionado
    player_data = df_players[df_players['Player Name'] == selected_player].copy()
    
    if player_data.empty:
        return html.P(f"No se encontraron datos para {selected_player}")
    
    # Calcular la carga crónica
    carga_cronica_valor, _ = calcular_carga_cronica(player_data)
    
    # Crear un medidor estilo speedometer para mostrar la carga crónica
    fig = go.Figure()
    
    # Definir el rango en función del valor (aproximadamente 2 veces el valor calculado)
    max_range = max(100, carga_cronica_valor * 2)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=carga_cronica_valor,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'color': 'white', 'size': 36, 'family': 'Arial, sans-serif'}},
        delta={'reference': carga_cronica_valor*0.9, 'relative': True, 'valueformat': '.1%', 'font': {'color': 'white'}},
        title={'text': "", 'font': {'color': 'white', 'size': 16}},
        gauge={
            'axis': {
                'range': [0, max_range], 
                'tickwidth': 1, 
                'tickcolor': "#ffffff",
                'tickfont': {'color': 'white'}
            },
            'bar': {'color': "darkblue"},
            'bgcolor': "rgba(0,0,0,0.2)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.5)",
            'steps': [
                {'range': [0, max_range*0.3], 'color': "lightgreen"},
                {'range': [max_range*0.3, max_range*0.7], 'color': "gold"},
                {'range': [max_range*0.7, max_range], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': carga_cronica_valor * 1.5
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font=dict(size=16, color='white'),
        margin=dict(l=40, r=40, t=40, b=40),  # Reducir margen superior
        paper_bgcolor='rgba(0,0,0,0.1)',
        plot_bgcolor='rgba(0,0,0,0.1)',
    )
    
    return dcc.Graph(figure=fig)

# Callback para actualizar el gráfico de distancia
@callback(
    Output("player-distance-graph", "figure"),
    Input("player-selector", "value")
)
def update_distance_graph(selected_player):
    if not selected_player or df_players is None:
        # Devolver un gráfico vacío
        return go.Figure()
    
    # Filtrar datos para el jugador seleccionado
    player_data = df_players[df_players['Player Name'] == selected_player].copy()
    
    if player_data.empty:
        return go.Figure()
    
    # Procesar los datos para el gráfico
    try:
        # Convertir fechas
        player_data['Date'] = pd.to_datetime(player_data['Date'], errors='coerce')
        player_data = player_data.sort_values('Date')
        
        # Verificar si tenemos la columna de distancia
        if 'Distance (km)' in player_data.columns:
            # Convertir a numérico si es necesario
            if player_data['Distance (km)'].dtype == 'object':
                player_data['Distance (km)'] = player_data['Distance (km)'].str.replace(',', '.', regex=False)
                player_data['Distance (km)'] = pd.to_numeric(player_data['Distance (km)'], errors='coerce')
            
            # Crear figura
            fig = go.Figure()
            
            # Añadir barras para distancia con texto en la parte superior
            fig.add_trace(go.Bar(
                x=player_data['Session Title'] if 'Session Title' in player_data.columns else player_data['Date'],
                y=player_data['Distance (km)'],
                name='Distancia',
                marker_color='rgb(70, 130, 180)',
                marker=dict(
                    line=dict(width=1, color='rgba(255, 255, 255, 0.4)')
                ),
                text=player_data['Distance (km)'].round(2),
                hovertemplate='<b>%{x}</b><br>Distancia: %{y:.2f} km<extra></extra>',
                textposition='outside'
            ))
            
            # Configurar layout sin título
            fig.update_layout(
                xaxis_title="Sesión",
                yaxis_title="Distancia (km)",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0.1)',
                paper_bgcolor='rgba(0,0,0,0.1)',
                margin=dict(l=40, r=40, t=20, b=60),  # Reducir el margen superior
                font=dict(color='white')
            )
            
            return fig
        else:
            return go.Figure()
    
    except Exception as e:
        print(f"Error al crear el gráfico: {e}")
        return go.Figure()

# Callback para actualizar la foto del jugador
@callback(
    Output("player-photo", "children"),
    Input("player-selector", "value")
)
def update_player_photo(selected_player):
    if not selected_player:
        return html.P("Seleccione un jugador para ver su foto")
    
    # Obtener la ruta de la imagen
    img_path = get_player_photo(selected_player)
    
    # Crear el componente de imagen
    return html.Img(
        src=img_path,
        style={
            "max-width": "100%", 
            "max-height": "320px", 
            "border-radius": "15px",
            "box-shadow": "0 8px 16px rgba(0,0,0,0.4)",
            "border": "3px solid white",
            "object-fit": "contain"
        },
        className="mt-2 player-photo-img"
    )

# Función para graficar la velocidad (sprint distance) del jugador
def graficar_velocidad_jugador(datos_jugador, n_sesiones=10):
    """
    Crea un gráfico de barras para visualizar la evolución de sprint distance
    por un jugador en sus últimas sesiones.
    
    Parámetros:
    - datos_jugador: DataFrame con los datos del jugador
    - n_sesiones: Número de últimas sesiones a mostrar
    
    Retorna:
    - fig: Figura de Plotly para visualización
    """
    # Verificar si hay datos válidos
    if datos_jugador is None or datos_jugador.empty:
        print("No hay datos para visualizar")
        return go.Figure()
    
    # Verificar que las columnas necesarias estén presentes
    columnas_requeridas = ['Date', 'Sprint Distance (m)']
    columnas_faltantes = [col for col in columnas_requeridas if col not in datos_jugador.columns]
    
    if columnas_faltantes:
        print(f"Faltan las siguientes columnas para el gráfico de velocidad: {', '.join(columnas_faltantes)}")
        return go.Figure()
    
    try:
        # Verificar tipo de datos de fecha
        if not pd.api.types.is_datetime64_dtype(datos_jugador['Date']):
            datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'], errors='coerce')
        
        # Eliminar filas donde Date es NaT (fecha no válida)
        datos_jugador = datos_jugador.dropna(subset=['Date'])
        
        # Verificar si aún quedan datos después de eliminar fechas no válidas
        if datos_jugador.empty:
            print("No hay fechas válidas en los datos")
            return go.Figure()
        
        # Ordenar datos por fecha de más reciente a más antigua
        datos_jugador_ordenado = datos_jugador.sort_values('Date', ascending=False)
        
        # Tomar las últimas n sesiones (ahora serán las más recientes)
        if len(datos_jugador_ordenado) > n_sesiones:
            datos_plot = datos_jugador_ordenado.head(n_sesiones)
        else:
            datos_plot = datos_jugador_ordenado.copy()
        
        # Reordenar para la visualización (de más antigua a más reciente para el eje X)
        datos_plot = datos_plot.sort_values('Date')
        
        # Verificar si 'Sprint Distance (m)' es numérico, si no, convertirlo
        if not pd.api.types.is_numeric_dtype(datos_plot['Sprint Distance (m)']):
            # Reemplazar comas por puntos si es necesario
            if datos_plot['Sprint Distance (m)'].dtype == 'object':
                datos_plot['Sprint Distance (m)'] = datos_plot['Sprint Distance (m)'].str.replace(',', '.', regex=False)
            datos_plot['Sprint Distance (m)'] = pd.to_numeric(datos_plot['Sprint Distance (m)'], errors='coerce')
            
        # Rellenar valores NaN en Sprint Distance con 0
        datos_plot['Sprint Distance (m)'] = datos_plot['Sprint Distance (m)'].fillna(0)
        
        # Asegurarse de que Session Title tenga valores (no nulos)
        if 'Session Title' in datos_plot.columns:
            datos_plot['Session Title'] = datos_plot['Session Title'].fillna('Sin título')
            # Crear etiquetas personalizadas con fecha y título de sesión
            etiquetas_eje_x = [f"{fecha.strftime('%d/%m/%Y')}<br><b>{sesion}</b>" 
                              for fecha, sesion in zip(datos_plot['Date'], datos_plot['Session Title'])]
        else:
            # Si no hay Session Title, usar solo las fechas
            etiquetas_eje_x = [fecha.strftime('%d/%m/%Y') for fecha in datos_plot['Date']]
        
        # Crear la figura para el gráfico
        fig = go.Figure()
        
        # Añadir barras para sprint distance
        fig.add_trace(go.Bar(
            x=etiquetas_eje_x,
            y=datos_plot['Sprint Distance (m)'],
            name='Sprint Distance',
            marker_color='#e74c3c',  # Rojo para sprint (mismo color que en los valores máximos)
            marker=dict(
                line=dict(width=1, color='rgba(255, 255, 255, 0.4)')
            ),
            text=datos_plot['Sprint Distance (m)'].round(2),
            hovertemplate='<b>%{x}</b><br>Sprint Distance: %{y:.2f} m<extra></extra>',
            textposition='outside'
        ))
        
        # Configurar layout sin título
        fig.update_layout(
            xaxis_title="Sesión",
            yaxis_title="Sprint Distance (m)",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            margin=dict(l=40, r=40, t=20, b=80),  # Reducir el margen superior
            font=dict(color='white'),
            xaxis=dict(
                tickangle=-45,  # Rotar etiquetas para mejor legibilidad
                tickfont=dict(size=10)
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error al crear el gráfico de velocidad: {e}")
        return go.Figure()

# Función para graficar aceleraciones y desaceleraciones del jugador
def graficar_aceleraciones_jugador(datos_jugador, n_sesiones=8):
    """
    Crea un gráfico de barras agrupadas para visualizar la evolución de aceleraciones
    y desaceleraciones por un jugador en sus últimas sesiones.
    
    Parámetros:
    - datos_jugador: DataFrame con los datos del jugador
    - n_sesiones: Número de últimas sesiones a mostrar
    
    Retorna:
    - fig: Figura de Plotly para visualización
    """
    # Verificar si hay datos válidos
    if datos_jugador is None or datos_jugador.empty:
        print("No hay datos para visualizar aceleraciones")
        return go.Figure()
    
    # Columnas de aceleraciones y desaceleraciones requeridas
    columnas_aceleraciones = [
        'Accelerations Zone Count: 2 - 3 m/s/s',
        'Accelerations Zone Count: 3 - 4 m/s/s',
        'Accelerations Zone Count: > 4 m/s/s'
    ]
    
    columnas_desaceleraciones = [
        'Deceleration Zone Count: 2 - 3 m/s/s',
        'Deceleration Zone Count: 3 - 4 m/s/s',
        'Deceleration Zone Count: > 4 m/s/s'
    ]
    
    # Verificar si las columnas necesarias están presentes
    columnas_requeridas = ['Date'] + columnas_aceleraciones + columnas_desaceleraciones
    columnas_faltantes = [col for col in columnas_requeridas if col not in datos_jugador.columns]
    
    if columnas_faltantes:
        print(f"Faltan las siguientes columnas para el gráfico de aceleraciones: {', '.join(columnas_faltantes)}")
        # Si faltan algunas columnas, verificar si al menos tenemos algunas para crear un gráfico parcial
        columnas_acc_disponibles = [col for col in columnas_aceleraciones if col in datos_jugador.columns]
        columnas_dec_disponibles = [col for col in columnas_desaceleraciones if col in datos_jugador.columns]
        
        if not columnas_acc_disponibles and not columnas_dec_disponibles:
            return go.Figure()  # No tenemos columnas suficientes para crear un gráfico
    
    try:
        # Verificar tipo de datos de fecha
        if not pd.api.types.is_datetime64_dtype(datos_jugador['Date']):
            datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'], errors='coerce')
        
        # Eliminar filas donde Date es NaT (fecha no válida)
        datos_jugador = datos_jugador.dropna(subset=['Date'])
        
        # Verificar si aún quedan datos después de eliminar fechas no válidas
        if datos_jugador.empty:
            print("No hay fechas válidas en los datos para aceleraciones")
            return go.Figure()
        
        # Ordenar datos por fecha de más reciente a más antigua
        datos_jugador_ordenado = datos_jugador.sort_values('Date', ascending=False)
        
        # Tomar las últimas n sesiones (ahora serán las más recientes)
        if len(datos_jugador_ordenado) > n_sesiones:
            datos_plot = datos_jugador_ordenado.head(n_sesiones)
        else:
            datos_plot = datos_jugador_ordenado.copy()
        
        # Reordenar para la visualización (de más antigua a más reciente para el eje X)
        datos_plot = datos_plot.sort_values('Date')
        
        # Procesar columnas numéricas y convertirlas si es necesario
        columnas_numericas = columnas_aceleraciones + columnas_desaceleraciones
        for col in columnas_numericas:
            if col in datos_plot.columns:
                if not pd.api.types.is_numeric_dtype(datos_plot[col]):
                    # Reemplazar comas por puntos si es necesario
                    if datos_plot[col].dtype == 'object':
                        datos_plot[col] = datos_plot[col].str.replace(',', '.', regex=False)
                    datos_plot[col] = pd.to_numeric(datos_plot[col], errors='coerce')
                
                # Rellenar valores NaN con 0
                datos_plot[col] = datos_plot[col].fillna(0)
        
        # Crear columnas sumarizadas para aceleraciones y desaceleraciones totales
        datos_plot['Total Aceleraciones'] = 0
        for col in columnas_aceleraciones:
            if col in datos_plot.columns:
                datos_plot['Total Aceleraciones'] += datos_plot[col]
        
        datos_plot['Total Desaceleraciones'] = 0
        for col in columnas_desaceleraciones:
            if col in datos_plot.columns:
                datos_plot['Total Desaceleraciones'] += datos_plot[col]
        
        # Asegurarse de que Session Title tenga valores (no nulos)
        if 'Session Title' in datos_plot.columns:
            datos_plot['Session Title'] = datos_plot['Session Title'].fillna('Sin título')
            # Crear etiquetas personalizadas con fecha y título de sesión
            etiquetas_eje_x = [f"{fecha.strftime('%d/%m/%Y')}<br><b>{sesion}</b>" 
                              for fecha, sesion in zip(datos_plot['Date'], datos_plot['Session Title'])]
        else:
            # Si no hay Session Title, usar solo las fechas
            etiquetas_eje_x = [fecha.strftime('%d/%m/%Y') for fecha in datos_plot['Date']]
        
        # Crear la figura para el gráfico
        fig = go.Figure()
        
        # Añadir barras para aceleraciones
        fig.add_trace(go.Bar(
            x=etiquetas_eje_x,
            y=datos_plot['Total Aceleraciones'],
            name='Aceleraciones',
            marker_color='#3498db',  # Azul para aceleraciones
            marker=dict(
                line=dict(width=1, color='rgba(255, 255, 255, 0.4)')
            ),
            text=datos_plot['Total Aceleraciones'].round(0).astype(int),
            hovertemplate='<b>%{x}</b><br>Aceleraciones: %{y}<extra></extra>',
            textposition='outside'
        ))
        
        # Añadir barras para desaceleraciones
        fig.add_trace(go.Bar(
            x=etiquetas_eje_x,
            y=datos_plot['Total Desaceleraciones'],
            name='Desaceleraciones',
            marker_color='#2ecc71',  # Verde para desaceleraciones
            marker=dict(
                line=dict(width=1, color='rgba(255, 255, 255, 0.4)')
            ),
            text=datos_plot['Total Desaceleraciones'].round(0).astype(int),
            hovertemplate='<b>%{x}</b><br>Desaceleraciones: %{y}<extra></extra>',
            textposition='outside'
        ))
        
        # Configurar layout sin título
        fig.update_layout(
            xaxis_title="Sesión",
            yaxis_title="Cantidad",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            margin=dict(l=40, r=40, t=50, b=80),  # Ajustar margen superior para la leyenda
            font=dict(color='white'),
            barmode='group',  # Barras agrupadas para comparar aceleraciones y desaceleraciones
            xaxis=dict(
                tickangle=-45,  # Rotar etiquetas para mejor legibilidad
                tickfont=dict(size=10)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error al crear el gráfico de aceleraciones: {e}")
        return go.Figure()

# Callback para mostrar el gráfico de velocidad (Sprint Distance)
@callback(
    Output("player-sprint-graph", "figure"),
    Input("player-selector", "value")
)
def update_sprint_graph(selected_player):
    """
    Actualiza el gráfico de Sprint Distance para el jugador seleccionado
    """
    if not selected_player or df_players is None:
        # Devolver un gráfico vacío
        return go.Figure()
    
    # Filtrar datos para el jugador seleccionado
    player_data = df_players[df_players['Player Name'] == selected_player].copy()
    
    if player_data.empty:
        return go.Figure()
    
    # Crear y devolver el gráfico de velocidad
    return graficar_velocidad_jugador(player_data, n_sesiones=8)

# Callback para mostrar el gráfico de aceleraciones y desaceleraciones
@callback(
    Output("player-acceleration-graph", "figure"),
    Input("player-selector", "value")
)
def update_acceleration_graph(selected_player):
    """
    Actualiza el gráfico de aceleraciones y desaceleraciones para el jugador seleccionado
    """
    if not selected_player or df_players is None:
        # Devolver un gráfico vacío
        return go.Figure()
    
    # Filtrar datos para el jugador seleccionado
    player_data = df_players[df_players['Player Name'] == selected_player].copy()
    
    if player_data.empty:
        return go.Figure()
    
    # Crear y devolver el gráfico de aceleraciones y desaceleraciones
    return graficar_aceleraciones_jugador(player_data, n_sesiones=8)