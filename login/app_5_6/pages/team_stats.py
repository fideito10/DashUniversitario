import dash
from dash import html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# Registrar la página
dash.register_page(__name__, path='/stats/team')

# Función para cargar datos
def load_data():
    try:
        sheet_id = "1kajUuZwL9l1suipRNy7t2g_SGJUcDVh6MeTu66yK-Z4"
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Leer directamente como DataFrame
        df = pd.read_csv(url)
        return df
    except Exception as e:
        return None


def procesar_sesiones(df, tipo_sesion='Partido', max_fechas=None, filtrar_por_tipo=True):
    """
    Procesa datos de sesiones específicas (Partido, Martes o Jueves)
    
    Parámetros:
    - df: DataFrame con los datos
    - tipo_sesion: Tipo de sesión a procesar ('Partido', 'Martes', 'Jueves')
    - max_fechas: Número máximo de fechas a mostrar (None para mostrar todas)
    - filtrar_por_tipo: Si es True, filtra por tipo_sesion; si es False, usa todos los datos
    """
    # Verificar si el DataFrame es válido
    if df is None or df.empty:
        # En Dash, no usamos st.warning. Simplemente retornamos None
        return None
    
    # Crear una copia para no modificar el original
    sesiones_df = df.copy()
    
    # Filtrar solo los registros que corresponden al tipo de sesión si se especifica
    if filtrar_por_tipo:
        sesiones_df = sesiones_df[sesiones_df['Session Title'].str.contains(tipo_sesion, case=False, na=False)]
    
    if sesiones_df.empty:
        # En Dash, no usamos st.warning. Simplemente retornamos None
        return None
    
    # Resto del código igual...
    try:
        # Convertir fecha a datetime si no lo está
        if not pd.api.types.is_datetime64_dtype(sesiones_df['Date']):
            sesiones_df['Date'] = pd.to_datetime(sesiones_df['Date'], errors='coerce')
        
        # Convertir distancia a numérica
        if isinstance(sesiones_df['Distance (km)'].iloc[0], str):
            sesiones_df['Distance (km)'] = sesiones_df['Distance (km)'].str.replace(',', '.').astype(float)
        elif not pd.api.types.is_numeric_dtype(sesiones_df['Distance (km)']):
            sesiones_df['Distance (km)'] = pd.to_numeric(sesiones_df['Distance (km)'], errors='coerce')
    except Exception as e:
        # En Dash, no usamos st.error. Simplemente retornamos None
        return None
    
    # Agrupar por fecha y obtener estadísticas
    sesiones_stats = sesiones_df.groupby([sesiones_df['Date'].dt.date, 'Session Title'])['Distance (km)'].agg(
        ['mean', 'count', 'sum']).reset_index()
    
    sesiones_stats.columns = ['Fecha', 'Sesión', 'Distancia Media (km)', 
                              'N° Jugadores', 'Distancia Total (km)']
    
    # Ordenar por fecha ascendente
    sesiones_stats = sesiones_stats.sort_values('Fecha', ascending=True)
    
    # Crear una columna de identificación única que combine fecha y sesión
    sesiones_stats['Fecha_Sesion'] = sesiones_stats['Sesión']  # Solo usar el título de la sesión
    
    # Limitar por cantidad de fechas si se especifica
    if max_fechas and max_fechas > 0 and max_fechas < len(sesiones_stats):
        # Tomar los últimos N registros (los más recientes)
        sesiones_stats = sesiones_stats.tail(max_fechas)
    
    return sesiones_stats



def graficar_distancia_sesion(sesiones_stats, n_sesiones, tipo_sesion):
    """
    Genera el gráfico de distancia por sesiones
    """
    if sesiones_stats is None or sesiones_stats.empty:
        # En Dash, no usamos st.warning. Simplemente retornamos un dict vacío
        return {}
    
    # Limitar cantidad de sesiones a mostrar
    # Tomamos las últimas n_sesiones (más recientes)
    df_plot = sesiones_stats.tail(n_sesiones).copy()

    # Crear figura con Plotly
    fig = go.Figure()
      # Añadir barras para distancia media
    fig.add_trace(go.Bar(
        x=df_plot['Fecha_Sesion'],
        y=df_plot['Distancia Media (km)'],
        name='Distancia Media',
        marker_color='rgb(70, 130, 180)',  # Azul más atractivo
        marker=dict(
            line=dict(width=1, color='rgba(255, 255, 255, 0.4)')
        ),
        text=df_plot['Distancia Media (km)'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Distancia Media: %{y:.2f} km<br>N° Jugadores: %{customdata[0]}<br>Total: %{customdata[1]:.2f} km<extra></extra>',
        customdata=df_plot[['N° Jugadores', 'Distancia Total (km)']]
    ))
    
    # Agregar línea de tendencia
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha_Sesion'],
        y=df_plot['Distancia Media (km)'],
        name='Tendencia',
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.7)', width=3, shape='spline'),
        hovertemplate='<b>%{x}</b><br>Tendencia: %{y:.2f} km<extra></extra>'
    ))    # Actualizar diseño
    fig.update_layout(
    # Eliminamos el título para evitar duplicación con el CardHeader
    xaxis=dict(
        title='Sesión',
        tickangle=-30,
        tickfont=dict(size=12, color='white'),
        title_font=dict(size=14, color='white')
    ),
    yaxis=dict(
        title='Distancia (km)',
        gridcolor='rgba(255, 255, 255, 0.2)',
        title_font=dict(color='white'),
        tickfont=dict(color='white')
    ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='white')),
        height=400,
        margin=dict(l=40, r=40, t=60, b=60)
    )
    
    return fig


def graficar_velocidad_alta(velocidad_stats, n_sesiones, tipo_sesion):
    """
    Genera el gráfico de velocidad alta por sesiones
    """
    if velocidad_stats is None or velocidad_stats.empty:
        return {}
    
    # Limitar cantidad de sesiones a mostrar
    # Tomamos las últimas n_sesiones (más recientes)
    df_plot = velocidad_stats.tail(n_sesiones).copy()
    
    # Crear figura con Plotly
    fig = go.Figure()      # Añadir barras para velocidad alta media
    fig.add_trace(go.Bar(
        x=df_plot['Fecha_Sesion'],
        y=df_plot['Velocidad Alta Media (m)'],
        name='Velocidad Alta Media',
        marker_color='#e74c3c',  # Rojo para sprint
        marker=dict(
            line=dict(width=1, color='rgba(255, 255, 255, 0.4)')
        ),
        text=df_plot['Velocidad Alta Media (m)'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Velocidad Alta Media: %{y:.2f} m<br>N° Jugadores: %{customdata[0]}<br>Total: %{customdata[1]:.2f} m<extra></extra>',
        customdata=df_plot[['N° Jugadores', 'Velocidad Alta Total (m)']]
    ))
    
    # Agregar línea de tendencia
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha_Sesion'],
        y=df_plot['Velocidad Alta Media (m)'],
        name='Tendencia',
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.7)', width=3, shape='spline'),
        hovertemplate='<b>%{x}</b><br>Tendencia: %{y:.2f} m<extra></extra>'
    ))    # Actualizar diseño
    fig.update_layout(
        # Eliminamos el título para evitar duplicación con el CardHeader
        xaxis=dict(
            title='Sesión',
            tickangle=-30,
            tickfont=dict(size=12, color='white'),
            title_font=dict(size=14, color='white')
        ),        
        yaxis=dict(
            title='Sprint Distance (m)',
            gridcolor='rgba(255, 255, 255, 0.2)',
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='white')),
        height=400,
        margin=dict(l=40, r=40, t=60, b=60)
    )
    
    return fig


def obtener_estadisticas_velocidad(df):
    """
    Procesa datos de Sprint Distance (m) por sesiones
    
    Parámetros:
    - df: DataFrame con los datos
    
    Retorna un DataFrame con las estadísticas de velocidad alta por sesión
    """
    # Verificar si el DataFrame es válido
    if df is None or df.empty:
        return None
    
    # Crear una copia para no modificar el original
    velocidad_df = df.copy()
    
    try:
        # Convertir fecha a datetime si no lo está
        if not pd.api.types.is_datetime64_dtype(velocidad_df['Date']):
            velocidad_df['Date'] = pd.to_datetime(velocidad_df['Date'], errors='coerce')
          # Verificar si 'Sprint Distance (m)' es numérico, si no, convertirlo
        if not pd.api.types.is_numeric_dtype(velocidad_df['Sprint Distance (m)']):
            if velocidad_df['Sprint Distance (m)'].dtype == 'object':
                velocidad_df['Sprint Distance (m)'] = velocidad_df['Sprint Distance (m)'].str.replace(',', '.', regex=False)
            velocidad_df['Sprint Distance (m)'] = pd.to_numeric(velocidad_df['Sprint Distance (m)'], errors='coerce')
        
        # Ya no convertimos a kilómetros, usamos directamente metros
        
        # Agrupar por fecha y obtener estadísticas
        velocidad_stats = velocidad_df.groupby([velocidad_df['Date'].dt.date, 'Session Title'])['Sprint Distance (m)'].agg(
            ['mean', 'count', 'sum']).reset_index()
        
        velocidad_stats.columns = ['Fecha', 'Sesión', 'Velocidad Alta Media (m)', 
                                'N° Jugadores', 'Velocidad Alta Total (m)']
        
        # Ordenar por fecha ascendente
        velocidad_stats = velocidad_stats.sort_values('Fecha', ascending=True)
        
        # Crear una columna de identificación única que combine fecha y sesión
        velocidad_stats['Fecha_Sesion'] = velocidad_stats['Sesión']  # Solo usar el título de la sesión
        
        return velocidad_stats
    
    except Exception as e:
        print(f"Error al procesar datos de velocidad alta: {e}")
        return None


 
def obtener_estadisticas_aceleraciones(sesiones_df):
   
    try:
        # Columnas de aceleraciones
        columnas_aceleraciones = [
            'Accelerations Zone Count: 2 - 3 m/s/s', 
            'Accelerations Zone Count: 3 - 4 m/s/s', 
            'Accelerations Zone Count: > 4 m/s/s'
        ]
        
        # Columnas de desaceleraciones
        columnas_desaceleraciones = [ 
            'Deceleration Zone Count: 2 - 3 m/s/s', 
            'Deceleration Zone Count: 3 - 4 m/s/s', 
            'Deceleration Zone Count: > 4 m/s/s'
        ]
        
        # Convertir columnas a numéricas si es necesario
        for col in columnas_aceleraciones + columnas_desaceleraciones:
            if isinstance(sesiones_df[col].iloc[0], str):
                sesiones_df[col] = sesiones_df[col].str.replace(',', '.').astype(float)
            elif not pd.api.types.is_numeric_dtype(sesiones_df[col]):
                sesiones_df[col] = pd.to_numeric(sesiones_df[col], errors='coerce')
        
        # Crear columnas con la suma de aceleraciones y desaceleraciones
        sesiones_df['Total Aceleraciones'] = sesiones_df[columnas_aceleraciones].sum(axis=1)
        sesiones_df['Total Desaceleraciones'] = sesiones_df[columnas_desaceleraciones].sum(axis=1)
        
        # Cambiar: Agrupar por fecha y Session Title para mantener el orden cronológico
        aceleraciones_stats = sesiones_df.groupby(['Date', 'Session Title'])[
            ['Total Aceleraciones', 'Total Desaceleraciones']].agg(['mean', 'count']).reset_index()
        
        # Reorganizar columnas para facilitar su uso
        aceleraciones_stats.columns = [
            'Fecha', 'Sesión',
            'Media Aceleraciones', 'Count Aceleraciones',
            'Media Desaceleraciones', 'Count Desaceleraciones'
        ]
        
        # Ordenar por fecha (cronológicamente)
        aceleraciones_stats = aceleraciones_stats.sort_values('Fecha', ascending=True)
        
        return aceleraciones_stats
        
    except Exception as e:
        print(f"Error al obtener estadísticas de aceleraciones: {e}")
        return None
    
    

# Layout completo con selector de tipo de sesión
layout = dbc.Container([      dbc.Row([
        dbc.Col([
            html.H2("Estadísticas del Equipo", className="text-center my-3", 
                   style={"fontWeight": "800", "letterSpacing": "1px", "color": "#ffffff", 
                          "textShadow": "2px 2px 4px rgba(0,0,0,0.5)", "padding": "10px 0"}),
            html.Hr(className="my-2", style={"borderColor": "rgba(255, 255, 255, 0.3)"})
        ], width=12)
    ]),
    
    # Selector de tipo de sesión y número de sesiones
    dbc.Row([
        dbc.Col([            dbc.Card([
                dbc.CardHeader("Filtros de Análisis", className="text-white font-weight-bold", 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),                dbc.CardBody([
                    html.Label("Seleccione el tipo de sesión:", className="text-white mb-2"),
                    dcc.Dropdown(
                        id='tipo-sesion-dropdown',
                        options=[
                            {'label': 'Partido', 'value': 'Partido'},
                            {'label': 'Entrenamiento Martes', 'value': 'Martes'},
                            {'label': 'Entrenamiento Jueves', 'value': 'Jueves'},
                            {'label': 'Toda la Semana', 'value': 'Semana'}
                        ],
                        value='Partido',
                        clearable=False,
                        className="mb-3",
                        style={"backgroundColor": "#343a40", "color": "white"}
                    ),
                    html.Label("Número de sesiones a mostrar:", className="text-white mb-2"),
                    dcc.Slider(
                        id='n-sesiones-slider',
                        min=1,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(1, 11)},
                        className="mb-3"
                    ),
                    html.Div(id="n-sesiones-output", className="text-white mt-2")
                ])
            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=12)
    ]),    # Gráficos de distancia y velocidad alta (en la misma fila)
    dbc.Row([
        # Gráfico de distancia (mitad izquierda)
        dbc.Col([            dbc.Card([
                dbc.CardHeader(html.Div(id="titulo-distancia"), 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),
                dbc.CardBody([
                    dcc.Graph(id="grafico-distancia"),
                ])
            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=6),  # Ancho de 6 (mitad de la fila)
        
        # Gráfico de velocidad alta (mitad derecha)
        dbc.Col([            dbc.Card([
                dbc.CardHeader(html.Div(id="titulo-velocidad-alta"), 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),
                dbc.CardBody([
                    dcc.Graph(id="grafico-velocidad-alta"),
                ])            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=6)  # Ancho de 6 (mitad de la fila)
    ]),
    
    # Nueva fila para gráfico de aceleraciones
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Div(id="titulo-aceleraciones"), 
                              style={"backgroundColor": "#414851", "borderBottom": "1px solid rgba(255,255,255,0.2)"}),
                dbc.CardBody([
                    dcc.Graph(id="grafico-aceleraciones"),
                ])
            ], className="mb-4", style={"backgroundColor": "#495057", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"})
        ], width=12)
    ]),
    
    # Espacio para mensaje de error
    dbc.Row([
        dbc.Col([
            html.Div(id="mensaje-error", className="alert alert-danger", style={"display": "none"})
        ], width=12)
    ])
], fluid=True, style={"backgroundColor": "#6c757d", "minHeight": "100vh"})

# Callbacks para la interactividad
@callback(
    [Output('grafico-distancia', 'figure'),
     Output('grafico-velocidad-alta', 'figure'),
     Output('grafico-aceleraciones', 'figure'),
     Output('mensaje-error', 'children'),
     Output('mensaje-error', 'style'),
     Output('n-sesiones-output', 'children'),
     Output('titulo-distancia', 'children'),
     Output('titulo-velocidad-alta', 'children'),
     Output('titulo-aceleraciones', 'children')],
    [Input('tipo-sesion-dropdown', 'value'),
     Input('n-sesiones-slider', 'value')]
)

def actualizar_grafico(tipo_sesion, n_sesiones):
    """
    Callback para actualizar el gráfico de distancia según el tipo de sesión seleccionado
    """    # Mostrar el valor del slider
    slider_output = f"Mostrando últimas {n_sesiones} sesiones"
      # Cargar los datos
    df = load_data()
    if df is None:
        empty_fig = {}
        error_msg = "Error al cargar los datos. Verifique la conexión."
        error_style = {"display": "block"}
        empty_title = ""
        return empty_fig, empty_fig, empty_fig, error_msg, error_style, slider_output, empty_title, empty_title, empty_title
    
    # Procesar los datos según el tipo seleccionado
    if tipo_sesion == "Semana":
        # Para "Semana", procesamos las últimas sesiones sin importar el tipo
        df_ordenado = df.copy()
        if not pd.api.types.is_datetime64_dtype(df_ordenado['Date']):
            df_ordenado['Date'] = pd.to_datetime(df_ordenado['Date'], errors='coerce')
        
        # Tomamos las últimas n_sesiones fechas (según el slider)
        ultimas_fechas = sorted(df_ordenado['Date'].unique())[-n_sesiones:]
        df_ultimas = df_ordenado[df_ordenado['Date'].isin(ultimas_fechas)]
          # Procesar datos para ambos gráficos
        sesiones_stats = procesar_sesiones(df_ultimas, tipo_sesion="Semana", filtrar_por_tipo=False)
        velocidad_stats = obtener_estadisticas_velocidad(df_ultimas)
        aceleraciones_stats = obtener_estadisticas_aceleraciones(df_ultimas)
    else:
        # Para los demás tipos, filtramos por tipo de sesión
        df_filtrado = df[df['Session Title'].str.contains(tipo_sesion, case=False, na=False)]
          # Verificar si hay datos filtrados
        if df_filtrado.empty:
            empty_fig = {}
            error_msg = f"No se encontraron datos para el tipo de sesión: {tipo_sesion}"
            error_style = {"display": "block"}
            empty_title = ""
            return empty_fig, empty_fig, empty_fig, error_msg, error_style, slider_output, empty_title, empty_title, empty_title
              # Convertir fecha a datetime para el procesamiento
        if not pd.api.types.is_datetime64_dtype(df_filtrado['Date']):
            df_filtrado['Date'] = pd.to_datetime(df_filtrado['Date'], errors='coerce')
            
        # Procesar datos para ambos gráficos
        sesiones_stats = procesar_sesiones(df_filtrado, tipo_sesion=tipo_sesion)
        velocidad_stats = obtener_estadisticas_velocidad(df_filtrado)
        aceleraciones_stats = obtener_estadisticas_aceleraciones(df_filtrado)
    
    # Verificar si hay datos    
    if sesiones_stats is None or sesiones_stats.empty:
        empty_fig = {}
        error_msg = f"No se encontraron datos para el tipo de sesión: {tipo_sesion}"
        error_style = {"display": "block"}
        empty_title = ""
        return empty_fig, empty_fig, empty_fig, error_msg, error_style, slider_output, empty_title, empty_title, empty_title
      # Generar gráfico de distancia
    fig_distancia = graficar_distancia_sesion(sesiones_stats, n_sesiones, tipo_sesion)
    
    # Crear títulos dinámicos para las cards    
    titulo_distancia = html.H5(f"    Distancia Km - {tipo_sesion}", className="mb-0 text-white", 
                              style={"fontWeight": "600", "letterSpacing": "0.5px"})
    titulo_velocidad = html.H5(f"    Sprint Distance Metros - {tipo_sesion}", className="mb-0 text-white", 
                              style={"fontWeight": "600", "letterSpacing": "0.5px"})
    titulo_aceleraciones = html.H5(f"    Aceleraciones y Desaceleraciones - {tipo_sesion}", className="mb-0 text-white", 
                              style={"fontWeight": "600", "letterSpacing": "0.5px"})
      # Inicializar gráficos vacíos para los casos de error
    fig_velocidad = {}
    fig_aceleraciones = {}
    error_msg = ""
    error_style = {"display": "none"}
    
    # Verificar si hay datos de velocidad    
    if velocidad_stats is None or velocidad_stats.empty:
        error_msg = "No se encontraron datos de velocidad para las sesiones seleccionadas."
        error_style = {"display": "block"}
    else:
        # Generar gráfico de velocidad alta
        fig_velocidad = graficar_velocidad_alta(velocidad_stats, n_sesiones, tipo_sesion)
    
    # Verificar si hay datos de aceleraciones
    if aceleraciones_stats is not None and not aceleraciones_stats.empty:
        # Generar gráfico de aceleraciones
        fig_aceleraciones = graficar_aceleraciones(aceleraciones_stats, n_sesiones, tipo_sesion)
    else:
        # Si ya hay un error de velocidad, añadimos a ese mensaje
        if error_msg:
            error_msg += " Tampoco se encontraron datos de aceleraciones."
        else:
            error_msg = "No se encontraron datos de aceleraciones para las sesiones seleccionadas."
            error_style = {"display": "block"}
    
    return fig_distancia, fig_velocidad, fig_aceleraciones, error_msg, error_style, slider_output, titulo_distancia, titulo_velocidad, titulo_aceleraciones


def graficar_aceleraciones(aceleraciones_stats, n_sesiones, tipo_sesion):
    """
    Genera el gráfico de aceleraciones y desaceleraciones por sesiones
    """
    if aceleraciones_stats is None or aceleraciones_stats.empty:
        return {}
    
    # Limitar cantidad de sesiones a mostrar
    # Tomamos las últimas n_sesiones (más recientes)
    df_plot = aceleraciones_stats.tail(n_sesiones).copy()
    
    # Crear figura con Plotly
    fig = go.Figure()    # Añadir barras para aceleraciones
    fig.add_trace(go.Bar(
        x=df_plot['Sesión'],
        y=df_plot['Media Aceleraciones'],
        name='Aceleraciones',
        marker_color='#1a1a2e',  # Azul muy oscuro casi negro
        marker=dict(
            line=dict(width=1, color='rgba(255, 255, 255, 0.5)'),
            opacity=0.9
        ),
        text=df_plot['Media Aceleraciones'].round(2),
        textposition='outside',
        textfont=dict(color='rgba(255, 255, 255, 0.9)'),
        hovertemplate='<b>%{x}</b><br>Media Aceleraciones: %{y:.2f}<br>N° Jugadores: %{customdata}<extra></extra>',
        customdata=df_plot['Count Aceleraciones']
    ))
    
    # Añadir barras para desaceleraciones
    fig.add_trace(go.Bar(
        x=df_plot['Sesión'],
        y=df_plot['Media Desaceleraciones'],
        name='Desaceleraciones',
        marker_color='#16213e',  # Azul oscuro
        marker=dict(
            line=dict(width=1, color='rgba(255, 255, 255, 0.5)'),
            opacity=0.9
        ),
        text=df_plot['Media Desaceleraciones'].round(2),
        textposition='outside',
        textfont=dict(color='rgba(255, 255, 255, 0.9)'),
        hovertemplate='<b>%{x}</b><br>Media Desaceleraciones: %{y:.2f}<br>N° Jugadores: %{customdata}<extra></extra>',
        customdata=df_plot['Count Desaceleraciones']
    ))
    
    # Actualizar diseño
    fig.update_layout(
        xaxis=dict(
            title='Sesión',
            tickangle=-30,
            tickfont=dict(size=12, color='white'),
            title_font=dict(size=14, color='white')
        ),        
        yaxis=dict(
            title='Cantidad',
            gridcolor='rgba(255, 255, 255, 0.2)',
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='white')),
        height=400,
        margin=dict(l=40, r=40, t=30, b=60)
    )
    
    return fig
