import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

# Registrar la página
dash.register_page(__name__, path='/', name='Inicio')

def layout():
    return html.Div([
        # Contenido principal (ahora visible)
        html.Div([
            # Título del Club con logos
            html.Div([
                # Contenedor con logos y título
                html.Div([
                    # Logo izquierdo (Universidad)
                    html.Img(
                        src="/assets/uni.jpg",
                        style={
                            'height': '80px',
                            'width': 'auto',
                            'marginRight': '20px'
                        }
                    ),
                    
                    # Título central
                    html.Div([
                        html.H1("Club Universitario de La Plata", 
                               className="text-center mb-2 text-white fw-bold"),  # Cambiado a text-white
                        html.Hr(className="mx-auto", style={'width': '200px', 'height': '3px'})
                    ], style={'flex': '1'}),
                    
                    # Logo derecho (Club)
                    html.Img(
                        src="/assets/logo.png",
                        style={
                            'height': '80px',
                            'width': 'auto',
                            'marginLeft': '20px'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'marginBottom': '30px'
                })
            ], className="mb-5"),
            
            # Explicación del Sistema GPS
            html.Div([
                html.H3("¿Qué es nuestro Sistema de Análisis GPS?", 
                       className="text-center mb-4 text-white"),  # Cambiado a text-white
                html.Div([
                    html.P([
                        "Nuestro sistema utiliza ",
                        html.Strong("tecnología GPS de última generación"),
                        " para monitorear el rendimiento de cada jugador durante entrenamientos y partidos."
                    ], className="text-center mb-3 text-white", style={'color': 'white', 'backgroundColor': 'transparent'}),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.I(className="fas fa-satellite-dish fa-3x text-info mb-3"),
                                    html.H5("Seguimiento Preciso", className="text-info"),
                                    html.P("Cada jugador lleva un dispositivo GPS que registra su posición exacta, velocidad y movimientos cada segundo del juego.")
                                ], className="text-center")
                            ], className="h-100 border-0 shadow-sm", style={'backgroundColor': '#333333', 'color': 'white'})
                        ], md=4, className="mb-4"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.I(className="fas fa-chart-line fa-3x text-success mb-3"),
                                    html.H5("Datos Inteligentes", className="text-success"),
                                    html.P("Convertimos millones de puntos de datos en información útil: distancias recorridas, intensidad del esfuerzo, zonas de calor y patrones de juego.")
                                ], className="text-center")
                            ], className="h-100 border-0 shadow-sm", style={'backgroundColor': '#333333', 'color': 'white'})
                        ], md=4, className="mb-4"),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.I(className="fas fa-trophy fa-3x text-warning mb-3"),
                                    html.H5("Mejora Continua", className="text-warning"),
                                    html.P("Los entrenadores pueden tomar decisiones basadas en datos reales para optimizar entrenamientos, prevenir lesiones y maximizar el rendimiento.")
                                ], className="text-center")
                            ], className="h-100 border-0 shadow-sm", style={'backgroundColor': '#333333', 'color': 'white'})
                        ], md=4, className="mb-4")
                    ]),
                    
                    html.Div([
                        html.P([
                            html.I(className="fas fa-lightbulb text-warning me-2"),
                            html.Strong("¿Por qué es importante? "),
                            "En l deporte,diferencia entre ganar y perder a menudo se encuentra en los detalles. ",
                            "Nuestro sistema GPS revela esos detalles invisibles al ojo humano, permitiendo un análisis científico ",
                            "del rendimiento que transforma datos en victorias."
                        ], className="text-center fst-italic p-3 rounded", style={'backgroundColor': '#333333', 'color': 'white'})
                    ], className="mt-4")
                ], className="container")
            ], className="mb-5"),
            
            # Footer
            html.Hr(className="my-5"),
            html.Div([
                html.P([
                    "Desarrollado para el análisis profesional de rendimiento deportivo | ",
                    html.Strong("Tecnología GPS"),
                    " | Datos reales para tomar decisiones"
                ], className="text-center text-white")  # Cambiado de text-muted a text-white
            ], style={'backgroundColor': '#333333', 'padding': '20px', 'borderRadius': '8px'})  # Agregado estilo de fondo negro
        ], id="main-content", className="container-fluid mt-4", style={'display': 'block'}),
    ])

