import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/team/analysis')

layout = dbc.Container([
    html.H1("Análisis del Equipo", className="text-center mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Análisis Detallado"),
                    html.P("Aquí encontrarás análisis detallados sobre el rendimiento y estrategias del equipo...")
                ])
            ])
        ])
    ])
])
