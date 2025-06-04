import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')

layout = dbc.Container([
    html.H1("Bienvenido al Club Deportivo", className="text-center mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Próximo Partido"),
                    html.P("vs. Equipo Rival - 15/12/2024")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Últimos Resultados"),
                    html.P("Victoria 2-1 vs. Equipo A")
                ])
            ])
        ], width=6)
    ])
])
