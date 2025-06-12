import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import os

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ]
)

# Configuración para despliegue en Render
server = app.server

# Crear el menú animado
animated_menu = html.Div([
    # Menú principal
    dbc.Nav([
        # Inicio
        dbc.NavLink([
            html.I(className="fas fa-home me-2"),
            html.Span("Inicio")
        ], 
        href="/",
        className="menu-link"),

        # Estadísticas con submenú
        html.Div([
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                html.Span("Fisicas"),
                html.I(className="fas fa-chevron-down ms-2")
            ],
            href="#",
            id="stats-dropdown",
            className="menu-link"),
            
            # Submenú de fisicas
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavLink("General", href="/stats/general", className="submenu-link"),
                    dbc.NavLink("Equipo", href="/stats/team", className="submenu-link"),
                    dbc.NavLink("Jugadores", href="/stats/players", className="submenu-link"),
                ],
                vertical=True,
                className="submenu_"),
                id="stats-submenu",
            )
        ], className="menu-item-with-submenu"),

        # Equipo con submenú
        html.Div([
            dbc.NavLink([
                html.I(className="fas fa-users me-2"),
                html.Span("Equipo"),
                html.I(className="fas fa-chevron-down ms-2")
            ],
            href="#",
            id="team-dropdown",
            className="menu-link"),
            
            # Submenú de equipo
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavLink("Plantilla", href="/team/roster", className="submenu-link"),
                    dbc.NavLink("Formación", href="/team/formation", className="submenu-link"),
                    dbc.NavLink("Análisis", href="/team/analysis", className="submenu-link"),
                ],
                vertical=True,
                className="submenu_"),
                id="team-submenu",
            )
        ], className="menu-item-with-submenu"),
    ],
    vertical=True,
    className="custom-menu")
], className="menu-container")

# Layout principal
app.layout = html.Div([
    animated_menu,
    html.Div(
        dash.page_container,
        id="page-content",
        className="content-container"
    )
])

# Callbacks para los submenús
@app.callback(
    Output("stats-submenu", "is_open"),
    [Input("stats-dropdown", "n_clicks")],
    [State("stats-submenu", "is_open")]
)
def toggle_stats_submenu(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("team-submenu", "is_open"),
    [Input("team-dropdown", "n_clicks")],
    [State("team-submenu", "is_open")]
)
def toggle_team_submenu(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    # Obtener el puerto del entorno (usado por Render) o usar 8050 por defecto
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port, debug=False)