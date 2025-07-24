import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import os

# Crear la aplicación PRIMERO
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ]
)

server = app.server

# IMPORTAR las páginas para que se registren automáticamente
import pages.home
import pages.stats  
import pages.players
import pages.analysis

# Debug: Verificar páginas registradas
print("Páginas registradas:")
for page_path, page_info in dash.page_registry.items():
    print(f"  - {page_path}: {page_info.get('title', 'Sin título')}")

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

        # Físicas con submenú
        html.Div([
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                html.Span("Físicas"),
                html.I(className="fas fa-chevron-down ms-2")
            ],
            href="#",
            id="stats-dropdown",
            className="menu-link"),
            
            # Submenú de físicas
              dbc.Collapse(
                dbc.Nav([
                    dbc.NavLink([
                        html.I(className="fas fa-chart-bar me-2"),
                        "Estadísticas"
                    ], href="/team/analysis", className="submenu-link"),  # Cambiado de /analysis a /team/analysis
                    dbc.NavLink([
                        html.I(className="fas fa-user-circle me-2"),
                        "Jugadores"
                    ], href="/stats/players", className="submenu-link"),  # Cambiado de /players a /stats/players
                    dbc.NavLink([
                        html.I(className="fas fa-users me-2"),
                        "Equipo"
                    ], href="/stats/general", className="submenu-link"),  # Cambiado de /stats a /stats/general
                ],
                vertical=True,
                className="submenu"),
                id="stats-submenu",
                is_open=False
            )
        ], className="menu-item-with-submenu")
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
    [State("stats-submenu", "is_open")],
    prevent_initial_call=True
)
def toggle_stats_submenu(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    # Obtener el puerto del entorno (usado por Render) o usar 8050 por defecto
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=False)