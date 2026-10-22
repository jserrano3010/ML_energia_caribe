"""Dashboard predictivo: calor y demanda eléctrica de Air-e (Sabanagrande) y el modelo original ATER.

Local:   python app.py            Render:  gunicorn app:server
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY,
                "https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap"],
                title="Calor y energía en el Caribe", suppress_callback_exceptions=True)
server = app.server

navbar = dbc.NavbarSimple(
    children=[dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
              dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
              dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact"))],
    brand="Calor y energía en el Caribe · ATER", color="primary", dark=True, sticky="top", fluid=True)

app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid=True, className="py-3 px-3", style={"maxWidth": "1280px"}),
                       html.Footer("Edwin Yunis y Jairo Serrano · Machine Learning, Maestría en Ingeniería Industrial · Prof. Lihki Rubio · v3.0, octubre de 2026",
                                   className="text-muted small text-center py-3")])

if __name__ == "__main__":
    app.run(debug=True)
