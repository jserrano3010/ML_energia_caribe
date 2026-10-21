"""Funciones de gráficos reutilizables con una plantilla Plotly única (paleta de 5 colores)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

TINTA, ROJO, VERDE, ORO, GRIS = "#14262B", "#C0392B", "#1F5A3A", "#C8892B", "#9AA5A0"
PALETA = [ROJO, VERDE, ORO, TINTA, GRIS]

pio.templates["energia"] = go.layout.Template(layout=dict(
    font=dict(family="Source Sans 3, Segoe UI, Arial, sans-serif", size=13, color=TINTA),
    colorway=PALETA, paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#E6ECEC", zeroline=False, linecolor="#C9D3D4"),
    yaxis=dict(gridcolor="#E6ECEC", zeroline=False, linecolor="#C9D3D4"),
    margin=dict(l=60, r=20, t=50, b=50), title=dict(x=0.01, font=dict(size=16)),
    legend=dict(orientation="h", y=-0.2), hoverlabel=dict(font_size=12)))
pio.templates.default = "energia"


def color_modelo(m):
    if m.startswith("ATER"):
        return ROJO
    if m.startswith("Random"):
        return VERDE
    if m.startswith("XGB"):
        return ORO
    if m.startswith(("Híbrido", "Ridge")):
        return TINTA
    return GRIS


def vacio(msg="Sin datos para los filtros elegidos"):
    f = go.Figure(); f.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=GRIS))
    f.update_xaxes(visible=False); f.update_yaxes(visible=False); return f


def barras_h(s: pd.Series, titulo, xlab, colores=None, texto=None):
    f = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colores or VERDE,
                         text=texto, textposition="outside", hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
    f.update_layout(title=titulo, xaxis_title=xlab, yaxis=dict(autorange="reversed"), height=max(300, 34 * len(s) + 120))
    return f


def acf(x, nlags=48):
    x = np.asarray(x, float); x = x - x.mean(); d = (x ** 2).sum()
    return np.array([1.0] + [(x[k:] * x[:-k]).sum() / d for k in range(1, nlags + 1)])


def fig_acf(valores, titulo, n):
    lim = 1.96 / np.sqrt(n)
    f = go.Figure(go.Bar(x=list(range(len(valores))), y=valores, marker_color=VERDE, hovertemplate="rezago %{x}: %{y:.3f}<extra></extra>"))
    f.add_hline(y=lim, line_dash="dot", line_color=GRIS); f.add_hline(y=-lim, line_dash="dot", line_color=GRIS)
    f.update_layout(title=titulo, xaxis_title="rezago", yaxis_title="autocorrelación", height=320)
    return f
