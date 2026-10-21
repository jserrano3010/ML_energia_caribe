"""Sección 2: EDA con filtros globales que actualizan todos los gráficos."""
import warnings
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import data as D
import plots as G

warnings.filterwarnings("ignore", category=FutureWarning)

dash.register_page(__name__, path="/eda", name="EDA", order=1)

E = D.eda()
NUM = {"mwh": "Demanda (MWh)", "calor": "Índice de calor (°C)", "T2M": "Temperatura (°C)", "RH2M": "Humedad relativa (%)",
       "WS2M": "Viento (m/s)", "lluvia": "Lluvia (mm/h)", "hora": "Hora", "mes": "Mes"}
CAT = {"tipo_dia": "Tipo de día", "anio": "Año", "festivo": "Festivo"}
TIPOS = ["laboral", "sábado", "domingo o festivo"]


def tarjeta(titulo, graf_id, insight, alto=None):
    return dbc.Card(dbc.CardBody([html.H5(titulo), dcc.Loading(dcc.Graph(id=graf_id, config={"displayModeBar": False}, style={"height": alto} if alto else None)),
                                  html.P(insight, className="small text-muted mb-0", id=f"{graf_id}-txt")]), className="mb-3")


layout = html.Div([
    html.H2("Exploración de los datos"),
    html.P("Los filtros actualizan todos los gráficos de la página.", className="text-muted"),
    dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([dbc.Label("Periodo"), dcc.DatePickerRange(id="f-fechas", min_date_allowed=E.index.min().date(), max_date_allowed=E.index.max().date(),
                                                           start_date=E.index.min().date(), end_date=E.index.max().date(), display_format="YYYY-MM-DD")], md=5),
        dbc.Col([dbc.Label("Tipo de día"), dcc.Dropdown(id="f-tipo", options=TIPOS, value=TIPOS, multi=True, clearable=False)], md=4),
        dbc.Col([dbc.Label("Horas del día"), dcc.RangeSlider(id="f-horas", min=0, max=23, step=1, value=[0, 23], marks={h: str(h) for h in range(0, 24, 3)})], md=3),
    ])), className="mb-3 sticky-filtros"),
    dbc.Alert(id="f-resumen", color="light", className="py-2"),
    dbc.Row([dbc.Col(tarjeta("Demanda en el tiempo", "g-serie", "La demanda crece año a año: la prueba 2025–2026 tiene niveles nunca vistos en el entrenamiento. Por eso los árboles aprenden la demanda relativa al nivel de 7 días."), lg=8),
             dbc.Col(tarjeta("Distribución de la variable objetivo", "g-hist", "La distribución es bimodal: horas de madrugada y horas de tarde y noche. No hay valores faltantes en la demanda."), lg=4)]),
    dbc.Row([dbc.Col(tarjeta("Perfil horario por tipo de día", "g-perfil", "Dos picos en días laborales: la tarde (13–15 h, el calor del día) y la noche (20–22 h, aire acondicionado en los hogares). Los domingos y festivos consumen menos."), lg=6),
             dbc.Col(tarjeta("El calor dispara la demanda", "g-calor", "Por encima de 34 °C de sensación térmica, la demanda sube con fuerza: la relación es no lineal y depende de la hora."), lg=6)]),
    dbc.Card(dbc.CardBody([html.H5("Explorador bivariado"),
        dbc.Row([dbc.Col([dbc.Label("Eje x"), dcc.Dropdown(id="b-x", options=[{"label": v, "value": k} for k, v in NUM.items()], value="calor", clearable=False)], md=4),
                 dbc.Col([dbc.Label("Eje y"), dcc.Dropdown(id="b-y", options=[{"label": v, "value": k} for k, v in NUM.items()], value="mwh", clearable=False)], md=4),
                 dbc.Col([dbc.Label("Color"), dcc.Dropdown(id="b-c", options=[{"label": v, "value": k} for k, v in CAT.items()], value="tipo_dia", clearable=False)], md=4)]),
        dcc.Loading(dcc.Graph(id="g-biv", config={"displayModeBar": False})),
        html.P("Muestra aleatoria de hasta 5.000 horas del filtro. Elija cualquier par de variables y el color.", className="small text-muted mb-0")]), className="mb-3"),
    dbc.Row([dbc.Col(tarjeta("Correlaciones", "g-corr", "La demanda se correlaciona sobre todo con la hora y con el calor; la humedad va en sentido contrario a la temperatura durante el día."), lg=6),
             dbc.Col(dbc.Card(dbc.CardBody([html.H5("Valores faltantes y atípicos"), html.Div(id="t-faltantes"),
                                            html.P("Los faltantes del clima (huecos de NASA POWER) se interpolan hasta 6 horas; la demanda no tiene faltantes. "
                                                   "Los atípicos (regla de 1,5 × RIC) de la demanda son horas de días festivos y de fin de año, no errores: se conservan.",
                                                   className="small text-muted mb-0")]), className="mb-3"), lg=6)]),
    html.H4("Serie de tiempo (demanda media diaria)", className="mt-2"),
    dbc.Row([dbc.Col(tarjeta("Descomposición: tendencia y estacionalidad semanal", "g-desc", "La tendencia sube y la estacionalidad semanal (domingos más bajos) es estable.", ), lg=12)]),
    dbc.Row([dbc.Col(tarjeta("ACF de la demanda diaria", "g-acf", "Picos cada 7 días: la memoria semanal que aprovechan los rezagos de 7 y 14 días."), lg=6),
             dbc.Col(tarjeta("PACF de la demanda diaria", "g-pacf", "La autocorrelación parcial se concentra en los primeros rezagos y en el rezago 7."), lg=6)]),
    dbc.Alert(id="t-adf", color="secondary"),
])


def filtrar(ini, fin, tipos, horas):
    d = E.loc[str(ini):str(fin)]
    d = d[d.tipo_dia.isin(tipos or TIPOS) & d.hora.between(horas[0], horas[1])]
    return d


@callback(Output("f-resumen", "children"), Output("g-serie", "figure"), Output("g-hist", "figure"), Output("g-perfil", "figure"),
          Output("g-calor", "figure"), Output("g-corr", "figure"), Output("t-faltantes", "children"),
          Output("g-desc", "figure"), Output("g-acf", "figure"), Output("g-pacf", "figure"), Output("t-adf", "children"),
          Input("f-fechas", "start_date"), Input("f-fechas", "end_date"), Input("f-tipo", "value"), Input("f-horas", "value"))
def actualizar(ini, fin, tipos, horas):
    return _actualizar(str(ini)[:10], str(fin)[:10], tuple(tipos or TIPOS), tuple(horas))


@lru_cache(maxsize=64)          # los mismos filtros no se recalculan
def _actualizar(ini, fin, tipos, horas):
    d = filtrar(ini, fin, list(tipos), list(horas))
    if len(d) < 48:
        v = G.vacio(); return ("Muy pocas horas con estos filtros: amplíe el periodo.", v, v, v, v, v, "", v, v, v, "")
    res = f"{len(d):,} horas seleccionadas · demanda media {d.mwh.mean():,.0f} MWh · máximo {d.mwh.max():,.0f} MWh".replace(",", ".")
    dia = d.mwh.resample("D").mean().dropna()
    s = go.Figure(go.Scatter(x=dia.index, y=dia.values, mode="lines", line=dict(color=G.VERDE, width=1), name="media diaria",
                             hovertemplate="%{x|%d-%b-%Y}: %{y:,.0f} MWh<extra></extra>"))
    s.add_vline(x=pd.Timestamp("2025-01-01").timestamp() * 1000, line_dash="dash", line_color=G.TINTA, annotation_text="inicio de la prueba")
    s.update_layout(title="Demanda media diaria (MWh)", yaxis_title="MWh", height=330)
    cnt, ed = np.histogram(d.mwh, bins=60)          # se agrega en el servidor: no se envían 52.000 valores al navegador
    h = go.Figure(go.Bar(x=(ed[:-1] + ed[1:]) / 2, y=cnt, width=np.diff(ed), marker_color=G.VERDE, hovertemplate="%{x:,.0f} MWh: %{y} horas<extra></extra>")); h.update_layout(title="Distribución de la demanda horaria", xaxis_title="MWh", yaxis_title="horas", height=330)
    p = go.Figure()
    for t, c in zip(TIPOS, [G.ROJO, G.ORO, G.VERDE]):
        g = d[d.tipo_dia == t].groupby("hora").mwh.mean()
        if len(g): p.add_trace(go.Scatter(x=g.index, y=g.values, name=t, mode="lines+markers", line=dict(color=c)))
    p.update_layout(title="Demanda media por hora", xaxis_title="hora del día", yaxis_title="MWh", height=360)
    b = pd.cut(d.calor, np.arange(20, 50, 2)); g = d.groupby(b, observed=True).mwh.agg(["mean", "count"]); g = g[g["count"] >= 20]
    c = go.Figure(go.Scatter(x=[i.mid for i in g.index], y=g["mean"], mode="lines+markers", line=dict(color=G.ROJO),
                             hovertemplate="%{x} °C: %{y:,.0f} MWh<extra></extra>"))
    c.update_layout(title="Demanda media según el índice de calor", xaxis_title="índice de calor en Sabanagrande (°C)", yaxis_title="MWh", height=360)
    cols = ["mwh", "calor", "T2M", "RH2M", "WS2M", "lluvia", "hora"]
    cm = d[cols].corr().round(2)
    k = go.Figure(go.Heatmap(z=cm.values, x=[NUM[x] for x in cols], y=[NUM[x] for x in cols], zmin=-1, zmax=1, colorscale=[[0, G.VERDE], [.5, "white"], [1, G.ROJO]],
                             text=cm.values, texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z}<extra></extra>"))
    k.update_layout(title="Matriz de correlación (Pearson)", height=420)
    falt = E.loc[d.index, ["mwh", "T2M", "RH2M", "WS2M", "lluvia"]].isna().sum()
    q1, q3 = d.mwh.quantile([.25, .75]); atip = int(((d.mwh < q1 - 1.5 * (q3 - q1)) | (d.mwh > q3 + 1.5 * (q3 - q1))).sum())
    tabla = dbc.Table([html.Thead(html.Tr([html.Th("variable"), html.Th("faltantes en el filtro")]))] +
                      [html.Tbody([html.Tr([html.Td(NUM.get(i, i)), html.Td(int(v))]) for i, v in falt.items()] +
                                  [html.Tr([html.Td(html.B("Atípicos de demanda (1,5 × RIC)")), html.Td(atip)])])], bordered=False, size="sm", striped=True)
    # descomposición semanal (media móvil de 7 días como tendencia + perfil semanal)
    tend = dia.rolling(7, center=True).mean(); est = (dia - tend).groupby(dia.index.dayofweek).transform("mean")
    dd = go.Figure([go.Scatter(x=dia.index, y=dia, name="observada", line=dict(color=G.GRIS, width=1)),
                    go.Scatter(x=tend.index, y=tend, name="tendencia (media móvil 7 días)", line=dict(color=G.ROJO, width=2)),
                    go.Scatter(x=est.index, y=est + dia.mean(), name="estacionalidad semanal (+ media)", line=dict(color=G.VERDE, width=1))])
    dd.update_layout(title="Descomposición aditiva de la demanda media diaria (MWh)", yaxis_title="MWh", height=360)
    if len(dia) > 60:
        from statsmodels.tsa.stattools import adfuller, pacf
        a = G.fig_acf(G.acf(dia.values, 35), "ACF (35 días)", len(dia))
        pa = G.fig_acf(pacf(dia.values, nlags=35), "PACF (35 días)", len(dia))
        st, pv = adfuller(dia.values)[:2]; st2, pv2 = adfuller(dia.diff(7).dropna().values)[:2]
        adf = (f"Prueba ADF de estacionariedad, demanda diaria: estadístico {st:.2f}, p = {pv:.3f} → "
               f"{'estacionaria' if pv < .05 else 'no estacionaria (tendencia)'}. Con diferencia semanal: p = {pv2:.4f} → "
               f"{'estacionaria' if pv2 < .05 else 'no estacionaria'}. Por eso los modelos trabajan con el nivel reciente y rezagos semanales.")
    else:
        a = pa = G.vacio("Se necesitan más de 60 días"); adf = "Se necesitan más de 60 días para la prueba ADF."
    return res, s, h, p, c, k, tabla, dd, a, pa, adf


@callback(Output("g-biv", "figure"), Input("f-fechas", "start_date"), Input("f-fechas", "end_date"), Input("f-tipo", "value"),
          Input("f-horas", "value"), Input("b-x", "value"), Input("b-y", "value"), Input("b-c", "value"))
def bivariado(ini, fin, tipos, horas, x, y, c):
    d = filtrar(ini, fin, tipos, horas)
    if len(d) == 0:
        return G.vacio()
    d = d.sample(min(5000, len(d)), random_state=1)
    f = go.Figure()
    for i, (k, g) in enumerate(d.groupby(c)):
        f.add_trace(go.Scattergl(x=g[x], y=g[y], mode="markers", name=str(k), marker=dict(size=4, opacity=.45, color=G.PALETA[i % 5]),
                                 hovertemplate=f"{NUM[x]}: %{{x:.1f}}<br>{NUM[y]}: %{{y:.1f}}<extra>{k}</extra>"))
    f.update_layout(title=f"{NUM[y]} vs {NUM[x]}", xaxis_title=NUM[x], yaxis_title=NUM[y], height=430, legend_title=CAT[c])
    return f


# precálculo de la vista inicial al arrancar el servidor (la primera visita no espera el cálculo)
_actualizar(str(E.index.min().date()), str(E.index.max().date()), tuple(TIPOS), (0, 23))
