"""Sección 3: Modelos: tabla de líderes, predicción en vivo con SHAP local, diagnóstico, comparación estadística y modelo original."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

import data as D
import plots as G

dash.register_page(__name__, path="/modelos", name="Modelos", order=2)

tp = D.csv("orig_prueba.csv")
an = D.csv("orig_validacion_anidada.csv").set_index("modelo")
cfg = D.js("eleccion.json")["configuraciones"]; hp = D.js("orig_hiperparametros.json")["hiperparametros"]
ver = D.js("orig_veredicto.json"); mcs = D.csv("orig_mcs.csv"); pr = D.csv("orig_pruebas_hipotesis.csv")
ab = D.csv("orig_ablacion.csv"); tri = D.csv("orig_optuna_trials.csv"); imp = D.csv("shap_importancia_global.csv", index_col=0)
q = D.csv("residuos_cuantiles.csv"); pesos = D.csv("pesos_por_hora.csv", index_col=0)
PT = D.pred_prueba(); XP = D.predictoras_prueba(); META = D.metadata()
RANGOS = META["columnas_entrada"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
D.modelo()          # carga el modelo una sola vez al iniciar

tipo = lambda m: "Original" if m == D.ORIG else ("Tradicional" if m in D.TRAD else "Machine Learning")
lider = pd.DataFrame({
    "modelo": [D.CORTO[m] for m in tp.modelo], "tipo": [tipo(m) for m in tp.modelo],
    "configuración": [", ".join(f"{k}={v:.3g}" for k, v in hp.items()) if m == D.ORIG else cfg.get(m, "—") for m in tp.modelo],
    "MSE validación anidada (media ± de)": [f"{an.loc[m, 'MSE_media']:,.0f} ± {an.loc[m, 'MSE_de']:,.0f}".replace(",", ".") for m in tp.modelo],
    "MSE prueba": tp.MSE.round(0), "MAPE prueba (%)": tp.MAPE.round(2), "AUC horas pico": tp.AUC_pico.round(3),
    "inferencia (s, 15.672 h)": tp.inferencia_s.round(3), "en el MCS (10 %)": np.where(tp.en_conjunto_10pct, "sí", "no")})
lider.insert(2, "optimizador", ["Bayesiano (Optuna TPE)" if m == D.ORIG else ("—" if m in D.TRAD else "Grid (validación)") for m in tp.modelo])

# ---------------- las 28 combinaciones del Entregable 2 (7 modelos x 4 optimizadores)
E2C = D.csv("e2_combinaciones.csv"); E2M = D.csv("e2_mejores.csv"); E2A = D.csv("e2_anytime.csv", index_col=0)
mcs_todos = D.csv("e2_mcs_todos.csv").set_index("modelo")
for c in ["en el MCS (10 %)"]:
    lider[c] = [("sí" if mcs_todos.en_conjunto_10pct.get(m, False) else "no") for m in tp.modelo]
lider = pd.concat([lider, pd.DataFrame({
    "modelo": "E2 · " + E2M.modelo, "tipo": "Entregable 2", "optimizador": E2M.optimizador,
    "configuración": [", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in __import__("json").loads(h).items()) for h in E2M.hiperparametros],
    "MSE validación anidada (media ± de)": [f"{a:,.0f} ± {b:,.0f}".replace(",", ".") for a, b in zip(E2M.MSE_media, E2M.MSE_de)],
    "MSE prueba": E2M.MSE_prueba.round(0), "MAPE prueba (%)": E2M.MAPE_prueba.round(2), "AUC horas pico": E2M.AUC_pico.round(3),
    "inferencia (s, 15.672 h)": E2M.inferencia_prueba_s.round(3), "en el MCS (10 %)": np.where(E2M.en_conjunto_10pct, "sí", "no")})],
    ignore_index=True).sort_values("MSE prueba").reset_index(drop=True)
tab_e2 = E2C.assign(**{"MSE externo (media ± de)": [f"{a:,.0f} ± {b:,.0f}".replace(",", ".") for a, b in zip(E2C.MSE_media, E2C.MSE_de)],
                       "MAPE (%)": E2C.MAPE_media.round(2), "búsqueda (min)": (E2C.tiempo_busqueda_s / 60).round(1),
                       "entrenamiento (s)": E2C.tiempo_entrenamiento_s.round(2), "inferencia (s)": E2C.tiempo_inferencia_s.round(3)})[
    ["modelo", "optimizador", "MSE externo (media ± de)", "MAPE (%)", "rango_en_modelo", "evaluaciones", "búsqueda (min)", "entrenamiento (s)", "inferencia (s)"]]
f_any = go.Figure([go.Scatter(x=E2A.index, y=E2A[c], mode="lines+markers", name=c) for c in E2A.columns])
f_any.update_layout(title="Desempeño en cualquier momento: mejor MSE interno hasta la evaluación t (relativo al mejor final)",
                    xaxis_title="evaluación", yaxis_title="MSE relativo (1 = mejor encontrado)", height=360)

EDIT = [("calor", "Índice de calor ayer, misma hora (°C)"), ("T2M", "Temperatura ayer (°C)"), ("RH2M", "Humedad ayer (%)"),
        ("lag24", "Demanda ayer, misma hora (MWh)"), ("media_7d", "Media de los últimos 7 días (MWh)"), ("max_dia_ant", "Máximo de ayer (MWh)")]


def campo(col, etiqueta):
    r = RANGOS[col]
    return dbc.Col([dbc.Label(etiqueta, html_for=f"in-{col}", className="small"),
                    dbc.Input(id=f"in-{col}", type="number", min=round(r["min"], 1), max=round(r["max"] * 1.15, 1), step=0.1)], md=4, className="mb-2")


def tabla(df, id_=None, alto=None):
    extra = {"id": id_} if id_ else {}
    return dash_table.DataTable(df.to_dict("records"), [{"name": c, "id": c} for c in df.columns], **extra, sort_action="native", filter_action="native",
                                style_cell={"textAlign": "left", "whiteSpace": "normal", "fontSize": 13, "fontFamily": "Source Sans 3, sans-serif"},
                                style_header={"fontWeight": "bold", "backgroundColor": "#EEF2F2"}, style_table={"overflowX": "auto", "maxHeight": alto, "overflowY": "auto"},
                                style_data_conditional=[{"if": {"filter_query": '{modelo} contains "ATER"'}, "backgroundColor": "#FBEAE8", "fontWeight": "bold"}])


# ---------------- figuras estáticas
f_conv = go.Figure([go.Scatter(x=tri.number, y=tri.value, mode="markers", marker=dict(color=G.GRIS, size=6), name="ensayo"),
                    go.Scatter(x=tri.number, y=tri.mejor_hasta_ahora, mode="lines", line=dict(color=G.ROJO, width=3), name="mejor hasta ahora")])
f_conv.update_layout(title="Curva de convergencia de Optuna (TPE)", xaxis_title="ensayo", yaxis_title="MSE medio en validación (MWh²)",
                     yaxis_range=[tri.value.min() * .99, tri.value.quantile(.9)], height=330)
a = ab.sort_values("MSE_prueba")
f_abl = go.Figure(go.Bar(x=a.MSE_prueba, y=a.variante, orientation="h", text=[f"{v:+.1f} %" for v in a["cambio_MSE_vs_completo_%"]], textposition="outside",
                         marker_color=[G.ROJO if v == "ATER completo" else {"componente": G.TINTA, "experto": G.ORO}.get(g, G.GRIS) for v, g in zip(a.variante, a.grupo)]))
f_abl.update_layout(title="Ablación: MSE en la prueba (texto = cambio frente a ATER completo)", xaxis_range=[a.MSE_prueba.min() * .93, a.MSE_prueba.max() * 1.04],
                    yaxis=dict(autorange="reversed", automargin=True), height=600, xaxis_title="MWh²", margin=dict(l=320))
ef = pr.drop_duplicates("benchmark")
f_ic = go.Figure(go.Scatter(x=ef.mejora_MSE_pct, y=[D.CORTO[b] for b in ef.benchmark], mode="markers", marker=dict(color=G.ROJO, size=10),
                            error_x=dict(type="data", symmetric=False, array=ef.IC95_hi - ef.mejora_MSE_pct, arrayminus=ef.mejora_MSE_pct - ef.IC95_lo, color=G.ROJO)))
f_ic.add_vline(x=0, line_color=G.TINTA); f_ic.update_layout(title="Reducción del MSE de ATER frente a cada benchmark (IC 95 %, bootstrap estacionario)",
                                                           xaxis_title="% de reducción del MSE", height=360, yaxis=dict(automargin=True))
f_pesos = go.Figure([go.Bar(x=pesos.index, y=pesos[c], name=D.CORTO[c], marker_color=G.color_modelo(c) if c != "Persistencia (misma hora de ayer)" else "#D5DBD8") for c in pesos.columns])
f_pesos.update_layout(barmode="stack", title="¿A quién le cree ATER a cada hora? (peso medio en la prueba)", xaxis_title="hora", yaxis_title="peso", height=360)

tab_pruebas = pr[["prueba", "benchmark", "estadistico", "p_crudo", "p_holm", "delta_cliff_pareado", "d_cohen", "conclusion"]].copy()
tab_pruebas["benchmark"] = tab_pruebas.benchmark.map(D.CORTO)
for c in ["estadistico", "delta_cliff_pareado", "d_cohen"]: tab_pruebas[c] = tab_pruebas[c].round(3)
for c in ["p_crudo", "p_holm"]: tab_pruebas[c] = tab_pruebas[c].map(lambda v: f"{v:.2e}")
mcs = D.csv("e2_mcs_todos.csv")
tab_mcs = mcs.assign(modelo=mcs.modelo.map(lambda m: D.CORTO.get(m, m)), p_MCS=mcs.p_MCS.round(4), en_conjunto_10pct=np.where(mcs.en_conjunto_10pct, "sí", "no"))

PSEUDO = """para cada día d (en orden):
    para cada hora t de d:                         # pronóstico emitido al cierre de d-1
        L[k] = Σ_j exp(-(c_t-τ_j)²/2σ²) · S[h_t, j, k]
        w[k] = exp(-η L[k]) / Σ_i exp(-η L[i])
        ŷ_t  = (1 + b[h_t]) · Σ_k w[k] f[t,k]
    al cierre de d (ya se conoce y):
        S ← λ S ;  S[h_t, banda(c_t), k] += min(1, (((1+b)f[t,k] - y_t)/n_t)²)
        b[h_t] ← b[h_t] - η_b · 2(ŷ_t - y_t)·f̄_t / n_t²"""

layout = html.Div([
    html.H2("Modelos"),
    dbc.Alert([html.B("Veredicto: "), f"ATER es estadísticamente superior a los 7 benchmarks (MCS, Giacomini-White y Diebold-Mariano con p ajustado por Holm < 0,05). "
               f"Frente al mejor benchmark (random forest relativo) reduce el MSE 3,9 % y gana el {pr.dias_gana_ATER_pct.iloc[12]:.1f} % de los días: "
               "un efecto pequeño (δ de Cliff pareado 0,25) pero consistente."], color="success"),
    dbc.Card(dbc.CardBody([html.H4("Tabla de líderes"), html.P("ATER, los 7 benchmarks del proyecto y la mejor combinación (modelo + optimizador) de cada uno de los 7 modelos base del Entregable 2. "
                                                           "Métrica del bucle externo de la validación anidada (media ± desviación estándar) y de la prueba única 2025–2026. Se puede ordenar y filtrar.",
                                                           className="small text-muted"), tabla(lider)]), className="mb-3"),
    dbc.Card(dbc.CardBody([html.H4("Entregable 2: las 28 combinaciones (7 modelos × 4 optimizadores)"),
        html.P("KNN, Ridge, Lasso, Decision Tree, Random Forest, XGBoost y SVR, cada uno optimizado con Grid Search, Random Search, Bayesiano (Optuna TPE) y genético, "
               "con el mismo presupuesto (12 evaluaciones por fold) y los mismos folds externos que ATER.", className="small text-muted"),
        tabla(tab_e2.rename(columns={"rango_en_modelo": "puesto en su modelo"}), alto="420px"),
        dcc.Graph(figure=f_any, config={"displayModeBar": False})]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Predicción en vivo"),
        html.P("Elija una hora del periodo de prueba: el formulario se llena con los datos reales que se conocían al emitir el pronóstico. "
               "Puede cambiar el calor o la demanda reciente para ver cómo responde el modelo.", className="small text-muted"),
        dbc.Row([dbc.Col([dbc.Label("Fecha", className="small"), dcc.DatePickerSingle(id="in-fecha", min_date_allowed=XP.index.min().date(), max_date_allowed=XP.index.max().date(),
                                                                                       date="2026-08-12", display_format="YYYY-MM-DD")], md=3),
                 dbc.Col([dbc.Label("Hora", className="small"), dcc.Dropdown(id="in-hora", options=list(range(24)), value=15, clearable=False)], md=2),
                 dbc.Col([dbc.Label("Modelo", className="small"), dcc.Dropdown(id="in-modelo", options=[{"label": D.CORTO[m], "value": m} for m in D.MODELOS_VIVO],
                                                                               value=D.ORIG, clearable=False)], md=4),
                 dbc.Col([dbc.Label("Festivo", className="small"), dbc.Switch(id="in-festivo", value=False)], md=3)], className="mb-2"),
        dbc.Row([campo(c, e) for c, e in EDIT]),
        dbc.Button("Pronosticar", id="btn-pred", color="danger", className="mt-1"),
        html.Div(id="out-alerta", className="mt-2"),
        dbc.Row([dbc.Col(html.Div(id="out-pred"), md=4), dbc.Col(dcc.Loading(dcc.Graph(id="out-shap", config={"displayModeBar": False})), md=8)], className="mt-2")]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Diagnóstico del modelo seleccionado (prueba 2025–2026)"),
        dcc.Dropdown(id="diag-modelo", options=[{"label": D.CORTO[m], "value": m} for m in tp.modelo], value=D.ORIG, clearable=False, style={"maxWidth": 420}),
        dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id="d-real")), lg=6), dbc.Col(dcc.Graph(id="d-hist"), lg=6)]),
        dbc.Row([dbc.Col(dcc.Graph(id="d-tiempo"), lg=6), dbc.Col(dcc.Graph(id="d-acf"), lg=6)]),
        html.P(id="d-txt", className="small text-muted")]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Interpretabilidad global (SHAP)"),
        dcc.Dropdown(id="shap-modelo", options=[{"label": "ATER (original)", "value": "ATER_media_abs_SHAP_MWh"}, {"label": "Random forest relativo", "value": "RF_media_abs_SHAP_MWh"}],
                     value="ATER_media_abs_SHAP_MWh", clearable=False, style={"maxWidth": 320}),
        dcc.Graph(id="shap-global", config={"displayModeBar": False}),
        html.P("Media del valor absoluto de SHAP (MWh) en 150 horas de la prueba. Los dos modelos dependen primero del nivel reciente; ATER pesa más la media de ayer "
               "y la interacción calor × hora (correlación de rangos 0,86).", className="small text-muted")]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Comparación estadística"),
        html.H6("Model Confidence Set (α = 0,10, bootstrap estacionario, bloque medio de 7 días)"), tabla(tab_mcs),
        html.P("Solo ATER queda en el conjunto de modelos superiores; el random forest y el XGBoost se eliminan con p = 0,005 y 0,001.", className="small text-muted mt-2"),
        dcc.Graph(figure=f_ic, config={"displayModeBar": False}),
        html.H6("Pruebas de hipótesis (21 comparaciones, ajuste de Holm)"), tabla(tab_pruebas, alto="420px")]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Modelo original: ATER"),
        html.P(["ATER (agregación de expertos por régimen horario y térmico, con corrección de nivel en línea) combina cinco pronósticos conocidos con pesos exponenciales "
                "que dependen de la hora y del calor, y corrige cada día el nivel con la demanda que acaba de observar. Está programado con numpy en ",
                html.Code("src/original_model.py"), " y tiene 13 pruebas unitarias."]),
        dbc.Row([dbc.Col(html.Pre(PSEUDO, className="bg-light p-2 small", style={"whiteSpace": "pre-wrap"}), lg=6),
                 dbc.Col(html.Ul([html.Li("Frente a Devaine et al. (2013) y Gaillard y Goude (2015): la especialización es continua (núcleo sobre el calor), no por reglas fijas, y se corrige el nivel."),
                                  html.Li("Frente al BOA suavizado (Ziel, 2022): suaviza en el calor, no entre horas."),
                                  html.Li("Frente al stacking: los pesos cambian cada día; en la ablación ATER le gana al stacking por MCO y al no negativo."),
                                  html.Li("Lo que aporta según la ablación: pesos por hora (7,9 % del MSE) y corrección de nivel (2,6 %). El núcleo de calor no aportó.")], className="small"), lg=6)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=f_conv, config={"displayModeBar": False}), lg=6), dbc.Col(dcc.Graph(figure=f_pesos, config={"displayModeBar": False}), lg=6)]),
        dcc.Graph(figure=f_abl, config={"displayModeBar": False}),
        dbc.Alert([html.B("Veredicto explícito: superior. "), "R1 novedad con matices (la familia existe para la demanda de Francia; la combinación y la aplicación al Caribe son nuevas). "
                   "R2 implementación propia. R3 superioridad significativa con MCS, Giacomini-White y Diebold-Mariano; efecto frente al RF pequeño (δ de Cliff pareado 0,25; "
                   "no pareado 0,07 y d de Cohen 0,12, despreciables)."], color="danger", className="mt-2")]), className="mb-3"),
])


@callback([Output(f"in-{c}", "value") for c, _ in EDIT] + [Output("in-festivo", "value")], Input("in-fecha", "date"), Input("in-hora", "value"))
def precargar(fecha, hora):
    if not fecha or hora is None:
        return [no_update] * (len(EDIT) + 1)
    ts = pd.Timestamp(fecha) + pd.Timedelta(hours=int(hora))
    if ts not in XP.index:
        return [no_update] * (len(EDIT) + 1)
    r = XP.loc[ts]
    return [round(float(r[c]), 1) for c, _ in EDIT] + [bool(r.festivo)]


@callback(Output("out-alerta", "children"), Output("out-pred", "children"), Output("out-shap", "figure"),
          Input("btn-pred", "n_clicks"), State("in-fecha", "date"), State("in-hora", "value"), State("in-modelo", "value"), State("in-festivo", "value"),
          [State(f"in-{c}", "value") for c, _ in EDIT], prevent_initial_call=True)
def pronosticar(_, fecha, hora, modelo, festivo, *vals):
    try:
        ts = pd.Timestamp(fecha) + pd.Timedelta(hours=int(hora))
        if ts not in XP.index:
            return dbc.Alert("Elija una fecha del periodo de prueba (2025-01-01 a 2026-10-15).", color="warning"), no_update, no_update
        errores = []
        for (c, e), v in zip(EDIT, vals):
            lo, hi = RANGOS[c]["min"], RANGOS[c]["max"] * 1.15
            if v is None:
                errores.append(f"Falta: {e}.")
            elif not (lo <= float(v) <= hi):
                errores.append(f"{e}: {v} está fuera del rango válido ({lo:,.1f} a {hi:,.1f}).")
        if errores:
            return dbc.Alert(html.Ul([html.Li(x) for x in errores]), color="danger"), no_update, no_update
        x = XP.loc[[ts]].copy()
        for (c, _), v in zip(EDIT, vals):
            x[c] = float(v)
        x["festivo"] = int(festivo); x["calor_x_hora"] = x.calor * x.hora_sin
        if festivo: x["tipo_dia"] = 2
        p = float(D.predecir(x, modelo)[0])
        qq = q[(q.modelo == modelo) & (q.hora == int(hora))].iloc[0]
        real = float(PT.loc[ts, "y"]) if ts in PT.index else None
        tarjeta = dbc.Card(dbc.CardBody([html.Div(D.CORTO[modelo], className="small text-muted"),
                                         html.Div(f"{p:,.0f} MWh".replace(",", "."), className="fs-2 fw-bold text-danger"),
                                         html.Div(f"Intervalo de predicción del 90 %: {p + qq.q05:,.0f} a {p + qq.q95:,.0f} MWh".replace(",", "."), className="small"),
                                         html.Div(f"Demanda real de esa hora: {real:,.0f} MWh".replace(",", ".") if real else "", className="small mt-1"),
                                         html.Div(f"{DIAS[ts.dayofweek]} {ts:%d-%m-%Y}, {ts:%H}:00", className="small text-muted mt-1")]))
        phi = D.shap_local(x, modelo).sort_values(key=np.abs, ascending=False).head(10)
        f = go.Figure(go.Bar(x=phi.values, y=phi.index, orientation="h", marker_color=[G.ROJO if v > 0 else G.VERDE for v in phi.values],
                             hovertemplate="%{y}: %{x:+,.1f} MWh<extra></extra>"))
        f.update_layout(title="Explicación local (Shapley): cuánto sube (rojo) o baja (verde) el pronóstico cada variable", xaxis_title="MWh",
                        yaxis=dict(autorange="reversed"), height=380)
        return None, tarjeta, f
    except Exception as ex:   # nunca tumbar la app por una entrada
        return dbc.Alert(f"No se pudo calcular el pronóstico: {ex}", color="danger"), no_update, no_update


@callback(Output("d-real", "figure"), Output("d-hist", "figure"), Output("d-tiempo", "figure"), Output("d-acf", "figure"), Output("d-txt", "children"),
          Input("diag-modelo", "value"))
def diagnostico(m):
    e = PT.y - PT[m]; s = PT.sample(4000, random_state=1)
    a = go.Figure(go.Scattergl(x=s.y, y=s[m], mode="markers", marker=dict(size=4, opacity=.35, color=G.color_modelo(m))))
    lim = [PT.y.min(), PT.y.max()]; a.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=G.TINTA, dash="dash"), name="ideal"))
    a.update_layout(title="Real frente a pronosticado (4.000 horas)", xaxis_title="real (MWh)", yaxis_title="pronóstico (MWh)", showlegend=False, height=360)
    h = go.Figure(go.Histogram(x=e, nbinsx=80, marker_color=G.color_modelo(m))); h.update_layout(title="Histograma de residuos", xaxis_title="real − pronóstico (MWh)", height=360)
    ed = e.resample("D").mean()
    t = go.Figure(go.Scatter(x=ed.index, y=ed, mode="lines", line=dict(color=G.color_modelo(m), width=1)))
    t.add_hline(y=0, line_color=G.TINTA); t.update_layout(title="Residuo medio diario en el tiempo", yaxis_title="MWh", height=330)
    ac = G.fig_acf(G.acf(e.values, 72), "ACF de los residuos horarios (72 rezagos)", len(e))
    txt = (f"MAPE {100 * np.mean(np.abs(e) / PT.y):.2f} %, sesgo medio {e.mean():+.1f} MWh. Los residuos tienen memoria a 24 h: "
           "parte del error de hoy se repite mañana, que es justo lo que la corrección de nivel de ATER aprovecha.")
    return a, h, t, ac, txt


@callback(Output("shap-global", "figure"), Input("shap-modelo", "value"))
def shap_global(col):
    s = imp[col].sort_values(ascending=False).head(12)
    return G.barras_h(s, "Importancia global (media de |SHAP|, MWh)", "MWh", colores=G.ROJO if col.startswith("ATER") else G.VERDE)
