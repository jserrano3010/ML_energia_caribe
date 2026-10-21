"""Sección 1: Contexto (narrativa estática y resumen visual)."""
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html

import data as D
import plots as G

dash.register_page(__name__, path="/", name="Contexto", order=0)

tp = D.csv("orig_prueba.csv").set_index("modelo")
res = D.js("resumen.json")
desc = D.csv("problemas_descartados.csv")
ver = D.js("orig_veredicto.json")
mape_or = tp.loc[D.ORIG, "MAPE"]; mej_per = tp.loc[D.ORIG, "mejora_vs_persistencia_%"]
mej_rf = 100 * (1 - tp.loc[D.ORIG, "MSE"] / tp.loc[D.RF, "MSE"])


def cifra(valor, texto):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(valor, className="fs-2 fw-bold text-danger"), html.Div(texto, className="small text-muted")])),
                   xs=12, sm=6, lg=3, className="mb-3")


def nf(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


diccionario = [
    {"variable": "mwh (objetivo)", "descripción": "Demanda comercial horaria del mercado CARIBE SOL (Air-e), MWh", "fuente": "XM"},
    {"variable": "hora, dow, mes, tipo_dia, festivo, día antes/después de festivo, navidad", "descripción": "Calendario conocido de antemano", "fuente": "holidays (Colombia)"},
    {"variable": "lag24, lag48, lag168, lag336", "descripción": "Demanda de la misma hora hace 1, 2, 7 y 14 días", "fuente": "XM"},
    {"variable": "media_dia_ant, max_dia_ant, media_7d, media_misma_hora_4sem", "descripción": "Nivel reciente de la demanda (conocido al emitir)", "fuente": "XM"},
    {"variable": "T2M, RH2M, WS2M, calor, tmax_dia, lluvia_24h", "descripción": "Temperatura, humedad, viento, índice de calor NOAA y lluvia de Sabanagrande del día anterior", "fuente": "NASA POWER"},
    {"variable": "calor_x_hora", "descripción": "Interacción índice de calor × hora (seno)", "fuente": "derivada"},
]

pasos = ["ETL (XM + NASA POWER, 52.272 h)", "EDA (calor, calendario, tendencia)", "7 benchmarks (3 tradicionales + 4 ML)",
         "Validación temporal (5 folds, purga 48 h)", "Modelo original ATER (Optuna, validación anidada)", "Pruebas: MCS, GW, DM + Holm", "Dashboard en Render"]

fig_desc = go.Figure(go.Bar(x=desc.mejora_pct, y=[p[:52] for p in desc.problema], orientation="h",
                            marker_color=[G.ROJO if c == 4 else G.GRIS for c in desc.criterios_cumplidos],
                            text=[f"{nf(v)} % · {c}/4 criterios" for v, c in zip(desc.mejora_pct, desc.criterios_cumplidos)], textposition="outside",
                            hovertemplate="%{y}<br>mejora: %{x:.1f} %<extra></extra>"))
fig_desc.add_vline(x=10, line_dash="dash", line_color=G.TINTA, annotation_text="umbral 10 %")
fig_desc.update_layout(title="Mejora del mejor ML frente a la mejor línea base tradicional (%)", yaxis=dict(autorange="reversed"),
                       xaxis=dict(range=[min(-10, desc.mejora_pct.min() - 5), 90]), height=360)

layout = html.Div([
    dbc.Row(dbc.Col([
        html.Div("Machine Learning · Proyecto final · Entregable 3", className="text-uppercase small text-danger fw-semibold"),
        html.H1("¿Cuánta energía consumirá mañana el Caribe cuando aprieta el calor?", className="fw-bold"),
        html.P(["En Sabanagrande, como en todo el Atlántico, el calor dispara el uso de aires, ventiladores y neveras. "
                "Este tablero pronostica al final de cada día la demanda de las 24 horas del día siguiente del mercado de Air-e, "
                "con el clima de Sabanagrande, y presenta ", html.B("ATER"), ", un modelo original de agregación de expertos."], className="lead"),
    ]), className="mb-2"),
    dbc.Row([cifra(f"{nf(mape_or, 2)} %", "error medio (MAPE) de ATER en 2025–2026"),
             cifra(f"{nf(mej_per)} %", "menos error cuadrático que la misma hora de ayer"),
             cifra(f"{nf(mej_rf)} %", "menos error cuadrático que el mejor modelo de ML conocido (random forest)"),
             cifra("1 de 8", "modelos en el Model Confidence Set: solo ATER")]),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Por qué importa"), html.Ul([
            html.Li("La tarifa regulada de Air-e subió cerca de 74 % hasta julio de 2023, más que la inflación (Char, s. f., con datos del Observatorio del Caribe Colombiano)."),
            html.Li("El Consumo Básico de Subsistencia no reconoce del todo que en clima cálido la refrigeración es una necesidad (Char, s. f.; Corte Constitucional, T-761 de 2015)."),
            html.Li("En Atlántico el 70,1 % de los hogares subsidiados no son pobres, y el Consumo Básico de Subsistencia es de 173 kWh/mes bajo 1.000 m de altitud (Fundesarrollo, 2023)."),
            html.Li("La energía se compra el día anterior: cada MWh mal pronosticado se liquida en bolsa. Con el precio de bolsa de 2025–2026, el mejor modelo ahorra del orden de $75.000 millones al año en desvíos frente a la persistencia (cota superior)."),
            html.Li("Con índice de calor de 38–42 °C, la demanda de las tardes laborales es 27,5 % mayor que con 30 °C o menos."),
        ])])), md=6, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Pregunta de investigación"),
            html.P("¿Puede un modelo propio de agregación en línea pronosticar la demanda horaria del día siguiente del mercado de Air-e mejor que los métodos tradicionales "
                   "y que los modelos de Machine Learning conocidos, con superioridad estadísticamente significativa?"),
            html.H5("Objetivos"), html.Ol([html.Li("Construir un dataset horario sin fuga de información (XM + NASA POWER)."),
                                          html.Li("Comparar 3 métodos tradicionales y 4 de ML con validación temporal."),
                                          html.Li("Diseñar, programar y probar el modelo original ATER."),
                                          html.Li("Verificar su superioridad con tres pruebas de hipótesis, intervalos y tamaños del efecto."),
                                          html.Li("Ponerlo en producción en este tablero.")])])), md=6, className="mb-3"),
    ]),
    dbc.Card(dbc.CardBody([html.H4("Datos y variables"),
        html.P(f"Fuente: XM (API Sinergox, demanda del mercado CARIBE SOL) y NASA POWER (clima horario de Sabanagrande, 10,79° N, −74,76° O). "
               f"Periodo: octubre de 2020 a octubre de 2026. {nf(res['filas'], 0)} horas utilizables, {res['n_predictoras']} predictoras. "
               "Tarea: regresión (pronóstico de la demanda en MWh). Entrenamiento hasta diciembre de 2024; prueba de enero de 2025 a octubre de 2026."),
        dash_table.DataTable(diccionario, [{"name": c, "id": c} for c in ["variable", "descripción", "fuente"]],
                             style_cell={"textAlign": "left", "whiteSpace": "normal", "fontFamily": "Source Sans 3, sans-serif", "fontSize": 14},
                             style_header={"fontWeight": "bold", "backgroundColor": "#EEF2F2"}, style_table={"overflowX": "auto"})]), className="mb-3"),
    dbc.Card(dbc.CardBody([html.H4("Metodología"),
        html.Div([html.Div([html.Span(f"{i + 1}", className="badge bg-danger me-2"), p], className="border rounded p-2 small bg-light")
                  for i, p in enumerate(pasos)], className="d-flex flex-wrap gap-2", role="list", **{"aria-label": "Pasos de la metodología"})]), className="mb-3"),
    dbc.Card(dbc.CardBody([html.H4("Problemas evaluados y por qué se descartaron"),
        html.P("Antes de la demanda eléctrica se evaluaron cuatro problemas con el mismo protocolo. Criterios: C1 ≥ 20.000 datos reales; "
               "C2 mejora ≥ 10 % frente a la mejor línea base tradicional; C3 mejora significativa (Diebold-Mariano) en el horizonte que importa; "
               "C4 la regla de validación no elige un modelo tradicional."),
        dcc.Graph(figure=fig_desc, config={"displayModeBar": False}),
        html.Ul([html.Li([html.B("Lluvia semanal (Entregable 2): "), "mejor pero no representativa. Mejora 7 % a la climatología, significativa solo a 1 semana."]),
                 html.Li([html.B("Precio del maíz: "), "lo gana un modelo tradicional. La regla de validación eligió el estacional; a 4 meses el RF no es significativo."]),
                 html.Li([html.B("Decisión de siembra: "), "lo gana un modelo tradicional. La persistencia tiene el menor error, y todos los modelos (incluido el oráculo) siembran las mismas fechas."]),
                 html.Li([html.B("Baloto: "), "azar por diseño. AUC ≈ 0,50, ningún modelo supera al promedio."]),
                 html.Li([html.B("Demanda de Air-e (definitivo): "), "cumple los 4 criterios, y el modelo original ATER le gana además a los modelos de ML conocidos."])]),
        dash_table.DataTable(desc[["orden", "problema", "n_datos", "mejor_ML", "linea_base", "mejora_pct", "p_DM", "criterios_cumplidos", "veredicto"]].round(1).to_dict("records"),
                             [{"name": c, "id": c} for c in ["orden", "problema", "n_datos", "mejor_ML", "linea_base", "mejora_pct", "p_DM", "criterios_cumplidos", "veredicto"]],
                             style_cell={"textAlign": "left", "whiteSpace": "normal", "fontSize": 13, "fontFamily": "Source Sans 3, sans-serif"},
                             style_header={"fontWeight": "bold", "backgroundColor": "#EEF2F2"}, style_table={"overflowX": "auto"})]), className="mb-3"),
    dbc.Card(dbc.CardBody([html.H4("Limitaciones, ética y autoría"), html.Ul([
        html.Li("La demanda es la del mercado de Air-e (Atlántico, Magdalena y La Guajira); XM no publica la de Sabanagrande sola. El clima de Sabanagrande actúa como impulsor local."),
        html.Li("No se pronostica el precio: la tarifa es mensual y es la misma en todo el mercado. El precio de bolsa se usa solo para valorar los desvíos."),
        html.Li("La prueba 2025–2026 se consultó más de una vez durante el proyecto (versión 1 de los árboles); se reporta en el libro."),
        html.Li("La ventaja de ATER frente al random forest es pequeña (3,9 % del MSE; δ de Cliff pareado 0,25). Frente a los métodos tradicionales es grande."),
        html.Li("Un error por debajo en días de calor extremo afectaría primero a los hogares vulnerables; por eso se reportan aparte las horas pico y los días más calurosos."),
        html.Li("Autor: Jairo Serrano. Asistencia de IA (Claude, Anthropic) declarada en el capítulo 8 del libro. Versión 3.0, octubre de 2026."),
    ])]), className="mb-3"),
])
