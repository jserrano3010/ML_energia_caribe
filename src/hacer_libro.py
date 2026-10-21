import nbformat as nbf
from pathlib import Path
L = Path("libro")
def nb(nombre, celdas):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:].lstrip("\n")) if c.startswith("#MD") else nbf.v4.new_code_cell(c) for c in celdas]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, L/nombre)
SETUP = """import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../src"); import pipeline_energia as P
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
R = "../resultados/"
plt.rcParams.update({"figure.figsize": (10, 4), "axes.spines.top": False, "axes.spines.right": False})
COL = {"Persistencia (misma hora de ayer)": "#9AA5A0", "Estacional (misma hora semana pasada)": "#B8C2BD", "Promedio por hora y día": "#D5DBD8",
       "Ridge (lineal)": "#5B7F96", "Random forest (relativo)": "#1F5A3A", "XGBoost (relativo)": "#C8892B", "Híbrido Ridge + XGBoost (residuo relativo)": "#8E4B7A"}"""

nb("01_datos_eda.ipynb", [
"#MD# Datos y análisis exploratorio\n\n**Demanda:** XM (operador del mercado eléctrico colombiano), API pública de Sinergox, métrica *Demanda Comercial por Mercado de Comercialización*, mercado **CARIBE SOL**: el área de Air-e (Atlántico, Magdalena y La Guajira), que existe desde el 1 de octubre de 2020. Resolución horaria.\n\n**Clima:** NASA POWER, serie horaria del punto de Sabanagrande (10,79° N, −74,76° O), en hora local: temperatura a 2 m, humedad relativa, viento y lluvia.\n\n**Calendario:** festivos oficiales de Colombia (paquete `holidays`), días antes y después de festivo, temporada de fin de año.",
SETUP,
"""df = pd.read_pickle(R + "datos_horarios.pkl")
print("Horas:", len(df), "| desde", df.index.min(), "hasta", df.index.max())
print("Horas de demanda faltantes:", int(df.mwh.isna().sum()))
df.mwh.describe().round(1)""",
"""a = df.mwh.resample("YE").agg(["mean", "max"]); a.index = a.index.year
a["energia_GWh"] = df.mwh.resample("YE").sum().values / 1e3; a.round(1)""",
"""print("Crecimiento de la demanda media 2021 → 2025: %.1f %%" % (100*(df.mwh["2025"].mean()/df.mwh["2021"].mean()-1)))
fig, ax = plt.subplots(); df.mwh.resample("D").mean().plot(ax=ax, lw=.8, color="#1F5A3A", label="media diaria")
df.mwh.resample("D").max().plot(ax=ax, lw=.6, color="#C8892B", alpha=.7, label="máximo diario")
ax.axvline(pd.Timestamp("2025-01-01"), color="k", ls="--"); ax.text(pd.Timestamp("2025-01-15"), df.mwh.max()*.98, "prueba →")
ax.set_ylabel("MWh por hora"); ax.set_title("Demanda del mercado CARIBE SOL (Air-e)"); ax.legend(); plt.show()""",
"#MD **Hallazgo 1 — tendencia fuerte.** La demanda media creció 17,5 % entre 2021 y 2025 y sigue subiendo en 2026: el periodo de prueba tiene niveles **nunca vistos** en el entrenamiento. Esto obliga a diseñar modelos que no dependan de haber visto ese nivel (capítulo 2).",
"""X, y = P.construir(df, expost=True)
perf = y.groupby([X.tipo_dia, X.hora]).mean().unstack(0); perf.columns = ["laboral", "sábado", "domingo o festivo"]
fig, ax = plt.subplots(); perf.plot(ax=ax, marker="o", ms=3); ax.set_xlabel("hora"); ax.set_ylabel("MWh"); ax.set_title("Perfil horario medio"); plt.show()
print("Festivo frente a día laboral (media): %.1f %%" % (100*(y[X.festivo == 1].mean()/y[(X.festivo == 0) & (X.dow < 5)].mean()-1)))""",
"#MD **Hallazgo 2 — calendario.** En días laborales la demanda tiene dos picos: la tarde (13–15 h, el calor del día) y la noche (20–22 h, aire acondicionado e iluminación en los hogares); un festivo consume 7,4 % menos que un día laboral. \"La misma hora de ayer\" falla justo después y antes de un festivo.",
"""m = X.hora.between(13, 16) & (X.tipo_dia == 0)
b = pd.cut(X.calor[m], [20, 30, 34, 38, 42, 46])
t = y[m].groupby(b).agg(["mean", "count"]).round(0); t["vs_<=30°C_%"] = (100*(t["mean"]/t["mean"].iloc[0]-1)).round(1); t""",
"""fig, ax = plt.subplots(figsize=(7, 4)); s = X[m].sample(4000, random_state=1)
ax.scatter(s.calor, y[s.index], s=4, alpha=.3, color="#C8892B"); ax.set_xlabel("índice de calor en Sabanagrande (°C)"); ax.set_ylabel("MWh")
ax.set_title("Días laborales, 13–16 h: la demanda sube con el calor"); plt.show()""",
"#MD **Hallazgo 3 — calor no lineal.** En días laborales de 13 a 16 h, con índice de calor de 34–38 °C la demanda es 19,7 % mayor que con 30 °C o menos, y con 38–42 °C, 27,5 % mayor. La relación se curva: un modelo lineal la captura solo en parte; los árboles la aprenden sola.",
"""print("Autocorrelación horaria — rezago 24 h: %.3f | 168 h: %.3f" % (y.autocorr(24), y.autocorr(168)))
Xo, yo = P.construir(df); ok = Xo.notna().all(axis=1)
print("Filas utilizables (con todas las predictoras):", int(ok.sum()), "| predictoras:", Xo.shape[1])
assert "mwh" not in Xo.columns
pd.Series(Xo.columns, name="predictoras").to_frame().T""",
"#MD **Predictoras (todas conocidas al emitir el pronóstico, al final del día anterior):** calendario (hora, día, mes, festivos, temporada, armónicos), historia de la demanda con rezagos de 24 h o más (misma hora de ayer, de anteayer, de la semana pasada y de hace dos semanas; media y máximo de ayer; media de 7 días; media de la misma hora en 4 semanas) y clima de Sabanagrande **del día anterior** (temperatura, humedad, viento, índice de calor, lluvia de 24 h). Como sensibilidad se prueba también con el clima real del día pronosticado (\"pronóstico del tiempo perfecto\").\n\n**Tamaño:** 52.272 filas horarias utilizables, por encima de las ~20.000 que sugiere la guía.",
])

nb("02_modelos_validacion.ipynb", [
"#MD# Modelos y validación\n\n**Emisión:** al final del día D se pronostican las 24 horas del día D+1 (así se programa la energía en Colombia). **Partición:** entrenamiento octubre 2020 – diciembre 2024 con purga de 48 h; prueba enero 2025 – octubre 2026, usada una sola vez. **Validación:** 5 folds de ventana creciente con separación de 48 h.\n\n**Modelos (7):**\n- Tradicionales: **persistencia** (misma hora de ayer), **estacional** (misma hora de la semana pasada) y **promedio por hora y día de la semana**.\n- Machine Learning: **Ridge** (perfiles hora × tipo de día, rezagos y calor), **random forest relativo**, **XGBoost relativo** e **híbrido Ridge + XGBoost** que corrige el residuo relativo de Ridge.\n\n**Objetivo relativo (la idea central).** La primera versión entrenaba los árboles con la demanda en MWh y en la prueba fallaron: con demanda récord, un árbol no puede predecir valores mayores a los que vio. En la versión 2, los árboles aprenden la **forma relativa** $y_t / \\\\bar y_{7d}$ (demanda dividida por la media de los 7 días anteriores) con predictoras también relativas; el pronóstico es $\\\\hat y_t = \\\\bar y_{7d} \\\\times \\\\hat r_t$. El nivel lo lleva la media reciente; el árbol solo aprende cómo el calor, la hora y los festivos deforman ese nivel.\n\n**Ajuste de hiperparámetros:** grillas pequeñas para todos los modelos de ML, elegidas con la validación.\n\n**Nota:** en este capítulo y los dos siguientes, \"el elegido\" es el **mejor benchmark** según la regla. El modelo original (ATER, capítulo 5) se compara contra todos ellos en el capítulo 6.\n\n**Regla fijada de antemano:** gana el menor MSE medio en validación; si la diferencia con el segundo es menor de 1 %, desempata el CSI de horas pico (> percentil 95 de entrenamiento). La prueba solo confirma.",
SETUP,
"""todas = pd.read_csv(R + "validacion_todas_config.csv")
todas.groupby(["modelo", "config"])[["MSE", "MAPE"]].mean().round(2)""",
"""val = pd.read_csv(R + "validacion.csv"); e = json.load(open(R + "eleccion.json"))
print("Configuración elegida por modelo:", e["configuraciones"]); print("ELEGIDO POR LA REGLA:", e["elegido"])
val[["modelo", "config", "MSE", "RMSE", "MAPE", "AUC_pico", "POD", "FAR", "CSI", "mejora_MSE_vs_Persistencia_%", "mejora_MSE_vs_Estacional_%"]].round(3)""",
"""t = val.sort_values("MSE"); print("Diferencia relativa de MSE entre el 1.º y el 2.º: %.2f %%" % (100*(t.MSE.iloc[1]/t.MSE.iloc[0]-1)))
fold = todas.merge(pd.DataFrame(e["configuraciones"].items(), columns=["modelo", "config"]).assign(ok=1), on=["modelo", "config"], how="left")
fold = fold[(fold.config == "-") | (fold.ok == 1)]
fold.pivot(index="modelo", columns="fold", values="MAPE").round(2)""",
"""fig, ax = plt.subplots(figsize=(9, 4)); v = val.sort_values("MSE", ascending=False)
ax.barh(v.modelo, v.RMSE, color=[COL[m] for m in v.modelo]); ax.set_xlabel("RMSE en validación (MWh)"); plt.show()""",
"#MD **Lectura.**\n\n- Todos los modelos de ML le ganan con amplitud a los tres tradicionales en validación: entre 56 y 62 % menos MSE que la persistencia y entre 62 y 67 % menos que la misma hora de la semana pasada.\n- Por la regla gana el **random forest relativo** (MSE 2.111, MAPE 2,81 %), con una ventaja de 1,58 % sobre el XGBoost relativo (MSE 2.144), por encima del umbral de empate de 1 %.\n- El híbrido Ridge + XGBoost queda tercero (MSE 2.209) y Ridge cuarto (2.440).\n- En los 5 folds, los cuatro modelos de ML tienen menos error que los tres tradicionales. Entre ellos, el random forest y el XGBoost relativos se alternan el primer lugar de un fold a otro: son prácticamente equivalentes.",
])

nb("03_prueba.ipynb", [
"#MD# Prueba 2025–2026: ¿se confirma?\n\nEste capítulo evalúa los **benchmarks**; el modelo original ATER se agrega en el capítulo 6 sobre la misma prueba. Los modelos se reentrenan con todo el entrenamiento (octubre 2020 – diciembre 2024) y se evalúan **una sola vez** en 15.672 horas de enero 2025 a octubre 2026, con niveles de demanda récord.",
SETUP,
"""tp = pd.read_csv(R + "prueba.csv")
tp[["modelo", "MSE", "MSE_lo", "MSE_hi", "RMSE", "MAE", "MAPE", "mejora_MSE_vs_Persistencia_%", "mejora_vs_Persistencia_lo", "mejora_vs_Persistencia_hi",
    "mejora_MSE_vs_Estacional_%", "mejora_MSE_vs_Promedio por hora y día_%"]].round(2)""",
"""dm = pd.read_csv(R + "diebold_mariano.csv")
print("Diebold-Mariano (pérdida diaria, HAC 7 rezagos, corrección HLN) — p-valor de cada modelo frente a cada referencia:")
dm.pivot(index="modelo", columns="referencia", values="p").round(4)""",
"""pt = pd.read_pickle(R + "pred_prueba.pkl"); s = pt["2026-08-03":"2026-08-16"]
fig, ax = plt.subplots(figsize=(11, 4)); ax.plot(s.index, s.y, color="k", lw=1.6, label="real")
for m in ["Random forest (relativo)", "Persistencia (misma hora de ayer)", "Híbrido Ridge + XGBoost (residuo relativo)"]: ax.plot(s.index, s[m], lw=1, label=m, color=COL[m])
ax.set_ylabel("MWh"); ax.set_title("Dos semanas de prueba (agosto 2026)"); ax.legend(fontsize=8); plt.show()""",
"""seg = pd.read_csv(R + "mape_por_segmento.csv"); print("MAPE (%) por tipo de día y por cuartil de calor:"); seg.round(2)""",
"""v1 = pd.read_csv("../resultados_v1/prueba.csv")[["modelo", "MAPE", "mejora_MSE_vs_Persistencia_%"]]
print("VERSIÓN 1 (árboles con la demanda en MWh) — prueba:"); v1.round(2)""",
"""print("Sensibilidad ex-post (clima real del día pronosticado):"); pd.read_csv(R + "sensibilidad_expost.csv", index_col=0).round(3)""",
"#MD **Resultado.**\n\n- **Se confirma el elegido:** el random forest relativo tiene MAPE de **2,54 %** y reduce el MSE **64 %** frente a la persistencia (IC 95 % por bootstrap de bloques de 7 días: 60–69 %), **72 %** frente a la misma hora de la semana pasada y **96 %** frente al promedio por hora y día. Las tres diferencias son significativas (Diebold-Mariano p < 0,001).\n- Con el XGBoost relativo la diferencia no es significativa (p = 0,25): los dos árboles relativos son equivalentes. Al híbrido y a Ridge sí les gana (p < 0,001).\n- **La lección de la versión 1:** con los árboles entrenados en MWh, el XGBoost tenía 60 % *más* error que la persistencia y el random forest 194 % más. El objetivo relativo cambió eso a 63 % y 64 % *menos*. Esa es la contribución metodológica del proyecto.\n- La ventaja es mayor en días laborales (2,33 % contra 3,87 % de la persistencia) y se mantiene en los días más calurosos (3,01 % contra 5,31 %).\n- Con el clima real del día pronosticado (\"pronóstico del tiempo perfecto\") el MAPE bajaría a 2,43 %: un pronóstico meteorológico bueno aportaría poco más.",
])

nb("04_picos_valor.ipynb", [
"#MD# Horas pico y valor económico\n\n**Hora pico:** demanda mayor que el percentil 95 del entrenamiento (1.373 MWh). Anticiparlas un día antes sirve para preparar generación de respaldo y reducir el riesgo de cortes. Ojo: por el crecimiento de la demanda, en la prueba ese umbral se supera en 44 % de las horas; en la prueba la tarea equivale a anticipar las horas de demanda alta.",
SETUP,
"""r = json.load(open(R + "resumen.json")); print("Umbral de hora pico (p95 de entrenamiento): %.0f MWh" % r["umbral_pico_p95_mwh"])
tp = pd.read_csv(R + "prueba.csv"); tp[["modelo", "AUC_pico", "POD", "FAR", "CSI"]].round(3)""",
"""from sklearn.metrics import roc_curve
pt = pd.read_pickle(R + "pred_prueba.pkl"); o = (pt.y > r["umbral_pico_p95_mwh"]).astype(int)
print("Horas pico en la prueba:", int(o.sum()), "de", len(o))
fig, ax = plt.subplots(figsize=(5.5, 5))
for m in ["Random forest (relativo)", "Híbrido Ridge + XGBoost (residuo relativo)", "Ridge (lineal)", "Persistencia (misma hora de ayer)", "Estacional (misma hora semana pasada)"]:
    f, t, _ = roc_curve(o, pt[m]); ax.plot(f, t, label=m, color=COL[m])
ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos"); ax.legend(fontsize=7); plt.show()""",
"#MD## Valor económico: costo de los desvíos\nLa energía del día siguiente se compra con el pronóstico; cada MWh de diferencia con la demanda real se liquida en la bolsa. Supuesto simplificado: todo desvío, por exceso o por defecto, se paga al **precio de bolsa promedio del día** (XM, métrica PrecBolsNaci). Es una cota superior del valor.",
"""v = pd.read_csv(R + "valor_desvios.csv"); print(json.load(open(R + "valor_supuestos.json"))); v.round(1)""",
"#MD## ¿Qué aprende el modelo elegido?\nImportancia por permutación del random forest relativo en una muestra de la prueba: cuánto **sube el error (MAE en MWh)** si se desordena cada predictora.",
"""df = pd.read_pickle(R + "datos_horarios.pkl"); X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
tr = X.index < pd.Timestamp("2025-01-01") - pd.Timedelta(hours=48); te = X.index >= pd.Timestamp("2025-01-01")
m = P.GRILLAS_V2["Random forest (relativo)"]["hoja5"]().fit(X[tr], y[tr])
Xs = X[te].sample(4000, random_state=2026); ys = y[Xs.index]
base = np.mean(np.abs(ys.values - m.predict(Xs))); rng = np.random.default_rng(2026); filas = []
for c in Xs.columns:
    d = []
    for _ in range(5):
        Xp = Xs.copy(); Xp[c] = rng.permutation(Xp[c].values); d.append(np.mean(np.abs(ys.values - m.predict(Xp))) - base)
    filas.append((c, np.mean(d), np.std(d)))
print("MAE base de la muestra: %.1f MWh" % base)
imp = pd.DataFrame(filas, columns=["variable", "aumento_MAE_MWh", "sd"]).sort_values("aumento_MAE_MWh", ascending=False)
imp.to_csv(R + "importancia_rf_relativo.csv", index=False); imp.head(15).round(2)""",
"""top = imp.head(12).iloc[::-1]; fig, ax = plt.subplots(figsize=(7, 4.5)); ax.barh(top.variable, top.aumento_MAE_MWh, xerr=top.sd, color="#1F5A3A")
ax.set_xlabel("aumento del MAE (MWh) al permutar"); plt.show()""",
"#MD **Resultado.**\n\n- **Picos:** el random forest relativo anticipa con un día el **91 %** de las horas pico extremas con **10 %** de falsas alarmas (AUC 0,974; CSI 0,83). La persistencia detecta el 83 % pero con 17 % de falsas alarmas (CSI 0,72), y el promedio por hora y día no detecta ninguna.\n- **Plata:** con el precio de bolsa real de 2025–2026 (en promedio \\\\$350/kWh), el modelo elegido reduce el desvío de 1.392 a 816 MWh por día y el costo de los desvíos en unos **\\\\$75.000 millones al año** frente a la persistencia. Es una cota superior por el supuesto de liquidación, pero muestra el orden de magnitud.\n- **Qué aprende:** lo que más pesa es el **nivel y la historia reciente** (media de los 7 días anteriores, máximo de ayer, misma hora de las últimas 4 semanas, misma hora de ayer: 15–17 MWh de aumento del MAE cada una), luego la **hora** (13,3) y la **interacción calor × hora** (7,3): el calor de Sabanagrande pesa sobre todo en ciertas horas del día. Le siguen el tipo de día (6,2) y el día de la semana (2,8).",
])

nb("05_rubrica.ipynb", [
"#MD# Rúbrica ponderada\n\nPesos fijados de antemano, iguales a los del proyecto anterior del curso: MSE en validación 25 %, MSE en prueba 15 %, CSI de picos en validación 15 %, CSI en prueba 15 %, AUC de picos en validación 10 %, AUC en prueba 10 %, significancia frente a los tres modelos tradicionales 10 %. Cada criterio se escala contra el mejor modelo (100 = el mejor). Niveles: Excelente ≥ 90, Bueno 75–90, Aceptable 60–75, Insuficiente < 60.",
SETUP,
"""val = pd.read_csv(R + "validacion.csv").set_index("modelo"); tp = pd.read_csv(R + "prueba.csv").set_index("modelo"); dm = pd.read_csv(R + "diebold_mariano.csv")
TRAD = P.TRADICIONALES; M = list(val.index)
sig = {m: np.mean([((dm.modelo == m) & (dm.referencia == t) & (dm.DM > 0) & (dm.p < 0.05)).any() for t in TRAD if t != m]) for m in M}
def esc(v, menor=True): v = pd.Series(v); return (100*v.min()/v) if menor else (100*v/v.max())
def rub(W):
    c = pd.DataFrame(index=M)
    c["MSE_val"] = esc(val.loc[M, "MSE"]); c["MSE_test"] = esc(tp.loc[M, "MSE"])
    c["CSI_val"] = esc(val.loc[M, "CSI"], False); c["CSI_test"] = esc(tp.loc[M, "CSI"], False)
    c["AUC_val"] = esc(val.loc[M, "AUC_pico"] - .5, False); c["AUC_test"] = esc(tp.loc[M, "AUC_pico"] - .5, False)
    c["signif"] = pd.Series(sig)*100
    c["puntaje"] = sum(c[k]*w for k, w in W.items())
    c["nivel"] = pd.cut(c.puntaje, [-1, 60, 75, 90, 101], labels=["Insuficiente", "Aceptable", "Bueno", "Excelente"])
    return c.sort_values("puntaje", ascending=False)
W = {"MSE_val": .25, "MSE_test": .15, "CSI_val": .15, "CSI_test": .15, "AUC_val": .10, "AUC_test": .10, "signif": .10}
R0 = rub(W); R0.to_csv(R + "rubrica.csv"); R0.round(1)""",
"""alt = {"pesos iguales": {k: 1/7 for k in W}, "solo error": {"MSE_val": .5, "MSE_test": .5, "CSI_val": 0, "CSI_test": 0, "AUC_val": 0, "AUC_test": 0, "signif": 0},
       "solo picos": {"MSE_val": 0, "MSE_test": 0, "CSI_val": .25, "CSI_test": .25, "AUC_val": .25, "AUC_test": .25, "signif": 0}}
pd.DataFrame({k: rub(w).puntaje.rank(ascending=False).astype(int) for k, w in alt.items()} | {"rúbrica": R0.puntaje.rank(ascending=False).astype(int)})""",
"#MD La rúbrica coloca primero al **random forest relativo** (98,6, Excelente), seguido del XGBoost relativo (97,5) y del híbrido (90,3); Ridge queda en Bueno (85,1), la persistencia en Aceptable (71,2) y los otros dos tradicionales en Insuficiente. El random forest relativo es primero con todos los esquemas de pesos. Solo cuando se califican únicamente los picos la persistencia sube al tercer lugar, porque detecta muchas horas altas aunque con el doble de falsas alarmas.",
])
print("ok")
