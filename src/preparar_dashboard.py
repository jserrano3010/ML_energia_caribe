"""Prepara los insumos livianos del dashboard (carpeta data/) y el artefacto del modelo para Render.

Render gratis tiene 512 MB de RAM: el random forest de 300 árboles (hojas de 5) ocupa demasiado. El artefacto del
dashboard usa los primeros 100 árboles del mismo bosque (reducción de árboles permitida por la guía, sección 4.2)
y se verifica que el desempeño en la prueba sea equivalente.
"""
import sys, json, copy, warnings, os, resource; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd, joblib
import pipeline_energia as P
from original_model import ModeloOriginal

R = Path("resultados"); D = Path("data"); D.mkdir(exist_ok=True); A = Path("artifacts")
df = pd.read_pickle(R / "datos_horarios.pkl")
X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
CORTE = pd.Timestamp("2025-01-01"); te = X.index >= CORTE
Xte, yte = X[te], y[te]

# ---------------- artefacto liviano: RF con 100 árboles
mo = joblib.load(A / "ater_modelo_original.joblib")
p300 = mo.predict(Xte)
ml = copy.deepcopy(mo)
rf = ml.modelos_["Random forest (relativo)"].m
rf.estimators_ = rf.estimators_[:100]; rf.n_estimators = 100
p100 = ml.predict(Xte)
rf300 = mo.modelos_["Random forest (relativo)"].predict(Xte); rf100 = ml.modelos_["Random forest (relativo)"].predict(Xte)
mse = lambda a: float(np.mean((yte.values - a) ** 2))
equiv = dict(MSE_ATER_300_arboles=mse(p300), MSE_ATER_100_arboles=mse(p100), MSE_RF_300=mse(rf300), MSE_RF_100=mse(rf100),
             cambio_MSE_ATER_pct=100 * (mse(p100) / mse(p300) - 1), cambio_MSE_RF_pct=100 * (mse(rf100) / mse(rf300) - 1))
print(equiv)
# estado final: recorrer la prueba en línea para que los pesos y la corrección de nivel queden al 15-oct-2026
ml.predict(Xte, conservar_estado=True)
ml.en_linea = False
joblib.dump(ml, A / "ater_dashboard.joblib", compress=3)
tam = os.path.getsize(A / "ater_dashboard.joblib") / 1e6
m2 = joblib.load(A / "ater_dashboard.joblib")
assert np.allclose(m2.predict(Xte.iloc[:240]), ml.predict(Xte.iloc[:240]))
meta = json.load(open(A / "metadata.json"))
meta_d = dict(meta, artefacto="ater_dashboard.joblib", reduccion="random forest con 100 de 300 árboles; estado de ATER al final de la prueba (congelado)",
              equivalencia_en_prueba=equiv, tamano_MB=tam)
json.dump(meta_d, open(A / "metadata_dashboard.json", "w"), indent=1, ensure_ascii=False, default=str)
print("artefacto del dashboard: %.1f MB" % tam, "| RSS máx del proceso: %.0f MB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

# ---------------- datos livianos
eda = pd.DataFrame({"mwh": df.mwh, "T2M": df.T2M, "RH2M": df.RH2M, "WS2M": df.WS2M, "lluvia": df.PRECTOTCORR,
                    "calor": P.indice_calor(df.T2M, df.RH2M)}, index=df.index)
Xe, _ = P.construir(df, expost=True)
eda["hora"] = df.index.hour; eda["dow"] = df.index.dayofweek; eda["mes"] = df.index.month; eda["anio"] = df.index.year
eda["festivo"] = Xe.festivo.values; eda["tipo_dia"] = Xe.tipo_dia.map({0: "laboral", 1: "sábado", 2: "domingo o festivo"}).values
eda.index.name = "ts"; eda.round(3).to_csv(D / "eda_horaria.csv.gz", compression="gzip")

pt = pd.read_pickle(R / "pred_prueba_con_original.pkl"); pt.index.name = "ts"; pt.round(2).to_csv(D / "pred_prueba.csv.gz", compression="gzip")
# predictoras de la prueba (para precargar el formulario de predicción en vivo) y fondo para SHAP local
Xte.rename_axis("ts").round(5).to_csv(D / "predictoras_prueba.csv.gz", compression="gzip")
rng = np.random.default_rng(2026)
X[X.index < CORTE].iloc[rng.choice((X.index < CORTE).sum(), 30, replace=False)].to_csv(D / "fondo_shap.csv")
# cuantiles de los residuos por hora y modelo (intervalo de predicción del 90 %)
q = []
for m in [c for c in pt.columns if c != "y"]:
    r = (pt.y - pt[m]).groupby(pt.index.hour)
    q.append(pd.DataFrame({"modelo": m, "hora": range(24), "q05": r.quantile(.05).values, "q95": r.quantile(.95).values}))
pd.concat(q).to_csv(D / "residuos_cuantiles.csv", index=False)
for f in ["orig_prueba.csv", "orig_validacion_anidada.csv", "orig_mcs.csv", "orig_pruebas_hipotesis.csv", "orig_ablacion.csv",
          "orig_optuna_trials.csv", "orig_sensibilidad.csv", "orig_semillas.csv", "shap_importancia_global.csv", "problemas_descartados.csv",
          "validacion.csv", "eleccion.json", "resumen.json", "orig_veredicto.json", "orig_hiperparametros.json", "valor_desvios.csv",
          "mape_por_segmento.csv"]:
    (D / f).write_bytes((R / f).read_bytes())
Wp = pd.read_pickle(R / "orig_pesos_prueba.pkl"); Wp.groupby(Wp.index.hour).mean().to_csv(D / "pesos_por_hora.csv")
print("datos listos:", sorted(p.name for p in D.iterdir()))
