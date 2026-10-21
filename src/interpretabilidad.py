"""Interpretabilidad: valores SHAP (explicador por permutación, independiente del modelo) para ATER y para el
mejor benchmark (random forest relativo), sobre las mismas 150 horas de la prueba y el mismo fondo.

ATER se explica con su estado congelado al final de la prueba (pesos y corrección de nivel fijos), que es el
modelo que usa el dashboard. Resultados: resultados/shap_*.csv
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd, joblib, shap
import pipeline_energia as P

OUT = Path("resultados"); SEED = P.SEED
df = pd.read_pickle(OUT / "datos_horarios.pkl")
X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
CORTE = pd.Timestamp("2025-01-01"); GAP = pd.Timedelta(hours=48)
Xtr, Xte = X[X.index < CORTE - GAP], X[X.index >= CORTE]

mo = joblib.load("artifacts/ater_modelo_original.joblib")
mo.predict(Xte, conservar_estado=True)            # recorre la prueba en línea: estado al 15-oct-2026
mo.en_linea = False                               # congelado
joblib.dump(mo, "artifacts/ater_estado_final.joblib", compress=3)
rf = mo.modelos_["Random forest (relativo)"]
COLS = list(Xte.columns)

def a_df(A):
    return pd.DataFrame(A, columns=COLS, index=pd.date_range("2030-01-01", periods=len(A), freq="h"))
f_ater = lambda A: mo.predict(a_df(A))
f_rf = lambda A: rf.predict(a_df(A))

rng = np.random.default_rng(SEED)
fondo = Xtr.iloc[rng.choice(len(Xtr), 50, replace=False)].values
filas = Xte.iloc[np.sort(rng.choice(len(Xte), 150, replace=False))]
res = {}
for nom, f in [("ATER", f_ater), ("RF", f_rf)]:
    ex = shap.PermutationExplainer(f, shap.maskers.Independent(fondo), seed=SEED)
    sv = ex(filas.values, max_evals=2 * len(COLS) + 1)
    v = pd.DataFrame(sv.values, columns=COLS, index=filas.index)
    v.to_csv(OUT / f"shap_valores_{nom}.csv")
    res[nom] = v.abs().mean()
    print(nom, "listo", flush=True)
imp = pd.DataFrame(res).rename(columns={"ATER": "ATER_media_abs_SHAP_MWh", "RF": "RF_media_abs_SHAP_MWh"})
imp = imp.sort_values("ATER_media_abs_SHAP_MWh", ascending=False)
imp["rank_ATER"] = imp.ATER_media_abs_SHAP_MWh.rank(ascending=False).astype(int)
imp["rank_RF"] = imp.RF_media_abs_SHAP_MWh.rank(ascending=False).astype(int)
imp.to_csv(OUT / "shap_importancia_global.csv")
filas.to_csv(OUT / "shap_filas_explicadas.csv")
print(imp.head(12).round(2))
from scipy.stats import spearmanr
print("Spearman rangos:", spearmanr(imp.ATER_media_abs_SHAP_MWh, imp.RF_media_abs_SHAP_MWh))
