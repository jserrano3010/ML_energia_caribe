"""Robustez ante la semilla: se re-entrenan los expertos aleatorios (RF, XGBoost, híbrido) con otras semillas
y se compara ATER (mismos hiperparámetros) contra el RF y el XGBoost entrenados con esa misma semilla."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
import pipeline_energia as P
from original_model import ModeloOriginal

df = pd.read_pickle("resultados/datos_horarios.pkl")
X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
CORTE = pd.Timestamp("2025-01-01"); GAP = pd.Timedelta(hours=48)
tr, te = X.index < CORTE - GAP, X.index >= CORTE
PAR = json.load(open("resultados/orig_hiperparametros.json"))["hiperparametros"]
filas = []
for s in [2026, 7, 42, 123, 999]:
    P.SEED = s                               # las fábricas leen P.SEED al construir cada modelo
    m = ModeloOriginal(**PAR, random_state=s).fit(X[tr], y[tr])
    F = m.pronosticos_expertos(X[te]); p = m.predict(X[te]); yt = y[te].values
    fila = dict(semilla=s, MSE_ATER=np.mean((yt - p) ** 2))
    for i, n in enumerate(m.nombres_):
        if n in ("Random forest (relativo)", "XGBoost (relativo)"):
            fila[f"MSE_{n}"] = np.mean((yt - F[:, i]) ** 2)
    fila["mejora_vs_RF_%"] = 100 * (1 - fila["MSE_ATER"] / fila["MSE_Random forest (relativo)"])
    filas.append(fila); print(fila, flush=True)
pd.DataFrame(filas).to_csv("resultados/orig_semillas.csv", index=False)
