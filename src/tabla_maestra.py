"""Tabla maestra de experimentos: todas las combinaciones evaluadas (benchmarks y modelo original),
con la métrica del bucle externo, tiempos y semillas."""
import json, pandas as pd
R = "resultados/"
SEED = 2026
v = pd.read_csv(R + "validacion_todas_config.csv").assign(etapa="validación (5 folds, ventana creciente)", semilla=SEED)
a = pd.read_csv(R + "orig_validacion_anidada_folds.csv")
a = a[a.modelo == "ATER (modelo original)"].assign(etapa="validación anidada (bucle externo)", config="Optuna TPE (folds anteriores)", semilla=SEED)
t = pd.read_csv(R + "orig_prueba.csv").assign(etapa="prueba única 2025-01 a 2026-10", fold="prueba", semilla=SEED)
cfg = json.load(open(R + "eleccion.json"))["configuraciones"]; hp = json.load(open(R + "orig_hiperparametros.json"))["hiperparametros"]
t["config"] = t.modelo.map(lambda m: cfg.get(m, "-") if m != "ATER (modelo original)" else ", ".join(f"{k}={v:.4g}" for k, v in hp.items()))
s = pd.read_csv(R + "orig_semillas.csv")
s = pd.concat([s[["semilla", "MSE_ATER"]].rename(columns={"MSE_ATER": "MSE"}).assign(modelo="ATER (modelo original)"),
               s[["semilla", "MSE_Random forest (relativo)"]].rename(columns={"MSE_Random forest (relativo)": "MSE"}).assign(modelo="Random forest (relativo)"),
               s[["semilla", "MSE_XGBoost (relativo)"]].rename(columns={"MSE_XGBoost (relativo)": "MSE"}).assign(modelo="XGBoost (relativo)")]).assign(etapa="prueba, robustez de semilla", fold="prueba")
cols = ["etapa", "modelo", "config", "fold", "semilla", "MSE", "RMSE", "MAPE", "AUC_pico", "CSI", "entrenamiento_s", "inferencia_s"]
m = pd.concat([v, a, t, s], ignore_index=True).reindex(columns=cols)
m.to_csv(R + "tabla_maestra_experimentos.csv", index=False)
print(m.shape); print(m.etapa.value_counts())
