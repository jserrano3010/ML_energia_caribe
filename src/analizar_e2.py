"""Análisis de las 28 combinaciones (7 modelos x 4 optimizadores): tabla de combinaciones, mejor combinación por modelo,
comparación de optimizadores (Friedman + curvas de desempeño en cualquier momento), prueba única con el reentrenamiento
de la mejor combinación de cada modelo, y comparación estadística con ATER (MCS, Giacomini-White, Diebold-Mariano, Holm)."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from arch.bootstrap import MCS, StationaryBootstrap
import pipeline_energia as P
import correr_e2 as C

OUT = Path("resultados"); SEED = 2026; ORIG = "ATER (modelo original)"
f = pd.read_csv(OUT / "e2_folds.csv"); cur = pd.read_csv(OUT / "e2_curvas.csv")

# ---------------------------------------------------------------- 1. las 28 combinaciones (bucle externo)
comb = f.groupby(["modelo", "optimizador"]).agg(MSE_media=("MSE", "mean"), MSE_de=("MSE", "std"), MAPE_media=("MAPE", "mean"),
                                                 MAPE_de=("MAPE", "std"), evaluaciones=("evaluaciones", "sum"),
                                                 tiempo_busqueda_s=("tiempo_busqueda_s", "sum"), tiempo_entrenamiento_s=("tiempo_entrenamiento_s", "mean"),
                                                 tiempo_inferencia_s=("tiempo_inferencia_s", "mean")).reset_index()
comb["rango_en_modelo"] = comb.groupby("modelo").MSE_media.rank().astype(int)
comb.sort_values(["modelo", "MSE_media"]).to_csv(OUT / "e2_combinaciones.csv", index=False)

mej = comb.loc[comb.groupby("modelo").MSE_media.idxmin()].sort_values("MSE_media").reset_index(drop=True)
# hiperparámetros finales: los del fold externo más reciente (el que usa más datos) de la mejor combinación
mej["hiperparametros"] = [f[(f.modelo == r.modelo) & (f.optimizador == r.optimizador)].sort_values("fold").hiperparametros.iloc[-1] for r in mej.itertuples()]
print(mej[["modelo", "optimizador", "MSE_media", "MSE_de", "MAPE_media"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- 2. comparación de optimizadores
# Friedman: bloques = (modelo, fold), tratamientos = 4 optimizadores, respuesta = MSE externo
pv = f.pivot_table(index=["modelo", "fold"], columns="optimizador", values="MSE")
fr = stats.friedmanchisquare(*[pv[c].values for c in pv.columns])
rangos = pv.rank(axis=1).mean().sort_values()
# curva de desempeño en cualquier momento, normalizada: mejor MSE interno hasta la evaluación t / mejor final de esa búsqueda (modelo, fold)
ref = cur.groupby(["modelo", "fold"]).mejor_mse_interno.transform("min")
cur["relativo"] = cur.mejor_mse_interno / ref
anytime = cur.groupby(["optimizador", "evaluacion"]).relativo.mean().unstack(0)
efic = pd.DataFrame({"rango_medio_Friedman": rangos,
                     "evaluaciones_para_quedar_a_1pct_del_mejor": [float(cur[cur.optimizador == o].assign(ok=lambda d: d.relativo <= 1.01)
                                                                         .query("ok").groupby(["modelo", "fold"]).evaluacion.min().mean()) for o in rangos.index],
                     "tiempo_busqueda_total_min": [comb[comb.optimizador == o].tiempo_busqueda_s.sum() / 60 for o in rangos.index],
                     "veces_mejor_optimizador": [int((mej.optimizador == o).sum()) for o in rangos.index]})
efic.index.name = "optimizador"; efic.to_csv(OUT / "e2_optimizadores.csv")
anytime.to_csv(OUT / "e2_anytime.csv")
json.dump(dict(friedman_chi2=float(fr.statistic), p=float(fr.pvalue), bloques=int(len(pv)), tratamientos=int(pv.shape[1])),
          open(OUT / "e2_friedman_optimizadores.json", "w"), indent=1)
print(efic.round(2)); print("Friedman optimizadores:", fr)

# diversidad genética (convergencia prematura)
div = f[f.optimizador == "Genético"][["modelo", "fold", "diversidad_genetica"]].copy()
div["diversidad"] = div.diversidad_genetica.apply(json.loads)
div = pd.DataFrame([dict(modelo=r.modelo, fold=r.fold, generacion=g + 1, diversidad=v) for r in div.itertuples() for g, v in enumerate(r.diversidad)])
div.to_csv(OUT / "e2_diversidad_genetica.csv", index=False)

# ---------------------------------------------------------------- 3. prueba única: mejor combinación de cada modelo, reentrenada con todo el entrenamiento
X, y = pd.read_pickle(OUT / "e2_Xy.pkl")
CORTE = pd.Timestamp("2025-01-01"); GAP = pd.Timedelta(hours=48)
tr, te = X.index < CORTE - GAP, X.index >= CORTE
pt = pd.read_pickle(OUT / "pred_prueba_con_original.pkl")
assert (pt.index == X.index[te]).all()
tiempos = []
for r in mej.itertuples():
    nom = f"E2 {r.modelo} ({r.optimizador})"
    t0 = time.perf_counter(); m = C.construir_modelo(r.modelo, json.loads(r.hiperparametros)).fit(X[tr], y[tr]); a = time.perf_counter() - t0
    t0 = time.perf_counter(); pt[nom] = m.predict(X[te]); b = time.perf_counter() - t0
    tiempos.append(dict(modelo=nom, entrenamiento_s=a, inferencia_s=b)); print(nom, "listo", flush=True)
pt.to_pickle(OUT / "pred_prueba_con_e2.pkl")
yv = pt.y.values; U95 = y[tr].quantile(0.95)
from sklearn.metrics import roc_auc_score
def met(p):
    e = yv - p; o = yv > U95; fz = p > U95; h = (o & fz).sum(); mi = (o & ~fz).sum(); fa = (~o & fz).sum()
    return dict(MSE=np.mean(e ** 2), MAPE=100 * np.mean(np.abs(e) / yv), AUC_pico=roc_auc_score(o, p), CSI=h / (h + mi + fa))
E2 = [t["modelo"] for t in tiempos]
tp = pd.DataFrame([dict(modelo=m, **met(pt[m].values)) for m in E2 + [ORIG]]).merge(pd.DataFrame(tiempos), on="modelo", how="left")
tp["mejora_vs_persistencia_%"] = 100 * (1 - tp.MSE / np.mean((yv - pt["Persistencia (misma hora de ayer)"].values) ** 2))

# ---------------------------------------------------------------- 4. MCS con TODOS los modelos y pruebas dirigidas ATER vs. cada mejor combinación
MOD = [c for c in pt.columns if c != "y"]
dias = pt.index.normalize()
L = pt.groupby(dias).apply(lambda d: pd.Series({m: np.mean((d.y - d[m]) ** 2) for m in MOD}))
mcs = MCS(L, size=0.10, reps=5000, block_size=7, method="R", bootstrap="stationary", seed=SEED); mcs.compute()
tm = mcs.pvalues.copy(); tm.columns = ["p_MCS"]; tm.index.name = "modelo"; tm = tm.reset_index(); tm["en_conjunto_10pct"] = tm.modelo.isin(mcs.included)
tm.to_csv(OUT / "e2_mcs_todos.csv", index=False)
print(tm.to_string(index=False))

def dm_hln(e1, e2, h=7):
    d = e1 - e2; n = len(d); db = d.mean(); g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(h)]
    v = (g[0] + 2 * sum((1 - k / h) * g[k] for k in range(1, h))) / n
    s = db / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n); return s, 2 * stats.t.sf(abs(s), n - 1)
def gw(lb, lo):
    d = lb - lo; Z = np.column_stack([np.ones(len(d) - 1), d[:-1]]) * d[1:, None]; n = len(Z); zb = Z.mean(0)
    s = n * zb @ np.linalg.solve(Z.T @ Z / n, zb); return s, stats.chi2.sf(s, 2)
SE = pt.groupby(dias).apply(lambda d: pd.Series({m: np.sum((d.y - d[m]) ** 2) for m in MOD} | {"n": len(d)}))
cols = list(SE.columns)
def mejora(arr):
    s = arr.sum(0); n_ = s[cols.index("n")]; mo = s[cols.index(ORIG)] / n_
    return np.array([100 * (1 - mo / (s[cols.index(b)] / n_)) for b in E2])
ci = StationaryBootstrap(7, SE.values, seed=SEED).conf_int(mejora, reps=2000, method="percentile"); mp = mejora(SE.values)
pr = []
for i, b in enumerate(E2):
    lb, lo = L[b].values, L[ORIG].values; d = lb - lo
    for nombre, (s, p) in [("Diebold-Mariano (HLN)", dm_hln(lb, lo)), ("Giacomini-White condicional", gw(lb, lo))]:
        pr.append(dict(prueba=nombre, benchmark=b, estadistico=s, p_crudo=p, mejora_MSE_pct=mp[i], IC95_lo=ci[0, i], IC95_hi=ci[1, i],
                       delta_cliff_pareado=float((d > 0).mean() - (d < 0).mean()), d_cohen=d.mean() / d.std(ddof=1), dif=d.mean()))
pr = pd.DataFrame(pr)
o = np.argsort(pr.p_crudo.values); m_ = len(pr); adj = np.empty(m_); run = 0
for j, idx in enumerate(o):
    run = max(run, min(1, (m_ - j) * pr.p_crudo.values[idx])); adj[idx] = run
pr["p_holm"] = adj; pr["n_comparaciones"] = m_
pr["conclusion"] = np.where((pr.p_holm < 0.05) & (pr.dif > 0), "ATER mejor (rechaza H0)", np.where(pr.p_holm < 0.05, "combinación E2 mejor", "sin diferencia significativa"))
pr.to_csv(OUT / "e2_pruebas_vs_ater.csv", index=False)
tp = tp.merge(tm, on="modelo", how="left").sort_values("MSE")
tp.to_csv(OUT / "e2_prueba.csv", index=False)
mej = mej.merge(tp.rename(columns={"modelo": "nombre_prueba"}).assign(modelo=lambda d: d.nombre_prueba.str.extract(r"^E2 (.*?) \(")[0])
                [["modelo", "MSE", "MAPE", "AUC_pico", "p_MCS", "en_conjunto_10pct", "inferencia_s"]].rename(columns={"MSE": "MSE_prueba", "MAPE": "MAPE_prueba", "inferencia_s": "inferencia_prueba_s"}),
                on="modelo", how="left")
mej.to_csv(OUT / "e2_mejores.csv", index=False)
print(tp.round(3).to_string(index=False)); print(pr[["benchmark", "prueba", "p_holm", "mejora_MSE_pct", "conclusion"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 5. tabla maestra: agregar las filas de las 28 combinaciones
tm_ = pd.read_csv(OUT / "tabla_maestra_experimentos.csv")
tm_ = tm_[~tm_.etapa.astype(str).str.startswith("Entregable 2")]
add = f.assign(etapa="Entregable 2: validación anidada (bucle externo)", config=f.optimizador + " | " + f.hiperparametros,
               entrenamiento_s=f.tiempo_entrenamiento_s, inferencia_s=f.tiempo_inferencia_s, modelo="E2 " + f.modelo)[
    ["etapa", "modelo", "config", "fold", "semilla", "MSE", "MAPE", "entrenamiento_s", "inferencia_s"]]
add2 = tp[tp.modelo != ORIG].assign(etapa="Entregable 2: prueba única (mejor combinación)", fold="prueba", semilla=SEED, config="mejor optimizador")[
    ["etapa", "modelo", "config", "fold", "semilla", "MSE", "MAPE", "AUC_pico", "CSI", "entrenamiento_s", "inferencia_s"]]
pd.concat([tm_, add, add2], ignore_index=True).to_csv(OUT / "tabla_maestra_experimentos.csv", index=False)
print("listo")
