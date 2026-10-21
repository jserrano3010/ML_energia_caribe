"""Modelo original ATER: optimización bayesiana (Optuna/TPE), validación anidada temporal, prueba única,
ablación, pruebas de hipótesis (MCS, Giacomini-White, Diebold-Mariano HLN, Wilcoxon), ajuste de Holm,
bootstrap estacionario, tamaños del efecto, tiempos y artefactos serializados.

Ejecutar desde la raíz del repositorio:  python src/correr_original.py
Los expertos usan exactamente las configuraciones y la semilla de los benchmarks (pipeline_energia.SEED).
"""
import sys, json, time, warnings, platform
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd, joblib, optuna, sklearn, xgboost
from scipy import stats
from arch.bootstrap import MCS, StationaryBootstrap
import pipeline_energia as P
from original_model import AgregacionTermica, ModeloOriginal

P.GRILLAS = P.GRILLAS_V2
OUT = Path("resultados"); ART = Path("artifacts"); ART.mkdir(exist_ok=True)
SEED = P.SEED
CORTE = pd.Timestamp("2025-01-01"); GAP = pd.Timedelta(hours=48)
ORIG = "ATER (modelo original)"
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_pickle(OUT / "datos_horarios.pkl")
X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
tr = X.index < CORTE - GAP; te = X.index >= CORTE
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
eleccion = json.load(open(OUT / "eleccion.json"))
CFG = eleccion["configuraciones"]
BENCH = P.TRADICIONALES + ["Ridge (lineal)", "Random forest (relativo)", "XGBoost (relativo)",
                           "Híbrido Ridge + XGBoost (residuo relativo)"]
from original_model import _expertos
EXPERTOS = list(_expertos("completo"))
COL_V = {n: (n if n in P.TRADICIONALES else f"{n}|{CFG[n]}") for n in BENCH}

# ------------------------------------------------------------------ 1. matrices del núcleo en validación
pv = pd.read_pickle(OUT / "pred_validacion.pkl")
def matriz(Xi, F):
    dia = pd.factorize(Xi.index.normalize())[0]
    return np.column_stack([Xi.hora.values, Xi.calor.values, Xi.media_7d.values, dia, F])

VAL = {}
for k in sorted(pv.fold.unique()):
    f = pv[pv.fold == k]; Xi = X.loc[f.index]
    F = f[[COL_V[n] for n in EXPERTOS]].values
    VAL[k] = dict(M=matriz(Xi, F), y=f.y.values, f=f)

def correr_nucleo(par, M, yv):
    m = AgregacionTermica(**par).iniciar_vacio(M.shape[1] - 4)
    return m.predict(M, yv)                    # en línea: actualiza con la demanda real al cierre de cada día

def mse(a, b): return float(np.mean((a - b) ** 2))

def espacio(trial):
    return dict(eta=trial.suggest_float("eta", 1.0, 1000.0, log=True),
                lam=trial.suggest_float("lam", 0.90, 1.0),
                sigma=trial.suggest_float("sigma", 0.5, 50.0, log=True),
                eta_b=trial.suggest_float("eta_b", 0.0, 0.1))

def optimizar(folds_, n=80, semilla=SEED):
    def obj(trial):
        par = espacio(trial)
        return np.mean([mse(VAL[k]["y"], correr_nucleo(par, VAL[k]["M"], VAL[k]["y"])) for k in folds_])
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=semilla))
    st.optimize(obj, n_trials=n)
    return st

# ------------------------------------------------------------------ 2. validación anidada temporal
# bucle externo: fold k = 1..4; bucle interno: Optuna (TPE) solo con los folds anteriores a k
filas = []
for k in range(1, 5):
    st = optimizar(list(range(k)), n=60)
    p = correr_nucleo(st.best_params, VAL[k]["M"], VAL[k]["y"])
    yk, f = VAL[k]["y"], VAL[k]["f"]
    filas.append(dict(fold=k, modelo=ORIG, MSE=mse(yk, p), MAPE=100 * np.mean(np.abs(yk - p) / yk), **{f"par_{a}": b for a, b in st.best_params.items()}))
    for n in BENCH:
        filas.append(dict(fold=k, modelo=n, MSE=mse(yk, f[COL_V[n]].values), MAPE=100 * np.mean(np.abs(yk - f[COL_V[n]].values) / yk)))
    print("anidada fold", k, "listo", flush=True)
anid = pd.DataFrame(filas); anid.to_csv(OUT / "orig_validacion_anidada_folds.csv", index=False)
res_anid = anid.groupby("modelo").agg(MSE_media=("MSE", "mean"), MSE_de=("MSE", "std"), MAPE_media=("MAPE", "mean"),
                                      MAPE_de=("MAPE", "std")).sort_values("MSE_media").reset_index()
res_anid.to_csv(OUT / "orig_validacion_anidada.csv", index=False)
print(res_anid.round(3).to_string(index=False))

# ------------------------------------------------------------------ 3. hiperparámetros finales (todos los folds)
st = optimizar(list(range(5)), n=80)
PAR = st.best_params
tri = st.trials_dataframe()[["number", "value", "params_eta", "params_lam", "params_sigma", "params_eta_b"]]
tri["mejor_hasta_ahora"] = tri.value.cummin(); tri.to_csv(OUT / "orig_optuna_trials.csv", index=False)
json.dump(dict(hiperparametros=PAR, mse_validacion=st.best_value, metodo="Optuna TPE (bayesiano), 80 ensayos, semilla 2026"),
          open(OUT / "orig_hiperparametros.json", "w"), indent=1)
print("hiperparámetros finales:", PAR)

# sensibilidad a los hiperparámetros propios (en validación: cada uno se mueve con los demás fijos)
sens = []
for nom, vals in dict(eta=[3, 10, 30, 100, 300, 1000], lam=[0.9, 0.95, 0.98, 0.99, 1.0], sigma=[0.5, 1, 2, 4, 8, 16, np.inf],
                      eta_b=[0, 0.005, 0.01, 0.02, 0.05, 0.1]).items():
    for v in vals:
        par = dict(PAR); par[nom] = v
        sens.append(dict(parametro=nom, valor=v, MSE_val=np.mean([mse(VAL[k]["y"], correr_nucleo(par, VAL[k]["M"], VAL[k]["y"])) for k in VAL])))
pd.DataFrame(sens).to_csv(OUT / "orig_sensibilidad.csv", index=False)

# ------------------------------------------------------------------ 4. prueba única 2025-01 a 2026-10
pt = pd.read_pickle(OUT / "pred_prueba.pkl")
t0 = time.perf_counter(); mo = ModeloOriginal(**PAR, random_state=SEED).fit(Xtr, ytr); t_fit = time.perf_counter() - t0
t0 = time.perf_counter(); p_or = mo.predict(Xte, guardar_pesos=True); t_inf = time.perf_counter() - t0
W = mo.nucleo_.historia_pesos_
F_te = mo.pronosticos_expertos(Xte)
# coherencia: los expertos re-entrenados dentro del modelo reproducen las predicciones de los benchmarks
dif = {n: float(np.max(np.abs(F_te[:, i] - pt[n].values))) for i, n in enumerate(EXPERTOS)}
print("máxima diferencia expertos vs benchmarks:", dif)
pt[ORIG] = p_or
pt.to_pickle(OUT / "pred_prueba_con_original.pkl")
pd.DataFrame(W, index=Xte.index, columns=EXPERTOS).to_pickle(OUT / "orig_pesos_prueba.pkl")

# artefactos: modelo completo + prueba de ida y vuelta
joblib.dump(mo, ART / "ater_modelo_original.joblib", compress=3)
mo2 = joblib.load(ART / "ater_modelo_original.joblib")
dif_iv = float(np.max(np.abs(mo2.predict(Xte) - p_or)))
ida_vuelta = bool(dif_iv < 1e-6)   # el RF paralelo suma árboles en distinto orden: diferencias de ~1e-12 MWh
print("prueba de ida y vuelta:", ida_vuelta, "máx. diferencia (MWh):", dif_iv)

# tiempos de entrenamiento e inferencia de los benchmarks (mismo hardware)
tiempos = [dict(modelo=ORIG, entrenamiento_s=t_fit, inferencia_s=t_inf)]
for n in BENCH:
    if n in P.TRADICIONALES:
        t0 = time.perf_counter(); P.pred_tradicional(n, Xtr, ytr, Xte); tiempos.append(dict(modelo=n, entrenamiento_s=0.0, inferencia_s=time.perf_counter() - t0))
    else:
        t0 = time.perf_counter(); m = P.GRILLAS[n][CFG[n]]().fit(Xtr, ytr); a = time.perf_counter() - t0
        t0 = time.perf_counter(); m.predict(Xte); b = time.perf_counter() - t0
        tiempos.append(dict(modelo=n, entrenamiento_s=a, inferencia_s=b))
pd.DataFrame(tiempos).to_csv(OUT / "orig_tiempos.csv", index=False)

# ------------------------------------------------------------------ 5. ablación (prueba y validación)
yv = pt.y.values
M_te = mo._matriz(Xte, F_te); y_real = ModeloOriginal.realizado(Xte)
def nucleo_te(par, cols=None):
    M = M_te if cols is None else np.column_stack([M_te[:, :4], M_te[:, 4:][:, cols]])
    return AgregacionTermica(**par).iniciar_vacio(M.shape[1] - 4).predict(M, y_real)
def nucleo_val(par, cols=None):
    out = []
    for k in VAL:
        M = VAL[k]["M"] if cols is None else np.column_stack([VAL[k]["M"][:, :4], VAL[k]["M"][:, 4:][:, cols]])
        out.append(mse(VAL[k]["y"], correr_nucleo(par, M, VAL[k]["y"])))
    return float(np.mean(out))

abl = {"ATER completo": dict(PAR),
       "sin régimen térmico (sigma = ∞)": dict(PAR, sigma=np.inf),
       "sin corrección de nivel (eta_b = 0)": dict(PAR, eta_b=0.0),
       "sin régimen horario (pesos comunes)": dict(PAR, por_hora=False),
       "sin pesos aprendidos (eta = 0)": dict(PAR, eta=0.0),
       "sin olvido (lambda = 1)": dict(PAR, lam=1.0),
       "promedio simple (eta = 0, eta_b = 0)": dict(PAR, eta=0.0, eta_b=0.0)}
filas = []
for nom, par in abl.items():
    p = nucleo_te(par)
    filas.append(dict(variante=nom, grupo="componente", MSE_prueba=mse(yv, p), MAPE_prueba=100 * np.mean(np.abs(yv - p) / yv), MSE_validacion=nucleo_val(par)))
for i, e in enumerate(EXPERTOS):
    cols = [j for j in range(len(EXPERTOS)) if j != i]
    p = nucleo_te(PAR, cols)
    filas.append(dict(variante=f"sin el experto {e}", grupo="experto", MSE_prueba=mse(yv, p), MAPE_prueba=100 * np.mean(np.abs(yv - p) / yv), MSE_validacion=nucleo_val(PAR, cols)))

# reglas estándar de combinación (contraste exigido): pesos fijos aprendidos en validación
Fv = np.vstack([VAL[k]["M"][:, 4:] for k in VAL]); Yv = np.concatenate([VAL[k]["y"] for k in VAL])
w_inv = 1 / ((Fv - Yv[:, None]) ** 2).mean(0); w_inv /= w_inv.sum()
from scipy.optimize import nnls
w_st, _ = nnls(Fv, Yv)
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(Fv, Yv)
for nom, p in {"regla estándar: promedio simple": F_te.mean(1),
               "regla estándar: pesos por error global (1/MSE)": F_te @ w_inv,
               "regla estándar: stacking lineal no negativo": F_te @ w_st,
               "regla estándar: stacking lineal (MCO)": lr.predict(F_te),
               "mejor experto individual (random forest)": F_te[:, EXPERTOS.index("Random forest (relativo)")]}.items():
    filas.append(dict(variante=nom, grupo="regla estándar", MSE_prueba=mse(yv, p), MAPE_prueba=100 * np.mean(np.abs(yv - p) / yv), MSE_validacion=np.nan))
ab = pd.DataFrame(filas)
ref = ab.loc[ab.variante == "ATER completo", "MSE_prueba"].iloc[0]
ab["cambio_MSE_vs_completo_%"] = 100 * (ab.MSE_prueba / ref - 1)
ab.to_csv(OUT / "orig_ablacion.csv", index=False)
print(ab.round(3).to_string(index=False))

# ------------------------------------------------------------------ 6. métricas de prueba y pruebas de hipótesis
MOD = [ORIG] + BENCH
U95 = ytr.quantile(0.95)
from sklearn.metrics import roc_auc_score
def met(p):
    e = yv - p; o = yv > U95; f = p > U95; h = (o & f).sum(); mi = (o & ~f).sum(); fa = (~o & f).sum()
    return dict(MSE=np.mean(e ** 2), RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)), MAPE=100 * np.mean(np.abs(e) / yv),
                AUC_pico=roc_auc_score(o, p), CSI=h / (h + mi + fa), POD=h / (h + mi), FAR=fa / (h + fa))
tab = pd.DataFrame([dict(modelo=m, **met(pt[m].values)) for m in MOD]).sort_values("MSE").reset_index(drop=True)
dias = pt.index.normalize()
L = pt.groupby(dias).apply(lambda d: pd.Series({m: np.mean((d.y - d[m]) ** 2) for m in MOD}))       # pérdida diaria (MSE del día)
SE = pt.groupby(dias).apply(lambda d: pd.Series({m: np.sum((d.y - d[m]) ** 2) for m in MOD} | {"n": len(d)}))

# bootstrap estacionario (Politis y Romano, bloque medio 7 días) de la mejora del MSE frente a cada modelo
bs = StationaryBootstrap(7, SE.values, seed=SEED)
cols = list(SE.columns)
def mejora(arr):
    s = arr.sum(0); n_ = s[cols.index("n")]; m_or = s[cols.index(ORIG)] / n_
    return np.array([100 * (1 - m_or / (s[cols.index(b)] / n_)) for b in BENCH])
ci = bs.conf_int(mejora, reps=2000, method="percentile")
mej_puntual = mejora(SE.values)

def dm_hln(e1, e2, h=7):
    d = e1 - e2; n = len(d); db = d.mean(); g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(h)]
    v = (g[0] + 2 * sum((1 - k / h) * g[k] for k in range(1, h))) / n
    s = db / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return s, 2 * stats.t.sf(abs(s), n - 1)

def giacomini_white(lb, lo):
    """Prueba condicional de Giacomini-White (2006), horizonte 1 día, instrumentos h_{t-1} = [1, d_{t-1}]."""
    d = lb - lo; Z = np.column_stack([np.ones(len(d) - 1), d[:-1]]) * d[1:, None]
    n = len(Z); zb = Z.mean(0); Om = (Z - 0).T @ Z / n
    s = n * zb @ np.linalg.solve(Om, zb)
    return s, stats.chi2.sf(s, Z.shape[1])

def cliff(a, b):
    """delta de Cliff = P(a > b) - P(a < b); a = pérdida diaria del benchmark, b = del original (> 0 favorece al original)."""
    a = np.sort(a); b = np.asarray(b)
    mayor = np.searchsorted(a, b, side="right"); menor = np.searchsorted(a, b, side="left")
    return float(((len(a) - mayor).sum() - menor.sum()) / (len(a) * len(b)))

def cliff_pareado(a, b):
    """delta de Cliff para muestras dependientes (Cliff, 1993; Feng, 2007): P(a_t > b_t) - P(a_t < b_t), mismo día t."""
    d = np.asarray(a) - np.asarray(b); return float((d > 0).mean() - (d < 0).mean())

def magnitud(dl):
    a = abs(dl); return "despreciable" if a < .147 else "pequeño" if a < .33 else "mediano" if a < .474 else "grande"

pruebas = []
for i, b in enumerate(BENCH):
    lb, lo = L[b].values, L[ORIG].values; d = lb - lo
    s, p = dm_hln(lb, lo); pruebas.append(dict(prueba="Diebold-Mariano (HLN)", benchmark=b, estadistico=s, p_crudo=p))
    s, p = giacomini_white(lb, lo); pruebas.append(dict(prueba="Giacomini-White condicional", benchmark=b, estadistico=s, p_crudo=p))
    s, p = stats.wilcoxon(d); pruebas.append(dict(prueba="Wilcoxon de rangos con signo", benchmark=b, estadistico=s, p_crudo=p))
    for r in pruebas[-3:]:
        r.update(mejora_MSE_pct=mej_puntual[i], IC95_lo=ci[0, i], IC95_hi=ci[1, i], delta_cliff=cliff(lb, lo),
                 magnitud_cliff=magnitud(cliff(lb, lo)), delta_cliff_pareado=cliff_pareado(lb, lo),
                 magnitud_cliff_pareado=magnitud(cliff_pareado(lb, lo)), dias_gana_ATER_pct=100 * (d > 0).mean(),
                 d_cohen=d.mean() / d.std(ddof=1), dif_media_perdida=d.mean())
pr = pd.DataFrame(pruebas)
# ajuste de Holm-Bonferroni sobre TODAS las comparaciones (3 pruebas x 7 benchmarks = 21)
o = np.argsort(pr.p_crudo.values); m_ = len(pr); adj = np.empty(m_); run = 0
for j, idx in enumerate(o):
    run = max(run, min(1, (m_ - j) * pr.p_crudo.values[idx])); adj[idx] = run
pr["p_holm"] = adj
pr["n_comparaciones"] = m_
pr["supuestos"] = pr.prueba.map({"Diebold-Mariano (HLN)": "serie de diferencias de pérdida estacionaria (ADF); varianza HAC con 7 rezagos",
                                 "Giacomini-White condicional": "diferencias de pérdida con momentos finitos; esquema de ventana fija (expertos entrenados una vez)",
                                 "Wilcoxon de rangos con signo": "diferencias simétricas alrededor de la mediana; no corrige autocorrelación (complementaria)"})
pr["conclusion"] = np.where((pr.p_holm < 0.05) & (pr.dif_media_perdida > 0), "ATER mejor (rechaza H0)",
                            np.where(pr.p_holm < 0.05, "benchmark mejor", "sin diferencia significativa"))
pr.to_csv(OUT / "orig_pruebas_hipotesis.csv", index=False)

# MCS (Hansen, Lunde y Nason 2011): estadístico T_R, bootstrap estacionario, alfa = 0,10
mcs = MCS(L[MOD], size=0.10, reps=5000, block_size=7, method="R", bootstrap="stationary", seed=SEED)
mcs.compute()
tm = mcs.pvalues.copy(); tm.columns = ["p_MCS"]; tm.index.name = "modelo"; tm = tm.reset_index()
tm["en_conjunto_10pct"] = tm.modelo.isin(mcs.included)
tm.to_csv(OUT / "orig_mcs.csv", index=False)
print(tm.to_string(index=False))

# estacionariedad de las diferencias de pérdida (supuesto de DM)
from statsmodels.tsa.stattools import adfuller
adf = [dict(benchmark=b, ADF=adfuller(L[b].values - L[ORIG].values)[0], p_ADF=adfuller(L[b].values - L[ORIG].values)[1]) for b in BENCH]
pd.DataFrame(adf).to_csv(OUT / "orig_adf.csv", index=False)

tab = tab.merge(tm[["modelo", "p_MCS", "en_conjunto_10pct"]], on="modelo", how="left")
tab = tab.merge(pd.DataFrame(tiempos), on="modelo", how="left")
tab["mejora_vs_persistencia_%"] = 100 * (1 - tab.MSE / tab.set_index("modelo").loc[P.TRADICIONALES[0], "MSE"])
tab.to_csv(OUT / "orig_prueba.csv", index=False)

# veredicto R3 (definición operativa de la guía, sección 6.1)
best_bench = tab[tab.modelo != ORIG].modelo.iloc[0]
cond1 = bool(res_anid.modelo.iloc[0] == ORIG and tab.modelo.iloc[0] == ORIG)
rech = pr[(pr.benchmark == best_bench) & (pr.conclusion == "ATER mejor (rechaza H0)")].prueba.tolist()
mcs_ok = bool(mcs.included == [ORIG]) if isinstance(mcs.included, list) else False
n_pruebas = len(rech) + int(mcs_ok)
cliff_bb = pr[pr.benchmark == best_bench].delta_cliff.iloc[0]
cliffp_bb = pr[pr.benchmark == best_bench].delta_cliff_pareado.iloc[0]
dz_bb = pr[pr.benchmark == best_bench].d_cohen.iloc[0]
# las pruebas son pareadas (mismos días), así que el tamaño del efecto principal es el delta de Cliff pareado;
# se reportan también el no pareado y la d de Cohen, que con el umbral de la guía dan "despreciable"
cond3 = bool(abs(cliffp_bb) >= 0.147)
ver = dict(mejor_benchmark=best_bench, condicion1_metrica_media_mejor=cond1, pruebas_que_rechazan_vs_mejor_benchmark=rech + (["MCS (único sobreviviente)"] if mcs_ok else []),
           n_pruebas_rechazan=n_pruebas, condicion2_tres_pruebas=n_pruebas >= 3, delta_cliff_pareado=cliffp_bb, delta_cliff_no_pareado=cliff_bb, d_cohen=dz_bb,
           efecto_con_medidas_no_pareadas="despreciable" if (abs(cliff_bb) < .147 and abs(dz_bb) < .2) else "no despreciable",
           condicion3_efecto_no_despreciable=cond3, R3=bool(cond1 and n_pruebas >= 3 and cond3),
           mcs_incluidos=list(mcs.included), ida_y_vuelta=ida_vuelta, ida_y_vuelta_max_dif_MWh=dif_iv, hiperparametros=PAR)
json.dump(ver, open(OUT / "orig_veredicto.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(ver, indent=1, ensure_ascii=False, default=float))

# metadata del artefacto
meta = dict(modelo=ORIG, clase="original_model.ModeloOriginal", semilla_global=SEED, hiperparametros=PAR,
            expertos={e: CFG.get(e, "-") for e in EXPERTOS}, entrenado_con=f"{Xtr.index.min()} a {Xtr.index.max()} ({len(Xtr)} horas)",
            columnas_entrada={c: dict(tipo=str(Xtr[c].dtype), min=float(Xtr[c].min()), max=float(Xtr[c].max())) for c in Xtr.columns},
            metricas_prueba=tab[tab.modelo == ORIG].iloc[0][["MSE", "MAPE", "AUC_pico", "CSI"]].astype(float).to_dict(),
            validacion_anidada=res_anid[res_anid.modelo == ORIG].iloc[0][["MSE_media", "MSE_de"]].astype(float).to_dict(),
            versiones=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                           xgboost=xgboost.__version__, optuna=optuna.__version__),
            prueba_ida_y_vuelta=ida_vuelta, prueba_ida_y_vuelta_max_dif_MWh=dif_iv)
json.dump(meta, open(ART / "metadata.json", "w"), indent=1, ensure_ascii=False, default=str)
print("listo")
