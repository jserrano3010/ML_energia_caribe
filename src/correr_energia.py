"""Protocolo completo: EDA numérico, ajuste en validación, regla de elección, prueba única, significancia y picos."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import pipeline_energia as P
P.GRILLAS = P.GRILLAS_V2   # versión 2

OUT = Path("resultados"); OUT.mkdir(exist_ok=True)
CORTE = pd.Timestamp("2025-01-01"); GAP = pd.Timedelta(hours=48)
df = P.cargar(); df.to_pickle(OUT/"datos_horarios.pkl")

def preparar(expost=False):
    X, y = P.construir(df, expost=expost); ok = X.notna().all(axis=1); return X[ok], y[ok]

X, y = preparar()
assert not any(c in X.columns for c in ["mwh"]), "fuga: el objetivo no puede estar en las predictoras"
tr = X.index < CORTE - GAP; te = X.index >= CORTE
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
U95 = ytr.quantile(0.95)                     # umbral de "hora pico extremo" (solo con entrenamiento)
info = dict(filas=len(X), entrenamiento=int(tr.sum()), prueba=int(te.sum()), inicio=str(X.index.min()), fin=str(X.index.max()),
            corte_prueba=str(CORTE.date()), umbral_pico_p95_mwh=float(U95), n_predictoras=X.shape[1], predictoras=list(X.columns))
json.dump(info, open(OUT/"resumen.json", "w"), indent=1, ensure_ascii=False)
print(info)

def csi(yv, p, u=U95):
    o, f = yv > u, p > u; h = (o & f).sum(); m = (o & ~f).sum(); fa = (~o & f).sum()
    return dict(POD=h/(h+m) if h+m else 0, FAR=fa/(h+fa) if h+fa else 0, CSI=h/(h+m+fa) if h+m+fa else 0)

def metricas(yv, p):
    e = yv - p; d = dict(MSE=np.mean(e**2), RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), MAPE=100*np.mean(np.abs(e)/yv))
    o = (yv > U95).astype(int)
    d["AUC_pico"] = roc_auc_score(o, p) if 0 < o.sum() < len(o) else np.nan
    d.update(csi(yv, p)); return d

# ---------------- 1. ajuste de hiperparámetros y validación (5 folds, ventana creciente, purga 48 h)
filas, preds_val = [], []
for k, (a, b) in enumerate(P.folds(Xtr.index)):
    Xa, ya, Xb, yb = Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], ytr.iloc[b]
    pv = pd.DataFrame({"y": yb.values}, index=Xb.index); pv["fold"] = k
    for n in P.TRADICIONALES:
        pv[n] = P.pred_tradicional(n, Xa, ya, Xb)
        filas.append(dict(fold=k, modelo=n, config="-", **metricas(yb.values, pv[n].values)))
    for n, grid in P.GRILLAS.items():
        for g, f in grid.items():
            m = f().fit(Xa, ya); p = m.predict(Xb); pv[f"{n}|{g}"] = p
            filas.append(dict(fold=k, modelo=n, config=g, **metricas(yb.values, p)))
    preds_val.append(pv); print("fold", k, "listo", flush=True)
val = pd.DataFrame(filas); val.to_csv(OUT/"validacion_todas_config.csv", index=False)
pd.concat(preds_val).to_pickle(OUT/"pred_validacion.pkl")
vm = val.groupby(["modelo", "config"])[["MSE", "RMSE", "MAPE", "AUC_pico", "CSI", "POD", "FAR"]].mean().reset_index()
mejor_cfg = vm[vm.config != "-"].loc[lambda d: d.groupby("modelo").MSE.idxmin()].set_index("modelo").config.to_dict()
tabla_val = pd.concat([vm[vm.config == "-"], vm[[ (m in mejor_cfg and c == mejor_cfg[m]) for m, c in zip(vm.modelo, vm.config)]]])
tabla_val = tabla_val.sort_values("MSE").reset_index(drop=True)
for b in P.TRADICIONALES:
    tabla_val[f"mejora_MSE_vs_{b.split(' (')[0]}_%"] = 100*(1 - tabla_val.MSE/tabla_val.set_index("modelo").loc[b, "MSE"])
tabla_val.to_csv(OUT/"validacion.csv", index=False)
# regla fijada: menor MSE medio en validación; si la diferencia con el 2.º < 1 %, desempata el CSI de picos
t = tabla_val.sort_values("MSE").reset_index(drop=True)
elegido = t.modelo[0] if (t.MSE[1] - t.MSE[0])/t.MSE[0] >= 0.01 else t.loc[:1].sort_values("CSI", ascending=False).modelo.iloc[0]
json.dump(dict(elegido=elegido, configuraciones=mejor_cfg), open(OUT/"eleccion.json", "w"), indent=1, ensure_ascii=False)
print("ELEGIDO (validación):", elegido, mejor_cfg)

# ---------------- 2. prueba única (2025-01 a 2026-10)
pt = pd.DataFrame({"y": yte.values}, index=Xte.index)
for n in P.TRADICIONALES: pt[n] = P.pred_tradicional(n, Xtr, ytr, Xte)
fitted = {}
for n, g in mejor_cfg.items():
    m = P.GRILLAS[n][g]().fit(Xtr, ytr); pt[n] = m.predict(Xte); fitted[n] = m
pt.to_pickle(OUT/"pred_prueba.pkl")
MOD = P.TRADICIONALES + list(mejor_cfg)
tp = pd.DataFrame([dict(modelo=m, **metricas(pt.y.values, pt[m].values)) for m in MOD]).sort_values("MSE")
for b in P.TRADICIONALES:
    tp[f"mejora_MSE_vs_{b.split(' (')[0]}_%"] = 100*(1 - tp.MSE/tp.set_index("modelo").loc[b, "MSE"])

# IC 95 % por bootstrap de bloques de 7 días sobre la serie diaria de errores
rng = np.random.default_rng(2026)
dias = pt.index.normalize()
SE = pt.groupby(dias).apply(lambda d: pd.Series({m: np.sum((d.y - d[m])**2) for m in MOD} | {"n": len(d)}))
n = len(SE); bl = 7; IDX = []
for _ in range(1000):
    ini = rng.integers(0, n - bl + 1, int(np.ceil(n/bl))); IDX.append(np.concatenate([np.arange(i, i+bl) for i in ini])[:n])
IDX = np.array(IDX); NN = SE["n"].values[IDX].sum(1)
MSEB = {m: SE[m].values[IDX].sum(1)/NN for m in MOD}
PER = "Persistencia (misma hora de ayer)"
for m in MOD:
    tp.loc[tp.modelo == m, ["MSE_lo", "MSE_hi"]] = np.percentile(MSEB[m], [2.5, 97.5])
    for b in P.TRADICIONALES:
        if m != b:
            tp.loc[tp.modelo == m, [f"mejora_vs_{b.split(' (')[0]}_lo", f"mejora_vs_{b.split(' (')[0]}_hi"]] = np.percentile(100*(1 - MSEB[m]/MSEB[b]), [2.5, 97.5])
tp.to_csv(OUT/"prueba.csv", index=False)

# Diebold-Mariano sobre la pérdida diaria (MSE de las 24 h), HAC con 7 rezagos, corrección HLN
def dm(e1, e2, h=7):
    d = e1 - e2; n = len(d); db = d.mean(); g = [np.mean((d[k:]-db)*(d[:n-k]-db)) for k in range(h)]
    v = (g[0] + 2*sum((1 - k/h)*g[k] for k in range(1, h)))/n; s = db/np.sqrt(v)*np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    return s, 2*stats.t.sf(abs(s), n-1)
L = pt.groupby(dias).apply(lambda d: pd.Series({m: np.mean((d.y - d[m])**2) for m in MOD}))
dmr = []
for m in MOD:
    for ref in P.TRADICIONALES + [elegido]:
        if m == ref: continue
        s, p = dm(L[ref].values, L[m].values); dmr.append(dict(modelo=m, referencia=ref, DM=s, p=p))
pd.DataFrame(dmr).to_csv(OUT/"diebold_mariano.csv", index=False)

# ---------------- 3. error por tipo de día y hora (dónde gana el modelo)
pt2 = pt.assign(tipo=np.where(Xte.festivo == 1, "festivo", np.where(Xte.dow >= 5, "fin de semana", "laboral")),
                calor=pd.qcut(Xte.calor, 4, labels=["frío", "templado", "caluroso", "muy caluroso"]))
seg = []
for col in ["tipo", "calor"]:
    for k, g in pt2.groupby(col):
        seg.append(dict(segmento=col, valor=k, n=len(g), **{m: 100*np.mean(np.abs(g.y - g[m])/g.y) for m in MOD}))
pd.DataFrame(seg).to_csv(OUT/"mape_por_segmento.csv", index=False)

# ---------------- 4. sensibilidad ex-post (clima real de la hora objetivo) para el elegido y XGBoost
Xe, ye = preparar(expost=True); tre, tee = Xe.index < CORTE - GAP, Xe.index >= CORTE
sens = {}
for n in [elegido, "XGBoost (relativo)", "Ridge (lineal)"]:
    m = P.GRILLAS[n][mejor_cfg[n]]().fit(Xe[tre], ye[tre]); sens[n] = metricas(ye[tee].values, m.predict(Xe[tee]))
pd.DataFrame(sens).T.to_csv(OUT/"sensibilidad_expost.csv")

# ---------------- 5. importancia (XGBoost del híbrido o XGBoost solo) por ganancia
try:
    h = fitted["Híbrido Ridge + XGBoost (residuo relativo)"]
    imp = pd.Series(h.xgb.feature_importances_, index=list(P.relativas(Xtr).columns) + ["base_rel"]).sort_values(ascending=False)
    imp.to_csv(OUT/"importancia_hibrido_etapa2.csv")
except Exception as ex:
    print("importancia:", ex)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n=== VALIDACIÓN (media 5 folds) ==="); print(tabla_val.round(3).to_string(index=False))
print("\n=== PRUEBA 2025-01 a 2026-10 ==="); print(tp.round(3).to_string(index=False))
print(pd.DataFrame(dmr).query("referencia != @elegido").pivot(index="modelo", columns="referencia", values="p").round(4))
print(pd.DataFrame(dmr).query("referencia == @elegido").round(4).to_string(index=False))
print(pd.DataFrame(seg).round(2).to_string(index=False)); print(pd.DataFrame(sens).T.round(3))
