"""Problema descartado: Baloto (simulación con las reglas reales, 5 de 43 + superbalota de 16)."""
import numpy as np, pandas as pd
from math import comb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.optimize import nnls
rng=np.random.default_rng(2026)
# --- Probabilidades exactas (5 de 43 + superbalota 1 de 16)
tot=comb(43,5)*16
print("Combinaciones totales:",tot)
for k in [5,4,3,2,1,0]:
    p5=comb(5,k)*comb(38,5-k)/comb(43,5)
    print(f"{k} aciertos + SB: 1 en {1/(p5/16):,.0f} | {k} aciertos sin SB: 1 en {1/(p5*15/16):,.0f}")
# --- Simulación de 2.700 sorteos con reglas reales
N=2700
draws=np.array([rng.choice(np.arange(1,44),5,replace=False) for _ in range(N)])
M=np.zeros((N,43),int)
for t,d in enumerate(draws): M[t,d-1]=1
# Clasificación: ¿sale la balota b en el sorteo t? predictoras solo con pasado
rows=[]
for t in range(200,N):
    past=M[:t]
    f10=past[-10:].mean(0); f50=past[-50:].mean(0); f200=past[-200:].mean(0); fall=past.mean(0)
    last=np.array([t-1-np.max(np.where(past[:,b])[0]) if past[:,b].any() else t for b in range(43)])
    for b in range(43):
        rows.append((t,b+1,f10[b],f50[b],f200[b],fall[b],last[b],M[t-1,b],M[t,b]))
df=pd.DataFrame(rows,columns="t balota f10 f50 f200 fhist sin_salir persist y".split())
X=["balota","f10","f50","f200","sin_salir"]
assert "y" not in X
cut=200+int(0.8*(N-200))
tr,te=df[df.t<cut],df[df.t>=cut]
res={}
res["Persistencia (salió en el anterior)"]=te.persist
res["Climatología (frecuencia histórica)"]=te.fhist
res["Dummy (promedio)"]=np.full(len(te),tr.y.mean())
lr=LogisticRegression(max_iter=500).fit(tr[X],tr.y); res["Logística (lineal)"]=lr.predict_proba(te[X])[:,1]
rf=RandomForestClassifier(300,min_samples_leaf=50,n_jobs=-1,random_state=0).fit(tr[X],tr.y); res["Random forest"]=rf.predict_proba(te[X])[:,1]
gb=HistGradientBoostingClassifier(random_state=0).fit(tr[X],tr.y); res["Gradient boosting"]=gb.predict_proba(te[X])[:,1]
res["Híbrido (lineal+RF)"]=(res["Logística (lineal)"]+res["Random forest"])/2
RES_C={}
print(f"\nClasificación ¿sale la balota?: {len(df):,} filas; prueba {len(te):,}")
for k,v in res.items():
    auc=roc_auc_score(te.y,v) if np.std(v)>0 else 0.5
    print(f"  {k:38s} AUC = {auc:.3f}  Brier = {mean_squared_error(te.y,v):.5f}"); RES_C[k]=auc
# Regresión: suma de las 5 balotas del próximo sorteo
s=draws.sum(1); R=pd.DataFrame({"y":s})
for L in [1,2,3,5,10]: R[f"lag{L}"]=pd.Series(s).shift(L)
R["m10"]=pd.Series(s).shift(1).rolling(10).mean(); R["m50"]=pd.Series(s).shift(1).rolling(50).mean()
R=R.dropna(); c=int(0.8*len(R)); a,b=R.iloc[:c],R.iloc[c:]; F=[x for x in R if x!="y"]
pr={"Dummy (promedio)":np.full(len(b),a.y.mean()),"Persistencia":b.lag1,"Media móvil 50":b.m50,
    "Ridge (lineal)":Ridge().fit(a[F],a.y).predict(b[F]),
    "Random forest":RandomForestRegressor(300,min_samples_leaf=20,random_state=0,n_jobs=-1).fit(a[F],a.y).predict(b[F]),
    "Gradient boosting":HistGradientBoostingRegressor(random_state=0).fit(a[F],a.y).predict(b[F])}
pr["Híbrido"]=(pr["Ridge (lineal)"]+pr["Random forest"])/2
base=mean_squared_error(b.y,pr["Dummy (promedio)"])
RES_R={}
print("\nRegresión: suma de las 5 balotas")
for k,v in pr.items():
    m=mean_squared_error(b.y,v); RES_R[k]=100*(m/base-1); print(f"  {k:22s} MSE = {m:8.1f}  vs promedio: {100*(m/base-1):+.1f} %")

from pathlib import Path
Path("datos_descartados").mkdir(exist_ok=True)
pd.DataFrame({"modelo": list(RES_C), "AUC_sale_balota": list(RES_C.values())}).to_csv("datos_descartados/baloto_clasificacion.csv", index=False)
pd.DataFrame({"modelo": list(RES_R), "cambio_MSE_vs_promedio_%": list(RES_R.values())}).to_csv("datos_descartados/baloto_regresion.csv", index=False)
