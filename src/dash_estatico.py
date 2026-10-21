"""Genera la versión estática (HTML + Plotly) del dashboard con el modelo original ATER, desde resultados/."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve
sys.path.insert(0, "src")
import pipeline_energia as P

R = Path("resultados"); ORIG = "ATER (modelo original)"
tp = pd.read_csv(R / "orig_prueba.csv"); an = pd.read_csv(R / "orig_validacion_anidada.csv")
pr = pd.read_csv(R / "orig_pruebas_hipotesis.csv"); mcs = pd.read_csv(R / "orig_mcs.csv"); ab = pd.read_csv(R / "orig_ablacion.csv")
rub = pd.read_csv(R / "rubrica.csv").rename(columns={"Unnamed: 0": "modelo"})
vd = pd.read_csv(R / "valor_desvios.csv"); imp = pd.read_csv(R / "shap_importancia_global.csv", index_col=0).head(12)
dia = pd.read_csv(R / "diario.csv"); perf = pd.read_csv(R / "perfil_horario.csv"); cal = pd.read_csv(R / "calor_demanda_muestra.csv")
pt = pd.read_pickle(R / "pred_prueba_con_original.pkl")
desc = pd.read_csv(R / "problemas_descartados.csv"); ver = json.load(open(R / "orig_veredicto.json"))
pesos = pd.read_pickle(R / "orig_pesos_prueba.pkl"); pesos = pesos.groupby(pesos.index.hour).mean()
sem = pd.read_csv(R / "orig_semillas.csv"); v1 = pd.read_csv("resultados_v1/prueba.csv")
res = json.load(open(R / "resumen.json")); U = res["umbral_pico_p95_mwh"]
MOD = [c for c in pt.columns if c != "y"]

# MAPE por segmento (incluye ATER)
df = pd.read_pickle(R / "datos_horarios.pkl"); X, _ = P.construir(df); Xt = X.loc[pt.index]
segs = {"tipo de día": np.where(Xt.festivo == 1, "festivo", np.where(Xt.dow >= 5, "fin de semana", "laboral")),
        "calor": pd.qcut(Xt.calor, 4, labels=["frío", "templado", "caluroso", "muy caluroso"]).astype(str)}
seg = [dict(segmento=s, valor=k, **{m: 100 * np.mean(np.abs(g.y - g[m]) / g.y) for m in MOD})
       for s, lab in segs.items() for k, g in pt.groupby(lab)]
roc = {}
for m in MOD:
    f, t, _ = roc_curve((pt.y > U).astype(int), pt[m]); k = np.linspace(0, len(f) - 1, 150).astype(int)
    roc[m] = [np.round(f[k], 4).tolist(), np.round(t[k], 4).tolist()]
ef = pr.drop_duplicates("benchmark")
D = dict(
    elegido=ORIG, U=U, res=res, ver=ver,
    val=an.round(3).to_dict("records"),
    tp=tp[["modelo", "MSE", "MAPE", "mejora_vs_persistencia_%", "AUC_pico", "POD", "FAR", "CSI", "p_MCS", "inferencia_s"]].round(4).to_dict("records"),
    pr=pr[["prueba", "benchmark", "estadistico", "p_crudo", "p_holm", "conclusion"]].to_dict("records"),
    ef=ef[["benchmark", "mejora_MSE_pct", "IC95_lo", "IC95_hi", "dias_gana_ATER_pct", "delta_cliff_pareado", "delta_cliff", "d_cohen"]].round(3).to_dict("records"),
    mcs=mcs.round(4).to_dict("records"), ab=ab.round(2).to_dict("records"), sem=sem.round(1).to_dict("records"),
    pesos=dict(hora=pesos.index.tolist(), **{c: pesos[c].round(3).tolist() for c in pesos.columns}),
    desc=desc[["orden", "problema", "n_datos", "mejor_ML", "linea_base", "mejora_pct", "p_DM", "criterios_cumplidos", "veredicto"]].round(1).to_dict("records"),
    seg=pd.DataFrame(seg).round(2).to_dict("records"), rub=rub.round(1).to_dict("records"),
    vd=vd.round(1).to_dict("records"), imp=imp.reset_index().rename(columns={"index": "variable"}).round(2).to_dict("records"),
    v1=v1[["modelo", "MAPE", "mejora_MSE_vs_Persistencia_%"]].round(2).to_dict("records"),
    dia=dict(f=dia.fecha.tolist(), m=dia.demanda_media.tolist(), x=dia.demanda_max.tolist(), t=dia.tmax.tolist()),
    perf=perf.to_dict("list"), cal=cal.to_dict("list"), roc=roc,
    pt=dict(t0=str(pt.index[0]), y=pt.y.round(0).astype(int).tolist(), **{m: pt[m].round(0).astype(int).tolist() for m in MOD}), mods=MOD)
html = open("src/dash_plantilla.html", encoding="utf-8").read().replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False))
Path("publicar").mkdir(exist_ok=True)
Path("publicar/dashboard.html").write_text(html, encoding="utf-8"); print(len(html) / 1e6, "MB")
