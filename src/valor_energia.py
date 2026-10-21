"""Valor económico del pronóstico: costo de los desvíos entre lo programado (pronóstico del día anterior)
y la demanda real, valorados al precio de bolsa diario (XM). Supuesto: cada MWh de desvío (por exceso o defecto)
se liquida al precio de bolsa del día."""
import json, numpy as np, pandas as pd
from pathlib import Path
pt = pd.read_pickle("resultados/pred_prueba_con_original.pkl" if Path("resultados/pred_prueba_con_original.pkl").exists() else "resultados/pred_prueba.pkl")
pb = pd.read_csv("datos/precio_bolsa_diario_2025_2026.csv", parse_dates=["fecha"]).set_index("fecha").precio_bolsa_medio_cop_kwh
precio = pb.reindex(pt.index.normalize()).values * 1000          # COP/MWh
MOD = [c for c in pt.columns if c != "y"]
dias = (pt.index.max() - pt.index.min()).days + 1
filas = []
for m in MOD:
    desvio = np.abs(pt.y.values - pt[m].values)
    costo = np.nansum(desvio * precio)
    filas.append(dict(modelo=m, desvio_MWh_total=desvio.sum(), desvio_MWh_dia=desvio.sum()/dias,
                      costo_desvios_COP_millones=costo/1e6, costo_anual_COP_millones=costo/1e6*365/dias))
v = pd.DataFrame(filas).sort_values("costo_desvios_COP_millones")
ref = v.set_index("modelo").loc["Persistencia (misma hora de ayer)", "costo_anual_COP_millones"]
v["ahorro_anual_vs_persistencia_COP_millones"] = ref - v.costo_anual_COP_millones
v.to_csv("resultados/valor_desvios.csv", index=False)
json.dump(dict(dias_prueba=int(dias), precio_bolsa_medio_cop_kwh=float(pb.mean())), open("resultados/valor_supuestos.json", "w"))
pd.set_option("display.width", 220); print(v.round(1).to_string(index=False)); print("días", dias, "precio medio", round(pb.mean(), 1))
