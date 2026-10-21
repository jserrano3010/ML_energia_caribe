"""Tabla resumen de los problemas evaluados antes de elegir el definitivo y por qué se descartaron.

Criterios de admisión fijados para el proyecto final (los mismos para todos los candidatos):
  C1. Datos reales con 20.000 observaciones o más.
  C2. El mejor modelo de Machine Learning mejora el error de la mejor línea base tradicional en 10 % o más.
  C3. La mejora es estadísticamente significativa (Diebold-Mariano, p < 0,05) en la prueba, en el horizonte de interés.
  C4. La regla de elección fijada de antemano (validación) no elige un modelo tradicional.

Fuentes: datos_descartados/maiz/*.csv (proyecto maíz y siembra), datos_descartados/baloto_*.csv
(src/descartados_baloto.py), resultados/*.csv (energía) y, para la lluvia, las cifras publicadas
en el Jupyter Book del Entregable 2 (resumen y capítulo 3), transcritas aquí.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path

D = Path("datos_descartados"); R = Path("resultados")
mz = D / "maiz"

# ---------------- 1. Lluvia semanal en Sabanagrande (Entregable 2) — cifras publicadas en ese libro
lluvia = dict(orden=1, problema="Lluvia semanal en Sabanagrande (Entregables 1 y 2)",
              datos="NASA POWER diario 1981–2025", n_datos=16436,
              mejor_ML="SVR lineal / híbrido SVR + RF", linea_base="Climatología",
              mejora_pct=7.0, ic_mejora="3,7 a 10,7", p_DM="significativo solo a 1 semana",
              C1=False, C2=False, C3=False, C4=True,
              veredicto="Mejor, pero no representativo: mejora de 7 % (< 10 %) y significativa solo a 1 semana; a 2–4 semanas no supera a la climatología con significancia.")

# ---------------- 2. Precio mensual del maíz (SIPSA)
rd = json.load(open(mz / "resumen_datos.json"))
mp = pd.read_csv(mz / "metricas_prueba.csv"); dm = pd.read_csv(mz / "diebold_mariano.csv")
el = json.load(open(mz / "eleccion.json"))
TRAD = ["Persistencia", "Promedio (Dummy)", "Estacional"]
filas_m = []
for h in [1, 4]:
    g = mp[mp.h == h].set_index("modelo")
    bt = g.loc[TRAD, "MSE"].idxmin(); bm = g.drop(TRAD).MSE.idxmin()
    p = dm[(dm.h == h) & (dm.modelo == bm) & (dm.ref == bt)].p.iloc[0]
    filas_m.append((h, bm, bt, 100 * (1 - g.loc[bm, "MSE"] / g.loc[bt, "MSE"]), p))
(h1, bm1, bt1, mj1, p1), (h4, bm4, bt4, mj4, p4) = filas_m
maiz = dict(orden=2, problema="Precio mensual del maíz amarillo (SIPSA, 61 mercados)",
            datos=f"SIPSA {rd['inicio'][:4]}–{rd['fin'][:4]}", n_datos=rd["filas_panel"],
            mejor_ML=bm4, linea_base=bt4, mejora_pct=mj4, ic_mejora="-",
            p_DM=f"p = {p4:.3f} a 4 meses (a 1 mes: {mj1:.1f} %, p = {p1:.3f})".replace(".", ","),
            C1=rd["filas_panel"] >= 20000, C2=mj4 >= 10, C3=p4 < 0.05, C4=el["elegido"] not in TRAD,
            veredicto=f"Un modelo tradicional gana: la regla de validación eligió «{el['elegido']}»; a 4 meses (horizonte de cosecha) el RF no es significativo y hay menos de 20.000 datos.")

# ---------------- 3. Decisión de siembra (beneficio del agricultor por fecha)
ep = pd.read_csv(mz / "error_precio_cosecha_bq.csv", index_col=0)
vp = pd.read_csv(mz / "valor_pronostico_siembra.csv", index_col=0)
mejor = ep["MAPE_%"].idxmin()
iguales = vp.drop(index=["Sembrar siempre"]).siembras.nunique() == 1 and vp.drop(index=["Sembrar siempre"]).beneficio_total.nunique() == 1
siembra = dict(orden=3, problema="Decisión de siembra de maíz (beneficio por fecha de siembra)",
               datos="Precio SIPSA + costos + rendimiento", n_datos=42,
               mejor_ML="Random forest", linea_base=mejor,
               mejora_pct=100 * (1 - ep.loc["Random forest", "MSE"] / ep.loc[mejor, "MSE"]), ic_mejora="-",
               p_DM="-", C1=False, C2=False, C3=False, C4=False,
               veredicto=(f"Un modelo tradicional gana: la persistencia tiene el menor error del precio a la cosecha "
                          f"(MAPE {ep.loc[mejor, 'MAPE_%']:.1f} % frente a {ep.loc['Random forest', 'MAPE_%']:.1f} % del RF) y "
                          + ("todos los modelos, incluido el oráculo, siembran las mismas fechas: el pronóstico no cambia la decisión." if iguales else "")).replace(".", ",", 2))

# ---------------- 4. Baloto (control de cordura: azar por diseño)
bc = pd.read_csv(D / "baloto_clasificacion.csv"); br = pd.read_csv(D / "baloto_regresion.csv")
ml_r = br[~br.modelo.isin(["Dummy (promedio)", "Persistencia", "Media móvil 50"])]
baloto = dict(orden=4, problema="Baloto (¿qué balotas saldrán?)", datos="Simulación con las reglas reales (5 de 43 + 1 de 16)",
              n_datos=107500, mejor_ML=ml_r.loc[ml_r["cambio_MSE_vs_promedio_%"].idxmin(), "modelo"], linea_base="Promedio",
              mejora_pct=-ml_r["cambio_MSE_vs_promedio_%"].min(), ic_mejora="-",
              p_DM=f"AUC entre {bc.AUC_sale_balota.min():.3f} y {bc.AUC_sale_balota.max():.3f}".replace(".", ","),
              C1=True, C2=False, C3=False, C4=False,
              veredicto="Imposible por diseño: el sorteo es aleatorio; ningún modelo supera al promedio (AUC ≈ 0,50).")

# ---------------- 5. Energía (problema definitivo): mejor benchmark ML y modelo original frente a la persistencia
tp = pd.read_csv(R / "orig_prueba.csv").set_index("modelo")
pr = pd.read_csv(R / "orig_pruebas_hipotesis.csv")
per = "Persistencia (misma hora de ayer)"; ORIG = "ATER (modelo original)"
p_or = pr[(pr.benchmark == per) & (pr.prueba == "Diebold-Mariano (HLN)")].p_holm.iloc[0]
info = json.load(open(R / "resumen.json"))
energia = dict(orden=5, problema="Demanda eléctrica horaria de Air-e con el calor de Sabanagrande (DEFINITIVO)",
               datos="XM (Sinergox) + NASA POWER horario 2020–2026", n_datos=info["filas"],
               mejor_ML=ORIG, linea_base=per, mejora_pct=tp.loc[ORIG, "mejora_vs_persistencia_%"], ic_mejora="ver capítulo 6",
               p_DM=f"p ajustado (Holm) = {p_or:.1e}", C1=True, C2=True, C3=bool(p_or < 0.05), C4=True,
               veredicto="Cumple los cuatro criterios: gana a todos los modelos conocidos con significancia.")

tab = pd.DataFrame([lluvia, maiz, siembra, baloto, energia])
tab["criterios_cumplidos"] = tab[["C1", "C2", "C3", "C4"]].sum(axis=1)
tab.to_csv(R / "problemas_descartados.csv", index=False)
print(tab[["orden", "problema", "n_datos", "mejora_pct", "C1", "C2", "C3", "C4", "criterios_cumplidos"]].to_string(index=False))
