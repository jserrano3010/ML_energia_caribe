"""Capítulo: las 28 combinaciones del Entregable 2 (7 modelos base x 4 optimizadores) sobre el problema de energía."""
import sys
import nbformat as nbf
from pathlib import Path
L = Path("libro")
LECTURAS = {}
exec(open("src/lecturas_e2.py", encoding="utf-8").read()) if Path("src/lecturas_e2.py").exists() else None


def nb(nombre, celdas):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:].lstrip("\n")) if c.startswith("#MD") else nbf.v4.new_code_cell(c) for c in celdas]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, L / nombre)


SETUP = r'''import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 90)
R = "../resultados/"
plt.rcParams.update({"figure.figsize": (10, 4), "axes.spines.top": False, "axes.spines.right": False})'''

nb("02b_entregable2.ipynb", [
r"""#MD# Las 28 combinaciones del Entregable 2

La guía del Entregable 2 pide, para regresión, **7 modelos base** (KNN, Ridge, Lasso, Decision Tree, Random Forest, XGBoost y SVR) combinados con **4 métodos de optimización** (Grid Search, Random Search, Bayesiano y Genético): $7 \times 4 = 28$ combinaciones. El balanceo de clases no aplica en regresión. Este capítulo las corre sobre el problema de energía con el mismo protocolo de ATER, elige la mejor combinación de cada modelo y la compara con ATER.

## Protocolo
- **Validación anidada temporal.** Bucle externo: folds 1 a 4 de la ventana creciente con purga de 48 h, los mismos que usa ATER en el capítulo 6. Bucle interno: dentro del entrenamiento de cada fold externo, el último 20 % (con purga de 48 h) es la validación donde el optimizador evalúa sus candidatos. La métrica que se reporta es la del bucle externo, nunca la del interno.
- **Presupuesto igual:** 12 evaluaciones por fold externo para cada optimizador, para comparar calidad por unidad de cómputo.
- **Sin fuga de información:** el escalado va dentro del pipeline y se ajusta solo con el entrenamiento de cada evaluación. Todos los modelos usan el objetivo relativo (demanda / media de 7 días), como los árboles del proyecto.
- **Semilla global 2026** en numpy, scikit-learn, XGBoost, Optuna y el algoritmo genético.

## Optimizadores
| Método | Implementación | Detalle |
|---|---|---|
| Grid Search | malla regular en el cubo unitario del espacio | niveles por dimensión $\lfloor 12^{1/d} \rfloor$; si la malla supera 12 puntos se toman 12 equiespaciados |
| Random Search | muestreo uniforme (o log-uniforme) | 12 puntos |
| Bayesiano | Optuna, *Tree-structured Parzen Estimator* | modelo sustituto TPE (densidades $l(x)$ y $g(x)$ de configuraciones buenas y malas); función de adquisición: mejora esperada, equivalente a maximizar $l(x)/g(x)$; 4 evaluaciones iniciales aleatorias (exploración) y 8 guiadas (explotación) |
| Genético | programado a mano con numpy | población de 4, 3 generaciones; selección por torneo ($k=2$); cruce uniforme ($p=0{,}8$); mutación gaussiana ($p=0{,}3$, $\sigma=0{,}15$ en el cubo unitario); elitismo de 1 individuo; se registra la diversidad genética |

## Espacios de búsqueda y complejidad
| Modelo | Hiperparámetros (rango) | Entrenamiento | Inferencia |
|---|---|---|---|
| KNN | vecinos 3–100 (log), pesos {uniforme, distancia}, $p \in \{1,2\}$ | $O(1)$ (guarda los datos; con árbol KD, $O(n \log n)$) | $O(n\,p)$ por consulta (fuerza bruta) |
| Ridge | $\alpha$ $10^{-3}$–$10^{3}$ (log) | $O(n\,p^2 + p^3)$ | $O(p)$ |
| Lasso | $\alpha$ $10^{-6}$–$10^{-1}$ (log) | $O(n\,p)$ por iteración (descenso por coordenadas) | $O(p)$ |
| Decision Tree | profundidad 3–25, hoja mínima 5–300 (log) | $O(p\,n \log n)$ | $O(\text{profundidad})$ |
| Random Forest | 100 árboles, muestra 50 %; profundidad 6–30, hoja 1–50 (log), fracción de variables 0,2–1 | $O(T\,p'\,n \log n)$ | $O(T \cdot \text{profundidad})$ |
| XGBoost (`hist`) | profundidad 2–10, tasa 0,01–0,3 (log), árboles 100–800 (log), peso mínimo 1–100 (log), submuestra 0,5–1 | $O(T\,p\,n)$ con histogramas | $O(T \cdot \text{profundidad})$ |
| SVR (Nyström + lineal) | $C$ $10^{-2}$–$10^{2}$ (log), $\gamma$ $10^{-3}$–1 (log), $\varepsilon$ 0–0,05 | $O(n\,m^2)$ con $m=300$ componentes, en vez de $O(n^2)$–$O(n^3)$ del kernel exacto | $O(m\,p)$ |

El SVR con kernel RBF exacto no es viable con ~30.000 filas de entrenamiento; la aproximación de Nyström más un SVR lineal (resuelto en el primal) es la alternativa que sugiere la guía. El random forest usa 100 árboles con muestras del 50 % para que las 28 combinaciones quepan en el presupuesto de cómputo.""",
SETUP,
r'''comb = pd.read_csv(R + "e2_combinaciones.csv")
print("Combinaciones evaluadas:", len(comb), "| evaluaciones totales del bucle interno:", int(comb.evaluaciones.sum()))
tab = comb.assign(MSE=lambda d: d.MSE_media.round(0).astype(int).astype(str) + " ± " + d.MSE_de.round(0).astype(int).astype(str))
tab.pivot(index="modelo", columns="optimizador", values="MSE")''',
r'''pv = comb.pivot(index="modelo", columns="optimizador", values="MSE_media")
fig, ax = plt.subplots(figsize=(9, 4.2)); im = ax.imshow(pv.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=15); ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]): ax.text(j, i, f"{pv.values[i, j]:,.0f}".replace(",", "."), ha="center", va="center", fontsize=8)
plt.colorbar(im, label="MSE externo medio (MWh²)"); ax.set_title("MSE del bucle externo: 7 modelos × 4 optimizadores"); plt.show()''',
r"""#MD## Mejor combinación de cada modelo""",
r'''mej = pd.read_csv(R + "e2_mejores.csv")
mej[["modelo", "optimizador", "MSE_media", "MSE_de", "MAPE_media", "MAPE_de", "hiperparametros", "MSE_prueba", "MAPE_prueba", "p_MCS"]].round(3)''',
LECTURAS.get("mejores", r"#MD"),
r"""#MD## ¿Qué optimizador rinde más por unidad de cómputo?""",
r'''efic = pd.read_csv(R + "e2_optimizadores.csv"); fr = json.load(open(R + "e2_friedman_optimizadores.json"))
print("Friedman (bloques = modelo × fold, %d bloques; tratamientos = %d optimizadores): chi2 = %.2f, p = %.4f" % (fr["bloques"], fr["tratamientos"], fr["friedman_chi2"], fr["p"]))
efic.round(2)''',
r'''an = pd.read_csv(R + "e2_anytime.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8, 4))
for c in an.columns: ax.plot(an.index, an[c], marker="o", label=c)
ax.axhline(1, color="k", lw=.8, ls=":"); ax.set_xlabel("evaluación"); ax.set_ylabel("mejor MSE interno hasta t / mejor final")
ax.set_title("Desempeño en cualquier momento (promedio de 7 modelos × 4 folds)"); ax.legend(); plt.show()''',
r'''dv = pd.read_csv(R + "e2_diversidad_genetica.csv").groupby("generacion").diversidad.agg(["mean", "std"])
print("Diversidad genética media por generación (desviación estándar de la población en el cubo unitario):"); dv.round(3)''',
LECTURAS.get("optimizadores", r"#MD"),
r"""#MD## Tiempos: complejidad teórica frente a la medida""",
r'''t = comb.groupby("modelo")[["tiempo_entrenamiento_s", "tiempo_inferencia_s"]].mean().sort_values("tiempo_entrenamiento_s")
t["busqueda_total_min"] = comb.groupby("modelo").tiempo_busqueda_s.sum() / 60
t.round(3)''',
LECTURAS.get("tiempos", r"#MD"),
r"""#MD## Las mejores combinaciones frente a ATER (prueba única 2025–2026)
Secuencia de la guía para regresión: **MCS** con todos los modelos (los 7 benchmarks del proyecto, las 7 mejores combinaciones del Entregable 2 y ATER), luego **Giacomini-White** y **Diebold-Mariano (HLN)** de ATER frente a cada mejor combinación, con ajuste de Holm sobre las 14 comparaciones, bootstrap estacionario para el intervalo y tamaño del efecto.""",
r'''tp = pd.read_csv(R + "e2_prueba.csv"); tp[["modelo", "MSE", "MAPE", "AUC_pico", "CSI", "mejora_vs_persistencia_%", "p_MCS", "en_conjunto_10pct", "inferencia_s"]].round(3)''',
r'''mcs = pd.read_csv(R + "e2_mcs_todos.csv"); print("Model Confidence Set con los 15 modelos (alfa = 0,10):"); mcs.round(4)''',
r'''pr = pd.read_csv(R + "e2_pruebas_vs_ater.csv")
pr[["prueba", "benchmark", "estadistico", "p_crudo", "p_holm", "mejora_MSE_pct", "IC95_lo", "IC95_hi", "delta_cliff_pareado", "d_cohen", "conclusion"]].round(4)''',
LECTURAS.get("vs_ater", r"#MD"),
])
print("ok")
