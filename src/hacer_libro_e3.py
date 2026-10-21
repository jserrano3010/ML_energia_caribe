"""Capítulos del Entregable 3: problemas descartados, modelo original y validación estadística."""
import nbformat as nbf
from pathlib import Path
L = Path("libro")


def nb(nombre, celdas):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:].lstrip("\n")) if c.startswith("#MD") else nbf.v4.new_code_cell(c) for c in celdas]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, L / nombre)


SETUP = r'''import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 120)
R = "../resultados/"
plt.rcParams.update({"figure.figsize": (10, 4), "axes.spines.top": False, "axes.spines.right": False})
ORIG = "ATER (modelo original)"
COL = {"Persistencia (misma hora de ayer)": "#9AA5A0", "Estacional (misma hora semana pasada)": "#B8C2BD", "Promedio por hora y día": "#D5DBD8",
       "Ridge (lineal)": "#5B7F96", "Random forest (relativo)": "#1F5A3A", "XGBoost (relativo)": "#C8892B",
       "Híbrido Ridge + XGBoost (residuo relativo)": "#8E4B7A", ORIG: "#C0392B"}'''

# ============================================================================ problemas descartados
nb("01b_problemas_descartados.ipynb", [
r"""#MD# Problemas evaluados y por qué se descartaron

Antes de llegar a la demanda eléctrica se evaluaron otros cuatro problemas con el mismo protocolo: partición cronológica, líneas base tradicionales, modelos lineal, no lineal e híbrido, y prueba de Diebold-Mariano. Para el proyecto final se fijaron cuatro **criterios de admisión**, iguales para todos los candidatos:

| Criterio | Exigencia |
|---|---|
| C1 | Datos reales con 20.000 observaciones o más |
| C2 | El mejor modelo de Machine Learning reduce el error de la mejor línea base tradicional en 10 % o más |
| C3 | La mejora es estadísticamente significativa en la prueba (Diebold-Mariano, p < 0,05) en el horizonte que importa |
| C4 | La regla de elección fijada de antemano no elige un modelo tradicional |

Las cifras se leen de los archivos de resultados de cada proyecto (`datos_descartados/`); las de la lluvia se transcriben del Jupyter Book del Entregable 2.""",
SETUP,
r'''t = pd.read_csv(R + "problemas_descartados.csv")
t[["orden", "problema", "datos", "n_datos", "mejor_ML", "linea_base", "mejora_pct", "p_DM", "C1", "C2", "C3", "C4", "criterios_cumplidos"]].round(1)''',
r'''fig, ax = plt.subplots(figsize=(9, 3.6))
cols = ["#B8C2BD"] * 4 + ["#C0392B"]
ax.barh(t.problema.str.slice(0, 48)[::-1], t.mejora_pct[::-1], color=cols[::-1])
ax.axvline(10, color="k", ls="--", lw=1); ax.text(10.5, 4.3, "umbral C2 = 10 %", fontsize=8)
ax.set_xlabel("mejora del error del mejor ML frente a la mejor línea base tradicional (%)")
ax.set_title("Solo la demanda eléctrica supera con holgura a los métodos tradicionales"); plt.show()''',
r'''for _, r in t.iterrows():
    print(f"{r.orden}. {r.problema}\n   → {r.veredicto}\n")''',
r"""#MD## Lectura

1. **Lluvia semanal (Entregables 1 y 2): mejor, pero no representativa.** El modelo de ML mejora 7 % el error de la climatología a una semana (IC 95 %: 3,7–10,7 %), pero esa ventaja solo es significativa a una semana; a dos, tres y cuatro semanas no supera a la climatología con significancia. Además, la serie tiene 16.436 días (menos de 20.000) y la mejora queda por debajo del 10 %.
2. **Precio del maíz: lo gana un modelo tradicional.** La regla de validación eligió el **modelo estacional**. En la prueba, el random forest reduce el error 11,6 % a cuatro meses (el horizonte de la cosecha), pero sin significancia (p = 0,51), y el panel tiene 8.658 precios.
3. **Decisión de siembra: lo gana un modelo tradicional.** La **persistencia** tiene el menor error del precio a la cosecha (MAPE 10,4 % frente a 10,5 % del random forest), y todos los modelos, incluido el oráculo que conoce el precio real, recomiendan sembrar en las mismas 24 de 42 fechas: el pronóstico no cambia la decisión.
4. **Baloto: imposible por diseño.** El sorteo es aleatorio; con más de 100.000 filas, todos los AUC quedan en 0,50 ± 0,01 y ningún modelo supera al promedio. Sirvió como control de que el protocolo no "encuentra" patrones donde no los hay.
5. **Demanda eléctrica de Air-e con el calor de Sabanagrande (definitivo).** Es el único problema que cumple los cuatro criterios: 52.272 horas reales, mejora de 65 % frente a la persistencia, significativa en todas las pruebas, y la regla elige un modelo de Machine Learning. Sobre él se construyó el modelo original de los capítulos 5 y 6, que además le gana a los modelos de Machine Learning conocidos.""",
])

# ============================================================================ modelo original
nb("06_modelo_original.ipynb", [
r"""#MD# Modelo original: ATER

**ATER** = *Agregación de expertos por régimen horario y Térmico con corrección de nivel En línea y Relativa*.

## Motivación
Los capítulos de benchmarks dejaron tres hechos:

1. El random forest y el XGBoost relativos están **empatados** y se alternan el primer lugar de un fold a otro: ninguno domina siempre.
2. El mejor modelo **cambia según la hora**: de madrugada la demanda es casi la de ayer; a las 15:00 con calor fuerte pesan las interacciones no lineales.
3. La demanda **sube sin parar** (17,5 % en cuatro años): un modelo entrenado una vez se va quedando corto en nivel durante los 21 meses de prueba.

ATER no entrena un modelo más: aprende **a quién creerle, a qué hora y con cuánto calor**, y corrige el nivel cada día con lo que acaba de pasar. Usa como expertos los cinco pronósticos ya conocidos (random forest, XGBoost, híbrido, Ridge y persistencia) y una regla de combinación propia.

## Formulación
Sea $f_{k,t}$ el pronóstico del experto $k \in \{1,\dots,K\}$ para la hora $t$, $h(t)\in\{0,\dots,23\}$ la hora del día, $c_t$ el índice de calor conocido al emitir (°C), $n_t$ el nivel (media de los 7 días anteriores, MWh) y $y_t$ la demanda real. La pérdida **relativa** del experto $k$ en $t$, acotada a $[0,1]$, es

$$\ell_{k,t} = \min\left\{1,\; \left(\frac{(1+b_{h(t)})\,f_{k,t} - y_t}{n_t}\right)^2\right\}$$ (eq-perdida)

ATER guarda, para cada hora $h$ y cada banda de calor $j$ (centros $\tau_j$ = 20, 21, …, 47 °C), la pérdida acumulada con olvido $\lambda$:

$$S_{h,j,k} \leftarrow \lambda\, S_{h,j,k} + \sum_{t \in \text{día},\, h(t)=h,\, \text{banda}(c_t)=j} \ell_{k,t}$$ (eq-acumulador)

Para pronosticar la hora $t$ suaviza esas pérdidas con un núcleo gaussiano sobre el calor y calcula pesos exponenciales:

$$L_{k,t} = \sum_j e^{-\frac{(c_t-\tau_j)^2}{2\sigma^2}}\, S_{h(t),j,k}, \qquad w_{k,t} = \frac{e^{-\eta L_{k,t}}}{\sum_{i=1}^K e^{-\eta L_{i,t}}}$$ (eq-pesos)

El pronóstico combina los expertos y aplica una corrección multiplicativa de nivel por hora:

$$\hat y_t = (1 + b_{h(t)}) \sum_{k=1}^K w_{k,t}\, f_{k,t}$$ (eq-pronostico)

Al cierre de cada día, $b_h$ se actualiza por **descenso de gradiente en línea** sobre la pérdida relativa $g(b)=\left((1+b)\bar f_t - y_t\right)^2/n_t^2$, con $\bar f_t=\sum_k w_{k,t} f_{k,t}$:

$$\frac{\partial g}{\partial b} = \frac{2\,(\hat y_t - y_t)\,\bar f_t}{n_t^2}, \qquad b_h \leftarrow b_h - \eta_b \sum_{t:\,h(t)=h} \frac{2\,(\hat y_t - y_t)\,\bar f_t}{n_t^2}$$ (eq-nivel)

**Hiperparámetros propios:** $\eta$ (tasa de los pesos), $\lambda$ (olvido), $\sigma$ (ancho del núcleo de calor, °C) y $\eta_b$ (tasa de la corrección de nivel).

**Sin fuga de información.** El pronóstico del día $D+1$ se emite al final del día $D$ y solo usa pérdidas hasta el día $D$. ATER no necesita $y$ para pronosticar: la demanda real del día $D$ es el `lag24` de las filas del día $D+1$, que ya está en las predictoras (y una prueba unitaria verifica que cambiar la demanda de un día no cambia los pronósticos de ese mismo día).""",
r"""#MD## Pseudocódigo

```text
Entradas : pronósticos de K expertos f[t,k], hora h[t], índice de calor c[t], nivel n[t],
           demanda real y[t] (se conoce al cierre de cada día), hiperparámetros η, λ, σ, η_b
Salida   : pronósticos ŷ[t]
Estado   : S[24, B, K] ← 0      (pérdidas acumuladas por hora, banda de calor y experto)
           b[24]       ← 0      (corrección de nivel por hora)

para cada día d en orden cronológico:
    # 1) pronóstico del día completo (emitido al cierre del día anterior)
    para cada hora t del día d:
        κ[j]   ← exp(-(c[t] - τ[j])² / (2σ²))                      para j = 1..B
        L[k]   ← Σ_j κ[j] · S[h[t], j, k]                           para k = 1..K
        w[k]   ← exp(-η (L[k] - min L)) / Σ_i exp(-η (L[i] - min L))
        f̄      ← Σ_k w[k] f[t,k]
        ŷ[t]   ← (1 + b[h[t]]) · f̄
    # 2) actualización al cierre del día d (cuando se conoce y)
    S ← λ · S
    para cada hora t del día d:
        ℓ[k]   ← min(1, (((1 + b[h[t]]) f[t,k] - y[t]) / n[t])²)    para k = 1..K
        S[h[t], banda(c[t]), :] += ℓ
        b[h[t]] ← b[h[t]] - η_b · 2 (ŷ[t] - y[t]) f̄ / n[t]²
devolver ŷ
Criterio de parada: se recorre cada día una sola vez (algoritmo en línea, una pasada).
```""",
r"""#MD## Sustento teórico

**Propiedad 1 (cota de arrepentimiento de los pesos).** Para un régimen fijo (una hora, sin núcleo de calor: $\sigma=\infty$, sin olvido: $\lambda=1$) la regla [](eq-pesos) es el pronosticador de pesos exponenciales (*Hedge*). Con pérdidas en $[0,1]$ —por eso la pérdida [](eq-perdida) se acota—, después de $T$ rondas

$$\sum_{t=1}^T \sum_k w_{k,t}\ell_{k,t} - \min_k \sum_{t=1}^T \ell_{k,t} \;\le\; \frac{\ln K}{\eta} + \frac{\eta\, T}{8}$$ (eq-cota)

(Cesa-Bianchi y Lugosi, 2006, teorema 2.2). *Esquema de la prueba:* con el potencial $\Phi_t=\frac{1}{\eta}\ln\sum_k e^{-\eta L_{k,t}}$, el lema de Hoeffding da $\Phi_{t}-\Phi_{t-1}\le -\sum_k w_{k,t}\ell_{k,t}+\eta/8$; sumando y usando $\Phi_0=\frac{\ln K}{\eta}$ y $\Phi_T\ge -\min_k L_{k,T}$ se obtiene [](eq-cota). Con $\eta^*=\sqrt{8\ln K/T}$ la cota es $\sqrt{T\ln K/2}$: el arrepentimiento promedio tiende a cero. El núcleo y el olvido convierten la regla en una versión *local* (en calor) y *de seguimiento* (en el tiempo) de Hedge; la cota exacta para esas variantes no se demuestra aquí y se verifica empíricamente abajo.

**Propiedad 2 (convexidad de la corrección de nivel).** $g(b)$ es una parábola en $b$ con $g''(b)=2\bar f_t^2/n_t^2>0$: es convexa, y [](eq-nivel) es descenso de gradiente en línea, que con paso decreciente $\eta_b\propto 1/\sqrt{t}$ tiene arrepentimiento $O(\sqrt T)$ (Zinkevich, 2003). Con paso constante sigue un sesgo que cambia en el tiempo, que es lo que se necesita con una demanda que crece. La prueba unitaria `test_correccion_de_nivel_aprende_un_sesgo` verifica que $b_h$ converge a un sesgo conocido de +5 %.

**Propiedad 3 (complejidad).** Por hora pronosticada: el núcleo cuesta $O(B)$, las pérdidas suavizadas $O(BK)$ y la actualización $O(K)$. Total $O(n\,B\,K)$ en tiempo y $O(24\,B\,K)$ en memoria (independiente de $n$). Con $B=28$ bandas y $K=5$, el estado ocupa 3.360 números. El entrenamiento cuesta lo que cuestan los expertos; el núcleo agrega una pasada lineal.""",
SETUP,
r'''# Verificación empírica de la Propiedad 1 en la prueba: Hedge puro por hora (σ = ∞, λ = 1, η_b = 0)
import joblib
from original_model import AgregacionTermica, ModeloOriginal
import pipeline_energia as P
df = pd.read_pickle(R + "datos_horarios.pkl"); X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
Xte = X[X.index >= pd.Timestamp("2025-01-01")]
mo = joblib.load("../artifacts/ater_modelo_original.joblib")
F = mo.pronosticos_expertos(Xte); M = mo._matriz(Xte, F); yr = ModeloOriginal.realizado(Xte); yv = y[Xte.index].values
par = json.load(open(R + "orig_hiperparametros.json"))["hiperparametros"]
h = AgregacionTermica(eta=par["eta"], lam=1.0, sigma=np.inf, eta_b=0.0).iniciar_vacio(F.shape[1])
p = h.predict(M, yr, guardar_pesos=True); W = h.historia_pesos_
ell = np.minimum(((F - yv[:, None]) / Xte.media_7d.values[:, None]) ** 2, 1)
filas = []
for hr in range(24):
    s = Xte.hora.values == hr; T = s.sum()
    reg = (W[s] * ell[s]).sum() - ell[s].sum(0).min()
    filas.append(dict(hora=hr, T=T, arrepentimiento=reg, cota=np.log(F.shape[1]) / par["eta"] + par["eta"] * T / 8))
reg = pd.DataFrame(filas)
print("Horas en que el arrepentimiento empírico es menor que la cota teórica: %d de 24" % (reg.arrepentimiento <= reg.cota).sum())
print("Arrepentimiento empírico medio por hora: %.4f | negativo (mejor que el mejor experto fijo) en %d de 24 horas" % (reg.arrepentimiento.mean(), (reg.arrepentimiento < 0).sum()))
reg.round(4).head(6)''',
r"""#MD**Lectura.** La cota [](eq-cota) se cumple en las 24 horas, pero es muy holgada: con el $\eta$ que eligió Optuna para minimizar el error (no para optimizar la cota) el término $\eta T/8$ domina. El arrepentimiento empírico frente al **mejor experto fijo elegido a posteriori** es positivo pero mínimo: 0,025 unidades de pérdida relativa acumuladas en 653 días por hora, del orden de $4\times10^{-5}$ por día. En la práctica, ese "mejor experto fijo" no se conoce de antemano, y la versión completa de ATER (con corrección de nivel) le gana al mejor experto individual en la prueba (ablación, abajo).""",
r'''# Complejidad empírica: tiempo del núcleo al variar n (horas) y K (expertos)
import time
rng = np.random.default_rng(0); filas = []
for K in [2, 5, 10]:
    for dias in [250, 500, 1000, 2000]:
        n = dias * 24; hora = np.tile(np.arange(24), dias); dia = np.repeat(np.arange(dias), 24)
        Fs = 1000 + rng.normal(0, 50, (n, K)); ys = 1000 + rng.normal(0, 30, n)
        Ms = np.column_stack([hora, rng.uniform(22, 45, n), np.full(n, 1000.0), dia, Fs])
        m = AgregacionTermica().iniciar_vacio(K); t0 = time.perf_counter(); m.predict(Ms, ys); filas.append(dict(K=K, n=n, segundos=time.perf_counter() - t0))
cx = pd.DataFrame(filas)
for K, g in cx.groupby("K"):
    pend = np.polyfit(np.log(g.n), np.log(g.segundos), 1)[0]; print(f"K = {K:2d}: pendiente log(tiempo) vs log(n) = {pend:.2f} (teoría: 1, lineal)")
fig, ax = plt.subplots(figsize=(6, 3.5))
for K, g in cx.groupby("K"): ax.loglog(g.n, g.segundos, "o-", label=f"K = {K}")
ax.set_xlabel("n (horas)"); ax.set_ylabel("segundos"); ax.set_title("Tiempo del núcleo de ATER: lineal en n"); ax.legend(); plt.show()''',
r"""#MD## Código
La implementación completa está en `src/original_model.py` (clases `AgregacionTermica`, el núcleo en numpy, y `ModeloOriginal`, que entrena los expertos). Estas son las tres piezas del algoritmo:""",
r'''import inspect
from original_model import AgregacionTermica
for f in [AgregacionTermica._pesos, AgregacionTermica._actualizar, AgregacionTermica._recorrer]:
    print(inspect.getsource(f))''',
r"""#MD## Pruebas unitarias
`tests/test_original_model.py` verifica: (i) que el modelo aprende en un problema sintético con solución conocida (los pesos se concentran en el experto exacto; la corrección de nivel recupera un sesgo de +5 %; con $\eta=\eta_b=0$ es exactamente el promedio), (ii) la reproducibilidad y que `predict` no altera el estado, (iii) el manejo de entradas e hiperparámetros inválidos, (iv) la compatibilidad con `sklearn.base.clone`, y además que **no usa el futuro**.""",
r'''import subprocess
out = subprocess.run([sys.executable, "-m", "pytest", "-q", "../tests", "-p", "no:cacheprovider"], capture_output=True, text=True)
print(out.stdout[-1500:])''',
r"""#MD## Optimización de hiperparámetros (bayesiana)
Los cuatro hiperparámetros de ATER se optimizan con **Optuna (TPE, bayesiano)**, 80 ensayos, semilla 2026, minimizando el MSE medio de los 5 folds de validación (los mismos folds, la misma purga y la misma semilla que los benchmarks). En la validación anidada del capítulo 6, la optimización se repite dentro de cada fold externo usando solo los folds anteriores.""",
r'''tri = pd.read_csv(R + "orig_optuna_trials.csv"); hp = json.load(open(R + "orig_hiperparametros.json")); print(hp)
fig, ax = plt.subplots(figsize=(8, 3.5)); ax.scatter(tri.number, tri.value, s=10, color="#9AA5A0", label="ensayo")
ax.plot(tri.number, tri.mejor_hasta_ahora, color="#C0392B", label="mejor hasta ahora"); ax.set_ylim(tri.value.min() * .99, tri.value.quantile(.9))
ax.set_xlabel("ensayo de Optuna"); ax.set_ylabel("MSE medio en validación (MWh²)"); ax.set_title("Curva de convergencia de la optimización"); ax.legend(); plt.show()''',
r'''sen = pd.read_csv(R + "orig_sensibilidad.csv")
fig, axs = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
for ax, (nom, g) in zip(axs, sen.groupby("parametro", sort=False)):
    xv = g.valor.replace(np.inf, 64) ; ax.plot(xv, g.MSE_val, "o-", color="#1F5A3A"); ax.set_title(nom)
    if nom in ("eta", "sigma"): ax.set_xscale("log")
axs[0].set_ylabel("MSE validación"); plt.suptitle("Sensibilidad a cada hiperparámetro (los demás en su óptimo; σ = ∞ se dibuja en 64)", y=1.05); plt.show()
sen.pivot_table(index="parametro", columns="valor", values="MSE_val").round(1)''',
r"""#MD**Lectura.** El MSE de validación es plano alrededor del óptimo para $\lambda$ y $\sigma$ y tiene un mínimo claro en $\eta$ (entre 30 y 300) y en $\eta_b$ (alrededor de 0,01): los dos hiperparámetros que importan son la tasa de los pesos y la de la corrección de nivel. Que $\sigma$ dé casi lo mismo con 4 °C que con $\infty$ anticipa el resultado de la ablación.""",
r"""#MD## Estudio de ablación
Se retira cada componente por separado (con los demás hiperparámetros fijos) y se compara con reglas estándar de combinación ajustadas en validación: promedio simple, pesos por error global ($1/\text{MSE}$), *stacking* lineal no negativo y *stacking* por mínimos cuadrados.""",
r'''ab = pd.read_csv(R + "orig_ablacion.csv"); ab.round(2)''',
r'''a = ab.sort_values("MSE_prueba"); fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(a.variante, a.MSE_prueba, color=["#C0392B" if v == "ATER completo" else "#5B7F96" if g == "componente" else "#C8892B" if g == "experto" else "#9AA5A0" for v, g in zip(a.variante, a.grupo)])
ax.invert_yaxis(); ax.set_xlim(a.MSE_prueba.min() * .95, a.MSE_prueba.max() * 1.01); ax.set_xlabel("MSE en la prueba (MWh²)")
ax.set_title("Ablación: azul = sin un componente, naranja = sin un experto, gris = reglas estándar"); plt.show()''',
r"""#MD**Lectura de la ablación.**

- **Lo que aporta:** los **pesos por hora** son el componente más valioso: con pesos comunes a todas las horas el MSE sube 7,9 %, y sin pesos aprendidos ($\eta=0$) sube 9,1 %. La **corrección de nivel en línea** aporta 2,6 % (sin ella, 2.315 en lugar de 2.256).
- **Lo que no aporta:** el **núcleo de calor** no mejora nada medible (sin él el MSE cambia −0,1 % en la prueba y +0,1 % en validación) y tampoco el olvido ($\lambda=1$ da −0,4 %). El calor de Sabanagrande ya lo usan los expertos (por eso es tan importante en los árboles); ATER no gana nada separando los pesos por calor. Se mantiene en la formulación porque Optuna lo eligió en validación, pero **el aporte original medible está en el régimen horario y en la corrección de nivel**, no en el régimen térmico.
- **Frente a las reglas estándar:** ATER (2.256) le gana al *stacking* por mínimos cuadrados (2.341), al *stacking* no negativo (2.366), a los pesos por error global (2.494), al promedio simple (2.612) y al mejor experto solo (random forest, 2.347). Ninguna regla de pesos fijos alcanza a una regla que se adapta cada día.
- **Expertos:** quitar el random forest es lo que más daño hace (+2,6 %), luego el XGBoost (+1,5 %). Quitar Ridge o la persistencia no empeora: ATER aprende a casi no usarlos.""",
r"""#MD## ¿A quién le cree ATER, y a qué hora?""",
r'''Wp = pd.read_pickle(R + "orig_pesos_prueba.pkl"); wm = Wp.groupby(Wp.index.hour).mean()
fig, ax = plt.subplots(figsize=(11, 4)); bottom = np.zeros(24)
for c in wm.columns: ax.bar(wm.index, wm[c], bottom=bottom, label=c, color=COL[c]); bottom += wm[c].values
ax.set_xlabel("hora del día"); ax.set_ylabel("peso medio en la prueba"); ax.set_title("Pesos de ATER por hora (prueba 2025–2026)"); ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.42)); plt.show()
wm.round(2).T''',
r"""#MD**Lectura.** ATER le cree sobre todo a los dos árboles relativos (entre 50 y 71 % del peso según la hora). De día (7–12 h) le sube el peso al híbrido (hasta 30 %) y a Ridge (hasta 19 %), que capturan mejor la rampa de la mañana. La persistencia casi no recibe peso (como máximo 7 % de madrugada). Esto explica la ablación: quitar Ridge o la persistencia no empeora el error, pero quitar un árbol sí.""",
r"""#MD## Interpretabilidad (SHAP)
Valores SHAP con el explicador por permutación de `shap` (independiente del modelo), para ATER (estado congelado al final de la prueba) y para el mejor benchmark (random forest relativo), sobre las mismas 150 horas de la prueba y el mismo fondo de 50 horas de entrenamiento.""",
r'''imp = pd.read_csv(R + "shap_importancia_global.csv", index_col=0)
from scipy.stats import spearmanr
print("Correlación de Spearman entre las importancias de ATER y del RF: %.2f" % spearmanr(imp.iloc[:, 0], imp.iloc[:, 1])[0])
top = imp.head(12).iloc[::-1]; fig, ax = plt.subplots(figsize=(8, 5)); yy = np.arange(len(top))
ax.barh(yy + .2, top.ATER_media_abs_SHAP_MWh, .4, color="#C0392B", label="ATER"); ax.barh(yy - .2, top.RF_media_abs_SHAP_MWh, .4, color="#1F5A3A", label="random forest relativo")
ax.set_yticks(yy); ax.set_yticklabels(top.index); ax.set_xlabel("media de |SHAP| (MWh)"); ax.legend(); ax.set_title("Importancia global (SHAP)"); plt.show()
imp.head(12).round(2)''',
r"""#MD**Lectura.** Los dos modelos usan la información de forma parecida (correlación de Spearman de 0,86 entre las importancias): primero el **nivel reciente** (media de 7 días, media de ayer, misma hora de las últimas 4 semanas, máximo de ayer), luego la **interacción calor × hora** y la misma hora de ayer. La diferencia es que en ATER la media de ayer y la interacción calor × hora pesan más que en el random forest: es el efecto de mezclar al random forest con el XGBoost y el híbrido, que usan más esas variables.""",
])

# ============================================================================ validación estadística
nb("07_validacion_estadistica.ipynb", [
r"""#MD# Validación estadística de la superioridad

## Protocolo común
- **Mismas particiones, misma semilla (2026), mismo preprocesamiento** para ATER y para los 7 benchmarks (3 tradicionales y 4 de Machine Learning).
- **Validación anidada temporal:** el bucle externo son los folds 1 a 4 de la validación de ventana creciente (purga de 48 h); dentro de cada uno, Optuna ajusta los hiperparámetros de ATER **solo con los folds anteriores**.
- **Prueba única:** enero 2025 – octubre 2026, 653 días. ATER se entrena con los mismos datos y los mismos expertos que los benchmarks y pronostica en línea.
- **Unidad de comparación:** la pérdida diaria (MSE de las 24 horas de cada día), que es la unidad en que se emite el pronóstico.
- **Pruebas (ruta de series de tiempo del Cuadro 3):** (1) **Model Confidence Set** con todos los modelos, (2) **Giacomini-White** condicional, (3) **Diebold-Mariano con corrección HLN**, y como complemento **Wilcoxon** de rangos con signo. Ajuste de **Holm-Bonferroni** sobre las 21 comparaciones (3 pruebas × 7 benchmarks). IC por **bootstrap estacionario** (bloque medio de 7 días) y tamaños del efecto (δ de Cliff y d de Cohen).""",
SETUP,
r'''an = pd.read_csv(R + "orig_validacion_anidada.csv"); af = pd.read_csv(R + "orig_validacion_anidada_folds.csv")
print("Validación anidada temporal (bucle externo = folds 1 a 4), media ± desviación estándar:")
an.assign(MSE=lambda d: d.MSE_media.round(0).astype(int).astype(str) + " ± " + d.MSE_de.round(0).astype(int).astype(str),
          MAPE=lambda d: d.MAPE_media.round(2).astype(str) + " ± " + d.MAPE_de.round(2).astype(str))[["modelo", "MSE", "MAPE"]]''',
r'''print("MSE por fold externo:"); af.pivot(index="modelo", columns="fold", values="MSE").round(0)''',
r'''tp = pd.read_csv(R + "orig_prueba.csv")
tp[["modelo", "MSE", "RMSE", "MAE", "MAPE", "AUC_pico", "CSI", "mejora_vs_persistencia_%", "entrenamiento_s", "inferencia_s"]].round(3)''',
r"""#MD## Supuestos
- **Diebold-Mariano** necesita que la serie de diferencias de pérdida sea estacionaria y usa una varianza HAC (7 rezagos) por la autocorrelación; se verifica con la prueba ADF.
- **Giacomini-White** se plantea con esquema de ventana fija (los expertos se entrenan una vez, como pide la prueba) e instrumentos $h_{t-1}=(1, d_{t-1})$; solo exige momentos finitos de $d_t$.
- **MCS** usa bootstrap estacionario (bloque medio de 7 días), así que tolera dependencia temporal.
- **Wilcoxon** supone simetría y no corrige la autocorrelación: se usa como prueba complementaria, no como una de las tres principales.""",
r'''adf = pd.read_csv(R + "orig_adf.csv"); adf.round(4)''',
r"""#MD La serie de diferencias es estacionaria (ADF, p < 0,01) frente a todos los benchmarks excepto el promedio por hora y día (p = 0,27): esa comparación de Diebold-Mariano es menos confiable, pero la diferencia frente a ese modelo es de 96 % y las pruebas MCS y Giacomini-White no dependen de ese supuesto.""",
r"""#MD## Prueba 1: Model Confidence Set (Hansen, Lunde y Nason, 2011)""",
r'''mcs = pd.read_csv(R + "orig_mcs.csv"); mcs''',
r"""#MD Con $\alpha=0{,}10$ (la convención del MCS), **el conjunto de modelos superiores contiene solo a ATER**: los siete benchmarks se eliminan, el random forest relativo con p = 0,005 y el XGBoost relativo con p = 0,001.""",
r"""#MD## Pruebas 2 y 3 (y complementaria): Giacomini-White, Diebold-Mariano y Wilcoxon""",
r'''pr = pd.read_csv(R + "orig_pruebas_hipotesis.csv")
print("Hipótesis nula de cada prueba: igual desempeño (pérdida diaria esperada igual) entre ATER y el benchmark.")
print("Comparaciones totales para el ajuste de Holm:", pr.n_comparaciones.iloc[0])
pr[["prueba", "benchmark", "supuestos", "estadistico", "p_crudo", "p_holm", "conclusion"]].round(5)''',
r"""#MD## Intervalos y tamaños del efecto""",
r'''ef = pr.drop_duplicates("benchmark")[["benchmark", "mejora_MSE_pct", "IC95_lo", "IC95_hi", "dias_gana_ATER_pct", "delta_cliff_pareado", "magnitud_cliff_pareado", "delta_cliff", "magnitud_cliff", "d_cohen"]]
ef.round(3)''',
r'''fig, ax = plt.subplots(figsize=(8, 3.8)); e = ef.sort_values("mejora_MSE_pct")
ax.errorbar(e.mejora_MSE_pct, e.benchmark, xerr=[e.mejora_MSE_pct - e.IC95_lo, e.IC95_hi - e.mejora_MSE_pct], fmt="o", color="#C0392B", capsize=3)
ax.axvline(0, color="k", lw=1); ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("reducción del MSE de ATER frente a cada benchmark (%), IC 95 % bootstrap estacionario")
plt.show()''',
r'''sem = pd.read_csv(R + "orig_semillas.csv"); print("Robustez ante la semilla de los expertos aleatorios (prueba):"); sem.round(1)''',
r'''# ¿Dónde NO gana ATER? Error por mes y por segmento frente al random forest relativo
import pipeline_energia as P
pt = pd.read_pickle(R + "pred_prueba_con_original.pkl"); df = pd.read_pickle(R + "datos_horarios.pkl"); X, y = P.construir(df); Xt = X.loc[pt.index]
RF = "Random forest (relativo)"
mes = pt.groupby(pt.index.to_period("M")).apply(lambda d: 100 * (1 - np.mean((d.y - d[ORIG]) ** 2) / np.mean((d.y - d[RF]) ** 2)))
print("Meses en que ATER tiene menos MSE que el RF: %d de %d" % ((mes > 0).sum(), len(mes)))
seg = pd.DataFrame({"tipo de día": np.where(Xt.festivo == 1, "festivo", np.where(Xt.dow >= 5, "fin de semana", "laboral")),
                    "calor": pd.qcut(Xt.calor, 4, labels=["frío", "templado", "caluroso", "muy caluroso"]).astype(str)}, index=pt.index)
for c in seg:
    print(pt.groupby(seg[c]).apply(lambda d: pd.Series({"MAPE_ATER": 100 * np.mean(np.abs(d.y - d[ORIG]) / d.y), "MAPE_RF": 100 * np.mean(np.abs(d.y - d[RF]) / d.y), "horas": len(d)})).round(2), "\n")
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.bar(mes.index.astype(str), mes.values, color=np.where(mes > 0, "#C0392B", "#9AA5A0"))
ax.axhline(0, color="k", lw=.8); ax.set_ylabel("reducción del MSE vs RF (%)"); ax.tick_params(axis="x", rotation=60); ax.set_title("Ventaja de ATER sobre el random forest, mes a mes"); plt.show()''',
r"""#MD**Dónde no gana.** ATER tiene menor MAPE que el random forest en todos los tipos de día y en los cuatro cuartiles de calor. La ventaja es mayor en los festivos (3,34 % frente a 3,72 %), donde la corrección de nivel en línea ayuda más, y casi nula en las horas frías (1,94 % frente a 1,97 %). Mes a mes pierde en 4 de 22 meses: la ventaja es pequeña y no es uniforme.""",
r"""#MD## Veredicto de superioridad (sección 6.1 de la guía)""",
r'''ver = json.load(open(R + "orig_veredicto.json")); print(json.dumps(ver, indent=1, ensure_ascii=False))''',
r"""#MD**Veredicto.**

1. **Métrica media mejor que todos los benchmarks:** sí, en la validación anidada (MSE 2.097 ± 821 frente a 2.210 ± 759 del híbrido, el mejor benchmark ahí) y en la prueba (2.256 frente a 2.347 del random forest relativo).
2. **Al menos tres pruebas rechazan la igualdad con p ajustado < 0,05:** sí, frente a los 7 benchmarks. Frente al mejor (random forest relativo): Diebold-Mariano HLN p ajustado = 0,004, Giacomini-White p ajustado = 0,003, Wilcoxon p ajustado < 0,001, y el MCS deja a ATER como **único** modelo superior (p del RF = 0,005).
3. **Tamaño del efecto no despreciable:** frente a los tradicionales, Ridge y el híbrido el efecto es mediano o grande con cualquier medida. Frente a los árboles relativos es **pequeño**: el δ de Cliff **pareado** (mismo día) es 0,25 frente al RF y 0,22 frente al XGBoost (umbral 0,147), y ATER gana el 62,6 % y el 61,1 % de los días. Con el δ de Cliff **no pareado** (0,07) y la d de Cohen (0,12) el efecto sería despreciable. Como todas las pruebas son pareadas por día, la medida que corresponde es la pareada; se reportan las tres para que el lector juzgue.

**Conclusión:** ATER es **estadísticamente superior** a todos los benchmarks (R3), con una ventaja práctica grande frente a los métodos tradicionales (65 % menos MSE que la persistencia) y **pequeña pero consistente** frente a los mejores árboles (3,9 % menos MSE que el random forest relativo, IC 95 %: 1,1–6,8 %; entre 3,6 y 3,9 % en las 5 semillas probadas y en 18 de 22 meses).""",
])
print("ok")
