"""Entregable 2 sobre el problema de energía: 7 modelos base de regresión x 4 métodos de optimización = 28 combinaciones.

Modelos (guía del Entregable 2, regresión): KNN, Ridge, Lasso, Decision Tree, Random Forest, XGBoost y SVR.
Optimizadores: Grid Search, Random Search, Bayesiano (Optuna, TPE) y Genético (implementado a mano con numpy).

Protocolo (el mismo de ATER, para que todo sea comparable):
* Validación anidada temporal. Bucle externo: folds 1 a 4 de la ventana creciente con purga de 48 h
  (pipeline_energia.folds). Bucle interno: dentro del entrenamiento de cada fold externo, el último 20 % (con purga
  de 48 h) es la validación interna donde cada optimizador evalúa sus candidatos.
* Presupuesto igual para los cuatro optimizadores: 12 evaluaciones por fold externo.
* Todos los modelos usan el objetivo relativo (demanda / media de 7 días) y predictoras relativas, igual que los
  árboles del proyecto; el escalado (StandardScaler) va dentro del pipeline y se ajusta solo con entrenamiento.
* SVR: el kernel RBF exacto cuesta O(n^2)–O(n^3) con ~30.000 filas; se usa la aproximación de Nyström (300 componentes)
  + SVR lineal, como sugiere la guía (sección 4.2).
* Random forest: 100 árboles con max_samples = 0,5 para que quepa en el presupuesto de cómputo.
Semilla global 2026.
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd, optuna
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import Nystroem
from xgboost import XGBRegressor
import pipeline_energia as P

OUT = Path("resultados"); SEED = 2026; BUDGET = 12
optuna.logging.set_verbosity(optuna.logging.WARNING)
CORTE = pd.Timestamp("2025-01-01"); GAP = pd.Timedelta(hours=48)


class Relativo(BaseEstimator, RegressorMixin):
    """Aprende y / media_7d con predictoras relativas; el pronóstico se re-escala por el nivel."""
    def __init__(self, modelo=None):
        self.modelo = modelo
    def fit(self, X, y):
        self.modelo.fit(P.relativas(X), np.asarray(y) / X[P.NIVEL].values); return self
    def predict(self, X):
        return self.modelo.predict(P.relativas(X)) * X[P.NIVEL].values


# ---------------------------------------------------------------- espacios de búsqueda (nombre, tipo, bajo, alto, log)
ESPACIOS = {
    "KNN": [("n_neighbors", "int", 3, 100, True), ("weights", "cat", ["uniform", "distance"]), ("p", "cat", [1, 2])],
    "Ridge": [("alpha", "float", 1e-3, 1e3, True)],
    "Lasso": [("alpha", "float", 1e-6, 1e-1, True)],
    "Decision Tree": [("max_depth", "int", 3, 25, False), ("min_samples_leaf", "int", 5, 300, True)],
    "Random Forest": [("max_depth", "int", 6, 30, False), ("min_samples_leaf", "int", 1, 50, True), ("max_features", "float", 0.2, 1.0, False)],
    "XGBoost": [("max_depth", "int", 2, 10, False), ("learning_rate", "float", 0.01, 0.3, True), ("n_estimators", "int", 100, 800, True),
                ("min_child_weight", "float", 1, 100, True), ("subsample", "float", 0.5, 1.0, False)],
    "SVR": [("C", "float", 1e-2, 1e2, True), ("gamma", "float", 1e-3, 1.0, True), ("epsilon", "float", 0.0, 0.05, False)],
}


def construir_modelo(nombre, h):
    if nombre == "KNN":
        m = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=int(h["n_neighbors"]), weights=h["weights"], p=int(h["p"]), n_jobs=1))
    elif nombre == "Ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=h["alpha"]))
    elif nombre == "Lasso":
        m = make_pipeline(StandardScaler(), Lasso(alpha=h["alpha"], max_iter=5000))
    elif nombre == "Decision Tree":
        m = DecisionTreeRegressor(max_depth=int(h["max_depth"]), min_samples_leaf=int(h["min_samples_leaf"]), random_state=SEED)
    elif nombre == "Random Forest":
        m = RandomForestRegressor(n_estimators=100, max_samples=0.5, max_depth=int(h["max_depth"]), min_samples_leaf=int(h["min_samples_leaf"]),
                                  max_features=h["max_features"], n_jobs=1, random_state=SEED)
    elif nombre == "XGBoost":
        m = XGBRegressor(max_depth=int(h["max_depth"]), learning_rate=h["learning_rate"], n_estimators=int(h["n_estimators"]),
                         min_child_weight=h["min_child_weight"], subsample=h["subsample"], tree_method="hist", n_jobs=1, random_state=SEED)
    elif nombre == "SVR":
        m = make_pipeline(StandardScaler(), Nystroem(gamma=h["gamma"], n_components=300, random_state=SEED),
                          LinearSVR(C=h["C"], epsilon=h["epsilon"], loss="squared_epsilon_insensitive", dual=False, max_iter=2000, random_state=SEED))
    return Relativo(m)


# ---------------------------------------------------------------- codificación en el cubo unitario (para Grid, Random y GA)
def decodificar(esp, u):
    h = {}
    for (spec, ui) in zip(esp, u):
        nombre, tipo = spec[0], spec[1]
        if tipo == "cat":
            ops = spec[2]; h[nombre] = ops[min(int(ui * len(ops)), len(ops) - 1)]
        else:
            lo, hi, log = spec[2], spec[3], spec[4]
            v = np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))) if log else lo + ui * (hi - lo)
            h[nombre] = int(round(v)) if tipo == "int" else float(v)
    return h


def malla(esp, n=BUDGET):
    d = len(esp); niveles = max(2, int(np.floor(n ** (1 / d) + 1e-9)))
    ejes = []
    for spec in esp:
        k = len(spec[2]) if spec[1] == "cat" else niveles
        ejes.append((np.arange(k) + 0.5) / k)
    puntos = np.array(np.meshgrid(*ejes, indexing="ij")).reshape(d, -1).T
    if len(puntos) > n:
        puntos = puntos[np.linspace(0, len(puntos) - 1, n).round().astype(int)]
    return puntos


# ---------------------------------------------------------------- optimizadores (todos con BUDGET evaluaciones)
def opt_grid(esp, evaluar, rng):
    for u in malla(esp):
        evaluar(decodificar(esp, u))


def opt_random(esp, evaluar, rng):
    for _ in range(BUDGET):
        evaluar(decodificar(esp, rng.random(len(esp))))


def opt_bayes(esp, evaluar, rng):
    def obj(trial):
        h = {}
        for spec in esp:
            if spec[1] == "cat": h[spec[0]] = trial.suggest_categorical(spec[0], spec[2])
            elif spec[1] == "int": h[spec[0]] = trial.suggest_int(spec[0], spec[2], spec[3], log=spec[4])
            else: h[spec[0]] = trial.suggest_float(spec[0], spec[2], spec[3], log=spec[4])
        return evaluar(h)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(rng.integers(1e9)), n_startup_trials=4))
    st.optimize(obj, n_trials=BUDGET)


def opt_genetico(esp, evaluar, rng, pob=4, generaciones=3, p_cruce=0.8, p_mut=0.3, sigma=0.15):
    """Algoritmo genético en el cubo unitario: selección por torneo (k=2), cruce uniforme, mutación gaussiana,
    elitismo de 1 individuo. 4 individuos x 3 generaciones = 12 evaluaciones. Se registra la diversidad genética."""
    d = len(esp); X = rng.random((pob, d)); fit = np.array([evaluar(decodificar(esp, x)) for x in X])
    diversidad = [float(np.mean(np.std(X, axis=0)))]
    for _ in range(generaciones - 1):
        elite = X[np.argmin(fit)].copy(); hijos = [elite]
        while len(hijos) < pob:
            padres = []
            for _ in range(2):
                i, j = rng.integers(pob, size=2); padres.append(X[i] if fit[i] < fit[j] else X[j])
            h = np.where(rng.random(d) < 0.5, padres[0], padres[1]) if rng.random() < p_cruce else padres[0].copy()
            mut = rng.random(d) < p_mut; h[mut] = np.clip(h[mut] + rng.normal(0, sigma, mut.sum()), 0, 1)
            hijos.append(h)
        X = np.array(hijos)
        fit = np.concatenate([[fit.min()], [evaluar(decodificar(esp, x)) for x in X[1:]]])
        diversidad.append(float(np.mean(np.std(X, axis=0))))
    return diversidad


OPTIMIZADORES = {"Grid Search": opt_grid, "Random Search": opt_random, "Bayesiano (Optuna TPE)": opt_bayes, "Genético": opt_genetico}


def mse(a, b): return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def correr_combinacion(modelo, optimizador):
    X, y = pd.read_pickle(OUT / "e2_Xy.pkl")
    tr = X.index < CORTE - GAP; Xtr, ytr = X[tr], y[tr]
    esp = ESPACIOS[modelo]; filas, curvas = [], []
    rng = np.random.default_rng(SEED)
    for k, (a, b) in enumerate(P.folds(Xtr.index)):
        if k == 0:
            continue                                       # bucle externo: folds 1 a 4 (igual que ATER)
        Xa, ya, Xb, yb = Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], ytr.iloc[b]
        m = int(0.2 * len(Xa)); g = 48
        Xi, yi, Xv, yv = Xa.iloc[:len(Xa) - m - g], ya.iloc[:len(Xa) - m - g], Xa.iloc[len(Xa) - m:], ya.iloc[len(Xa) - m:]
        hist = []
        def evaluar(h):
            t0 = time.perf_counter()
            try:
                p = construir_modelo(modelo, h).fit(Xi, yi).predict(Xv); s = mse(yv.values, p)
            except Exception:
                s = float("inf")
            hist.append(dict(h=h, mse_val=s, seg=time.perf_counter() - t0)); return s
        t0 = time.perf_counter()
        extra = OPTIMIZADORES[optimizador](esp, evaluar, rng)
        t_busqueda = time.perf_counter() - t0
        mejor = min(hist, key=lambda r: r["mse_val"])
        t0 = time.perf_counter(); mod = construir_modelo(modelo, mejor["h"]).fit(Xa, ya); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); p = mod.predict(Xb); t_inf = time.perf_counter() - t0
        filas.append(dict(modelo=modelo, optimizador=optimizador, fold=k, MSE=mse(yb.values, p), MAPE=100 * float(np.mean(np.abs(yb.values - p) / yb.values)),
                          mejor_mse_interno=mejor["mse_val"], hiperparametros=json.dumps(mejor["h"]), evaluaciones=len(hist),
                          tiempo_busqueda_s=t_busqueda, tiempo_entrenamiento_s=t_fit, tiempo_inferencia_s=t_inf,
                          diversidad_genetica=json.dumps(extra) if extra else "", semilla=SEED))
        mejor_hasta = np.minimum.accumulate([r["mse_val"] for r in hist])
        for t, v in enumerate(mejor_hasta):
            curvas.append(dict(modelo=modelo, optimizador=optimizador, fold=k, evaluacion=t + 1, mejor_mse_interno=v,
                               seg_acumulados=float(np.sum([r["seg"] for r in hist[:t + 1]]))))
    return filas, curvas


if __name__ == "__main__":
    df = pd.read_pickle(OUT / "datos_horarios.pkl")
    X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
    pd.to_pickle((X, y), OUT / "e2_Xy.pkl")
    combos = [(m, o) for m in ["Random Forest", "XGBoost", "KNN", "SVR", "Decision Tree", "Ridge", "Lasso"] for o in OPTIMIZADORES]
    solo = sys.argv[1:] and sys.argv[1]
    if solo:
        combos = [c for c in combos if c[0] == solo]
    t0 = time.perf_counter()
    res = Parallel(n_jobs=2, verbose=10)(delayed(correr_combinacion)(m, o) for m, o in combos)
    print("tiempo total: %.1f min" % ((time.perf_counter() - t0) / 60))
    folds = pd.DataFrame([f for r in res for f in r[0]]); cur = pd.DataFrame([c for r in res for c in r[1]])
    suf = f"_{solo.replace(' ', '_')}" if solo else ""
    folds.to_csv(OUT / f"e2_folds{suf}.csv", index=False); cur.to_csv(OUT / f"e2_curvas{suf}.csv", index=False)
    print(folds.groupby(["modelo", "optimizador"]).agg(MSE=("MSE", "mean"), de=("MSE", "std"), t=("tiempo_busqueda_s", "sum")).round(1))
