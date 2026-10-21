"""Carga única (al iniciar) de datos livianos y del artefacto del modelo para el dashboard.

Nada se entrena aquí: solo se leen CSV de data/ y el modelo serializado de artifacts/.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
ART = RAIZ / "artifacts"
sys.path.insert(0, str(RAIZ / "src"))          # para que joblib encuentre original_model y pipeline_energia

ORIG = "ATER (modelo original)"
RF = "Random forest (relativo)"
TRAD = ["Persistencia (misma hora de ayer)", "Estacional (misma hora semana pasada)", "Promedio por hora y día"]
CORTO = {ORIG: "ATER (original)", RF: "Random forest relativo", "XGBoost (relativo)": "XGBoost relativo",
         "Híbrido Ridge + XGBoost (residuo relativo)": "Híbrido Ridge + XGBoost", "Ridge (lineal)": "Ridge",
         "Persistencia (misma hora de ayer)": "Persistencia", "Estacional (misma hora semana pasada)": "Estacional",
         "Promedio por hora y día": "Promedio hora × día"}


def csv(nombre, **kw):
    return pd.read_csv(DATA / nombre, **kw)


def js(nombre):
    return json.load(open(DATA / nombre, encoding="utf-8"))


@lru_cache(maxsize=1)
def eda():
    d = pd.read_csv(DATA / "eda_horaria.csv.gz", parse_dates=["ts"], index_col="ts")
    return d


@lru_cache(maxsize=1)
def pred_prueba():
    return pd.read_csv(DATA / "pred_prueba.csv.gz", parse_dates=["ts"], index_col="ts")


@lru_cache(maxsize=1)
def predictoras_prueba():
    return pd.read_csv(DATA / "predictoras_prueba.csv.gz", parse_dates=["ts"], index_col="ts")


@lru_cache(maxsize=1)
def fondo_shap():
    return pd.read_csv(DATA / "fondo_shap.csv", parse_dates=[0], index_col=0)


@lru_cache(maxsize=1)
def modelo():
    """ATER con sus expertos (random forest de 100 árboles; estado congelado al 15-oct-2026)."""
    f = ART / "ater_dashboard.joblib"
    if not f.exists():          # en GitHub el artefacto va partido en trozos < 10 MB: se reensambla al iniciar
        partes = sorted(ART.glob("ater_dashboard.joblib.part*"))
        with open(f, "wb") as out:
            for p in partes:
                out.write(p.read_bytes())
    return joblib.load(f)


@lru_cache(maxsize=1)
def metadata():
    return json.load(open(ART / "metadata_dashboard.json", encoding="utf-8"))


MODELOS_VIVO = [ORIG, RF, "XGBoost (relativo)", "Híbrido Ridge + XGBoost (residuo relativo)", "Ridge (lineal)",
                "Persistencia (misma hora de ayer)"]


def predecir(X: pd.DataFrame, nombre: str) -> np.ndarray:
    """Pronóstico de cualquiera de los modelos disponibles en vivo para las filas de X."""
    m = modelo()
    if nombre == ORIG:
        return m.predict(X)
    if nombre == "Persistencia (misma hora de ayer)":
        return X.lag24.values
    return m.modelos_[nombre].predict(X[m.columnas_])


def shap_local(x: pd.DataFrame, nombre: str, n_perm: int = 8, semilla: int = 2026) -> pd.Series:
    """Valores de Shapley de una fila por muestreo de permutaciones (Strumbelj y Kononenko, 2014), con
    permutaciones antitéticas y el mismo fondo de 30 horas de entrenamiento. Suman pred(x) - E[pred(fondo)]."""
    B = fondo_shap()[x.columns].values
    cols = list(x.columns); p = len(cols); xv = x.values[0]
    rng = np.random.default_rng(semilla)
    perms = []
    for _ in range(n_perm // 2):
        o = rng.permutation(p); perms += [o, o[::-1]]
    filas = []
    for o in perms:                      # cada permutación: p + 1 coaliciones crecientes x fondo
        Z = np.repeat(B[None, :, :], p + 1, axis=0).copy()
        for j in range(1, p + 1):
            Z[j:, :, o[j - 1]] = xv[o[j - 1]]
        filas.append(Z.reshape(-1, p))
    A = np.vstack(filas)
    idx = pd.date_range("2030-01-01", periods=len(A), freq="h")
    pred = predecir(pd.DataFrame(A, columns=cols, index=idx), nombre).reshape(len(perms), p + 1, len(B)).mean(axis=2)
    phi = np.zeros(p)
    for k, o in enumerate(perms):
        phi[o] += np.diff(pred[k])
    return pd.Series(phi / len(perms), index=cols)
