"""Pruebas unitarias del modelo original (ATER). Ejecutar con:  pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from original_model import AgregacionTermica, ModeloOriginal  # noqa: E402


def sintetico(dias=120, semilla=0, sesgo=0.0):
    """Demanda sintética con solución conocida: el experto 0 es la verdad, el 1 tiene ruido, el 2 está sesgado."""
    rng = np.random.default_rng(semilla)
    n = dias * 24
    hora = np.tile(np.arange(24), dias)
    dia = np.repeat(np.arange(dias), 24)
    calor = 28 + 8 * np.sin(np.pi * hora / 24) + rng.normal(0, 1, n)
    y = 1000 + 200 * np.sin(np.pi * hora / 24) + rng.normal(0, 5, n)
    y = y * (1 + sesgo)
    F = np.column_stack([y / (1 + sesgo), y + rng.normal(0, 80, n), y * 1.15])
    X = np.column_stack([hora, calor, np.full(n, 1000.0), dia, F])
    return X, y


def test_aprende_al_experto_correcto():
    """(i) Con solución conocida, los pesos se concentran en el experto exacto y el error baja al suyo."""
    X, y = sintetico()
    m = AgregacionTermica(eta=200, lam=1.0, sigma=4.0, eta_b=0.0).fit(X[:1440], y[:1440])
    w = m.pesos(np.arange(24), np.full(24, 30.0))
    assert np.all(w[:, 0] > 0.95)
    p = m.predict(X[1440:])
    assert np.mean((p - y[1440:]) ** 2) < 1.1 * np.mean((X[1440:, 4] - y[1440:]) ** 2) + 1e-9


def test_correccion_de_nivel_aprende_un_sesgo():
    """La corrección multiplicativa b_h converge al sesgo común de todos los expertos (+5 %)."""
    X, y = sintetico(dias=300, sesgo=0.05)
    X = X[:, :6]                       # expertos 0 y 1: ambos sin el sesgo del +5 %
    m = AgregacionTermica(eta=200, lam=0.99, sigma=4.0, eta_b=0.5).fit(X, y)
    assert np.allclose(m.b_, 0.05, atol=0.01)


def test_sin_aprendizaje_es_promedio_simple():
    """Con eta = 0 y eta_b = 0 el modelo es exactamente el promedio de los expertos."""
    X, y = sintetico(dias=10)
    m = AgregacionTermica(eta=0.0, eta_b=0.0).fit(X, y)
    assert np.allclose(m.predict(X), X[:, 4:].mean(axis=1))


def test_reproducibilidad():
    """(ii) Misma entrada y mismos hiperparámetros: mismo resultado; predict no altera el estado."""
    X, y = sintetico(semilla=3)
    a = AgregacionTermica().fit(X[:2000], y[:2000])
    b = AgregacionTermica().fit(X[:2000], y[:2000])
    p1 = a.predict(X[2000:], y[2000:])
    p2 = a.predict(X[2000:], y[2000:])
    assert np.array_equal(p1, p2)
    assert np.array_equal(p1, b.predict(X[2000:], y[2000:]))


def test_no_usa_el_futuro():
    """Cambiar la demanda real de un día no cambia los pronósticos de ese mismo día (solo los siguientes)."""
    X, y = sintetico(dias=20)
    m = AgregacionTermica().fit(X[:240], y[:240])
    y2 = y.copy()
    y2[24 * 15:24 * 16] += 500
    p1, p2 = m.predict(X, y), m.predict(X, y2)
    assert np.array_equal(p1[:24 * 16], p2[:24 * 16])
    assert not np.array_equal(p1[24 * 16:], p2[24 * 16:])


@pytest.mark.parametrize("par", [dict(eta=-1), dict(lam=0), dict(lam=1.5), dict(sigma=0), dict(eta_b=np.nan)])
def test_parametros_invalidos(par):
    """(iii) Hiperparámetros inválidos generan ValueError."""
    X, y = sintetico(dias=3)
    with pytest.raises(ValueError):
        AgregacionTermica(**par).fit(X, y)


def test_entradas_invalidas():
    """(iii) Entradas inválidas: NaN, hora fuera de rango, orden temporal roto, tamaños distintos."""
    X, y = sintetico(dias=3)
    m = AgregacionTermica()
    Xn = X.copy(); Xn[0, 5] = np.nan
    Xh = X.copy(); Xh[0, 0] = 25
    Xo = X[::-1].copy()
    for Z, w in [(Xn, y), (Xh, y), (Xo, y), (X, y[:-1])]:
        with pytest.raises(ValueError):
            m.fit(Z, w)
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :-1])          # número de expertos distinto


def test_clone_scikit_learn():
    """(iv) Compatibilidad con sklearn.base.clone y get_params."""
    m = AgregacionTermica(eta=7.0, sigma=np.inf)
    c = clone(m)
    assert c.get_params() == m.get_params()
    mo = ModeloOriginal(eta=3.0, expertos="ligero")
    assert clone(mo).get_params()["eta"] == 3.0


def test_modelo_completo_ligero():
    """El modelo completo (expertos ligeros) entrena, lee la demanda real de lag24 y no usa y en predict."""
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(1)
    y = pd.Series(1000 + 200 * np.sin(np.pi * idx.hour / 24) + rng.normal(0, 10, len(idx)), index=idx)
    X = pd.DataFrame(index=idx)
    X["hora"] = idx.hour; X["tipo_dia"] = 0; X["dow"] = idx.dayofweek
    for c in ["festivo", "dia_antes_festivo", "dia_despues_festivo", "navidad"]:
        X[c] = 0
    X["doy_sin"], X["doy_cos"] = np.sin(idx.dayofyear), np.cos(idx.dayofyear)
    X["lag24"] = y.shift(24); X["lag168"] = y.shift(168)
    X["media_dia_ant"] = X["media_7d"] = X["media_misma_hora_4sem"] = 1000.0
    X["calor"] = 30.0; X["calor_x_hora"] = 30 * np.sin(idx.hour)
    X = X.dropna(); y = y[X.index]
    m = ModeloOriginal(expertos="ligero").fit(X[:800], y[:800])
    assert np.allclose(ModeloOriginal.realizado(X)[:-24], y.values[:-24])
    p = m.predict(X[800:])
    assert p.shape == (len(X) - 800,) and np.all(np.isfinite(p))
