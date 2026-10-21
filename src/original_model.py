"""Modelo original: ATER, agregación térmica de expertos con corrección de nivel en línea.

Idea
----
Cada día, al cierre, se conocen las 24 horas reales del día que termina. ATER usa ese dato para
re-ponderar a sus K expertos (Ridge, random forest, XGBoost, híbrido y persistencia) y para corregir
el nivel del pronóstico. Las ponderaciones no son globales:

* son distintas para cada hora del día (24 regímenes horarios), y
* dentro de cada hora se suavizan con un núcleo gaussiano sobre el índice de calor,

así que el experto que mejor pronostica las 15:00 con 38 °C de sensación térmica puede no ser el que
mejor pronostica las 03:00 con 27 °C.

Pronóstico para la hora h del día d (índice de calor c, nivel n = media de 7 días, f_k = experto k):

    L_k(h, c) = sum_j K_sigma(c - c_j) * lambda^(edad_j) * l_{k,j}        (pérdida pasada en la hora h)
    w_k(h, c) = exp(-eta * L_k) / sum_i exp(-eta * L_i)                 (pesos exponenciales)
    p        = (1 + b_h) * sum_k w_k * f_k                                (corrección de nivel)

Actualización al cierre del día d, con la pérdida relativa l = ((p - y)/n)^2, acotada a [0, 1]:

    S[h, bin(c), k] <- lambda * S + l_k                (acumulador de pérdidas por hora y banda de calor)
    b_h             <- b_h - eta_b * 2 * (p - y)/n * (sum_k w_k f_k)/n      (descenso de gradiente)

Las dos clases son compatibles con scikit-learn:

* ``AgregacionTermica``: el núcleo algorítmico, programado solo con numpy. Recibe la matriz de pronósticos
  de los expertos y las columnas de régimen.
* ``ModeloOriginal``: entrena los expertos y delega la combinación en el núcleo. Recibe el DataFrame de
  predictoras de ``pipeline_energia.construir`` y no necesita ``y`` para pronosticar: la demanda real de
  cada día se lee de la columna ``lag24`` del día siguiente, que ya está en las predictoras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# columnas de la matriz que recibe el núcleo: hora, calor, nivel, día, luego los K expertos
_META = 4


class AgregacionTermica(BaseEstimator, RegressorMixin):
    """Núcleo de ATER (numpy puro).

    Parámetros
    ----------
    eta : float >= 0
        Tasa de aprendizaje de los pesos exponenciales. ``eta = 0`` da el promedio simple.
    lam : float en (0, 1]
        Factor de olvido diario de las pérdidas acumuladas (1 = memoria completa).
    sigma : float > 0
        Ancho de banda (°C) del núcleo sobre el índice de calor. ``np.inf`` quita el régimen térmico.
    eta_b : float >= 0
        Tasa de la corrección de nivel en línea. ``0`` la desactiva.
    por_hora : bool
        Si es False, todas las horas comparten pesos y corrección (un solo régimen horario).
    t_min, t_max, paso : float
        Rejilla de bandas del índice de calor (°C).

    Entrada ``X`` (n, 4 + K): [hora 0–23, índice de calor °C, nivel MWh, id de día, f_1 ... f_K].
    Las filas deben estar en orden temporal. Un día se pronostica completo antes de actualizar.
    """

    def __init__(self, eta=100.0, lam=0.98, sigma=4.0, eta_b=0.01, por_hora=True, t_min=20.0, t_max=47.0, paso=1.0):
        self.eta = eta
        self.lam = lam
        self.sigma = sigma
        self.eta_b = eta_b
        self.por_hora = por_hora
        self.t_min = t_min
        self.t_max = t_max
        self.paso = paso

    # ------------------------------------------------------------------ validación
    def _validar_parametros(self):
        if not (np.isfinite(self.eta) and self.eta >= 0):
            raise ValueError("eta debe ser un número finito >= 0")
        if not (0 < self.lam <= 1):
            raise ValueError("lam debe estar en (0, 1]")
        if not (self.sigma > 0):
            raise ValueError("sigma debe ser > 0 (use np.inf para quitar el régimen térmico)")
        if not (np.isfinite(self.eta_b) and self.eta_b >= 0):
            raise ValueError("eta_b debe ser un número finito >= 0")
        if not (self.t_max > self.t_min and self.paso > 0):
            raise ValueError("la rejilla de calor requiere t_max > t_min y paso > 0")

    @staticmethod
    def _validar_X(X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < _META + 1:
            raise ValueError(f"X debe ser una matriz con {_META} columnas de régimen y al menos un experto")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contiene valores faltantes o infinitos")
        h = X[:, 0]
        if np.any((h < 0) | (h > 23) | (h != np.round(h))):
            raise ValueError("la columna hora debe ser entera entre 0 y 23")
        if np.any(X[:, 2] <= 0):
            raise ValueError("el nivel debe ser positivo")
        if np.any(np.diff(X[:, 3]) < 0):
            raise ValueError("las filas deben estar en orden temporal (id de día no decreciente)")
        if y is not None:
            y = np.asarray(y, dtype=float).ravel()
            if len(y) != len(X):
                raise ValueError("X e y deben tener el mismo número de filas")
            if not np.all(np.isfinite(y)):
                raise ValueError("y contiene valores faltantes o infinitos")
        return X, y

    # ------------------------------------------------------------------ piezas del algoritmo
    def _iniciar(self, K):
        self.bandas_ = np.arange(self.t_min, self.t_max + 1e-9, self.paso)
        R = 24 if self.por_hora else 1
        self.S_ = np.zeros((R, len(self.bandas_), K))      # pérdidas acumuladas (descontadas)
        self.b_ = np.zeros(R)                               # corrección de nivel por régimen horario
        self.n_expertos_ = K

    def _regimen(self, hora):
        return hora.astype(int) if self.por_hora else np.zeros(len(hora), int)

    def _pesos(self, r, c):
        """Pesos exponenciales w (m, K) para las filas con régimen horario r y calor c."""
        c = np.clip(c, self.bandas_[0], self.bandas_[-1])
        if np.isinf(self.sigma):
            Kc = np.ones((len(c), len(self.bandas_)))
        else:
            Kc = np.exp(-0.5 * ((c[:, None] - self.bandas_[None, :]) / self.sigma) ** 2)
        L = np.einsum("mb,mbk->mk", Kc, self.S_[r])
        W = np.exp(-self.eta * (L - L.min(axis=1, keepdims=True)))   # restar el mínimo: estabilidad numérica
        return W / W.sum(axis=1, keepdims=True)

    def _pronosticar(self, Fd, r, c):
        W = self._pesos(r, c)
        comb = (W * Fd).sum(axis=1)
        return comb * (1.0 + self.b_[r]), comb, W

    def _actualizar(self, Fd, yd, r, c, nivel, comb, p):
        ell = np.minimum(((Fd * (1.0 + self.b_[r])[:, None] - yd[:, None]) / nivel[:, None]) ** 2, 1.0)
        self.S_ *= self.lam
        cb = np.clip(np.round((np.clip(c, self.bandas_[0], self.bandas_[-1]) - self.bandas_[0]) / self.paso).astype(int),
                     0, len(self.bandas_) - 1)
        np.add.at(self.S_, (r, cb), ell)
        if self.eta_b > 0:
            grad = 2.0 * (p - yd) / nivel * comb / nivel
            np.add.at(self.b_, r, -self.eta_b * grad)

    def _recorrer(self, X, y, guardar=False):
        """Recorre los días en orden: pronostica el día completo y, si hay y, actualiza al cierre."""
        hora, calor, nivel, dia, F = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, _META:]
        r_all = self._regimen(hora)
        out = np.empty(len(X))
        hist_w = np.empty((len(X), F.shape[1])) if guardar else None
        cortes = np.flatnonzero(np.diff(dia)) + 1
        for idx in np.split(np.arange(len(X)), cortes):
            r, c = r_all[idx], calor[idx]
            p, comb, W = self._pronosticar(F[idx], r, c)
            out[idx] = p
            if guardar:
                hist_w[idx] = W
            if y is not None:
                ok = np.isfinite(y[idx])
                if ok.any():
                    self._actualizar(F[idx][ok], y[idx][ok], r[ok], c[ok], nivel[idx][ok], comb[ok], p[ok])
        if guardar:
            self.historia_pesos_ = hist_w
        return out

    # ------------------------------------------------------------------ API scikit-learn
    def fit(self, X, y):
        """Calienta el estado (pérdidas acumuladas y corrección de nivel) recorriendo X en línea."""
        self._validar_parametros()
        X, y = self._validar_X(X, y)
        self._iniciar(X.shape[1] - _META)
        self._recorrer(X, y)
        return self

    def iniciar_vacio(self, K):
        """Estado inicial sin historia (pesos uniformes, sin corrección)."""
        self._validar_parametros()
        self._iniciar(K)
        return self

    def predict(self, X, y_realizado=None, guardar_pesos=False, conservar_estado=False):
        """Pronostica. Si se pasa ``y_realizado`` (valores reales que se conocerán al cierre de cada día;
        NaN donde aún no se conocen) el estado se actualiza en línea; si no, el estado queda congelado."""
        check_is_fitted(self, "S_")
        X, _ = self._validar_X(X)
        if X.shape[1] - _META != self.n_expertos_:
            raise ValueError(f"se esperaban {self.n_expertos_} expertos y llegaron {X.shape[1] - _META}")
        y = None if y_realizado is None else np.asarray(y_realizado, dtype=float).ravel()
        if y is not None and len(y) != len(X):
            raise ValueError("y_realizado debe tener el mismo número de filas que X")
        S0, b0 = self.S_.copy(), self.b_.copy()
        out = self._recorrer(X, y, guardar=guardar_pesos)
        if not conservar_estado:              # predict no modifica el modelo: dos llamadas dan lo mismo
            self.S_, self.b_ = S0, b0
        return out

    def pesos(self, hora, calor):
        """Pesos actuales de los expertos para una hora (0–23) y un índice de calor (°C)."""
        check_is_fitted(self, "S_")
        r = self._regimen(np.atleast_1d(np.asarray(hora)))
        return self._pesos(r, np.atleast_1d(np.asarray(calor, dtype=float)))


# ============================================================================ modelo completo
def _expertos(nombre):
    """Fábricas de expertos (mismas configuraciones elegidas en validación para los benchmarks)."""
    import pipeline_energia as P
    G = P.GRILLAS_V2
    base = {
        "Random forest (relativo)": G["Random forest (relativo)"]["hoja5"],
        "XGBoost (relativo)": G["XGBoost (relativo)"]["d6_n600"],
        "Híbrido Ridge + XGBoost (residuo relativo)": G["Híbrido Ridge + XGBoost (residuo relativo)"]["d3_n300"],
        "Ridge (lineal)": G["Ridge (lineal)"]["a100"],
        "Persistencia (misma hora de ayer)": None,
    }
    if nombre == "completo":
        return base
    if nombre == "ligero":                       # para pruebas unitarias rápidas
        return {k: base[k] for k in ["Ridge (lineal)", "Persistencia (misma hora de ayer)"]}
    if nombre.startswith("sin:"):
        quitar = nombre[4:]
        return {k: v for k, v in base.items() if k != quitar}
    raise ValueError(f"conjunto de expertos desconocido: {nombre}")


class ModeloOriginal(BaseEstimator, RegressorMixin):
    """ATER completo: expertos + núcleo ``AgregacionTermica``.

    ``fit(X, y)`` entrena los expertos con (X, y). ``predict(X)`` pronostica en línea: antes de emitir el
    día d+1 actualiza pesos y corrección con la demanda real del día d, que lee de ``lag24`` (no usa ``y``).
    """

    def __init__(self, eta=100.0, lam=0.98, sigma=4.0, eta_b=0.01, por_hora=True, expertos="completo",
                 en_linea=True, random_state=2026):
        self.eta = eta
        self.lam = lam
        self.sigma = sigma
        self.eta_b = eta_b
        self.por_hora = por_hora
        self.expertos = expertos
        self.en_linea = en_linea
        self.random_state = random_state

    def _nucleo(self):
        return AgregacionTermica(eta=self.eta, lam=self.lam, sigma=self.sigma, eta_b=self.eta_b, por_hora=self.por_hora)

    def _validar_df(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser el DataFrame de predictoras de pipeline_energia.construir")
        falt = {"hora", "calor", "media_7d", "lag24"} - set(X.columns)
        if falt:
            raise ValueError(f"faltan columnas en X: {sorted(falt)}")
        if X.isna().any().any():
            raise ValueError("X contiene valores faltantes")
        if not X.index.is_monotonic_increasing:
            raise ValueError("X debe estar en orden temporal")

    def fit(self, X, y):
        self._validar_df(X)
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        if y.isna().any():
            raise ValueError("y contiene valores faltantes")
        np.random.seed(self.random_state)
        self.nombres_ = list(_expertos(self.expertos))
        self.modelos_ = {}
        for n, f in _expertos(self.expertos).items():
            self.modelos_[n] = None if f is None else f().fit(X, y)
        self.nucleo_ = self._nucleo().iniciar_vacio(len(self.nombres_))
        self.columnas_ = list(X.columns)
        return self

    def pronosticos_expertos(self, X):
        check_is_fitted(self, "modelos_")
        return np.column_stack([X.lag24.values if m is None else m.predict(X[self.columnas_])
                                for m in self.modelos_.values()])

    def _matriz(self, X, F):
        dia = pd.factorize(X.index.normalize())[0]
        return np.column_stack([X.hora.values, X.calor.values, X.media_7d.values, dia, F])

    @staticmethod
    def realizado(X):
        """Demanda real de cada fila tomada de lag24 de la misma hora del día siguiente (NaN si no existe)."""
        s = pd.Series(X.lag24.values, index=X.index)
        return s.reindex(X.index + pd.Timedelta(hours=24)).values

    def predict(self, X, y_realizado=None, guardar_pesos=False, conservar_estado=False):
        check_is_fitted(self, "modelos_")
        self._validar_df(X)
        F = self.pronosticos_expertos(X)
        if self.en_linea:
            yr = self.realizado(X) if y_realizado is None else y_realizado
        else:
            yr = None
        return self.nucleo_.predict(self._matriz(X, F), yr, guardar_pesos=guardar_pesos, conservar_estado=conservar_estado)
