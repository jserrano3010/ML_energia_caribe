"""Pronóstico horario de la demanda eléctrica del mercado CARIBE SOL (Air-e) con un día de anticipación.

Emisión: al final del día D (se conoce la demanda hasta las 24:00 de D). Objetivo: las 24 horas del día D+1.
Predictoras: solo información disponible al emitir (rezagos >= 24 h, clima rezagado 24 h, calendario conocido).
Variante ex-post (sensibilidad): clima real de la hora objetivo ("pronóstico del tiempo perfecto").
"""
import numpy as np, pandas as pd, holidays
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

D = Path(__file__).resolve().parent.parent / "datos"
SEED = 2026

def cargar():
    d = pd.read_csv(D/"demanda_caribe_sol_horaria.csv")
    d["ts"] = pd.to_datetime(d.fecha) + pd.to_timedelta(d.hora - 1, unit="h")     # Hour01 = 00:00-01:00
    y = d.set_index("ts").demanda_kwh.div(1000).rename("mwh")                       # MWh
    c = pd.read_csv(D/"clima_sabanagrande_horario.csv", dtype={"fecha_hora_lst": str})
    c["ts"] = pd.to_datetime(c.fecha_hora_lst, format="%Y%m%d%H")
    c = c.set_index("ts")[["T2M", "RH2M", "WS2M", "PRECTOTCORR"]].replace(-999, np.nan)
    df = pd.concat([y, c], axis=1).loc[y.index.min():y.index.max()]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    assert df.mwh.isna().sum() == 0, "faltan horas de demanda"
    df[["T2M", "RH2M", "WS2M", "PRECTOTCORR"]] = df[["T2M", "RH2M", "WS2M", "PRECTOTCORR"]].interpolate(limit=6)
    return df

def indice_calor(T, RH):
    """Índice de calor (NOAA, Rothfusz) en °C: la sensación térmica que mueve el aire acondicionado."""
    F = T*9/5 + 32
    HI = (-42.379 + 2.04901523*F + 10.14333127*RH - .22475541*F*RH - .00683783*F*F - .05481717*RH*RH
          + .00122874*F*F*RH + .00085282*F*RH*RH - .00000199*F*F*RH*RH)
    HI = np.where(F < 80, F, HI)
    return (HI - 32) * 5/9

def construir(df, expost=False):
    X = pd.DataFrame(index=df.index)
    y = df.mwh
    co = holidays.Colombia(years=range(2019, 2028))
    dia = df.index.normalize()
    fest = pd.Series([d in co for d in dia.date], index=df.index)
    # --- calendario (conocido de antemano)
    X["hora"] = df.index.hour; X["dow"] = df.index.dayofweek; X["mes"] = df.index.month
    X["festivo"] = fest.astype(int)
    X["dia_antes_festivo"] = pd.Series([(d + pd.Timedelta(days=1)).date() in co for d in dia], index=df.index).astype(int)
    X["dia_despues_festivo"] = pd.Series([(d - pd.Timedelta(days=1)).date() in co for d in dia], index=df.index).astype(int)
    X["tipo_dia"] = np.where(X.festivo == 1, 2, np.where(X.dow == 6, 2, np.where(X.dow == 5, 1, 0)))  # 0 laboral, 1 sábado, 2 dom/festivo
    doy = df.index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    X["hora_sin"], X["hora_cos"] = np.sin(2*np.pi*X.hora/24), np.cos(2*np.pi*X.hora/24)
    X["navidad"] = ((df.index.month == 12) & (df.index.day >= 20) | (df.index.month == 1) & (df.index.day <= 6)).astype(int)
    X["t"] = (df.index - df.index[0]).days / 365.25        # tendencia (años)
    # --- historia de la demanda: solo lo conocido al cierre del día D (objetivo en D+1)
    #     la hora objetivo h de D+1 conoce: misma hora de D (rezago 24), de D-1 (48), de D-6 (168) ...
    X["lag24"], X["lag48"], X["lag168"] = y.shift(24), y.shift(48), y.shift(168)
    X["lag336"] = y.shift(336)
    dmean = y.resample("D").mean(); dmax = y.resample("D").max()
    X["media_dia_ant"] = dmean.shift(1).reindex(dia).values
    X["max_dia_ant"] = dmax.shift(1).reindex(dia).values
    X["media_7d"] = dmean.rolling(7).mean().shift(1).reindex(dia).values
    X["media_misma_hora_4sem"] = (y.shift(168) + y.shift(336) + y.shift(504) + y.shift(672)) / 4
    # --- clima de Sabanagrande
    hi = pd.Series(indice_calor(df.T2M, df.RH2M), index=df.index)
    tmax = df.T2M.resample("D").max()
    if expost:   # ex-post: clima real de la hora/día objetivo (cota superior)
        X["T2M"], X["RH2M"], X["WS2M"], X["calor"] = df.T2M, df.RH2M, df.WS2M, hi
        X["tmax_dia"] = tmax.reindex(dia).values
    else:        # operativo: clima de la misma hora del día anterior
        X["T2M"], X["RH2M"], X["WS2M"], X["calor"] = df.T2M.shift(24), df.RH2M.shift(24), df.WS2M.shift(24), hi.shift(24)
        X["tmax_dia"] = tmax.shift(1).reindex(dia).values
    X["lluvia_24h"] = df.PRECTOTCORR.rolling(24).sum().shift(24)
    X["calor_x_hora"] = X.calor * X.hora_sin          # interacción explícita para el modelo lineal
    return X, y

LINEAL = ["festivo", "dia_antes_festivo", "dia_despues_festivo", "navidad", "doy_sin", "doy_cos",
          "lag24", "lag168", "media_dia_ant", "media_7d", "media_misma_hora_4sem", "calor", "calor_x_hora"]

def dummies_lineal(X):
    """Para Ridge: dummies hora x tipo de día (72 perfiles) + variables continuas."""
    hd = pd.get_dummies(X.hora.astype(str) + "_" + X.tipo_dia.astype(str), prefix="hd", dtype=float)
    return pd.concat([X[LINEAL], hd], axis=1)

class Hibrido:
    """Etapa 1: Ridge (perfil horario x tipo de día, tendencia, rezagos, calor).
    Etapa 2: XGBoost aprende el residuo de Ridge con todas las predictoras (no linealidades e interacciones).
    El residuo de entrenamiento se toma fuera de muestra (K-fold temporal) para no sobreajustar la etapa 2."""
    def __init__(self, alpha=1.0, k=4, oos=False, **par):
        self.alpha, self.k, self.oos = alpha, k, oos
        self.par = par or dict(max_depth=3, n_estimators=300, learning_rate=0.03, min_child_weight=50)
    def fit(self, X, y):
        L = dummies_lineal(X); self.cols = L.columns
        self.ridge = make_pipeline(StandardScaler(), Ridge(self.alpha)).fit(L, y)
        res = y.values - self.ridge.predict(L)       # Ridge tiene pocos parámetros: residuo en muestra ~ fuera de muestra
        if self.oos:
            res = np.full(len(y), np.nan); bloques = np.array_split(np.arange(len(y)), self.k + 1)
            for i in range(1, self.k + 1):
                tr = np.concatenate(bloques[:i]); te = bloques[i]
                r = make_pipeline(StandardScaler(), Ridge(self.alpha)).fit(L.iloc[tr], y.iloc[tr])
                res[te] = y.iloc[te].values - r.predict(L.iloc[te])
        ok = ~np.isnan(res)
        Xr = X.assign(pred_ridge=self.ridge.predict(L))
        self.xgb = XGBRegressor(subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=-1,
                                **self.par)
        self.xgb.fit(Xr[ok], res[ok])
        return self
    def predict(self, X):
        L = dummies_lineal(X).reindex(columns=self.cols, fill_value=0.0)
        base = self.ridge.predict(L)
        return base + self.xgb.predict(X.assign(pred_ridge=base))

class RidgeM:
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        L = dummies_lineal(X); self.cols = L.columns
        self.m = make_pipeline(StandardScaler(), Ridge(self.alpha)).fit(L, y); return self
    def predict(self, X):
        return self.m.predict(dummies_lineal(X).reindex(columns=self.cols, fill_value=0.0))

# Grillas pequeñas de hiperparámetros, para TODOS los modelos de ML, elegidas solo con validación
GRILLAS = {
    "Ridge (lineal)": {"a1": lambda: RidgeM(1.0), "a10": lambda: RidgeM(10.0), "a100": lambda: RidgeM(100.0)},
    "Random forest": {"hoja5": lambda: RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED),
                      "hoja20": lambda: RandomForestRegressor(300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=SEED)},
    "XGBoost": {"d3_n300": lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=-1),
                "d6_n600": lambda: XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=-1),
                "d8_n1000": lambda: XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.02, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=-1)},
    "Híbrido (Ridge + XGBoost residual)": {"d3_n300": lambda: Hibrido(max_depth=3, n_estimators=300, learning_rate=0.03, min_child_weight=50),
                                           "d4_n400": lambda: Hibrido(max_depth=4, n_estimators=400, learning_rate=0.03, min_child_weight=20),
                                           "d6_n600": lambda: Hibrido(max_depth=6, n_estimators=600, learning_rate=0.03, min_child_weight=5)},
}

def modelos_ml():
    return {
        "Ridge (lineal)": RidgeM(),
        "Random forest": RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, random_state=SEED, n_jobs=-1),
        "Híbrido (Ridge + XGBoost residual)": Hibrido(),
    }

TRADICIONALES = ["Persistencia (misma hora de ayer)", "Estacional (misma hora semana pasada)", "Promedio por hora y día"]

def pred_tradicional(nombre, Xtr, ytr, X):
    if nombre.startswith("Persistencia"): return X.lag24.values
    if nombre.startswith("Estacional"): return X.lag168.values
    perf = ytr.groupby([Xtr.dow, Xtr.hora]).mean()
    return np.array([perf.get((d, h), ytr.mean()) for d, h in zip(X.dow, X.hora)])

def folds(index, n=5, gap_h=48):
    """Ventana creciente por bloques de tiempo con separación (purga) de 48 h."""
    u = np.arange(len(index)); tam = len(u) // (n + 1)
    for k in range(1, n + 1):
        fin_tr = tam*k - gap_h
        yield u[:fin_tr], u[tam*k: min(tam*(k+1), len(u))]


# ======================= VERSIÓN 2: objetivo relativo (los árboles no extrapolan niveles) =======================
NIVEL = "media_7d"   # nivel de la demanda: promedio de los 7 días anteriores (conocido al emitir)
REL = ["lag24", "lag48", "lag168", "lag336", "media_dia_ant", "max_dia_ant", "media_misma_hora_4sem"]

def relativas(X):
    """Predictoras en términos relativos al nivel reciente; se quitan los niveles absolutos y la tendencia."""
    R = X.drop(columns=REL + ["t", NIVEL])
    for c in REL: R[c + "_rel"] = X[c] / X[NIVEL]
    return R

class ArbolRelativo:
    """RF o XGBoost que aprenden la demanda relativa y/nivel; el pronóstico se re-escala por el nivel."""
    def __init__(self, f): self.f = f
    def fit(self, X, y):
        self.m = self.f().fit(relativas(X), y / X[NIVEL]); return self
    def predict(self, X):
        return self.m.predict(relativas(X)) * X[NIVEL].values
    def __getstate__(self):          # la fábrica (lambda) no se serializa; el modelo entrenado sí
        d = self.__dict__.copy(); d.pop("f", None); return d

class HibridoV2:
    """Etapa 1: Ridge (nivel, perfil hora x tipo de día, rezagos, calor) -> pronóstico base en MWh.
    Etapa 2: XGBoost aprende el residuo RELATIVO (y - base)/nivel con predictoras relativas (calor, festivos,
    interacciones). Pronóstico = base + nivel x corrección. Ningún árbol necesita extrapolar el nivel."""
    def __init__(self, alpha=100.0, **par):
        self.alpha = alpha
        self.par = par or dict(max_depth=3, n_estimators=300, learning_rate=0.03, min_child_weight=50)
    def fit(self, X, y):
        L = dummies_lineal(X); self.cols = L.columns
        self.ridge = make_pipeline(StandardScaler(), Ridge(self.alpha)).fit(L, y)
        base = self.ridge.predict(L)
        R = relativas(X).assign(base_rel=base / X[NIVEL].values)
        self.xgb = XGBRegressor(subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=-1, **self.par)
        self.xgb.fit(R, (y.values - base) / X[NIVEL].values)
        return self
    def predict(self, X):
        base = self.ridge.predict(dummies_lineal(X).reindex(columns=self.cols, fill_value=0.0))
        R = relativas(X).assign(base_rel=base / X[NIVEL].values)
        return base + X[NIVEL].values * self.xgb.predict(R)

def _xgb(d, n, lr, mcw): return lambda: XGBRegressor(n_estimators=n, max_depth=d, learning_rate=lr, min_child_weight=mcw,
                                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=-1)
def _rf(h): return lambda: RandomForestRegressor(300, min_samples_leaf=h, max_features=0.5, n_jobs=-1, random_state=SEED)

GRILLAS_V2 = {
    "Ridge (lineal)": {"a10": lambda: RidgeM(10.0), "a100": lambda: RidgeM(100.0)},
    "Random forest (relativo)": {"hoja5": lambda: ArbolRelativo(_rf(5)), "hoja20": lambda: ArbolRelativo(_rf(20))},
    "XGBoost (relativo)": {"d3_n300": lambda: ArbolRelativo(_xgb(3, 300, .03, 50)), "d6_n600": lambda: ArbolRelativo(_xgb(6, 600, .03, 5))},
    "Híbrido Ridge + XGBoost (residuo relativo)": {"d3_n300": lambda: HibridoV2(max_depth=3, n_estimators=300, learning_rate=0.03, min_child_weight=50),
                                                  "d4_n400": lambda: HibridoV2(max_depth=4, n_estimators=400, learning_rate=0.03, min_child_weight=20),
                                                  "d6_n600": lambda: HibridoV2(max_depth=6, n_estimators=600, learning_rate=0.03, min_child_weight=5)},
}
