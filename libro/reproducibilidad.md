# Reproducibilidad

## Entorno
- Python 3.11 (desarrollo en 3.13), versiones exactas en `requirements.txt`. Las principales: numpy, pandas, scikit-learn, xgboost, statsmodels, arch, optuna, shap, dash, dash-bootstrap-components, gunicorn.
- **Semilla global:** 2026 (`pipeline_energia.SEED`). La robustez se probó con las semillas 7, 42, 123 y 999.

## Orden de ejecución
```bash
pip install -r requirements.txt
python src/correr_energia.py        # benchmarks: validación, prueba, Diebold-Mariano, segmentos
python src/valor_energia.py         # valor económico de los desvíos
python src/correr_original.py       # ATER: Optuna, validación anidada, prueba, ablación, pruebas de hipótesis, artefactos
python src/semillas.py              # robustez ante la semilla
python src/interpretabilidad.py     # SHAP
python src/descartados_baloto.py && python src/descartados.py   # problemas descartados
python src/tabla_maestra.py         # tabla maestra de experimentos
pytest -q tests/                    # pruebas unitarias
python src/hacer_libro.py && python src/hacer_libro_e3.py        # genera los notebooks
jupyter nbconvert --to notebook --execute --inplace libro/*.ipynb
jupyter-book build libro/
```

## Datos
- **Demanda:** API pública de XM (`https://servapibi.xm.com.co/hourly`), métrica `DemaCome`, entidad `MercadoComercializacion`, filtro `CARIBE SOL`, en ventanas de 30 días.
- **Precio de bolsa:** métrica `PrecBolsNaci`.
- **Clima:** API de NASA POWER, horaria, punto 10,79° N, −74,76° O, parámetros T2M, RH2M, WS2M, ALLSKY_SFC_SW_DWN y PRECTOTCORR, hora local.
- Los CSV descargados están versionados en `datos/`.

## Tabla maestra de experimentos
`resultados/tabla_maestra_experimentos.csv` reúne cada combinación evaluada: modelo, configuración, fold, métrica del bucle externo, tiempos y semilla.

## Bibliografía

```{bibliography}
:all:
```
