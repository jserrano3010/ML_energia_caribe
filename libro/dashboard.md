# Dashboard y despliegue

**URL pública (Render):** <https://ml-energia-caribe.onrender.com>. En el plan gratuito el servicio se suspende por inactividad: la primera visita puede tardar cerca de un minuto.

## Arquitectura
Aplicación **multipágina** de Dash (`use_pages=True`) con `dash-bootstrap-components` (tema FLATLY) y una barra de navegación fija:

```text
ML_energia_caribe/
├── app.py                 # punto de entrada: crea la app y expone "server" (gunicorn app:server)
├── pages/
│   ├── contexto.py        # 1. Contexto: problema, pregunta, datos, metodología, problemas descartados, limitaciones
│   ├── eda.py             # 2. EDA: filtros globales, distribución, perfiles, calor, explorador bivariado, ACF
│   └── modelos.py         # 3. Modelos: tabla de líderes, predicción en vivo + SHAP local, diagnóstico, MCS, modelo original
├── src/
│   ├── original_model.py  # clase del modelo original (ATER)
│   ├── data.py            # carga y caché de datos y artefactos (una sola vez al iniciar)
│   ├── plots.py           # funciones de gráficos reutilizables (plantilla Plotly única)
│   └── pipeline_energia.py, correr_*.py ...   # ETL, benchmarks, experimentos
├── artifacts/             # modelo serializado (joblib, compress=3) + metadata.json
├── data/                  # datos procesados livianos (CSV)
├── tests/                 # pytest
├── libro/                 # fuentes del Jupyter Book
├── requirements.txt       # versiones exactas
└── render.yaml            # configuración de despliegue
```

## Decisiones de diseño
- **No entrena nada en el servidor.** Carga una sola vez los CSV de resultados y el artefacto `ater_dashboard.joblib`. Para caber en la memoria del plan gratuito de Render, este artefacto usa 100 de los 300 árboles del random forest. Se verificó que el desempeño es equivalente (ver `artifacts/metadata_dashboard.json`).
- **Predicción en vivo:** formulario generado a partir del esquema de variables (rangos válidos de `metadata.json`), con validación y mensajes `dbc.Alert`. Devuelve el pronóstico, un intervalo de predicción del 90 % (cuantiles de los residuos de ATER en la prueba para esa hora) y la explicación local SHAP.
- **Paleta limitada** (5 colores), tipografía única, gráficos con títulos, unidades y *hover*.
- **Accesibilidad:** el color nunca es la única señal (las tablas repiten las conclusiones en texto).

## Despliegue
1. `python app.py` en local para verificar las páginas y que `app.server` está expuesto.
2. `requirements.txt` con versiones exactas, incluido `gunicorn`; Python 3.11.9 fijado en `render.yaml`.
3. Push al repositorio de GitHub `ML_energia_caribe`.
4. En Render: *New → Blueprint* con el repositorio. Toma `render.yaml`: `pip install -r requirements.txt` y `gunicorn app:server`.
5. Revisar los *logs* y probar las tres páginas y sus *callbacks* en la URL pública.
