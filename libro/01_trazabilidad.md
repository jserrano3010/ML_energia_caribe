# Trazabilidad de correcciones

Este capítulo documenta cada observación recibida en las entregas anteriores y cada corrección hecha en este entregable. Para cada una indica qué se cambió, dónde está en este libro y con qué evidencia.

## Observaciones del docente

Observaciones del Dr. Lihki Rubio a las entregas del proyecto (Entregables 1 y 2). Se aplican a cualquier proyecto del curso.

| # | Observación del docente | Acción realizada | Ubicación en el Jupyter Book | Evidencia |
|---|---|---|---|---|
| D1 | Alinear el texto con el dataset real (se había descrito un diseño regional que no correspondía a un dataset de un solo sitio) | El libro describe exactamente los datos usados: la demanda del mercado CARIBE SOL (Air-e) y un único punto de clima en Sabanagrande. Las limitaciones aclaran que la demanda es regional, no municipal | Cap. 2, *Datos y análisis exploratorio*; cap. 8, *Limitaciones* | Celdas de carga de datos (52.944 horas, mercado CARIBE SOL, coordenadas 10,79° N, −74,76° O) |
| D2 | Cada cifra citada en el texto debe aparecer en las salidas de los notebooks | Auditoría completa del texto contra las salidas, antes de la entrega | Todos los capítulos con notebooks (2, 3, 5 y 6) | Cada afirmación numérica tiene al lado la celda que la produce; correcciones de la auditoría en la fila C9 |
| D3 | Agregar una línea base de persistencia (el último valor observado) | La persistencia (misma hora de ayer) es la referencia principal, junto con la estacional (misma hora de la semana pasada) y el promedio por hora y día | Cap. 3, *Modelos y validación* y *Prueba 2025–2026*; cap. 6 | Tablas de validación y prueba; mejora de ATER de 65 % frente a la persistencia |
| D4 | Probar modelos no lineales (random forest) para los eventos extremos | Random forest, XGBoost, el híbrido y ATER se evalúan también en las horas pico extremas (POD, FAR, CSI y AUC) | Cap. 3, *Horas pico y valor económico*; cap. 6 | Tabla de picos: ATER detecta el 91 % de las horas pico con 9 % de falsas alarmas |
| D5 | La guía sugiere al menos 20.000 observaciones; si hay menos, justificarlo | Se cambió a un problema con 52.272 horas utilizables; la lluvia (16.436 días) se descartó en parte por este criterio (C1) | Cap. 1, *Problemas evaluados y descartados*; cap. 2 | Tabla `problemas_descartados.csv`; celda con el número de filas utilizables |
| D6 | En el dashboard, el modelo final y la comparación con todos los modelos deben verse de primera (el docente preguntó dónde estaban los demás modelos) | La página *Modelos* abre con el veredicto y una tabla de líderes con los 8 modelos (ordenable y filtrable), seguida del MCS y de las pruebas de hipótesis | Cap. 7, *Dashboard y despliegue* | Página `/modelos` del dashboard |
| D7 | La elección del modelo debe hacerse con la validación, no con la prueba (un error del proyecto anterior) | Regla fijada de antemano (menor MSE en validación); los hiperparámetros de ATER se eligen con Optuna solo en validación, y dentro de la validación anidada solo con los folds anteriores | Cap. 3, *Modelos y validación*; caps. 5 y 6 | `eleccion.json`, `orig_hiperparametros.json`, tabla de validación anidada |
| D8 | Explicar con evidencia por qué funciona un modelo combinado (no basta con decir que "los modelos se compensan") | Estudio de ablación (cada componente y cada experto retirado por separado), pesos aprendidos por hora y comparación con reglas estándar de combinación | Cap. 5, *Estudio de ablación* y *¿A quién le cree ATER?* | `orig_ablacion.csv`: los pesos por hora aportan 7,9 % del MSE y la corrección de nivel 2,6 % |

## Correcciones identificadas en la revisión propia

| # | Problema detectado | Acción realizada | Ubicación en el Jupyter Book | Evidencia |
|---|---|---|---|---|
| C1 | El problema del Entregable 2 (lluvia semanal) mejora a la climatología solo 7 %, con significancia solo a 1 semana | Se evaluaron cuatro problemas alternativos con criterios fijados (C1–C4) y se adoptó la demanda eléctrica de Air-e | Cap. 1, *Problemas evaluados y descartados* | Tabla y gráfico de mejoras |
| C2 | Riesgo de fuga de información en las predictoras (en el proyecto anterior una variable derivada del objetivo dio AUC = 1,000) | Todas las predictoras usan datos con rezago de 24 h o más, el clima es el del día anterior, la columna objetivo se excluye con un `assert`, y una prueba unitaria verifica que ATER no usa el futuro | Cap. 2; cap. 5, *Pruebas unitarias* | `assert "mwh" not in X.columns`; `test_no_usa_el_futuro` |
| C3 | Validación sin respetar el orden temporal | Ventana creciente con 5 folds y purga de 48 h; prueba cronológica 2025–2026 | Cap. 3, *Modelos y validación* | Tabla por fold de validación |
| C4 | Los árboles en MWh fallaron en la prueba (versión 1): no extrapolan niveles récord | Objetivo relativo $y/\bar y_{7d}$ para los árboles (versión 2) | Cap. 3, *Prueba 2025–2026* | Tabla "VERSIÓN 1" frente a la versión 2 |
| C5 | La prueba se consultó más de una vez (versión 1, versión 2 y ATER) | Se reporta de forma explícita; las elecciones se hicieron solo con validación | Cap. 8, *Limitaciones* | Texto de limitaciones |
| C6 | Comparación estadística limitada a Diebold-Mariano | MCS, Giacomini-White, Diebold-Mariano HLN y Wilcoxon, con ajuste de Holm, bootstrap estacionario y δ de Cliff / d de Cohen | Cap. 6, *Validación estadística* | `orig_mcs.csv`, `orig_pruebas_hipotesis.csv` |
| C7 | Optimización solo por grilla | Optimización bayesiana (Optuna, TPE) de los hiperparámetros de ATER | Caps. 5 y 6 | Curva de convergencia; `orig_optuna_trials.csv` |
| C8 | Interpretabilidad solo por permutación | SHAP (explicador por permutación) para ATER y el mejor benchmark | Cap. 5, *Interpretabilidad* | `shap_importancia_global.csv` |
| C9 | Cifras del texto que no coincidían con las salidas | Corregidos: picos del perfil horario, estabilidad del ranking RF/XGB, posición de la persistencia en la rúbrica y frecuencia de horas pico en la prueba | Caps. 2 y 3 | Celdas de salida junto a cada afirmación |
| C10 | Artefactos no serializados | Modelo serializado con `joblib` (compresión 3), `metadata.json` y prueba de ida y vuelta | Caps. 7 y 9 | Diferencia máxima de 1,1 × 10⁻¹² MWh entre el modelo en memoria y el cargado |
| C12 | El Entregable 3 exige la mejor combinación de cada uno de los 7 modelos base del Entregable 2 | Se corrieron las 28 combinaciones (7 modelos × 4 optimizadores) con validación anidada sobre los mismos folds externos que ATER; las mejores entran a la tabla de líderes, al MCS y a la tabla maestra | Cap. 3, *Las 28 combinaciones del Entregable 2*; dashboard, página Modelos | `e2_combinaciones.csv`, `e2_mejores.csv`, `e2_mcs_todos.csv` |
| C11 | Seguridad al publicar (en el proyecto anterior un error de Python mostró el token de GitHub) | El token se pide con `getpass`, va en variables de entorno de git (`GIT_CONFIG_*`) y se borra de los mensajes de salida | `publicar_colab.ipynb` (cap. 7) | Celda de publicación |
