# Revisión de literatura y novedad

## Protocolo de búsqueda
- **Fecha de consulta:** 21 de octubre de 2026.
- **Bases y buscadores:** Google Scholar, arXiv, IEEE Xplore (por medio de un buscador web con los términos de cada base), y repositorios latinoamericanos (Redalyc, DOAJ y el repositorio de la Universidad Nacional de Colombia).
- **Cadenas:** combinan (i) la familia del modelo (agregación de expertos en línea, pesos exponenciales), (ii) el problema (pronóstico de demanda eléctrica horaria del día siguiente) y (iii) sinónimos en inglés y español.
- **Revisión:** título y resumen de todos los resultados mostrados, y texto completo de los más cercanos.

```{important}
La columna "Result." cuenta los resultados que mostró el buscador para cada cadena, no el total de la base. Para la versión final conviene repetir las cadenas en Scopus o Web of Science con el acceso de la universidad y anotar los conteos oficiales. Las cadenas están listas para copiar.
```

## Matriz de novedad

| Base | Cadena de búsqueda | Result. | Revis. | Relev. | Observación |
|---|---|---|---|---|---|
| Google Scholar | `online aggregation of experts electricity load forecasting Gaillard Goude` | 9 | 9 | 3 | Devaine et al. (2013), Gaillard y Goude (2015) y el paquete `opera`: agregación de expertos para la demanda de Francia (EDF) |
| arXiv | `online expert aggregation load forecasting temperature dependent weights` | 9 | 9 | 2 | Agregación de modelos adaptativos (Francia, COVID-19); los pesos no dependen de la temperatura |
| IEEE Xplore | `"expert aggregation" day-ahead load forecasting exponentially weighted average forecaster` | 10 | 10 | 3 | BOA suavizado entre horas (Ziel, 2022); modelos de espacio de estados (de Vilmarest y Goude, 2022) |
| Google Scholar | `"online aggregation" OR "aggregation of experts" "load forecasting" heat index regime kernel weights` | 10 | 10 | 2 | Agregación de expertos no lineales; agregación para pronosticar temperatura (no demanda) |
| Redalyc / DOAJ | `pronóstico demanda eléctrica región Caribe Colombia machine learning temperatura` | 9 | 9 | 0 | Redes neuronales y ARIMA para Colombia; ninguna agregación en línea |
| Repositorios colombianos | `agregación de expertos en línea pronóstico de demanda de energía eléctrica Colombia` | 9 | 9 | 0 | Proyecciones de la UPME y modelos econométricos; ninguna agregación en línea |
| Google Scholar | `forecasting hourly electricity demand Colombia XM machine learning random forest XGBoost` | 9 | 9 | 1 | Consumo horario de Antioquia con modelos estadísticos y de ML; sin agregación de expertos ni Caribe |

## Trabajos más cercanos y en qué se diferencia ATER

| Método (familia) | Qué hace | Qué conserva ATER | Qué es distinto en ATER |
|---|---|---|---|
| Pesos exponenciales / Hedge (Cesa-Bianchi y Lugosi, 2006) | Pondera expertos con $e^{-\eta L_k}$ de la pérdida acumulada | La regla exponencial y su cota de arrepentimiento | Pesos distintos por hora, suavizados por calor, con olvido, sobre pérdida relativa al nivel |
| Expertos especializados (Freund et al., 1997; Devaine et al., 2013) | Cada experto está activo o "dormido" según reglas fijas (invierno, festivos) | La idea de que el mejor experto depende de la condición | La condición es continua (núcleo gaussiano sobre el índice de calor) y no hay que diseñar reglas de activación |
| Fixed-share (Herbster y Warmuth, 1998) | Sigue al mejor experto cuando este cambia en el tiempo | El seguimiento (aquí por olvido $\lambda$) | Además corrige el nivel del pronóstico combinado |
| BOA / ML-Poly (Wintenberger, 2017; Gaillard et al., 2014) | Agregación con cotas de segundo orden y tasas adaptativas | La agregación en línea | No ajusta tasas en línea; tiene corrección de nivel |
| BOA suavizado (Ziel, 2022; Berrisch y Ziel, 2023) | Pesos por hora suavizados con splines entre horas | Pesos por hora | Suaviza en el calor, no entre horas; incluye corrección de nivel |
| Espacio de estados / Kalman adaptativo (de Vilmarest y Goude, 2022) | Adapta en línea los coeficientes de un modelo (GAM) | La adaptación diaria | Adapta la combinación de modelos de caja negra y un factor de nivel, no los coeficientes internos |
| Stacking (Wolpert, 1992) | Pesos fijos aprendidos por regresión sobre predicciones fuera de muestra | La combinación de modelos | Los pesos cambian cada día; en la ablación, ATER supera al stacking por MCO y al no negativo |
| FFORMA (Montero-Manso et al., 2020) | Meta-aprendizaje de pesos según rasgos de la serie | Pesos condicionados a rasgos | Los rasgos son la hora y el calor de cada predicción, y el aprendizaje es en línea |

## Declaración de novedad
Al mejor de nuestro conocimiento, y según la búsqueda descrita, **no se encontraron estudios que apliquen a la demanda eléctrica horaria del día siguiente del Caribe colombiano (mercado de Air-e) una agregación de expertos en línea** que reúna:
- pesos exponenciales por hora,
- suavizado de esos pesos sobre el índice de calor,
- pérdida relativa al nivel de 7 días,
- y corrección multiplicativa del nivel aprendida en línea junto con los pesos.

**Trabajos más cercanos:**
- Devaine et al. (2013) y Gaillard y Goude (2015) agregan expertos para la demanda de Francia. Usan especialización por reglas discretas y no corrigen el nivel.
- Ziel (2022) suaviza los pesos entre horas, no en función del clima.
- De Vilmarest y Goude (2022) adaptan los coeficientes de un solo modelo.

**Alcance honesto de la novedad.** La familia algorítmica (agregación exponencial de expertos) es conocida y se ha usado para la demanda eléctrica. La contribución de ATER está en su **combinación de mecanismos** y en su **aplicación**. La ablación del capítulo 5 muestra que el aporte medible viene de los pesos por hora y de la corrección de nivel. El suavizado por calor, que es el mecanismo más distintivo frente a la literatura, **no** mejoró el error en estos datos.

## Literatura de soporte del proyecto (20 trabajos)

Además de los trabajos de agregación de expertos de la matriz, el proyecto se apoya en la siguiente literatura. Están agrupados por el papel que cumplen en el libro.

### A. Calor, clima y demanda eléctrica (por qué el calor de Sabanagrande importa)

| # | Referencia | Aporte | Uso en el proyecto |
|---|---|---|---|
| 1 | Auffhammer y Mansur (2014), *Energy Economics* | Revisión de la evidencia empírica del efecto del clima sobre el consumo de energía | Justifica el índice de calor como predictora (cap. 2) |
| 2 | Auffhammer, Baylis y Hausman (2017), *PNAS* | El calentamiento aumenta la frecuencia e intensidad de los picos de demanda eléctrica | Motiva el análisis de horas pico (cap. 3) |
| 3 | Davis y Gertler (2015), *PNAS* | La adopción de aire acondicionado crece con el ingreso y la temperatura y dispara el consumo | Mecanismo del calor → aires → demanda en el Caribe (marco) |
| 4 | Wang y Bielicki (2018), *Energy* | La respuesta horaria de la carga a las variables meteorológicas depende de la hora y de la aclimatación | Respalda la interacción calor × hora y los pesos por hora de ATER (caps. 2 y 5) |
| 5 | Rothfusz (1990), NWS | Ecuación del índice de calor | Cálculo de la sensación térmica (cap. 2) |

### B. Pronóstico de demanda eléctrica (métodos y benchmarks)

| # | Referencia | Aporte | Uso en el proyecto |
|---|---|---|---|
| 6 | Hippert, Pedreira y Souza (2001), *IEEE Trans. Power Systems* | Revisión del pronóstico de carga de corto plazo y de la importancia de comparar con métodos ingenuos | Exigencia de líneas base tradicionales (cap. 3) |
| 7 | Taylor (2003), *J. Operational Research Society* | Suavizamiento exponencial con doble estacionalidad (diaria y semanal) para la demanda horaria | Fundamenta los benchmarks estacionales (misma hora de ayer y de la semana pasada) |
| 8 | Fan y Hyndman (2012), *IEEE Trans. Power Systems* | Modelo aditivo semiparamétrico para la carga de corto plazo con temperatura y calendario | Diseño de predictoras de calendario y temperatura (cap. 2) |
| 9 | Hyndman y Fan (2010), *IEEE Trans. Power Systems* | Pronóstico de densidad de los picos de demanda | Trabajo futuro: pronóstico probabilístico (cap. 8) |
| 10 | Hong y Fan (2016), *Int. J. Forecasting* | Tutorial de pronóstico probabilístico de carga | Marco general del pronóstico de demanda (marco) |
| 11 | Hong et al. (2016), *Int. J. Forecasting* (GEFCom2014) | Competencia global de pronóstico de energía: prácticas que ganan | Protocolo de evaluación fuera de muestra (cap. 3) |
| 12 | Petropoulos et al. (2022), *Int. J. Forecasting* | *Forecasting: theory and practice*, revisión enciclopédica que incluye la combinación de pronósticos | Respalda la combinación de modelos como estrategia (caps. 4 y 5) |
| 13 | Weron (2014), *Int. J. Forecasting* | Revisión del pronóstico de precios de la electricidad | Por qué no se pronostica el precio y cómo se valora el error (cap. 3) |
| 14 | Lago et al. (2021), *Applied Energy* | Buenas prácticas y benchmark abierto para el pronóstico del día siguiente en mercados eléctricos | Prueba de Diebold-Mariano y comparación justa con benchmarks (cap. 6) |

### C. Colombia y el Caribe (contexto y antecedentes)

| # | Referencia | Aporte | Uso en el proyecto |
|---|---|---|---|
| 15 | Restrepo Gil (2024), tesis de maestría, Universidad Nacional de Colombia | Pronóstico horario del consumo eléctrico de Antioquia con LSTM, día tipo y temperatura máxima, frente al modelo de XM | Antecedente colombiano más cercano: otra región y un método neuronal (cap. 4) |
| 16 | Mosquera Cabra (2024), tesis de maestría, Universidad Nacional de Colombia | Pronóstico probabilístico de la demanda diaria de Colombia; el Lasso supera al aprendizaje profundo y los rezagos son lo más importante | Respalda la importancia de los rezagos y de comparar con modelos simples (caps. 3 y 5) |
| 17 | Lara-Montañez, Cisneros-Sosa y Herrera-Murillo (2022), *Prospectiva* 20(2) | Pronóstico del IPC y peso del costo de la electricidad en la Costa Caribe | Contexto del impacto económico de la energía en la región (marco) |
| 18 | Fundesarrollo (2023), *Recomendaciones para la racionalización de las tarifas de energía en la región Caribe* | En Atlántico el 70,1 % de los hogares subsidiados no son pobres; el Consumo Básico de Subsistencia es de 173 kWh/mes bajo 1.000 m de altitud | Contexto de tarifas y subsidios (marco) |
| 19 | Fundesarrollo (2022, septiembre), *Impacto de las alzas tarifarias en la región Caribe* | Efectos de las alzas de tarifas sobre los hogares y la economía del Caribe | Contexto de tarifas (marco) |
| 20 | Char (s. f.), *Tarifas que oscurecen derechos* | Tarifas de Air-e (+74 % hasta julio de 2023), pérdidas, opción tarifaria, Consumo Básico de Subsistencia y mínimo vital (T-761 de 2015) | Contexto social y ético (marco y cap. 8) |

**Referencia complementaria:** Corficolombiana (2024, 30 de septiembre), *Informe Semanal: Una mirada a la evolución tarifaria de la energía*, sobre la trayectoria de las tarifas de energía en Colombia y el Caribe; se usa como contexto del marco.
