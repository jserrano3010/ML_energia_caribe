# Conclusiones y limitaciones

## Hallazgos
1. **El problema importa y es predecible.**
   - La demanda horaria de Air-e depende del ciclo diario, del calendario y del calor de Sabanagrande: con índice de calor de 38–42 °C sube 27,5 % en las tardes laborales.
   - Creció 17,5 % entre 2021 y 2025.
   - Los modelos de Machine Learning reducen el error de los métodos tradicionales entre 45 y 64 %.
2. **La clave de los benchmarks fue el objetivo relativo.** Los árboles entrenados con la demanda en MWh fallaron con niveles récord. Entrenados con la demanda relativa al nivel de 7 días, ganaron con amplitud.
3. **El modelo original ATER es el mejor.**
   - Tiene 2,44 % de MAPE en la prueba.
   - Reduce el MSE 65 % frente a la persistencia y 3,9 % frente al random forest relativo (IC 95 %: 1,1–6,8 %).
   - Gana en las 5 semillas probadas (3,6–3,9 %) y en la validación anidada.
4. **Las 28 combinaciones del Entregable 2 no cambian el veredicto.** Los 7 modelos base (KNN, Ridge, Lasso, Decision Tree, Random Forest, XGBoost y SVR) con 4 optimizadores (Grid, Random, Bayesiano y genético) dan como mejores al random forest y al XGBoost. ATER les gana en la validación anidada y en la prueba, y en el MCS con los 15 modelos sigue siendo el único sobreviviente.
5. **Lo que aporta ATER** son los pesos que cambian según la hora (7,9 % del MSE) y la corrección de nivel en línea (2,6 %). El suavizado por calor no aportó.

## Veredicto sobre las condiciones del modelo original

| Condición | Veredicto | Evidencia |
|---|---|---|
| **R1. Novedad en la aplicación** | **Sí, con matices.** No se encontró la combinación de mecanismos de ATER aplicada a la demanda del Caribe colombiano; la familia (agregación exponencial de expertos) sí existe para la demanda eléctrica en Francia | Cap. 4: matriz de novedad, tabla comparativa y declaración |
| **R2. Implementación algorítmica propia** | **Sí.** Clase propia compatible con scikit-learn; el núcleo está escrito solo con numpy (pesos, acumuladores, núcleo y gradiente de la corrección de nivel); 13 pruebas unitarias | Cap. 5; `src/original_model.py`; `tests/` |
| **R3. Superioridad estadísticamente verificada** | **Sí.** Mejor métrica media que los 7 benchmarks. MCS, Giacomini-White y Diebold-Mariano rechazan la igualdad frente a todos con p ajustado (Holm) < 0,05. El efecto frente al mejor benchmark es pequeño (δ de Cliff pareado 0,25) y despreciable con las medidas no pareadas | Cap. 6 |

## Limitaciones
- **Demanda regional, no municipal.** XM no publica la demanda de Sabanagrande sola. Se usa la del mercado de Air-e (Atlántico, Magdalena y La Guajira) con el clima de Sabanagrande como impulsor; Barranquilla pesa mucho en ese mercado.
- **No se pronostica el precio.** La tarifa es mensual y es la misma en todo el mercado de Air-e, así que habría muy pocos datos para un modelo. El precio entra solo para valorar los desvíos, con el precio de bolsa diario y un supuesto simplificado (cota superior).
- **La prueba se usó más de una vez.** La versión 1 de los benchmarks (árboles en MWh) se evaluó y falló; la versión 2 se rediseñó y se volvió a evaluar. ATER se diseñó y ajustó solo con validación, pero se evaluó sobre esa misma prueba. Las cifras de prueba no son completamente "ciegas".
- **Hiperparámetros de los benchmarks.** Se eligieron con todos los folds de validación, incluidos los que sirven de bucle externo en la validación anidada de ATER. Ese sesgo favorece a los benchmarks, no a ATER.
- **Tamaño del efecto frente a los mejores árboles.** Es pequeño (3,9 % del MSE): la ventaja práctica grande es frente a los métodos tradicionales, no frente al random forest.
- **El suavizado por calor no aportó.** El nombre de ATER conserva la "T" de térmico porque está en la formulación, pero en estos datos ese mecanismo no mejora el error.
- **Un solo punto de clima** representa una región de tres departamentos, y se usa el clima del día anterior. Con el clima real del día pronosticado, el MAPE del random forest bajaría de 2,54 % a 2,43 %.
- "Mejor" significa el mejor **de los evaluados** bajo este protocolo.

## Consideraciones éticas y de sesgo
- Los datos son agregados y públicos; no hay información personal.
- Un pronóstico de demanda puede usarse para programar racionamientos. Un error por debajo en días de calor extremo afectaría primero a los hogares vulnerables, que son los que más necesitan refrigeración (Char, s. f.). Por eso se reportan aparte el error en los días más calurosos y la detección de horas pico.

## Trabajo futuro
- Usar el pronóstico meteorológico real del día siguiente en lugar del clima rezagado.
- Agregar expertos más diversos (modelos de espacio de estados, GAM) y probar tasas adaptativas (BOA).
- Pronóstico probabilístico (cuantiles) para dimensionar reservas.
- Estimar la demanda residencial de Sabanagrande con datos de facturación, si Air-e los publica.

## Declaración de autoría y uso de herramientas de IA
**Autores:** Edwin Yunis y Jairo Serrano.

Se usó **Claude (Anthropic)** como asistente de programación y redacción para:
- buscar y descargar datos,
- escribir y depurar el código de los pipelines, del modelo original, de las pruebas unitarias y del dashboard,
- redactar borradores de los capítulos,
- y auditar que cada cifra del texto coincida con una salida de los notebooks.

Las decisiones de diseño, la elección del problema y la interpretación de los resultados son responsabilidad de los autores, que revisaron el código y deben poder explicarlo y modificarlo en la sustentación.

**Firmas:**

______________________ Edwin Yunis

______________________ Jairo Serrano
