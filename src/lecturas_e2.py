LECTURAS = {
"mejores": r"""#MD**Lectura.**

- **Mejores combinaciones en la validación anidada:** random forest con Grid Search (MSE 2.259 ± 818) y XGBoost con Random Search (2.268 ± 783). Les siguen KNN (2.482), SVR (2.493), Lasso (2.788), Ridge (2.800) y el árbol de decisión (2.956).
- **Estos modelos coinciden con los benchmarks del proyecto.** Los mejores modelos del Entregable 2 son los mismos árboles relativos que ya eran benchmarks, con resultados casi iguales: el random forest relativo del proyecto tuvo 2.240 en los mismos folds. Por eso las conclusiones de los capítulos 3 y 6 no cambian.
- **ATER sigue primero:** su MSE externo en esos mismos 4 folds es 2.097 ± 821, el menor de todos.
- **El optimizador importa poco frente a la elección del modelo.** Dentro de cada modelo, la diferencia entre el mejor y el peor optimizador es de 0,1 % (Ridge) a 5,8 % (SVR). Entre modelos, la diferencia llega a 31 %.""",

"optimizadores": r"""#MD**Lectura.**

- **Con este presupuesto no hay un optimizador significativamente mejor.** La prueba de Friedman sobre los 28 bloques (7 modelos × 4 folds) no rechaza la igualdad (χ² = 0,39, p = 0,94). Los rangos medios van de 2,39 (Grid) a 2,61 (Genético).
- **El Bayesiano (TPE) arranca peor y termina igual.** Sus 4 primeras evaluaciones son aleatorias (exploración), por eso empieza 19,5 % por encima de su mejor final. Desde la evaluación 9 se acerca más rápido que Random Search y termina a 0,5 % del mejor, prácticamente igual que Grid (0,3 %).
- **El genético converge rápido, pero se queda corto.** La diversidad de su población cae de 0,23 a 0,10 en tres generaciones, una señal de convergencia prematura con solo 4 individuos. Termina a 3,3 % de su mejor y no fue el mejor optimizador en ningún modelo. Además usó 10 evaluaciones y no 12, porque el individuo élite no se vuelve a evaluar.
- **Por qué no se nota la ventaja teórica del Bayesiano y del genético:** los espacios tienen de 1 a 5 dimensiones y el presupuesto es de 12 evaluaciones. Con tan pocas dimensiones una malla bien puesta cubre bien el espacio, y el modelo sustituto del TPE apenas tiene datos para aprender. La ventaja de los métodos guiados aparece con más dimensiones o con más presupuesto; aquí no se observó, y se reporta así.""",

"tiempos": r"""#MD**Lectura.** Los tiempos medidos siguen el orden que anticipa la complejidad:

- **Entrenamiento.** Lineales y árbol: 0,2 s o menos. SVR con Nyström: 1,2 s, lejos del costo cuadrático del kernel exacto. XGBoost con histogramas: 2,4 s. Random forest: 5,2 s, el más costoso por sus 100 árboles.
- **Inferencia.** KNN invierte el patrón: entrena en 0,01 s pero tarda 2,7 s en predecir un fold, porque compara cada hora con todo el entrenamiento ($O(n\,p)$ por consulta).
- **Búsqueda.** El random forest consumió 12,9 de los 25,3 minutos de búsqueda de las 28 combinaciones.""",

"vs_ater": r"""#MD**Lectura.**

- **ATER gana a las 7 mejores combinaciones del Entregable 2 en la prueba 2025–2026.** Las más cercanas son el random forest (MSE 2.361; ATER reduce el error 4,4 %, IC 95 %: 1,7–7,3 %) y el XGBoost (2.448; reducción de 7,9 %, IC: 5,2–10,3 %).
- **Las pruebas lo confirman.** Giacomini-White y Diebold-Mariano rechazan la igualdad frente a las 7, con p ajustado por Holm < 0,001 en las 14 comparaciones.
- **En el MCS con los 15 modelos, ATER es otra vez el único sobreviviente.** El random forest del Entregable 2 queda fuera con p = 0,002.
- **Tamaño del efecto.** Frente a las combinaciones de árboles es pequeño (δ de Cliff pareado 0,28). Frente a KNN, SVR, Lasso, Ridge y el árbol de decisión es de mediano a grande (0,35 a 0,73).
- **Conclusión:** sumar las 28 combinaciones del Entregable 2 no cambia el veredicto de R3. ATER sigue siendo el mejor modelo con significancia estadística.""",
}
