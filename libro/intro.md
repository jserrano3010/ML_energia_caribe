# Calor y energía en el Caribe

**Pronóstico horario de la demanda eléctrica de Air-e con el clima de Sabanagrande y un modelo original de agregación de expertos (ATER)**

**Curso:** Machine Learning, Maestría en Ingeniería Industrial. **Profesor:** Dr. Lihki Rubio. **Autores:** Edwin Yunis y Jairo Serrano. **Versión:** Entregable 3, octubre de 2026.

**Dashboard desplegado (Render):** <https://ml-energia-caribe.onrender.com> · **Repositorio:** <https://github.com/jserrano3010/ML_energia_caribe>

## Resumen ejecutivo
En Sabanagrande (Atlántico), como en todo el Caribe colombiano, el calor dispara el consumo de electricidad: aires acondicionados, ventiladores y neveras trabajan más cuando sube la sensación térmica. Los hogares de la región pagan una de las tarifas más altas del país. La tarifa regulada de Air-e subió alrededor de 74 % hasta julio de 2023, y el Consumo Básico de Subsistencia no reconoce que en clima cálido la energía es una necesidad para refrigerar (Char, s. f.). En el mercado mayorista la energía se compra **el día anterior**, y cada megavatio-hora mal pronosticado se liquida después en la bolsa.

El proyecto pronostica, al final de cada día, **la demanda de cada una de las 24 horas del día siguiente** del mercado de Air-e (al que pertenece Sabanagrande) con el calendario y el clima de Sabanagrande. Para eso:

- Compara **7 modelos de referencia**: 3 tradicionales y 4 de Machine Learning (Ridge, random forest, XGBoost y un híbrido).
- Propone un **modelo original, ATER**. ATER aprende cada día a qué experto creerle según la hora y corrige en línea el nivel del pronóstico.

## Respuesta corta
- **ATER es el mejor modelo.** En la prueba 2025–2026 (653 días, nunca usados para ajustar) tiene **2,44 % de error** (MAPE) y reduce el error cuadrático:
  - **65 %** frente a la persistencia (misma hora de ayer).
  - **3,9 %** frente al mejor benchmark, el random forest relativo (IC 95 %: 1,1–6,8 %).
- **La superioridad es estadísticamente significativa** frente a los 7 benchmarks, con p ajustado por Holm menor que 0,05 en tres pruebas (Model Confidence Set, Giacomini-White y Diebold-Mariano). En el MCS, ATER es el **único** modelo que sobrevive.
- **El efecto es pequeño frente a los mejores árboles.** El δ de Cliff pareado es 0,25 y ATER gana el 62,6 % de los días; frente a los métodos tradicionales el efecto es grande.
- Anticipa el **91 %** de las horas de demanda alta con 9 % de falsas alarmas.
- **Es el quinto problema evaluado.** Lluvia, precio del maíz, decisión de siembra y Baloto se descartaron porque no cumplían los criterios de evaluación (capítulo 1).

## Pregunta de investigación
¿Puede un modelo propio de agregación en línea pronosticar la demanda horaria del día siguiente del mercado de Air-e mejor que los métodos tradicionales y que los modelos de Machine Learning conocidos? ¿Lo logra con superioridad estadísticamente significativa?

## Objetivos
- **General:** desarrollar y validar un modelo original de pronóstico de la demanda horaria del día siguiente para el mercado de Air-e que supere a los benchmarks con significancia estadística.
- **Específicos:**
  1. Construir un dataset horario de demanda (XM) y clima (NASA POWER) sin fuga de información.
  2. Comparar 3 métodos tradicionales y 4 de Machine Learning con validación temporal.
  3. Diseñar, implementar y probar el modelo original ATER.
  4. Verificar su superioridad con al menos tres pruebas de hipótesis, intervalos y tamaños del efecto.
  5. Ponerlo en producción en un dashboard.

## Mapa del libro

| Cap. | Contenido |
|---|---|
| 1 | Trazabilidad de correcciones y problemas descartados |
| 2 | Datos: ETL y EDA |
| 3 | Modelos benchmark: validación, prueba, horas pico, valor y rúbrica |
| 4 | Revisión de literatura y novedad |
| 5 | Modelo original ATER |
| 6 | Validación estadística |
| 7 | Dashboard y despliegue |
| 8 | Conclusiones y limitaciones |
| 9 | Reproducibilidad y bibliografía |

```{tableofcontents}
```
