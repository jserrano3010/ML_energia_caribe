# Marco de referencia

## El mercado eléctrico y el pronóstico de demanda
En el Mercado de Energía Mayorista de Colombia, XM programa el despacho del día siguiente con las ofertas de los generadores y la demanda esperada; los comercializadores compran por contratos y en bolsa, y las diferencias entre lo previsto y lo real se liquidan al precio de bolsa. El pronóstico de demanda de corto plazo (horas a días) es por eso una tarea central de la operación (Hong y Fan, 2016).

## Tarifas y calor en el Caribe
El servicio de energía en la Costa Caribe es más caro y menos estable que en el resto del país, en una región con temperaturas altas todo el año y altos niveles de vulnerabilidad socioeconómica. En 2020 se intervino a Electricaribe y el servicio pasó a Air-e (Atlántico, Magdalena y La Guajira) y a Afinia. Hasta julio de 2023 los precios regulados habían subido cerca de 74 % en Air-e y 60 % en Afinia, por encima de la inflación, en parte por las pérdidas de energía y por el cobro diferido de la *opción tarifaria* de la pandemia. El Consumo Básico de Subsistencia que se subsidia no reconoce del todo que en clima cálido los ventiladores, los aires y la refrigeración son necesidades básicas (Char, s. f.). En Atlántico, además, el 70,1 % de los hogares que reciben subsidios de energía no son pobres, y el consumo de subsistencia es de 173 kWh/mes por debajo de 1.000 m de altitud (Fundesarrollo, 2023); las alzas tarifarias han golpeado a los hogares y a la economía de la región (Fundesarrollo, 2022; Corficolombiana, 2024). En climas cálidos, el ingreso y la temperatura empujan la compra de aires acondicionados, que es lo que dispara la demanda en las horas de calor (Davis y Gertler, 2015; Auffhammer y Mansur, 2014). Pronosticar bien **cuánta energía se va a consumir mañana, hora por hora, y en especial en las horas de calor**, es la base para comprarla a menor costo y para operar la red sin cortes.

## Determinantes de la demanda en el Caribe
La demanda horaria depende del ciclo diario de actividad, del día de la semana, de los festivos y, en climas cálidos y húmedos, de la **sensación térmica**, que dispara el uso de aire acondicionado y refrigeración. El índice de calor de la NOAA (Rothfusz, 1990) combina temperatura y humedad en una sola variable.

## Métodos
- **Tradicionales:** persistencia (misma hora de ayer), estacional ingenuo (misma hora de la semana pasada) y perfiles promedio por hora y día (Hyndman y Athanasopoulos, 2021).
- **Lineal:** Ridge con perfiles hora × tipo de día.
- **No lineales:** random forest (Breiman, 2001) y gradient boosting / XGBoost (Friedman, 2001; Chen y Guestrin, 2016).
- **Híbrido:** Ridge para la parte regular y XGBoost para el residuo, en la línea de los modelos de dos etapas para carga eléctrica (Hong y Fan, 2016).
- **Agregación de expertos en línea:** combinar varios pronósticos con pesos que se actualizan con el desempeño reciente (Cesa-Bianchi y Lugosi, 2006; Devaine et al., 2013). Es la familia del modelo original de este proyecto (capítulos 4 y 5).
- **Objetivo relativo:** los árboles no extrapolan fuera del rango de entrenamiento; normalizar por el nivel reciente es la práctica estándar con series con tendencia.

## Evaluación honesta
Validación temporal con separación (Roberts et al., 2017), regla de elección fijada de antemano, prueba usada una vez, Diebold-Mariano con corrección HLN (Diebold y Mariano, 1995; Harvey et al., 1997) e intervalos por bootstrap de bloques (Künsch, 1989).
