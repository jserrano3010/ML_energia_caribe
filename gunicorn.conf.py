# Configuración de gunicorn (se carga sola desde la raíz del repositorio).
# 1 proceso (memoria del plan gratuito) con 4 hilos: un cálculo lento no bloquea los demás clics.
workers = 1
threads = 4
timeout = 120
