
# Comandos usados

## docker compose:
- Levantar dockers:

    `docker compose up -d `

- Detener los dockers del compose :

    `docker compose down`

## Para acceder desde el docker de prometheus al puerto 8765 para metricas se necesita identificar la IP de la maquina host:

    `ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'`

Tal vez se necesite instalar ifconfig pero ahi sale la sugerencia si no hay.

- PERMITIR CONEXION POR EL PUERTO:

    `sudo ufw allow 8765`

- Correr kafka monitoring y levantar servidor en el puerto 8765

    `python kafka-monitoring.py`