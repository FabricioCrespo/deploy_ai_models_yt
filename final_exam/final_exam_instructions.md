
# Lanzar AIRFLOW

Lanzar ambiente de conda de airflow: `conda activate airflow`

Lanzar scheduler: `airflow scheduler`

Lanzar webapplication: `airflow webserver -p 8080`

# En una ventana de firefox tener las siguientes pestañas:

`http://localhost:8080` Airflow web server

`http://localhost:9090` Prometheus web server

`http://localhost:3000` Grafana web server

`http://localhost:8020/` Jenkins web server


# Lanzar docker para simular streamings

- Inicializar los dockers de streaming: 
    - ir al path `/home/fabricio/Documents/maestria_ai/AI_MODELS_DEPLOY_FINAL2/deploy_ai_models_yt/final_exam/data_ingestion`
    y luego ejecutar `docker compose up -d`

- Para asegurarnos que sean topics limpios puedo correr: `docker-compose rm -sf` (elimina todos los dockers y empieza desde 0) y de nuevo ejecutar `docker compose up -d`

# PASOS PARA EXPOSICION 

1. Ejecutar el dag 01 que simula el streaming del dataset como fuente de datos.
2. ejecutar el dag 02 que lee los mensajes enviado a los topics de kakka y los almacena en SQLITE DB.

3. Podemos marcar el dag 01 y 02 como success debido a que la ingesta es lenta y no hay problema si la detenemos antes de que acabe.
    
    3.1 Aqu'i debo ir a detener los dockers en la misma direccion: 
    - ir al path `/home/fabricio/Documents/maestria_ai/AI_MODELS_DEPLOY_FINAL2/deploy_ai_models_yt/final_exam/data_ingestion`
    y luego ejecutar `docker compose down`
4. Ejecutar el dag 03 de entrenamiento y evaluacion de modelos.
5. Ejecutar el dag 04 que es el deploy del servidor que sirve para hacer request al modelo. Este se va a quedar corriendo eternamente.
6. Ejecutar el monitoreo con el dag 05. Aqui se debe abrir tanto las apps de Prometheus como Grafana. En grafana hay que importar el json que tiene el dashboard pero hay que entrar a editar cada panel y ejecutar la query para que se muestren las graficas. 
7. Prueba de CI con jenkins: ejecutar el dag 06, entrar a jenkins y ejecutar el pipline. Mostrar el log y ver que se testearon 5 cosas y se tuvieron 4 OK y 1 BAD debido a que no esta disponible el modelo dentro del docker. Se arregla si metes el modelo en el repo o en el docker para que si lo encuentre. Tambien puedes ir a la carpeta de app y ejecutar el comando `PYTHONPATH=. pytest App/tests/`.

