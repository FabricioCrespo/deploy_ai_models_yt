# MT: MOVIE RECOMMENDATION SYSTEM

## DATA PREPROCESSING, MODELS TRAINING AND EVALUATION, MODEL SERVING AND KAFKA CONSUMER.

### Step a step:

1. Get datasets: `movie.csv` and `rating.csv`.
2. `preprocessing.py` script preprocesses movie and rating data: extracts movie years and one-hot encodes genres from movies, processes rating timestamps into years/months, saves processed data as CSV files, and stores a user label encoder and genre list for later use. Outputs are saved in a specified directory. (50 words)
3.
    - 3.1. `train_model.py` trains, and evaluates a **RandomForestRegressor** meanwhile `predict.py` lets us to make a prediction using the saved model. 
    - 3.2. `train_xgboost.py` trains, and evaluates a **XGboost model** meanwhile `predict_xgboost.py` lets us to make a prediction using the saved model. 
4. Define a docker compose file to have two services:
    - Zookeeper
    - Kafka with a predefined topic named **mt_topic** to get data from the MOVIE RECOMMENDATION SERVER. We can use the following command to run the docker compose: `docker compose up -d` and `docker compose down` to turn off the services.
5. Define a script to build a server using Flask. We should be able to use the API to make a prediction given an user ID but also, send metrics to a Kafka broker using **mt_topic** and KafkaProducer: `python app.py`

6. Make request to the Server API Ex:
    
    `curl http://localhost:8082/recommend/3` where 3 is de user_id

7. We can consume the data in the kafka topic using the following command:

    `docker exec -it 452fa9cfb940 /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic mt_topic --from-beginning`. Please replace **452fa9cfb940** with your current docker_id where kafka is running (Get ir running `docker ps` and find the kafka docker id). Finally, we can get something like:

        2025-03-29 02:37:31,3,recommendation request ubuntu-MS-7D31, status 200, result: 319,324,326,773,1162,1234,298,750,320,321,1252,1136,770,775,293,1260,1178,755,1212,1226, 1397
    which follows the required format:
        
        expect to see an entry <time>,<userid>,recommendation request <server>, status <200 for success>, result: <parsed recommendations>, <responsetime> with status code 200

## RESULTS:

- RandomForestRegressor training and evaluation results:

<img src="images_rd/1.png" alt="Imagen 1" width="300">

- XGBoost training and evaluation results:

<img src="images_rd/2.png" alt="Imagen 2" width="300">

- Docker compose with Kafka and Zookeper services:

<img src="images_rd/3.png" alt="Imagen 2" width="300">

- Movie recommendation system with FLASK:

<img src="images_rd/4.png" alt="Imagen 2" width="300">

- Kafka consumer: We can see all required data about the request.

<img src="images_rd/5.png" alt="Imagen 2" width="300">