from flask import Flask, jsonify, request
import xgboost as xgb
import joblib
import pandas as pd
from datetime import datetime
import time
import logging
from functools import wraps
from kafka import KafkaProducer
import socket

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración de Kafka
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'mt_topic'
SERVER_NAME = socket.gethostname()

class KafkaLogProducer:
    def __init__(self):
        self.producer = None
        
    def connect(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda x: x.encode('utf-8'),
                acks=1,
                retries=3
            )
            logger.info(f"✅ Conectado a Kafka en {KAFKA_BROKER}")
            return True
        except Exception as e:
            logger.error(f"❌ Error conectando a Kafka: {str(e)}")
            return False
    
    def send_log(self, user_id, status, recommendations, response_time):
        """Envía log al topic de Kafka en el formato especificado"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            recommendations_str = ",".join(map(str, recommendations)) if recommendations else "none"
            
            log_entry = (
                f"{timestamp},{user_id},recommendation request {SERVER_NAME}, "
                f"status {status}, result: {recommendations_str}, {response_time}"
            )
            
            self.producer.send(KAFKA_TOPIC, value=log_entry)
            self.producer.flush()
            logger.info(f"📨 Log enviado a Kafka: {log_entry}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando log a Kafka: {str(e)}")
            return False

class RecommendationService:
    def __init__(self, kafka_producer):
        self.model = None
        self.user_encoder = None
        self.movies_info = None
        self.movies_processed = None
        self.ratings_path = './data_processed/ratings_processed.csv'
        self.loaded = False
        self.kafka = kafka_producer
        
    def load_resources(self):
        """Carga todos los recursos necesarios para el servicio"""
        try:
            # Cargar modelo
            self.model = xgb.Booster()
            self.model.load_model('./model_xgb/xgboost_recommender.model')
            
            # Cargar encoder
            self.user_encoder = joblib.load('./data_processed/user_encoder.pkl')
            
            # Cargar datos de películas
            self.movies_info = pd.read_csv('./movie.csv')[['movieId', 'title']]
            self.movies_processed = pd.read_csv('./data_processed/movies_processed.csv')[['movieId', 'year']]
            
            self.loaded = True
            logger.info("✅ Todos los recursos cargados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando recursos: {str(e)}")
            self.loaded = False
            return False
    
    def get_unseen_movies(self, user_id):
        """Obtiene películas no vistas por el usuario"""
        try:
            ratings = pd.read_csv(self.ratings_path, usecols=['userId', 'movieId'])
            user_ratings = ratings[ratings['userId'] == user_id]
            all_movies = self.movies_processed['movieId'].unique()
            return list(set(all_movies) - set(user_ratings['movieId']))
        except Exception as e:
            logger.error(f"Error obteniendo películas no vistas: {str(e)}")
            return []
    
    def prepare_features(self, user_id, movie_ids):
        """Prepara características para predicción"""
        try:
            features = pd.DataFrame({'movieId': movie_ids})
            features['user_encoded'] = self.user_encoder.transform([user_id])[0]
            
            now = datetime.now()
            features['year_rated'] = now.year
            features['month_rated'] = now.month
            
            features = features.merge(self.movies_processed, on='movieId', how='left')
            
            # Asegurar el orden correcto de características (sin movieId)
            return features[self.model.feature_names]
        except Exception as e:
            logger.error(f"Error preparando características: {str(e)}")
            raise
    
    def predict(self, user_id, max_recommendations=20):
        """Genera recomendaciones para un usuario"""
        if not self.loaded:
            raise RuntimeError("Modelo no cargado")
        
        start_time = time.time()
        
        try:
            # 1. Obtener películas no vistas
            unseen_movies = self.get_unseen_movies(user_id)
            if not unseen_movies:
                logger.info(f"Usuario {user_id} ya ha calificado todas las películas")
                return []
            
            # 2. Predecir ratings
            features = self.prepare_features(user_id, unseen_movies)
            dmatrix = xgb.DMatrix(features)
            predictions = self.model.predict(dmatrix)
            
            # 3. Crear DataFrame con resultados
            results = pd.DataFrame({
                'movieId': unseen_movies,
                'predicted_rating': predictions
            })
            
            # 4. Ordenar y seleccionar las mejores
            top_movies = results.sort_values('predicted_rating', ascending=False)
            recommendations = top_movies.head(max_recommendations)['movieId'].tolist()
            
            # 5. Registrar en Kafka
            response_time = int((time.time() - start_time) * 1000)  # en ms
            self.kafka.send_log(user_id, 200, recommendations, response_time)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {str(e)}")
            raise

# Inicializar el producer de Kafka
kafka_producer = KafkaLogProducer()
if not kafka_producer.connect():
    logger.warning("⚠️ El servicio funcionará sin Kafka")

# Inicializar el servicio de recomendación
service = RecommendationService(kafka_producer)
service.load_resources()

def measure_time(f):
    """Decorador para medir tiempo de respuesta"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        response_time = int((time.time() - start_time) * 1000)
        logger.info(f"Tiempo de respuesta: {response_time}ms")
        return result
    return wrapper

@app.route('/recommend/<int:user_id>', methods=['GET'])
@measure_time
def recommend(user_id):
    """Endpoint para obtener recomendaciones"""
    try:
        if not service.loaded:
            # Enviar log de error a Kafka
            service.kafka.send_log(user_id, 500, [], 0)
            return jsonify({
                "status": 500,
                "message": "Model not loaded"
            }), 500
        
        # Obtener recomendaciones
        start_time = time.time()
        recommendations = service.predict(user_id)
        response_time = int((time.time() - start_time) * 1000)
        
        # Verificar tiempo de respuesta
        if response_time > 800:
            logger.warning(f"⚠️ Tiempo de respuesta {response_time}ms excede el límite de 800ms")
        
        # Formatear respuesta como CSV
        recommendations_csv = ",".join(map(str, recommendations)) if recommendations else ""
        
        return recommendations_csv, 200
    
    except Exception as e:
        logger.error(f"Error en endpoint /recommend: {str(e)}")
        service.kafka.send_log(user_id, 500, [], 0)
        return jsonify({
            "status": 500,
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar salud del servicio"""
    kafka_status = "connected" if kafka_producer.producer else "disconnected"
    return jsonify({
        "status": "healthy" if service.loaded else "unhealthy",
        "model_loaded": service.loaded,
        "kafka": kafka_status
    }), 200 if service.loaded else 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, threaded=True)