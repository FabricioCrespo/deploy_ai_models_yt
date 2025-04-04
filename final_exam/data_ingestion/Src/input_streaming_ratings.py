import socket
import random
import logging
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime

# Configuración de Kafka
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'input_streaming_ratings'
SERVER_NAME = socket.gethostname()
CSV_FILE = '../datasets/ratings.csv'  # Ruta del archivo CSV
CHUNK_SIZE = 100  # Número de filas a leer por lote

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaCSVProducer:
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
    
    def send_message(self, message):
        try:
            self.producer.send(KAFKA_TOPIC, value=message)
            self.producer.flush()
            logger.info(f"📨 Mensaje enviado a Kafka: {message}")
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje a Kafka: {str(e)}")
    
    def simulate_transmission(self, csv_file):
        try:
            for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE):
                for _, row in chunk.iterrows():
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if random.random() < 0.1:  # 10% de los casos
                        message = f"{timestamp}, Mensaje aleatorio generado"
                    else:
                        message = f"{timestamp}, {', '.join(map(str, row.values))}"
                    
                    self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error leyendo CSV: {str(e)}")

if __name__ == "__main__":
    producer = KafkaCSVProducer()
    if producer.connect():
        producer.simulate_transmission(CSV_FILE)
