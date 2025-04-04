import socket
import logging
import sqlite3
from kafka import KafkaConsumer
from datetime import datetime
import re

# Configuración de Kafka
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'input_streaming_movies'
DB_FILE = '../db/movies.db'  # Archivo de la base de datos SQLite

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear base de datos y tablas
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS movies (
    etl_timestamp TEXT,
    movieId INTEGER,
    title STRING,
    genres STRING
)
               
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    insert_timestamp TEXT,
    status TEXT
)
''')
conn.commit()
conn.close()

class KafkaCSVConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='earliest',
            value_deserializer=lambda x: x.decode('utf-8')
        )
    
    def consume_messages(self):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        for message in self.consumer:
            print(message.value)
            message_content = message.value
            insert_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if "Mensaje aleatorio generado" in message_content:
                cursor.execute("INSERT INTO logs (insert_timestamp, status) VALUES (?, ?)",
                               (insert_timestamp, "BAD"))
            else:
                try:
                    parts = message_content.split(", ")
                    #parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', message_content)
                    movieId = parts[1]
                    genres = parts[-1]
                    title = ','.join(parts[1:-1]).strip()
                    etl_timestamp = parts[0]
                    #movieId, title, genres = map(str.strip, parts[1:])
                    
                    cursor.execute("INSERT INTO movies (etl_timestamp, movieId, title, genres) VALUES (?, ?, ?, ?)",
                                   (etl_timestamp, movieId, title, genres))
                    cursor.execute("INSERT INTO logs (insert_timestamp, status) VALUES (?, ?)",
                                   (insert_timestamp, "OK"))
                except Exception as e:
                    cursor.execute("INSERT INTO logs (insert_timestamp, status) VALUES (?, ?)",
                               (insert_timestamp, "BAD"))
                    logger.error(f"❌ Error procesando mensaje: {str(e)}, {parts}")
            
            conn.commit()
        conn.close()

if __name__ == "__main__":
    consumer = KafkaCSVConsumer()
    consumer.consume_messages()
