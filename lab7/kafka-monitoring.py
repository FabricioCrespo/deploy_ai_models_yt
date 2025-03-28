from kafka import KafkaConsumer
from prometheus_client import Counter, Histogram, start_http_server
import random
import time

# Configuración
topic = 'lab07'  # Cambia por tu topic real
start_http_server(8765)

# Métricas
REQUEST_COUNT = Counter(
    'request_count', 
    'Recommendation Request Count',
    ['http_status']
)

REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 
    'Request latency',
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

def generate_synthetic_data():
    """Genera datos sintéticos cuando no hay mensajes reales"""
    status_codes = [200, 400, 404, 500]
    return {
        'user_id': f"user_{random.randint(1, 10000)}",
        'timestamp': int(time.time()),
        'request_type': 'recommendation request',
        'status': random.choice(status_codes),
        'latency_ms': random.uniform(10, 500)
    }

def process_message(message_value):
    """Procesa el mensaje de Kafka o genera datos sintéticos"""
    try:
        if message_value:
            event = message_value.decode('utf-8')
            values = event.split(',')
            
            if len(values) >= 5 and 'recommendation request' in values[2]:
                # Extraer status (formato: "status=200")
                status_part = [p for p in values if p.startswith('status=')][0]
                status = status_part.split('=')[1].strip()
                
                # Extraer latencia (formato: "123.45ms")
                latency_part = values[-1].strip()
                time_taken = float(latency_part.replace('ms', ''))
                
                return status, time_taken
        return None, None
    except Exception as e:
        print(f"Error processing message: {e}")
        return None, None

def main():
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        group_id=f'{topic}-consumer',
        enable_auto_commit=True,
        auto_commit_interval_ms=1000,
        consumer_timeout_ms=10000  # Timeout para evitar bloqueo
    )

    while True:
        # Procesar mensajes reales
        for message in consumer:
            status, latency = process_message(message.value)
            if status and latency:
                REQUEST_COUNT.labels(http_status=status).inc()
                REQUEST_LATENCY.observe(latency / 1000)  # Convertir a segundos
        
        # Generar datos sintéticos si no hay mensajes
        synthetic_data = generate_synthetic_data()
        REQUEST_COUNT.labels(http_status=str(synthetic_data['status'])).inc()
        REQUEST_LATENCY.observe(synthetic_data['latency_ms'] / 1000)
        print(f"Generated synthetic data: {synthetic_data}")

        time.sleep(1)  # Esperar antes de generar más datos

if __name__ == "__main__":
    main()