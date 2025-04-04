import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, 
                            mean_absolute_error, 
                            r2_score)
import joblib
import os
import time
import gc
from memory_profiler import memory_usage
import sys

def print_memory_usage():
    """Imprime el uso actual de memoria"""
    usage = memory_usage(-1, interval=0.1)[0]
    print(f"Uso actual de memoria: {usage:.2f} MiB")

def load_data_optimized(movies_path, ratings_path, user_encoder_path, sample_fraction=None):
    """
    Carga datos optimizando memoria, con opción de muestreo
    """
    print("\nCargando datos...")
    start_time = time.time()
    
    # Cargar solo columnas esenciales de películas
    movies = pd.read_csv(movies_path, usecols=['movieId', 'year'])
    
    # Cargar ratings con muestreo si se especifica
    ratings = pd.read_csv(ratings_path, 
                         usecols=['userId', 'movieId', 'rating', 'year_rated', 'month_rated'])
    
    if sample_fraction and sample_fraction < 1.0:
        ratings = ratings.sample(frac=sample_fraction, random_state=42)
        print(f"Muestreo aplicado: {len(ratings)} registros")
    
    # Codificar usuarios
    user_encoder = joblib.load(user_encoder_path)
    ratings['user_encoded'] = user_encoder.transform(ratings['userId'])
    
    # Combinar datos
    data = pd.merge(ratings, movies, on='movieId')
    
    # Liberar memoria
    del movies, ratings
    gc.collect()
    
    load_time = time.time() - start_time
    print(f"Datos cargados en {load_time:.2f} segundos")
    print_memory_usage()
    
    return data

def evaluate_model(model, X_test, y_test):
    """Calcula múltricas métricas de evaluación"""
    print("\nEvaluando modelo...")
    start_time = time.time()
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'Tiempo evaluación': time.time() - start_time
    }
    
    print("Métricas de evaluación:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

def train_and_evaluate(data, model_dir='./models'):
    """Entrena y evalúa el modelo con tracking de métricas"""
    print("\nPreparando entrenamiento...")
    
    # 1. Preparar datos
    X = data[['user_encoded', 'movieId', 'year_rated', 'month_rated', 'year']]
    y = data['rating']
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Liberar memoria
    del data
    gc.collect()
    
    # 3. Configurar modelo optimizado
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # 4. Entrenar con tracking de tiempo y memoria
    print("\nIniciando entrenamiento...")
    train_start = time.time()
    
    # Medir memoria máxima usada durante el entrenamiento
    mem_usage = memory_usage((model.fit, [X_train, y_train]), 
                           interval=0.1,
                           max_usage=True)
    
    train_time = time.time() - train_start
    
    print("\nEntrenamiento completado!")
    print(f"Tiempo de entrenamiento: {train_time:.2f} segundos")
    print(f"Memoria máxima usada: {mem_usage:.2f} MiB")
    
    # 5. Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    metrics.update({
        'Tiempo entrenamiento': train_time,
        'Memoria máxima': mem_usage
    })
    
    # 6. Guardar modelo y métricas
    os.makedirs(model_dir, exist_ok=True)
    model_path = f'{model_dir}/optimized_random_forest_recommender.pkl'
    joblib.dump(model, model_path)
    print(f"\nModelo guardado en {model_path}")
    
    metrics_path = f'{model_dir}/training_random_forest_metrics.json'
    pd.Series(metrics).to_json(metrics_path)
    print(f"Métricas guardadas en {metrics_path}")
    
    return model, metrics

if __name__ == '__main__':
    # Configuración
    data_dir = './data'
    movies_path = f'{data_dir}/movies_processed.csv'
    ratings_path = f'{data_dir}/ratings_processed.csv'
    user_encoder_path = f'{data_dir}/user_encoder.pkl'
    models_path = './models'
    
    # Opción para usar muestra de datos (ej. 0.5 = 50% de los datos)
    SAMPLE_FRACTION = None  # Cambiar a un valor entre 0.1-1.0 si es necesario
    
    try:
        # 1. Cargar datos
        data = load_data_optimized(
            movies_path, ratings_path, user_encoder_path, 
            sample_fraction=SAMPLE_FRACTION
        )
        
        # 2. Entrenar y evaluar
        model, metrics = train_and_evaluate(data, model_dir=models_path)
        
        # 3. Resultados finales
        print("\nResumen de métricas:")
        print(f"- RMSE: {metrics['RMSE']:.4f}")
        print(f"- R²: {metrics['R²']:.4f}")
        print(f"- Tiempo entrenamiento: {metrics['Tiempo entrenamiento']:.2f}s")
        print(f"- Memoria máxima usada: {metrics['Memoria máxima']:.2f} MiB")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}", file=sys.stderr)
        sys.exit(1)