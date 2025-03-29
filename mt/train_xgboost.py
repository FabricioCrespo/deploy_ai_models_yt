import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import (mean_squared_error, 
                            mean_absolute_error, 
                            r2_score)
import joblib
import os
import time
import gc
import json
import sys
from memory_profiler import memory_usage

def load_data(movies_path, ratings_path, user_encoder_path, sample_fraction=None):
    """Carga y prepara los datos optimizando memoria"""
    print("Cargando datos...")
    start_time = time.time()
    
    try:
        # Cargar solo columnas esenciales
        movies = pd.read_csv(movies_path, usecols=['movieId', 'year'])
        ratings = pd.read_csv(ratings_path,
                            usecols=['userId', 'movieId', 'rating', 'year_rated', 'month_rated'])
        
        if sample_fraction and sample_fraction < 1.0:
            ratings = ratings.sample(frac=sample_fraction, random_state=42)
            print(f"✅ Muestreo aplicado: {len(ratings)} registros")
        
        # Codificar usuarios
        user_encoder = joblib.load(user_encoder_path)
        ratings['user_encoded'] = user_encoder.transform(ratings['userId'])
        
        # Combinar datos
        data = pd.merge(ratings, movies, on='movieId')
        
        # Liberar memoria
        del movies, ratings
        gc.collect()
        
        print(f"⏱️ Datos cargados en {time.time() - start_time:.2f} segundos")
        return data
    
    except Exception as e:
        print(f"❌ Error cargando datos: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo con múltiples métricas"""
    print("\nEvaluando modelo...")
    eval_start_time = time.time()  # Definimos el start_time aquí
    
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'Tiempo evaluación': time.time() - eval_start_time  # Usamos eval_start_time
    }
    
    print("\n📊 Métricas de evaluación:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Entrena y evalúa un modelo XGBoost"""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 0.1,
        'lambda': 1.0,
        'seed': 42,
        'n_jobs': -1
    }
    
    try:
        # Convertir a formato DMatrix óptimo para XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        print("\n🔍 Entrenando modelo XGBoost...")
        train_start_time = time.time()
        
        # Entrenar con early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        train_time = time.time() - train_start_time
        print(f"\n✅ Entrenamiento completado en {train_time:.2f} segundos")
        
        return model, train_time
    
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        return None, 0

def save_model_and_metrics(model, metrics, model_dir='./model_xgb'):
    """Guarda el modelo y las métricas"""
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar modelo
        model_path = f'{model_dir}/xgboost_recommender.model'
        model.save_model(model_path)
        print(f"\n💾 Modelo XGBoost guardado en {model_path}")
        
        # Guardar métricas
        metrics_path = f'{model_dir}/training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"📝 Métricas guardadas en {metrics_path}")
        
    except Exception as e:
        print(f"\n❌ Error guardando modelo: {str(e)}")
        raise

if __name__ == '__main__':
    # Configuración
    data_dir = './data_processed'
    movies_path = f'{data_dir}/movies_processed.csv'
    ratings_path = f'{data_dir}/ratings_processed.csv'
    user_encoder_path = f'{data_dir}/user_encoder.pkl'
    model_dir = './model_xgb'
    
    # Opción para usar muestra de datos (ej. 0.5 = 50% de los datos)
    SAMPLE_FRACTION = None  # Cambiar a un valor entre 0.1-1.0 si es necesario
    
    print("🚀 Iniciando proceso de entrenamiento XGBoost")
    
    try:
        # 1. Cargar datos
        data = load_data(movies_path, ratings_path, user_encoder_path, sample_fraction=SAMPLE_FRACTION)
        if data is None:
            sys.exit(1)
        
        # 2. Preparar características
        X = data[['user_encoded', 'movieId', 'year_rated', 'month_rated', 'year']]
        y = data['rating']
        
        # 3. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 4. Entrenar modelo
        model, train_time = train_xgboost_model(X_train, y_train, X_test, y_test)
        if model is None:
            sys.exit(1)
        
        # 5. Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test)
        metrics.update({
            'Tiempo entrenamiento': train_time,
            'Memoria máxima': memory_usage(-1, interval=0.1, max_usage=True)
        })
        
        # 6. Guardar resultados
        save_model_and_metrics(model, metrics, model_dir)
        
        # 7. Resultados finales
        print("\n🎉 Resumen final del entrenamiento:")
        print(f"- RMSE: {metrics['RMSE']:.4f}")
        print(f"- R²: {metrics['R²']:.4f}")
        print(f"- Tiempo total: {metrics['Tiempo entrenamiento']:.2f}s")
        print(f"- Memoria máxima usada: {metrics['Memoria máxima']:.2f} MiB")
        
    except Exception as e:
        print(f"\n❌ Error fatal en el proceso: {str(e)}")
        sys.exit(1)