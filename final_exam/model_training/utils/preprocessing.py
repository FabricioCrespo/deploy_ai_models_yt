import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import sqlite3
import os

DB_FILE_MOVIES = './db/movies.db'  # Archivo de la base de datos SQLite
DB_FILE_RATINGS = './db/ratings.db'  # Archivo de la base de datos SQLite

def preprocess_data(output_dir='./data_processed'):
    """
    Preprocesa los datos de películas y ratings y guarda los resultados en archivos CSV
    """
    # Cargar datos

    # DB CONNECTION
    conn_movies = sqlite3.connect(DB_FILE_MOVIES)

    #Movies
    movies = pd.read_sql_query("SELECT * FROM movies", conn_movies)
    #movies = pd.read_csv(movies_path)
    # delete column etl_timestamp from movies
    movies.drop(columns=['etl_timestamp'], inplace=True)
    
    #Ratings
    conn_ratings = sqlite3.connect(DB_FILE_RATINGS)
    ratings = pd.read_sql_query("SELECT * FROM ratings", conn_ratings)
    # delete column etl_timestamp from ratings
    ratings.drop(columns=['etl_timestamp'], inplace=True)
    #ratings = pd.read_csv(ratings_path)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Procesamiento de películas
    print("Procesando datos de películas...")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    genres = movies['genres'].str.get_dummies(sep='|')
    movies_processed = pd.concat([movies, genres], axis=1)
    movies_processed.drop(['title', 'genres'], axis=1, inplace=True)
    
    # Guardar películas procesadas
    movies_processed.to_csv(f'{output_dir}/movies_processed.csv', index=False)
    print(f"Películas procesadas guardadas en {output_dir}/movies_processed.csv")
    
    # Procesamiento de ratings
    print("Procesando datos de ratings...")
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'])
    ratings['year_rated'] = ratings['timestamp'].dt.year
    ratings['month_rated'] = ratings['timestamp'].dt.month
    ratings_processed = ratings.drop('timestamp', axis=1)
    
    # Guardar ratings procesados
    ratings_processed.to_csv(f'{output_dir}/ratings_processed.csv', index=False)
    print(f"Ratings procesados guardados en {output_dir}/ratings_processed.csv")
    
    # Codificar usuarios y guardar encoder
    user_encoder = LabelEncoder()
    user_encoder.fit(ratings_processed['userId'])
    joblib.dump(user_encoder, f'{output_dir}/user_encoder.pkl')
    print(f"User encoder guardado en {output_dir}/user_encoder.pkl")
    
    # Guardar lista de géneros
    genres_list = list(genres.columns)
    joblib.dump(genres_list, f'{output_dir}/genres_list.pkl')
    print(f"Lista de géneros guardada en {output_dir}/genres_list.pkl")
    
    print("Preprocesamiento completado!")

if __name__ == '__main__':
    # Configuración de paths
    #movies_path = './data/movies.csv'
    #ratings_path = './data/ratings.csv'
    output_dir = './data'
    
    # Ejecutar preprocesamiento
    preprocess_data(output_dir)