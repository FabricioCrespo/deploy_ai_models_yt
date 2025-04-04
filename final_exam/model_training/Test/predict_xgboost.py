import pandas as pd
import xgboost as xgb
import joblib
import os
import sys
from datetime import datetime

class XGBoostRecommender:
    def __init__(self, model_path, encoder_path, movies_path, processed_movies_path):
        """
        Inicializa el sistema de recomendación cargando todos los recursos necesarios.
        
        Args:
            model_path: Ruta al modelo XGBoost (.model o .json)
            encoder_path: Ruta al encoder de usuarios (.pkl)
            movies_path: Ruta al archivo original de películas (movie.csv)
            processed_movies_path: Ruta al archivo procesado de películas (movies_processed.csv)
        """
        try:
            # Cargar modelo XGBoost
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            
            # Verificar características esperadas por el modelo
            self.expected_features = self.model.feature_names
            print(f"✅ Características esperadas por el modelo: {self.expected_features}")
            
            # Cargar encoder de usuarios
            self.user_encoder = joblib.load(encoder_path)
            
            # Cargar datos de películas originales (para mostrar información)
            self.movies_info = pd.read_csv(movies_path)[['movieId', 'title', 'genres']]
            
            # Cargar datos procesados de películas (para características)
            self.movies_processed = pd.read_csv(processed_movies_path)[['movieId', 'year']]
            
            print("✅ Sistema de recomendación inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error al inicializar el recomendador: {str(e)}")
            raise

    def get_unseen_movies(self, user_id, ratings_path):
        """
        Identifica películas no vistas por el usuario.
        """
        try:
            ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId'])
            user_ratings = ratings[ratings['userId'] == user_id]
            all_movies = self.movies_processed['movieId'].unique()
            unseen_movies = list(set(all_movies) - set(user_ratings['movieId']))
            return unseen_movies
            
        except Exception as e:
            print(f"Error al obtener películas no vistas: {str(e)}")
            return []

    def prepare_features(self, user_id, movie_ids):
        """
        Prepara los datos para hacer predicciones, asegurando el orden correcto de características.
        """
        try:
            # Crear DataFrame base con movieIds
            features = pd.DataFrame({'movieId': movie_ids})
            
            # Añadir características del usuario
            features['user_encoded'] = self.user_encoder.transform([user_id])[0]
            
            # Añadir características temporales
            now = datetime.now()
            features['year_rated'] = now.year
            features['month_rated'] = now.month
            
            # Añadir características de las películas
            features = features.merge(self.movies_processed, on='movieId', how='left')
            
            # Asegurar el orden correcto de características
            # Eliminar 'movieId' ya que no debe usarse como característica
            final_features = features[self.expected_features]
            
            return final_features
            
        except Exception as e:
            print(f"Error preparando características: {str(e)}")
            raise

    def predict_ratings(self, user_id, movie_ids):
        """
        Predice ratings para un conjunto de películas.
        """
        try:
            # Preparar características (sin movieId como característica)
            features = self.prepare_features(user_id, movie_ids)
            
            # Convertir a formato DMatrix
            dmatrix = xgb.DMatrix(features)
            
            # Hacer predicciones
            predictions = pd.DataFrame({
                'movieId': movie_ids,
                'predicted_rating': self.model.predict(dmatrix)
            })
            
            return predictions.sort_values('predicted_rating', ascending=False)
            
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            raise

    def generate_recommendations(self, user_id, ratings_path, top_n=10):
        """
        Genera recomendaciones personalizadas para un usuario.
        """
        try:
            unseen_movies = self.get_unseen_movies(user_id, ratings_path)
            
            if not unseen_movies:
                print("⚠️ El usuario ya ha calificado todas las películas disponibles")
                return []
            
            predictions = self.predict_ratings(user_id, unseen_movies)
            top_movies = predictions.head(top_n)['movieId'].values
            
            recommendations = self.movies_info[
                self.movies_info['movieId'].isin(top_movies)
            ].to_dict('records')
            
            # Ordenar según el ranking de predicción
            movie_ranking = {movie_id: idx for idx, movie_id in enumerate(top_movies, 1)}
            for rec in recommendations:
                rec['ranking'] = movie_ranking[rec['movieId']]
            
            return sorted(recommendations, key=lambda x: x['ranking'])
            
        except Exception as e:
            print(f"Error generando recomendaciones: {str(e)}")
            return []

def main():
    # Configuración de rutas
    MODEL_PATH = './models/xgboost_recommender.model'
    ENCODER_PATH = './data/user_encoder.pkl'
    MOVIES_PATH = './data/movies.csv'
    PROCESSED_MOVIES_PATH = './data/movies_processed.csv'
    RATINGS_PATH = './data/ratings_processed.csv'
    
    # ID de usuario de ejemplo
    USER_ID = 3
    
    try:
        print("\n🚀 Iniciando sistema de recomendación XGBoost")
        
        # Inicializar recomendador
        recommender = XGBoostRecommender(
            model_path=MODEL_PATH,
            encoder_path=ENCODER_PATH,
            movies_path=MOVIES_PATH,
            processed_movies_path=PROCESSED_MOVIES_PATH
        )
        
        # Generar recomendaciones
        print(f"\n🔍 Generando recomendaciones para el usuario {USER_ID}...")
        recommendations = recommender.generate_recommendations(USER_ID, RATINGS_PATH)
        
        # Mostrar resultados
        if recommendations:
            print("\n🎬 Top 10 Recomendaciones:")
            for movie in recommendations:
                print(f"{movie['ranking']}. {movie['title']}")
                print(f"   Géneros: {movie['genres']}")
                print(f"   ID: {movie['movieId']}\n")
        else:
            print("No se encontraron recomendaciones.")
            
    except Exception as e:
        print(f"\n❌ Error en el sistema: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()