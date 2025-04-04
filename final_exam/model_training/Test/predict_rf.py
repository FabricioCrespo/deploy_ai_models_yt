# predict_optimized.py
import pandas as pd
import joblib
import numpy as np

def load_lightweight_model():
    """Carga el modelo optimizado"""
    try:
        model = joblib.load('./models/optimized_random_forest_recommender.pkl')
        user_encoder = joblib.load('./data/user_encoder.pkl')
        return model, user_encoder
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

def recommend_for_user(user_id, model, user_encoder, top_n=10):
    """Genera recomendaciones optimizadas"""
    try:
        user_encoded = user_encoder.transform([user_id])[0]
    except ValueError:
        return []
    
    # Cargar películas (solo las necesarias)
    movies = pd.read_csv('./data/movies_processed.csv', 
                        usecols=['movieId', 'year'])
    
    # Cargar ratings del usuario (eficiente)
    user_ratings = pd.read_csv('./data/ratings_processed.csv',
                             usecols=['userId', 'movieId'])
    user_ratings = user_ratings[user_ratings['userId'] == user_id]
    
    # Obtener películas no vistas
    unseen_movies = movies[~movies['movieId'].isin(user_ratings['movieId'])]
    
    if unseen_movies.empty:
        return []
    
    # Preparar datos para predicción
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().month
    
    X_pred = unseen_movies.copy()
    X_pred['user_encoded'] = user_encoded
    X_pred['year_rated'] = current_year
    X_pred['month_rated'] = current_month
    
    # Columnas en el orden correcto
    X_pred = X_pred[['user_encoded', 'movieId', 'year_rated', 'month_rated', 'year']]
    
    # Predecir
    X_pred['predicted_rating'] = model.predict(X_pred)
    
    # Obtener mejores películas
    top_movies = X_pred.nlargest(top_n, 'predicted_rating')['movieId'].values
    
    # Obtener títulos
    movie_titles = pd.read_csv('./data/movies.csv', usecols=['movieId', 'title'])
    recommendations = movie_titles[movie_titles['movieId'].isin(top_movies)]
    
    return recommendations.to_dict('records')

if __name__ == '__main__':
    model, user_encoder = load_lightweight_model()
    if model:
        user_id = 3  # Ejemplo
        recs = recommend_for_user(user_id, model, user_encoder)
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['title']} (ID: {rec['movieId']})")