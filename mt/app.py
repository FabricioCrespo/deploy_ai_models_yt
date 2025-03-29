import pandas as pd
from flask import Flask, request, render_template
import joblib
import numpy as np

# Flask App
app = Flask(__name__)

# Cargar datos de películas y calificaciones
movies = pd.read_csv("movies2.csv")  # Asegúrate de que el archivo esté correctamente generado
ratings = pd.read_csv("ratings.csv")  # Archivo de calificaciones

# Cargar modelos entrenados
naive_bayes_model = joblib.load("models/naive_bayes_recommender.pkl")
decision_tree_model = joblib.load("models/decision_tree_recommender.pkl")

# Función para generar recomendaciones personalizadas basadas en userId
def recommend_movies_for_user(user_id, model, top_n=5):
    # Obtener las películas que el usuario ya calificó
    user_ratings = ratings[ratings['userId'] == user_id]
    watched_movie_ids = user_ratings['movieId'].tolist()

    # Filtrar películas que el usuario no ha visto
    unwatched_movies = movies[~movies['movieId'].isin(watched_movie_ids)]

    # Extraer géneros en formato one-hot para las películas no vistas
    available_genres = [col for col in movies.columns if col not in ['movieId', 'title', 'genres']]
    genre_features = unwatched_movies[available_genres]

    # Predicción para las películas no vistas
    predictions = model.predict(genre_features)

    # Filtrar películas con predicción positiva
    unwatched_movies['prediction'] = predictions
    recommended_movies = unwatched_movies[unwatched_movies['prediction'] == 1]

    # Ordenar por predicción y devolver las primeras N recomendaciones
    recommended_titles = recommended_movies['title'].head(top_n).tolist()
    return recommended_titles

@app.route('/')
def home():
    # Página principal para introducir el userId
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Obtener datos ingresados por el usuario
    user_id = int(request.form['userId'])  # UserId proporcionado
    model_type = request.form['model']  # Modelo seleccionado por el usuario

    # Selección del modelo basado en el tipo
    if model_type == 'naive_bayes':
        recommendations = recommend_movies_for_user(user_id, naive_bayes_model)
    elif model_type == 'decision_tree':
        recommendations = recommend_movies_for_user(user_id, decision_tree_model)
    else:
        recommendations = ["Modelo no válido"]

    # Renderizar la página de resultados con las recomendaciones
    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)