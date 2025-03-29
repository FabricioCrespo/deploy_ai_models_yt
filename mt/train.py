import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# Reading datasets
dataset_relative_path = "./datasets"

movies = pd.read_csv(os.path.join(dataset_relative_path, "movies_preprocessed.csv"))
ratings = pd.read_csv(os.path.join(dataset_relative_path,"rating.csv"))

# Ensable dataset with movies + ratings
movies_grouped = movies.groupby('movieId')['genres'].apply(list).reset_index()
ratings = pd.merge(ratings, movies_grouped, on='movieId')

# Using one hot encoding for movie genre
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(ratings['genres'])
genre_columns = mlb.classes_
genres_df = pd.DataFrame(genre_encoded, columns=genre_columns)

# Añadir columnas de géneros al DataFrame
ratings = pd.concat([ratings, genres_df], axis=1)

# Crear etiqueta binaria (si el usuario disfrutó la película)
ratings['liked'] = ratings['rating'].apply(lambda x: 1 if x >= 3 else 0)

# Características (X) y etiquetas (y)
X = ratings[genre_columns]  # Usamos los géneros como características
y = ratings['liked']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Naive Bayes

print("========================== ** TRAINING NAIVE BAYES ALGORITHM ** ========================== ")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
print("Reporte de clasificación para Naive Bayes:")
print(classification_report(y_test, nb_predictions))

# Entrenar modelo Árbol de Decisión

print("========================== ** TRAINING DECISION TREE ALGORITHM ** ========================== ")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
print("Reporte de clasificación para Árbol de Decisión:")
print(classification_report(y_test, dt_predictions))

# Guardar los modelos
import joblib

models_path = "./models"

joblib.dump(nb_model, os.path.join(models_path,"naive_bayes_recommender.pkl"))
joblib.dump(dt_model, os.path.join(models_path,"decision_tree_recommender.pkl"))

print("Modelos guardados como 'naive_bayes_recommender.pkl' y 'decision_tree_recommender.pkl'")