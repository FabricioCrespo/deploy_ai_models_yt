import pandas as pd
import os 


# Cargar el dataset desde el archivo movies.csv
dataset_relative_path = "./datasets"
movies = pd.read_csv(os.path.join(dataset_relative_path,"movie.csv"))

# Dividir los géneros en una lista
movies['genres'] = movies['genres'].str.split('|')

# Crear un nuevo DataFrame donde cada género tenga su propia fila
movies_exploded = movies.explode('genres')

# Guardar el DataFrame preprocesado en un nuevo archivo, si lo deseas
movies_exploded.to_csv(os.path.join(dataset_relative_path,"movies_preprocessed.csv"), index=False)

# Mostrar el nuevo DataFrame
#print(movies_exploded)