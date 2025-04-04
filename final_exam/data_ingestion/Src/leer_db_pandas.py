import sqlite3
import pandas as pd

DB_FILE = '../db/movies.db'  # Archivo de la base de datos SQLite

def fetch_data():
    conn = sqlite3.connect(DB_FILE)
    
    print("\nDatos de la tabla 'movies':")
    df_rating = pd.read_sql_query("SELECT * FROM movies", conn)
    print(df_rating)
    
    print("\nDatos de la tabla 'logs':")
    df_log = pd.read_sql_query("SELECT * FROM logs", conn)
    print(df_log)
    
    conn.close()
    return df_rating, df_log

if __name__ == "__main__":
    df_rating, df_log = fetch_data()
