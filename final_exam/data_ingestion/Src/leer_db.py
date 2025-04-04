import sqlite3

DB_FILE = '../db/ratings.db'  # Archivo de la base de datos SQLite

def fetch_data():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    print("\nDatos de la tabla 'rating':")
    cursor.execute("SELECT * FROM rating")
    ratings = cursor.fetchall()
    for row in ratings:
        print(row)
    
    print("\nDatos de la tabla 'log':")
    cursor.execute("SELECT * FROM log")
    logs = cursor.fetchall()
    for row in logs:
        print(row)
    
    conn.close()

if __name__ == "__main__":
    fetch_data()
