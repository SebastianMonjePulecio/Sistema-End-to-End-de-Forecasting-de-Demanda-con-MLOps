import sqlite3
import os

def init_database():
    # Crear carpeta data si no existe
    os.makedirs('data', exist_ok=True)
    
    # Conectar (crea el archivo si no existe)
    conn = sqlite3.connect('data/demand_forecast.db')
    cursor = conn.cursor()
    
    # Leer y ejecutar el esquema
    with open('sql/schema_sqlite.sql', 'r', encoding='utf-8') as f:
        schema = f.read()
    
    cursor.executescript(schema)
    conn.commit()
    conn.close()
    
    print("✅ Base de datos SQLite creada: data/demand_forecast.db")

if __name__ == "__main__":
    init_database()