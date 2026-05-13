import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "everwod"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("Conexión exitosa a PostgreSQL.")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chat_messages;")
    result = cur.fetchone()
    print(f"Número de filas en 'chat_messages': {result[0]}")

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cur.fetchall()
    print("Tablas en la base de datos:")
    for table in tables:
        print(f"- {table[0]}")

    cur.close()
    conn.close()
    print("Conexión cerrada.")

except psycopg2.Error as e:
    print(f"Error de PostgreSQL: {e}")
except Exception as e:
    print(f"Error general: {e}")
