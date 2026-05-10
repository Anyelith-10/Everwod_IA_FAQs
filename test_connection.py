import os
from dotenv import load_dotenv
import psycopg2

# Carga las credenciales desde .env para no escribir passwords en el codigo.
load_dotenv()

# Configuracion de conexion con valores por defecto para desarrollo local.
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "everwod_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

try:
    # Abre la conexion con PostgreSQL usando psycopg2.
    conn = psycopg2.connect(**DB_CONFIG)
    print("Conexión exitosa a PostgreSQL.")

    # Crea un cursor para ejecutar consultas SQL.
    cur = conn.cursor()

    # Verifica que la tabla principal de conversaciones existe y tiene datos.
    cur.execute("SELECT COUNT(*) FROM chat_messages;")
    result = cur.fetchone()
    print(f"Número de filas en 'chat_messages': {result[0]}")

    # Lista las tablas publicas para confirmar que el dump se restauro correctamente.
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cur.fetchall()
    print("Tablas en la base de datos:")
    for table in tables:
        print(f"- {table[0]}")

    # Cierra recursos de base de datos al finalizar la prueba.
    cur.close()
    conn.close()
    print("Conexión cerrada.")

except psycopg2.Error as e:
    # Captura errores especificos de PostgreSQL, como credenciales o tablas faltantes.
    print(f"Error de PostgreSQL: {e}")
except Exception as e:
    # Captura cualquier otro error inesperado de Python.
    print(f"Error general: {e}")
