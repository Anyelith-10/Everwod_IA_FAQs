import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import psycopg2

# Carpeta raiz del proyecto; se usa para construir rutas absolutas estables.
ROOT_DIR = Path(__file__).resolve().parent

# Carpeta donde se guardan archivos intermedios como conversaciones y FAQs.
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Archivo local de variables de entorno para credenciales de PostgreSQL.
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)


def get_db_config() -> Dict[str, str]:
    """Construye la configuracion de conexion a partir del archivo .env."""
    return {
        "dbname": os.getenv("DB_NAME", "everwod_db"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }


def get_db_connection() -> psycopg2.extensions.connection:
    """Abre una conexion nueva con PostgreSQL usando la configuracion centralizada."""
    return psycopg2.connect(**get_db_config())


def json_text(payload: Any) -> str:
    """Extrae texto limpio desde strings, listas o estructuras JSON anidadas."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        if "value" in payload:
            return json_text(payload["value"])
        if "text" in payload:
            return json_text(payload["text"])
        if "content" in payload:
            return json_text(payload["content"])
        parts = [json_text(value) for value in payload.values()]
        return " ".join(part for part in parts if part)
    if isinstance(payload, list):
        parts = [json_text(item) for item in payload]
        return " ".join(part for part in parts if part)
    return str(payload).strip()


def normalize_text(text: str) -> str:
    """Elimina espacios sobrantes para comparar y guardar textos de forma consistente."""
    return " ".join(text.strip().split()) if text else ""


def save_json(data: Any, path: Path) -> None:
    """Guarda datos en formato JSON legible."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_json_lines(records: List[Dict[str, Any]], path: Path) -> None:
    """Guarda una lista de registros como JSONL, un JSON por linea."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    """Lee un archivo JSON y devuelve su contenido."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_lines(path: Path) -> List[Dict[str, Any]]:
    """Lee un archivo JSONL y devuelve una lista de diccionarios."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                records.append(json.loads(payload))
    return records
