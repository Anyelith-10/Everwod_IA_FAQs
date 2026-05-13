import json
import os
import psycopg2

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

JsonDict = Dict[str, Any]


def get_env(name: str, default: str = "") -> str:
    """Lee una variable de entorno y elimina espacios accidentales."""
    return os.getenv(name, default).strip()

def get_db_config() -> Dict[str, str]:
    """Construye la configuracion de conexion a partir del archivo .env."""
    return {
        "dbname": get_env("DB_NAME", "everwod_db"),
        "user": get_env("DB_USER", "postgres"),
        "password": get_env("DB_PASSWORD", ""),
        "host": get_env("DB_HOST", "localhost"),
        "port": get_env("DB_PORT", "5432"),
        "connect_timeout": get_env("DB_CONNECT_TIMEOUT", "10"),
    }

def get_db_connection() -> psycopg2.extensions.connection:
    """Abre una conexion nueva con PostgreSQL usando la configuracion centralizada."""
    return psycopg2.connect(**get_db_config())

def normalize_text(text: Optional[str]) -> str:
    """Elimina espacios sobrantes para comparar y guardar textos de forma consistente."""
    if not text:
        return ""
    return " ".join(str(text).strip().split())


def json_text(payload: Any) -> str:
    """Extrae texto limpio desde strings, listas o estructuras JSON anidadas."""
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        try:
            return json_text(payload.decode("utf-8"))
        except UnicodeDecodeError:
            return ""
    if isinstance(payload, str):
        value = payload.strip()
        if not value:
            return ""
        # Algunos campos pueden venir como JSON serializado en texto.
        if value.startswith("{") or value.startswith("["):
            try:
                parsed = json.loads(value)
                return json_text(parsed)
            except json.JSONDecodeError:
                return normalize_text(value)
        return normalize_text(value)
    if isinstance(payload, dict):
        # Prioriza campos frecuentes de contenido conversacional.
        for key in ("value", "text", "content", "message", "body"):
            if key in payload:
                extracted = json_text(payload[key])
                if extracted:
                    return extracted
        parts = [json_text(value) for value in payload.values()]
        return normalize_text(" ".join(part for part in parts if part))
    if isinstance(payload, list):
        parts = [json_text(item) for item in payload]
        return normalize_text(" ".join(part for part in parts if part))
    return normalize_text(str(payload))


def save_json(data: Any, path: Path) -> None:
    """Guarda datos en formato JSON legible usando escritura segura."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def save_json_lines(records: List[JsonDict], path: Path) -> None:
    """Guarda una lista de registros como JSONL, un JSON por linea."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    """Lee un archivo JSON y devuelve su contenido."""
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_lines(path: Path) -> List[JsonDict]:
    """Lee un archivo JSONL y devuelve una lista de diccionarios."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}, line {line_number}: {exc}"
                ) from exc
            if isinstance(item, dict):
                records.append(item)
    return records