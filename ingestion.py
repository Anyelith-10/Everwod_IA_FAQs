import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException

from faq_core import (
    DATA_DIR,
    get_db_connection,
    json_text,
    normalize_text,
    save_json_lines,
)
from faq_model import IngestRequest, IngestResponse

app = FastAPI(title="Everwod FAQ Ingestion Service")

OUTPUT_PATH = DATA_DIR / "conversations.jsonl"

DEFAULT_LIMIT = 15000
MAX_LIMIT = 50000
DEFAULT_SINCE_DAYS = 365
MAX_SINCE_DAYS = 1500


def parse_message_payload(payload: Any) -> Any:
    """Convierte mensajes serializados en JSON a estructuras Python cuando aplica."""
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, str):
        value = payload.strip()
        if not value:
            return ""
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    return payload


def extract_role_from_message(payload: Any) -> Optional[str]:
    """Extrae el role del mensaje si esta disponible."""
    payload = parse_message_payload(payload)
    if isinstance(payload, dict):
        role = payload.get("role") or payload.get("sender") or payload.get("type")

        if isinstance(role, str):
            return role.strip().lower()
    return None


def extract_text_from_message(payload: Any) -> str:
    """Extrae el texto util desde el JSON de un mensaje de chat."""
    payload = parse_message_payload(payload)
    if not payload:
        return ""
    if isinstance(payload, dict):
        for key in ("content", "text", "message", "body", "value"):
            if key in payload:
                text = normalize_text(json_text(payload[key]))
                if text:
                    return text
        return normalize_text(json_text(payload))
    return normalize_text(json_text(payload))


def is_useful_assistant_text(text: str) -> bool:
    """Descarta respuestas tecnicas que no sirven como evidencia para FAQs."""
    lowered = normalize_text(text).lower()
    if not lowered:
        return False

    technical_fragments = (
        "function_call",
        "tool_call",
        "verifydatetime",
        "verifydate",
        "makeappointment",
        "reasoning\nfunction_call",
    )
    return not any(fragment in lowered for fragment in technical_fragments)


def sanitize_limit(limit: int) -> int:
    """Normaliza el limite de registros para evitar valores invalidos."""
    if limit <= 0:
        return DEFAULT_LIMIT
    return min(limit, MAX_LIMIT)


def sanitize_since_days(since_days: int) -> int:
    """Normaliza el rango de dias para evitar valores invalidos."""
    if since_days <= 0:
        return DEFAULT_SINCE_DAYS
    return min(since_days, MAX_SINCE_DAYS)


def fetch_conversation_records(
    limit: int = DEFAULT_LIMIT,
    since_days: int = DEFAULT_SINCE_DAYS,
) -> List[Dict[str, Any]]:
    """Consulta PostgreSQL y arma pares usuario/asistente listos para analizar."""
    limit = sanitize_limit(limit)
    since_days = sanitize_since_days(since_days)
    since = datetime.now(UTC) - timedelta(days=since_days)
    query = (
        "SELECT cm.agent_chat_id, cm.message, cm.created_at, ac.workspace_id, w.name "
        "FROM chat_messages cm "
        "JOIN agent_chats ac ON ac.id = cm.agent_chat_id "
        "LEFT JOIN workspaces w ON w.id = ac.workspace_id "
        "WHERE cm.created_at >= %s "
        "ORDER BY ac.workspace_id, cm.agent_chat_id, cm.created_at "
        "LIMIT %s"
    )
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (since, limit))
                rows = cursor.fetchall()
    except Exception as exc:
        raise RuntimeError(f"Error querying PostgreSQL during ingestion: {exc}") from exc

    conversations: List[Dict[str, Any]] = []
    messages_by_chat: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        agent_chat_id = row[0]
        message_payload = parse_message_payload(row[1])
        created_at = row[2]
        workspace_id = row[3]
        workspace_name = row[4]

        messages_by_chat[agent_chat_id].append(
            {
                "message": message_payload,
                "created_at": created_at,
                "company_id": str(workspace_id) if workspace_id is not None else "unknown",
                "company_name": workspace_name,
            }
        )

    for conversation_id, events in messages_by_chat.items():
        last_user_text = ""
        for event in sorted(events, key=lambda item: item["created_at"]):
            payload = event["message"]
            role = extract_role_from_message(payload)
            text = extract_text_from_message(payload)
            if not text:
                continue
            if role == "user":
                last_user_text = text
                continue
            if role == "assistant" and last_user_text:
                if not is_useful_assistant_text(text):
                    continue
                created_at = event["created_at"]
                created_at_value = (
                    created_at.isoformat()
                    if hasattr(created_at, "isoformat")
                    else str(created_at)
                )
                conversations.append(
                    {
                        "company_id": event["company_id"],
                        "company_name": event["company_name"],
                        "conversation_id": str(conversation_id),
                        "user_text": last_user_text,
                        "assistant_text": text,
                        "created_at": created_at_value,
                    }
                )
                last_user_text = ""
    return conversations


def ingest(
    limit: int = DEFAULT_LIMIT,
    since_days: int = DEFAULT_SINCE_DAYS,
) -> IngestResponse:
    """Ejecuta la ingesta y guarda el resultado en data/conversations.jsonl."""
    records = fetch_conversation_records(limit=limit, since_days=since_days)
    save_json_lines(records, OUTPUT_PATH)
    return IngestResponse(
        imported_records=len(records),
        output_file=str(OUTPUT_PATH),
    )


@app.get("/health")
def health() -> Dict[str, str]:
    """Endpoint simple para confirmar que el servicio esta activo."""
    return {
        "status": "ok",
        "service": "ingest",
    }


@app.post("/ingest", response_model=IngestResponse)
def run_ingest(request: IngestRequest) -> IngestResponse:
    """Endpoint principal: recibe parametros y dispara la ingesta."""
    try:
        return ingest(
            limit=request.limit,
            since_days=request.since_days,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ingestion:app", host="127.0.0.1", port=8001, log_level="info")