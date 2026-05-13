from datetime import UTC, datetime
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from faq_core import DATA_DIR, load_json, save_json
from faq_model import ValidationRequest, ValidationResponse

app = FastAPI(title="Everwod FAQ Validation Service")

SUGGESTIONS_PATH = DATA_DIR / "faq_suggestions.json"
VALIDATIONS_PATH = DATA_DIR / "faq_validations.json"

ALLOWED_STATUSES = {"approved", "rejected", "needs_changes"}


def load_suggestions() -> List[Dict[str, Any]]:
    """Carga las sugerencias disponibles para revision humana."""
    if not SUGGESTIONS_PATH.exists():
        raise FileNotFoundError(
            "No FAQ suggestions available. Ejecute el servicio de sugerencias primero."
        )

    payload = load_json(SUGGESTIONS_PATH, default={})
    suggestions = payload.get("suggestions", [])
    if not isinstance(suggestions, list):
        raise ValueError("Invalid suggestions file format: suggestions must be a list.")
    return suggestions


def load_validations() -> List[Dict[str, Any]]:
    """Carga validaciones guardadas; devuelve lista vacia si aun no hay archivo."""
    if not VALIDATIONS_PATH.exists():
        return []
    payload = load_json(VALIDATIONS_PATH, default=[])
    if not isinstance(payload, list):
        raise ValueError("Invalid validations file format: expected a list.")
    return payload


def save_validations(validations: List[Dict[str, Any]]) -> None:
    """Persiste todas las validaciones en un archivo JSON."""
    save_json(validations, VALIDATIONS_PATH)


def find_suggestion(suggestion_id: str, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Busca una sugerencia por ID."""
    for suggestion in suggestions:
        if suggestion.get("id") == suggestion_id:
            return suggestion
    raise HTTPException(status_code=404, detail="Suggestion ID not found.")


def normalize_status(status: str) -> str:
    """Normaliza y valida el estado de revision."""
    clean_status = status.strip().lower()
    if clean_status not in ALLOWED_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid status. Use one of: "
                "approved, rejected, needs_changes."
            ),
        )
    return clean_status


@app.get("/health")
def health() -> Dict[str, str]:
    """Endpoint simple para confirmar que el servicio esta activo."""
    return {
        "status": "ok",
        "service": "validation",
    }


@app.get("/suggestions")
def get_suggestions() -> List[Dict[str, Any]]:
    """Devuelve las sugerencias pendientes o disponibles para validar."""
    try:
        return load_suggestions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading suggestions: {exc}")


@app.get("/validations")
def get_validations() -> List[Dict[str, Any]]:
    """Devuelve el historial de validaciones realizadas."""
    try:
        return load_validations()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading validations: {exc}")


@app.post("/validate", response_model=ValidationResponse)
def validate(request: ValidationRequest) -> ValidationResponse:
    """Registra o actualiza la revision de una sugerencia especifica."""
    try:
        suggestions = load_suggestions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading suggestions: {exc}")
    find_suggestion(request.suggestion_id, suggestions)
    status = normalize_status(request.status)
    try:
        validations = load_validations()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading validations: {exc}")
    reviewed_at = request.reviewed_at or datetime.now(UTC)
    entry = {
        "suggestion_id": request.suggestion_id,
        "reviewer": request.reviewer,
        "status": status,
        "notes": request.notes,
        "reviewed_at": reviewed_at.isoformat(),
    }

    updated = False
    for index, validation in enumerate(validations):
        if validation.get("suggestion_id") == request.suggestion_id:
            validations[index] = entry
            updated = True
            break
    if not updated:
        validations.append(entry)
    try:
        save_validations(validations)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error saving validation: {exc}")

    return ValidationResponse(**entry)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("validation:app", host="127.0.0.1", port=8004, log_level="info")