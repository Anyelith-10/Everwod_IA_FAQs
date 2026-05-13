from typing import List, Optional
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from faq_core import normalize_text
from faq_model import EncodeRequest, EncodeResponse

app = FastAPI(title="Everwod FAQ")
MODEL_NAME = "all-MiniLM-L6-v2"
model: Optional[SentenceTransformer] = None


@app.on_event("startup")
def startup_event() -> None:
    """Carga el modelo cuando arranca el servicio."""
    load_model()

def load_model() -> None:
    """Carga el modelo de embeddings si aun no esta cargado."""
    global model

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

@app.get("/health")
def health() -> dict:
    """Endpoint simple para confirmar que el servicio esta activo."""
    return {
        "status": "ok",
        "service": "embed",
        "model": MODEL_NAME,
        "model_loaded": model is not None,
    }

@app.post("/encode", response_model=EncodeResponse)
def encode(request: EncodeRequest) -> EncodeResponse:
    """Convierte textos limpios en embeddings numericos."""
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded")

    normalized_texts = [normalize_text(text) for text in request.texts]
    texts: List[str] = [text for text in normalized_texts if text]

    if not texts:
        return EncodeResponse(embeddings=[], count=0)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        )
    return EncodeResponse(
        embeddings=embeddings.tolist(),
        count=len(texts),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("embeddings:app", host="127.0.0.1", port=8002, log_level="info")