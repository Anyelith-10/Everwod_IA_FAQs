from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Parametros que recibe el servicio de ingesta para leer conversaciones historicas.
class IngestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "limit": 15000,
                "since_days": 365,
            }
        }
    )

    # Cantidad maxima de mensajes que se consultaran en PostgreSQL.
    limit: int = Field(default=15000, ge=100, le=50000)

    # Ventana de tiempo, en dias, para traer solo conversaciones recientes.
    since_days: int = Field(default=365, ge=1, le=1500)


# Respuesta del servicio de ingesta despues de crear el archivo JSONL.
class IngestResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "imported_records": 2258,
                "output_file": "data/conversations.jsonl",
            }
        }
    )

    # Numero de pares usuario/asistente importados.
    imported_records: int = Field(ge=0)

    # Ruta del archivo donde quedaron guardadas las conversaciones procesadas.
    output_file: str


# Cuerpo que recibe el servicio de embeddings.
class EncodeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Cuanto cuesta la mensualidad?",
                    "Donde estan ubicados?",
                ]
            }
        }
    )

    # Lista de textos que se convertiran en vectores numericos.
    texts: List[str] = Field(default_factory=list, max_length=1000)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: List[str]) -> List[str]:
        cleaned = [text.strip() for text in value if text and text.strip()]

        if not cleaned:
            raise ValueError("texts must contain at least one non-empty text.")

        return cleaned


# Respuesta con los embeddings generados.
class EncodeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embeddings": [[0.0123, -0.0456, 0.0789]],
                "count": 1,
            }
        }
    )

    # Un embedding por cada texto valido recibido.
    embeddings: List[List[float]]

    # Cantidad final de textos procesados.
    count: int = Field(ge=0)


# Representa una sugerencia individual de FAQ.
class SuggestionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a50c7574-bcbe-415b-803b-e494b96637b8",
                "company_id": "118",
                "company_name": "Mové",
                "question": "Cuándo sería la clase?",
                "answer": "La clase depende del horario disponible y debe confirmarse con el equipo.",
                "cluster_size": 3,
                "support_examples": [
                    "Cuándo sería la clase?",
                    "Me gustaría iniciar con las clases",
                ],
                "cluster_score": 75.0,
                "support_score": 0.72,
            }
        }
    )

    # Identificador unico para validar la sugerencia mas adelante.
    id: str

    # Empresa a la que pertenece la sugerencia.
    company_id: str

    # Nombre de empresa si pudo extraerse desde la base de datos.
    company_name: Optional[str] = None

    # Pregunta representativa del grupo de conversaciones similares.
    question: str = Field(min_length=3)

    # Respuesta sugerida segun las respuestas historicas del asistente.
    answer: str = Field(min_length=1)

    # Cantidad de ejemplos que cayeron en el mismo cluster.
    cluster_size: int = Field(ge=1)

    # Ejemplos reales que soportan la sugerencia.
    support_examples: List[str] = Field(default_factory=list)

    # Porcentaje aproximado de soporte del cluster sobre el total.
    cluster_score: float = Field(ge=0, le=100)

    # Similitud promedio interna del cluster. Opcional para compatibilidad.
    support_score: Optional[float] = Field(default=None, ge=0, le=1)


# Resumen completo de la ejecucion del servicio de sugerencias.
class SuggestionSummary(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "company_count": 18,
                "cluster_count": 4,
                "total_examples": 58,
                "average_cluster_size": 14.5,
                "silhouette_score": 0.2866,
                "suggestions": [],
            }
        }
    )

    # Numero de empresas analizadas en la ejecucion.
    company_count: int = Field(ge=0)

    # Numero total de grupos detectados.
    cluster_count: int = Field(ge=0)

    # Cantidad de textos usados para generar los clusters.
    total_examples: int = Field(ge=0)

    # Tamano promedio de cada cluster.
    average_cluster_size: float = Field(ge=0)

    # Metrica opcional de separacion entre clusters.
    silhouette_score: Optional[float] = None

    # Lista de sugerencias generadas.
    suggestions: List[SuggestionResponse] = Field(default_factory=list)


# Datos que envia una persona para aprobar, rechazar o pedir cambios.
class ValidationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggestion_id": "a50c7574-bcbe-415b-803b-e494b96637b8",
                "reviewer": "Anyelith",
                "status": "approved",
                "notes": "Aprobada para revision posterior",
            }
        }
    )

    # ID de la sugerencia que se va a revisar.
    suggestion_id: str = Field(min_length=1)

    # Nombre o identificador de quien revisa.
    reviewer: str = Field(min_length=1)

    # Estado de la revision.
    status: Literal["approved", "rejected", "needs_changes"]

    # Comentario opcional de la revision.
    notes: Optional[str] = None

    # Fecha opcional; si no se envia, el servicio usa la fecha actual.
    reviewed_at: Optional[datetime] = None


# Respuesta guardada despues de validar una sugerencia.
class ValidationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggestion_id": "a50c7574-bcbe-415b-803b-e494b96637b8",
                "reviewer": "Anyelith",
                "status": "approved",
                "notes": "Aprobada para revision posterior",
                "reviewed_at": "2026-05-10T04:58:42.461675",
            }
        }
    )

    # ID de la sugerencia revisada.
    suggestion_id: str

    # Revisor que hizo la validacion.
    reviewer: str

    # Estado final registrado.
    status: Literal["approved", "rejected", "needs_changes"]

    # Comentarios asociados a la revision.
    notes: Optional[str] = None

    # Fecha exacta en la que quedo registrada la revision.
    reviewed_at: datetime