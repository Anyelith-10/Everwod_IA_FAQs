import json
import os
import re
import uuid
import numpy as np

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from faq_core import (
    DATA_DIR,
    get_db_connection,
    load_json,
    load_json_lines,
    normalize_text,
    save_json,
)

from faq_model import (
    SuggestionResponse,
    SuggestionSummary,
)

app = FastAPI(title="Everwod FAQ Sugerencias")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "all-MiniLM-L6-v2"

FAQ_LLM_MODEL = os.getenv(
    "FAQ_LLM_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

FAQ_LLM_ENABLED = (
    os.getenv(
        "FAQ_LLM_ENABLED",
        "true",
    ).lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

FAQ_CLUSTER_EPS = float(
    os.getenv(
        "FAQ_CLUSTER_EPS",
        "0.34",
    )
)

FAQ_MIN_CLUSTER_SIZE = int(
    os.getenv(
        "FAQ_MIN_CLUSTER_SIZE",
        "3",
    )
)

FAQ_SKIP_EXISTING = (
    os.getenv(
        "FAQ_SKIP_EXISTING",
        "true",
    ).lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

FAQ_DUPLICATE_THRESHOLD = float(
    os.getenv(
        "FAQ_DUPLICATE_THRESHOLD",
        "0.78",
    )
)

EMBEDDING_MODEL: Optional[
    SentenceTransformer
] = None

ANSWER_GENERATOR: Optional[Any] = None

MODELS_READY = False

SUGGESTIONS_PATH = (
    DATA_DIR / "faq_suggestions.json"
)

CONVERSATIONS_PATH = (
    DATA_DIR / "conversations.jsonl"
)


@app.on_event("startup")
def startup_event() -> None:
    """Carga modelos."""

    load_models()


def load_models() -> None:
    """Carga modelos IA."""

    global EMBEDDING_MODEL
    global ANSWER_GENERATOR
    global MODELS_READY

    if MODELS_READY:
        return

    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = SentenceTransformer(
            MODEL_NAME
        )

    if FAQ_LLM_ENABLED:

        try:

            tokenizer = (
                AutoTokenizer.from_pretrained(
                    FAQ_LLM_MODEL
                )
            )

            model = (
                AutoModelForCausalLM.from_pretrained(
                    FAQ_LLM_MODEL
                )
            )

            ANSWER_GENERATOR = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

        except Exception as exc:

            print(
                f"No se pudo cargar "
                f"{FAQ_LLM_MODEL}. "
                f"Error: {exc}"
            )

            ANSWER_GENERATOR = None

    MODELS_READY = True


def load_conversation_pairs() -> List[
    Dict[str, str]
]:
    """Carga conversaciones."""

    if not CONVERSATIONS_PATH.exists():

        raise FileNotFoundError(
            f"Conversation file not found: "
            f"{CONVERSATIONS_PATH}"
        )

    return load_json_lines(
        CONVERSATIONS_PATH
    )


def redact_personal_data(
    text: str,
    protected_terms: Optional[
        List[str]
    ] = None,
) -> str:
    """Elimina datos sensibles."""

    text = normalize_text(text)

    if not text:
        return ""

    protected_values: Dict[
        str,
        str,
    ] = {}

    for index, term in enumerate(
        protected_terms or []
    ):

        clean_term = normalize_text(term)

        if clean_term:

            token = (
                f"__PROTECTED_{index}__"
            )

            protected_values[token] = (
                clean_term
            )

            text = text.replace(
                clean_term,
                token,
            )

    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
        "[correo]",
        text,
    )

    text = re.sub(
        r"\b(?:\+?\d[\s-]?){7,}\b",
        "[telefono]",
        text,
    )

    for token, value in (
        protected_values.items()
    ):
        text = text.replace(
            token,
            value,
        )

    return normalize_text(text)


def clean_generated_answer(
    answer: str,
) -> str:
    """Limpia respuestas."""

    answer = normalize_text(answer)

    if not answer:
        return ""

    cleanup_patterns = [
        r"(?i)^¡?claro!?\s*",
        r"(?i)^aqui tienes una respuesta final:?\s*",
        r"(?i)^respuesta final:?\s*",
        r"(?i)^faq:?\s*",
    ]

    for pattern in cleanup_patterns:

        answer = re.sub(
            pattern,
            "",
            answer,
        ).strip()

    answer = answer.replace(
        "---",
        "",
    ).strip()

    return normalize_text(answer)


def is_good_faq_candidate(
    text: str,
) -> bool:
    """Filtra preguntas válidas."""

    text = normalize_text(text)

    if not text:
        return False

    lowered = (
        text.lower()
        .strip(" ¿?¡!.,;:")
    )

    word_count = len(text.split())

    if len(text) < 8:
        return False

    if word_count < 3:
        return False

    if len(text) > 280:
        return False

    trivial_messages = {
        "hola",
        "buenas",
        "ok",
        "si",
        "sí",
        "no",
        "gracias",
        "dale",
        "listo",
        "quien soy",
        "hola quien soy",
    }

    if lowered in trivial_messages:
        return False

    has_question_signal = (
        "?" in text
        or lowered.startswith("como")
        or lowered.startswith("cómo")
        or lowered.startswith("donde")
        or lowered.startswith("dónde")
        or lowered.startswith("cuando")
        or lowered.startswith("cuándo")
        or lowered.startswith("quiero")
        or lowered.startswith("puedo")
        or lowered.startswith("necesito")
    )

    return has_question_signal


def company_key(
    item: Dict[str, str],
) -> str:
    """Obtiene company_id."""

    return normalize_text(
        str(
            item.get("company_id")
            or item.get("workspace_id")
            or "unknown"
        )
    )


def most_common_answer(
    answers: List[str],
    protected_terms: Optional[
        List[str]
    ] = None,
) -> str:
    """Respuesta fallback."""

    clean_answers: List[str] = []

    for answer in answers:

        clean_answer = (
            redact_personal_data(
                answer,
                protected_terms=
                protected_terms,
            )
        )

        if clean_answer:
            clean_answers.append(
                clean_answer
            )

    if not clean_answers:

        return (
            "No tengo información "
            "suficiente para responder "
            "esta FAQ."
        )

    return max(
        set(clean_answers),
        key=clean_answers.count,
    )


def load_existing_faqs_by_company(
) -> Dict[str, List[str]]:
    """Carga FAQs existentes."""

    if not FAQ_SKIP_EXISTING:
        return {}

    query = (
        "SELECT a.workspace_id, af.question "
        "FROM agent_faqs af "
        "JOIN agents a ON a.id = af.agent_id "
        "WHERE af.deleted_at IS NULL "
        "AND af.question IS NOT NULL"
    )

    faqs_by_company: Dict[
        str,
        List[str],
    ] = defaultdict(list)

    try:

        with get_db_connection() as conn:

            with conn.cursor() as cursor:

                cursor.execute(query)

                for (
                    workspace_id,
                    question,
                ) in cursor.fetchall():

                    clean_question = (
                        normalize_text(
                            question
                        )
                    )

                    if clean_question:

                        faqs_by_company[
                            str(
                                workspace_id
                            )
                        ].append(
                            clean_question
                        )

    except Exception as exc:

        print(
            "No se pudieron cargar FAQs. "
            f"Error: {exc}"
        )

    return faqs_by_company


def save_suggestions_to_db(
    suggestions: List[
        SuggestionResponse
    ],
) -> None:
    """Guarda FAQs."""

    query = """
    INSERT INTO faq_suggestions (
        id,
        company_id,
        company_name,
        question,
        answer,
        cluster_size,
        cluster_score,
        support_examples,
        status
    )
    VALUES (
        %s,%s,%s,%s,%s,%s,%s,%s,%s
    )
    ON CONFLICT (id) DO NOTHING
    """

    try:

        with get_db_connection() as conn:

            with conn.cursor() as cursor:

                for item in suggestions:

                    cursor.execute(
                        query,
                        (
                            item.id,
                            item.company_id,
                            item.company_name,
                            item.question,
                            item.answer,
                            item.cluster_size,
                            item.cluster_score,
                            json.dumps(
                                item.support_examples
                            ),
                            "pending",
                        ),
                    )

            conn.commit()

    except Exception as exc:

        print(
            f"Error guardando FAQs: {exc}"
        )


def is_existing_faq(
    question: str,
    existing_questions: List[str],
) -> bool:
    """Valida duplicados."""

    if not existing_questions:
        return False

    load_models()

    if EMBEDDING_MODEL is None:
        return False

    texts = [
        question,
        *existing_questions,
    ]

    embeddings = EMBEDDING_MODEL.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarities = (
        embeddings[1:]
        @ embeddings[0]
    )

    return bool(
        np.max(similarities)
        >= FAQ_DUPLICATE_THRESHOLD
    )


def generate_answer(
    question: str,
    examples: List[str],
    historical_answers: List[str],
    company_name: Optional[str],
) -> str:
    """Genera respuestas IA."""

    fallback = most_common_answer(
        historical_answers
    )

    if not ANSWER_GENERATOR:
        return fallback

    answer_context = "\n".join(
        f"- {answer}"
        for answer in historical_answers[:4]
    )

    example_context = "\n".join(
        f"- {example}"
        for example in examples[:5]
    )

    company_context = (
        company_name
        or "esta empresa"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un especialista en "
                "experiencia al cliente y "
                "documentación corporativa. "
                "Redacta respuestas FAQ "
                "profesionales y útiles."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Empresa: {company_context}\n"
                f"Pregunta FAQ: {question}\n\n"
                f"Ejemplos:\n"
                f"{example_context}\n\n"
                f"Respuestas:\n"
                f"{answer_context}"
            ),
        },
    ]

    try:

        tokenizer = (
            ANSWER_GENERATOR.tokenizer
        )

        prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        generated = ANSWER_GENERATOR(
            prompt,
            max_new_tokens=80,
            do_sample=False,
            return_full_text=False,
            pad_token_id=
            tokenizer.eos_token_id,
        )

        answer = generated[0].get(
            "generated_text",
            "",
        )

        answer = clean_generated_answer(
            answer
        )

        return answer or fallback

    except Exception as exc:

        print(
            f"No se pudo generar respuesta: "
            f"{exc}"
        )

        return fallback

def generate_faq_question(
    examples: List[str],
    company_name: Optional[str],
) -> str:
    """Genera preguntas FAQ profesionales."""

    if not examples:
        return "¿Cómo puedo obtener ayuda?"

    fallback = normalize_text(examples[0])

    if len(fallback) < 5:
        fallback = "¿Cómo puedo obtener ayuda?"

    if not ANSWER_GENERATOR:
        return fallback

    company_context = company_name or "la empresa"

    example_context = "\n".join(
        f"- {example}"
        for example in examples[:5]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un especialista en "
                "documentación corporativa y FAQs. "
                "Transforma conversaciones reales "
                "en preguntas FAQ profesionales, "
                "claras y útiles."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Empresa: {company_context}\n\n"
                f"Mensajes de usuarios:\n"
                f"{example_context}\n\n"
                "Genera UNA pregunta FAQ profesional."
            ),
        },
    ]

    try:

        tokenizer = ANSWER_GENERATOR.tokenizer

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        generated = ANSWER_GENERATOR(
            prompt,
            max_new_tokens=30,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        question = generated[0].get(
            "generated_text",
            "",
        )

        question = clean_generated_answer(
            question
        )

        if (
            not question
            or len(question) < 5
        ):
            return fallback

        return normalize_text(question)

    except Exception:
        return fallback


def build_company_suggestions(
    conversations: List[Dict[str, str]],
    existing_questions: Optional[
        List[str]
    ] = None,
) -> Tuple[
    List[SuggestionResponse],
    Optional[float],
]:
    """Genera FAQs por empresa."""

    load_models()

    if EMBEDDING_MODEL is None:

        raise RuntimeError(
            "Embedding model is not loaded."
        )

    existing_questions = (
        existing_questions or []
    )

    valid_items: List[
        Dict[str, str]
    ] = []

    for item in conversations:

        user_text = normalize_text(
            item.get("user_text", "")
        )

        if (
            user_text
            and is_good_faq_candidate(
                user_text
            )
        ):

            valid_items.append(
                {
                    **item,
                    "user_text": user_text,
                }
            )

    if (
        len(valid_items)
        < FAQ_MIN_CLUSTER_SIZE
    ):
        return [], None

    user_texts = [
        item["user_text"]
        for item in valid_items
    ]

    embeddings = EMBEDDING_MODEL.encode(
        user_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    clustering_model = DBSCAN(
        eps=FAQ_CLUSTER_EPS,
        min_samples=
        FAQ_MIN_CLUSTER_SIZE,
        metric="cosine",
    )

    labels = (
        clustering_model.fit_predict(
            embeddings
        )
    )

    suggestions: List[
        SuggestionResponse
    ] = []

    cluster_groups: Dict[
        int,
        List[int],
    ] = {}

    for index, label in enumerate(
        labels
    ):

        if label != -1:

            cluster_groups.setdefault(
                int(label),
                [],
            ).append(index)

    for label, indices in (
        cluster_groups.items()
    ):

        center = np.mean(
            embeddings[indices],
            axis=0,
        )

        norm = np.linalg.norm(center)

        if norm == 0:
            continue

        center = center / norm

        best_index = max(
            indices,
            key=lambda idx: float(
                np.dot(
                    embeddings[idx],
                    center,
                )
            ),
        )

        representative = valid_items[
            best_index
        ]

        examples: List[str] = []

        for idx in indices[:5]:

            candidate = (
                redact_personal_data(
                    user_texts[idx]
                )
            )

            if (
                candidate
                and candidate
                not in examples
            ):
                examples.append(
                    candidate
                )

        if not examples:
            continue

        question_text = (
            generate_faq_question(
                examples=examples,
                company_name=
                representative.get(
                    "company_name"
                ),
            )
        )

        question_text = normalize_text(
            question_text
        )

        if (
            not question_text
            or len(
                question_text.strip()
            )
            < 5
        ):
            continue

        if is_existing_faq(
            question_text,
            existing_questions,
        ):
            continue

        answers: List[str] = []

        for i in indices:

            assistant_text = (
                normalize_text(
                    valid_items[i].get(
                        "assistant_text",
                        "",
                    )
                )
            )

            if assistant_text:
                answers.append(
                    assistant_text
                )

        answer_text = generate_answer(
            question=question_text,
            examples=examples,
            historical_answers=answers,
            company_name=
            representative.get(
                "company_name"
            ),
        )

        answer_text = normalize_text(
            answer_text
        )

        if not answer_text:

            answer_text = (
                "No tengo información "
                "suficiente para responder "
                "esta FAQ."
            )

        cluster_score = round(
            min(
                100.0,
                100.0
                * len(indices)
                / len(valid_items),
            ),
            2,
        )

        suggestions.append(
            SuggestionResponse(
                id=str(uuid.uuid4()),
                company_id=company_key(
                    representative
                ),
                company_name=
                representative.get(
                    "company_name"
                ),
                question=question_text,
                answer=answer_text,
                cluster_size=len(
                    indices
                ),
                support_examples=
                examples[:3],
                cluster_score=
                cluster_score,
            )
        )

    silhouette = None

    clean_labels = [
        label
        for label in labels
        if label != -1
    ]

    clean_embeddings = embeddings[
        labels != -1
    ]

    if (
        len(set(clean_labels)) > 1
        and len(clean_embeddings)
        > len(
            set(clean_labels)
        )
    ):

        silhouette = round(
            silhouette_score(
                clean_embeddings,
                clean_labels,
                metric="cosine",
            ),
            4,
        )

    return suggestions, silhouette


def build_suggestions(
    conversations: List[
        Dict[str, str]
    ],
) -> SuggestionSummary:
    """Genera FAQs."""

    if not conversations:

        summary = (
            SuggestionSummary(
                company_count=0,
                cluster_count=0,
                total_examples=0,
                average_cluster_size=0,
                silhouette_score=None,
                suggestions=[],
            )
        )

        save_json(
            summary.dict(),
            SUGGESTIONS_PATH,
        )

        return summary

    conversations_by_company: Dict[
        str,
        List[Dict[str, str]],
    ] = defaultdict(list)

    for item in conversations:

        conversations_by_company[
            company_key(item)
        ].append(item)

    suggestions: List[
        SuggestionResponse
    ] = []

    silhouettes: List[float] = []

    total_examples = 0

    existing_faqs_by_company = (
        load_existing_faqs_by_company()
    )

    for items in (
        conversations_by_company.values()
    ):

        current_company = (
            company_key(items[0])
            if items
            else "unknown"
        )

        total_examples += sum(
            1
            for item in items
            if is_good_faq_candidate(
                item.get(
                    "user_text",
                    "",
                )
            )
        )

        (
            company_suggestions,
            company_silhouette,
        ) = build_company_suggestions(
            items,
            existing_questions=
            existing_faqs_by_company.get(
                current_company,
                [],
            ),
        )

        suggestions.extend(
            company_suggestions
        )

        if (
            company_silhouette
            is not None
        ):
            silhouettes.append(
                company_silhouette
            )

    silhouette = (
        round(
            sum(silhouettes)
            / len(silhouettes),
            4,
        )
        if silhouettes
        else None
    )

    summary = SuggestionSummary(
        company_count=len(
            conversations_by_company
        ),
        cluster_count=len(
            suggestions
        ),
        total_examples=total_examples,
        average_cluster_size=round(
            total_examples
            / len(suggestions),
            2,
        )
        if suggestions
        else 0,
        silhouette_score=silhouette,
        suggestions=suggestions,
    )

    save_suggestions_to_db(
        summary.suggestions
    )

    save_json(
        summary.dict(),
        SUGGESTIONS_PATH,
    )

    return summary


@app.get("/health")
def health() -> dict:

    load_models()

    return {
        "status": "ok",
        "service": "suggestion",
        "embedding_model":
        MODEL_NAME,
        "answer_model": (
            FAQ_LLM_MODEL
            if ANSWER_GENERATOR
            else "historical_fallback"
        ),
    }


@app.post(
    "/suggest",
    response_model=
    SuggestionSummary,
)
def suggest() -> SuggestionSummary:
    """Genera FAQs."""

    try:

        conversations = (
            load_conversation_pairs()
        )

        return build_suggestions(
            conversations
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        import traceback

        print(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Error generating "
                f"suggestions: {exc}"
            ),
        )


@app.get(
    "/suggestions",
    response_model=
    SuggestionSummary,
)
def get_suggestions(
) -> SuggestionSummary:
    """Obtiene FAQs."""

    if not SUGGESTIONS_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "No suggestions have "
                "been generated yet."
            ),
        )

    raw = load_json(
        SUGGESTIONS_PATH
    )

    return SuggestionSummary(
        **raw
    )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "suggestion:app",
        host="127.0.0.1",
        port=8003,
        log_level="info",
    )