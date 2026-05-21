import uuid
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faq_core import get_db_connection

app = FastAPI(title="FAQ Review Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/faq/pending")
def get_pending_faqs():

    query = """
    SELECT
        id,
        company_id,
        company_name,
        question,
        answer,
        cluster_size,
        cluster_score,
        support_examples,
        status,
        created_at
    FROM faq_suggestions
    WHERE status = 'pending'
    ORDER BY created_at DESC
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(query)

                rows = cursor.fetchall()

        results = []

        for row in rows:
            results.append(
                {
                    "id": str(row[0]),
                    "company_id": row[1],
                    "company_name": row[2],
                    "question": row[3],
                    "answer": row[4],
                    "cluster_size": row[5],
                    "cluster_score": row[6],
                    "support_examples": row[7],
                    "status": row[8],
                    "created_at": row[9],
                }
            )

        return results

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/faq/{faq_id}/approve")
def approve_faq(faq_id: str):

    select_query = """
    SELECT
        company_id,
        question,
        answer
    FROM faq_suggestions
    WHERE id = %s
    """

    update_query = """
    UPDATE faq_suggestions
    SET
        status = 'approved',
        reviewed_at = %s
    WHERE id = %s
    """

    insert_query = """
    INSERT INTO agent_faqs (
        id,
        agent_id,
        question,
        answer,
        created_at,
        updated_at
    )
    VALUES (
        %s,%s,%s,%s,%s,%s
    )
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    select_query,
                    (faq_id,),
                )

                faq = cursor.fetchone()

                if not faq:
                    raise HTTPException(
                        status_code=404,
                        detail="FAQ not found",
                    )

                company_id, question, answer = faq

                # buscar agent_id
                cursor.execute(
                    """
                    SELECT id
                    FROM agents
                    WHERE workspace_id = %s
                    LIMIT 1
                    """,
                    (company_id,),
                )

                agent = cursor.fetchone()

                if not agent:
                    raise HTTPException(
                        status_code=404,
                        detail="Agent not found",
                    )

                agent_id = agent[0]

                now = datetime.utcnow()

                cursor.execute(
                    insert_query,
                    (
                        str(uuid.uuid4()),
                        agent_id,
                        question,
                        answer,
                        now,
                        now,
                    ),
                )

                cursor.execute(
                    update_query,
                    (
                        now,
                        faq_id,
                    ),
                )

            conn.commit()

        return {
            "success": True,
            "message": "FAQ approved",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/faq/{faq_id}/reject")
def reject_faq(faq_id: str):

    query = """
    UPDATE faq_suggestions
    SET
        status = 'rejected',
        reviewed_at = %s
    WHERE id = %s
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    query,
                    (
                        datetime.utcnow(),
                        faq_id,
                    ),
                )

            conn.commit()

        return {
            "success": True,
            "message": "FAQ rejected",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "review:app",
        host="127.0.0.1",
        port=8004,
        log_level="info",
    )