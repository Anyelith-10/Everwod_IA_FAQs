from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from ingest_service import ingest
from suggestion_service import build_suggestions, load_conversation_pairs


def scheduled_pipeline() -> None:
    """Ejecuta la canalizacion completa de ingesta y sugerencias."""
    start = datetime.utcnow()
    print(f"[{start.isoformat()}] Iniciando pipeline de FAQ automática...")

    # Primero actualiza el archivo intermedio con conversaciones recientes.
    ingest_response = ingest(limit=15000, since_days=365)
    print(f"  - Ingested {ingest_response.imported_records} conversation pairs into data/conversations.jsonl")

    # Luego carga las conversaciones procesadas para generar FAQs.
    conversations = load_conversation_pairs()
    if not conversations:
        print("  - No hay conversaciones procesadas para generar sugerencias.")
        return

    # Finalmente agrupa preguntas similares y guarda las sugerencias.
    summary = build_suggestions(conversations)
    print(f"  - Generadas {len(summary.suggestions)} sugerencias de FAQ")
    print(f"  - Métrica silhouette: {summary.silhouette_score}")
    print(f"  - Cluster count: {summary.cluster_count}")
    print(f"[{datetime.utcnow().isoformat()}] Pipeline completada.")


if __name__ == "__main__":
    # Scheduler bloqueante: mantiene el proceso vivo y ejecuta el job cada 24 horas.
    scheduler = BlockingScheduler()
    scheduler.add_job(scheduled_pipeline, "interval", hours=24, next_run_time=datetime.now())
    print("Scheduler iniciado: el job se ejecutará cada 24 horas.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler detenido manualmente.")
