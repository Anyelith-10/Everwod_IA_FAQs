from datetime import UTC, datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from ingestion import ingest
from suggestion import build_suggestions, load_conversation_pairs


def scheduled_pipeline() -> None:
    """Ejecuta la canalizacion completa de ingesta y sugerencias."""
    start = datetime.now(UTC)
    print(f"[{start.isoformat()}] Iniciando pipeline de FAQ automática...")
    ingest_response = ingest(limit=15000, since_days=365)
    print(
        f"  - Ingested {ingest_response.imported_records} "
        "conversation pairs into data/conversations.jsonl"
    )
    conversations = load_conversation_pairs()
    if not conversations:
        print("  - No hay conversaciones procesadas para generar sugerencias.")
        return
    try:
        summary = build_suggestions(conversations)
    except Exception as exc:
        print(f"  - No se pudieron generar sugerencias. Error: {exc}")
        return
    print(f"  - Generadas {len(summary.suggestions)} sugerencias de FAQ")
    print(f"  - Métrica silhouette: {summary.silhouette_score}")
    print(f"  - Cluster count: {summary.cluster_count}")
    print(f"[{datetime.now(UTC).isoformat()}] Pipeline completada.")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(
        scheduled_pipeline,
        "interval",
        hours=24,
        next_run_time=datetime.now(),
    )
    print("Scheduler iniciado: el job se ejecutará cada 24 horas.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler detenido manualmente.")