from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.schemas import OnlineEvalRun


def _chunked(entries: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(int(batch_size or 1), 1)
    return [entries[index:index + size] for index in range(0, len(entries), size)]


def _build_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(entry.get("request_id") or "").strip(),
        "conversation_id": str(entry.get("conversation_id") or ""),
        "thread_id": str(entry.get("thread_id") or ""),
        "event_date": str(entry.get("event_date") or ""),
        "request_created_at": str(entry.get("request_created_at") or ""),
        "evaluation_created_at": str(entry.get("evaluation_created_at") or ""),
        "query_type": str(entry.get("query_type") or "unknown"),
        "hop_count": str(entry.get("hop_count") or "unknown"),
        "abstraction_level": str(entry.get("abstraction_level") or "unknown"),
        "query_type_source": str(entry.get("query_type_source") or "unknown"),
        "status": str(entry.get("status") or "unknown"),
        "latency_ms": entry.get("latency_ms"),
        "rewrite_count": entry.get("rewrite_count"),
        "generate_count": entry.get("generate_count"),
        "retrieved_doc_count": entry.get("retrieved_doc_count"),
        "retrieved_file_count": entry.get("retrieved_file_count"),
        "retrieved_context_count": entry.get("retrieved_context_count"),
        "graph_message_count": entry.get("graph_message_count"),
        "graph_event_count": entry.get("graph_event_count"),
        "query_type_confidence": entry.get("query_type_confidence"),
        "successful_metric_count": entry.get("successful_metric_count"),
        "metric_timeout_seconds": entry.get("metric_timeout_seconds"),
        "user_input": str(entry.get("user_input") or ""),
        "actual_response": str(entry.get("actual_response") or ""),
        "reference_answer": entry.get("reference_answer"),
        "error_message": entry.get("error_message"),
        "actual_contexts": list(entry.get("actual_contexts") or []),
        "actual_doc_ids": list(entry.get("actual_doc_ids") or []),
        "actual_file_ids": list(entry.get("actual_file_ids") or []),
        "graph_messages": list(entry.get("graph_messages") or []),
        "graph_events": list(entry.get("graph_events") or []),
        "reference_contexts": list(entry.get("reference_contexts") or []),
        "reference_context_ids": list(entry.get("reference_context_ids") or []),
        "metrics": dict(entry.get("metrics") or {}),
        "skipped_metrics": list(entry.get("skipped_metrics") or []),
        "query_type_reasons": list(entry.get("query_type_reasons") or []),
        "metric_names": list(entry.get("metric_names") or []),
        "metric_failures": dict(entry.get("metric_failures") or {}),
        "metadata_json": dict(entry.get("metadata") or {}),
    }


def _upsert_batch(db: Session, batch: list[dict[str, Any]]) -> tuple[int, int]:
    if not batch:
        return 0, 0

    payloads = [_build_payload(entry) for entry in batch]
    request_ids = [payload["request_id"] for payload in payloads if payload["request_id"]]
    if not request_ids:
        return 0, 0

    stmt = select(OnlineEvalRun).where(OnlineEvalRun.request_id.in_(request_ids))
    existing_rows = {
        row.request_id: row
        for row in db.execute(stmt).scalars().all()
    }

    inserted = 0
    updated = 0
    for payload in payloads:
        request_id = payload["request_id"]
        if not request_id:
            continue

        existing = existing_rows.get(request_id)
        if existing is None:
            db.add(OnlineEvalRun(**payload))
            inserted += 1
            continue

        for key, value in payload.items():
            setattr(existing, key, value)
        updated += 1

    return inserted, updated


def upsert_online_eval_entries(
    db: Session,
    entries: list[dict[str, Any]],
    *,
    batch_size: int = 100,
) -> tuple[int, int, list[str]]:
    if not entries:
        return 0, 0, []

    inserted = 0
    updated = 0
    failed_request_ids: list[str] = []
    failed_batches: list[list[dict[str, Any]]] = []

    for batch in _chunked(entries, batch_size):
        batch_request_ids = [
            str(item.get("request_id") or "").strip()
            for item in batch
            if str(item.get("request_id") or "").strip()
        ]
        if not batch_request_ids:
            continue
        try:
            batch_inserted, batch_updated = _upsert_batch(db, batch)
            db.commit()
            inserted += batch_inserted
            updated += batch_updated
        except Exception as exc:
            db.rollback()
            failed_batches.append(batch)
            print(
                "ONLINE EVAL IMPORT: "
                f"batch_failed batch_request_ids={batch_request_ids} "
                f"error={exc.__class__.__name__}: {exc}"
            )

    for batch in failed_batches:
        for entry in batch:
            request_id = str(entry.get("request_id") or "").strip()
            if not request_id:
                continue
            try:
                single_inserted, single_updated = _upsert_batch(db, [entry])
                db.commit()
                inserted += single_inserted
                updated += single_updated
            except Exception as exc:
                db.rollback()
                failed_request_ids.append(request_id)
                print(
                    "ONLINE EVAL IMPORT: "
                    f"single_failed request_id={request_id} "
                    f"error={exc.__class__.__name__}: {exc}"
                )

    return inserted, updated, failed_request_ids
