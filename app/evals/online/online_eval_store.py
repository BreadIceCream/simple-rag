from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evals.online.online_eval_schema import OnlineEvalRecord, OnlineEvalSummary

_JSONL_WRITE_LOCK = threading.Lock()


def online_eval_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "store" / "evals" / "online"


def online_eval_day_dir(created_at: str | None = None) -> Path:
    stamp = created_at or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    day = stamp[:10]
    output_dir = online_eval_root() / day
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def daily_online_eval_jsonl_path(created_at: str | None = None) -> Path:
    return online_eval_day_dir(created_at) / "online_eval.jsonl"


def materialize_online_eval_entry(record: OnlineEvalRecord, summary: OnlineEvalSummary) -> dict[str, Any]:
    record_payload = record.to_dict()
    summary_payload = summary.to_dict()
    record_metadata = dict(record_payload.get("metadata") or {})
    summary_metadata = dict(summary_payload.get("metadata") or {})
    created_at = str(record_payload.get("created_at") or summary_payload.get("created_at") or "")

    return {
        "request_id": record_payload["request_id"],
        "conversation_id": record_payload["conversation_id"],
        "thread_id": record_payload["thread_id"],
        "event_date": created_at[:10] if created_at else "",
        "request_created_at": created_at,
        "evaluation_created_at": str(summary_payload.get("created_at") or ""),
        "user_input": record_payload["user_input"],
        "actual_response": record_payload["actual_response"],
        "actual_contexts": list(record_payload.get("actual_contexts") or []),
        "actual_doc_ids": list(record_payload.get("actual_doc_ids") or []),
        "actual_file_ids": list(record_payload.get("actual_file_ids") or []),
        "query_type": record_payload.get("query_type", "unknown"),
        "hop_count": record_payload.get("hop_count", "unknown"),
        "abstraction_level": record_payload.get("abstraction_level", "unknown"),
        "query_type_source": record_payload.get("query_type_source", "unknown"),
        "latency_ms": record_payload.get("latency_ms"),
        "rewrite_count": record_payload.get("rewrite_count"),
        "generate_count": record_payload.get("generate_count"),
        "graph_messages": list(record_payload.get("graph_messages") or []),
        "graph_events": list(record_payload.get("graph_events") or []),
        "reference_answer": record_payload.get("reference_answer"),
        "reference_contexts": list(record_payload.get("reference_contexts") or []),
        "reference_context_ids": list(record_payload.get("reference_context_ids") or []),
        "status": summary_payload.get("status", "unknown"),
        "metrics": dict(summary_payload.get("metrics") or {}),
        "skipped_metrics": list(summary_payload.get("skipped_metrics") or []),
        "error_message": summary_payload.get("error_message"),
        "retrieved_doc_count": record_metadata.get("retrieved_doc_count", len(record_payload.get("actual_doc_ids") or [])),
        "retrieved_file_count": record_metadata.get("retrieved_file_count", len(record_payload.get("actual_file_ids") or [])),
        "retrieved_context_count": record_metadata.get("retrieved_context_count", len(record_payload.get("actual_contexts") or [])),
        "graph_message_count": record_metadata.get("graph_message_count", len(record_payload.get("graph_messages") or [])),
        "graph_event_count": record_metadata.get("graph_event_count", len(record_payload.get("graph_events") or [])),
        "query_type_confidence": record_metadata.get("query_type_confidence"),
        "query_type_reasons": list(record_metadata.get("query_type_reasons") or []),
        "metric_names": list(summary_metadata.get("metric_names") or []),
        "metric_failures": dict(summary_metadata.get("metric_failures") or {}),
        "metric_timeout_seconds": summary_metadata.get("metric_timeout_seconds"),
        "successful_metric_count": summary_metadata.get("successful_metric_count"),
        "metadata": {
            "record": record_metadata,
            "summary": summary_metadata,
        },
    }


def append_online_eval_entry(record: OnlineEvalRecord, summary: OnlineEvalSummary) -> Path:
    output_path = daily_online_eval_jsonl_path(record.created_at)
    payload = materialize_online_eval_entry(record, summary)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _JSONL_WRITE_LOCK:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return output_path


def load_online_eval_entries(event_date: str) -> list[dict[str, Any]]:
    path = online_eval_root() / event_date / "online_eval.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
    return entries
