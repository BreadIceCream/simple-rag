from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


@dataclass
class OnlineEvalRecord:
    request_id: str
    conversation_id: str
    thread_id: str
    user_input: str
    actual_response: str
    actual_contexts: list[str] = field(default_factory=list)
    actual_doc_ids: list[str] = field(default_factory=list)
    actual_file_ids: list[str] = field(default_factory=list)
    query_type: str = "unknown"
    hop_count: str = "unknown"
    abstraction_level: str = "unknown"
    query_type_source: str = "unknown"
    latency_ms: float | None = None
    rewrite_count: int | None = None
    generate_count: int | None = None
    graph_messages: list[dict[str, Any]] = field(default_factory=list)
    graph_events: list[dict[str, Any]] = field(default_factory=list)
    reference_answer: str | None = None
    reference_contexts: list[str] = field(default_factory=list)
    reference_context_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actual_contexts"] = [str(item) for item in self.actual_contexts if str(item).strip()]
        payload["actual_doc_ids"] = _ordered_unique(self.actual_doc_ids)
        payload["actual_file_ids"] = _ordered_unique(self.actual_file_ids)
        payload["reference_contexts"] = [str(item) for item in self.reference_contexts if str(item).strip()]
        payload["reference_context_ids"] = _ordered_unique(self.reference_context_ids)
        payload["graph_messages"] = [dict(item) for item in self.graph_messages if isinstance(item, dict)]
        payload["graph_events"] = [dict(item) for item in self.graph_events if isinstance(item, dict)]
        payload["metadata"] = dict(self.metadata or {})
        return payload


@dataclass
class OnlineEvalSummary:
    request_id: str
    conversation_id: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    skipped_metrics: list[str] = field(default_factory=list)
    query_type: str = "unknown"
    latency_ms: float | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = {str(key): float(value) for key, value in (self.metrics or {}).items()}
        payload["skipped_metrics"] = [str(item) for item in self.skipped_metrics]
        payload["metadata"] = dict(self.metadata or {})
        return payload
