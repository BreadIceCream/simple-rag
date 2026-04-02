from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.evals.online.online_eval_schema import OnlineEvalRecord


def _detect_query_type_heuristic(user_input: str, reference_doc_ids: list[str], scope_file_ids: list[str]) -> tuple[str, str, str, float, list[str]]:
    text = str(user_input or "").strip().lower()
    abstract_patterns = [
        r"\bwhy\b",
        r"\bhow\b",
        r"\bcompare\b",
        r"\bdifference\b",
        r"\bsummar",
        r"\boverview\b",
        r"\bexplain\b",
    ]
    multi_patterns = [
        r"\bbetween\b",
        r"\brelationship\b",
        r"\bimpact\b",
        r"\bdepend\b",
        r"\balso\b",
    ]

    reasons: list[str] = []
    is_abstract = any(re.search(pattern, text) for pattern in abstract_patterns)
    if is_abstract:
        reasons.append("abstract_keyword")

    unique_doc_count = len({str(item).strip() for item in (reference_doc_ids or []) if str(item).strip()})
    unique_file_count = len({str(item).strip() for item in (scope_file_ids or []) if str(item).strip()})
    multi_from_evidence = unique_doc_count >= 2 or unique_file_count >= 2
    multi_from_query = any(re.search(pattern, text) for pattern in multi_patterns)
    is_multi = multi_from_evidence or multi_from_query
    if multi_from_evidence:
        reasons.append("multi_evidence")
    if multi_from_query:
        reasons.append("multi_keyword")

    hop_count = "multi" if is_multi else "single"
    abstraction_level = "abstract" if is_abstract else "specific"
    query_type = f"{hop_count}_hop_{abstraction_level}"

    confidence = 0.55
    if multi_from_evidence:
        confidence += 0.2
    if is_abstract:
        confidence += 0.15
    if multi_from_query:
        confidence += 0.1
    confidence = round(min(confidence, 0.95), 2)

    return query_type, hop_count, abstraction_level, confidence, reasons


def serialize_langchain_message(message: BaseMessage) -> dict[str, Any]:
    message_type = getattr(message, "type", message.__class__.__name__)
    content = getattr(message, "content", "")
    payload: dict[str, Any] = {
        "type": str(message_type),
        "content": str(content) if content is not None else "",
    }
    if isinstance(message, AIMessage):
        payload["tool_calls"] = list(getattr(message, "tool_calls", []) or [])
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = getattr(message, "tool_call_id", None)
        payload["status"] = getattr(message, "status", None)
    if isinstance(message, HumanMessage):
        payload["role"] = "human"
    return payload


def classify_online_query_type(user_input: str, actual_doc_ids: list[str], actual_file_ids: list[str]) -> tuple[str, str, str, str, float, list[str]]:
    query_type, hop_count, abstraction_level, confidence, reasons = _detect_query_type_heuristic(
        user_input=user_input,
        reference_doc_ids=actual_doc_ids,
        scope_file_ids=actual_file_ids,
    )
    return query_type, hop_count, abstraction_level, "online_heuristic", confidence, reasons


def build_online_eval_record(
    *,
    request_id: str,
    conversation_id: str,
    thread_id: str,
    user_input: str,
    actual_response: str,
    actual_contexts: list[str],
    actual_doc_ids: list[str],
    actual_file_ids: list[str],
    latency_ms: float | None,
    rewrite_count: int | None,
    generate_count: int | None,
    graph_messages: list[dict[str, Any]],
    graph_events: list[dict[str, Any]],
) -> OnlineEvalRecord:
    query_type, hop_count, abstraction_level, query_type_source, confidence, reasons = classify_online_query_type(
        user_input=user_input,
        actual_doc_ids=actual_doc_ids,
        actual_file_ids=actual_file_ids,
    )
    return OnlineEvalRecord(
        request_id=request_id,
        conversation_id=conversation_id,
        thread_id=thread_id,
        user_input=user_input,
        actual_response=actual_response,
        actual_contexts=list(actual_contexts),
        actual_doc_ids=list(actual_doc_ids),
        actual_file_ids=list(actual_file_ids),
        query_type=query_type,
        hop_count=hop_count,
        abstraction_level=abstraction_level,
        query_type_source=query_type_source,
        latency_ms=latency_ms,
        rewrite_count=rewrite_count,
        generate_count=generate_count,
        graph_messages=list(graph_messages),
        graph_events=list(graph_events),
        metadata={
            "query_type_confidence": confidence,
            "query_type_reasons": reasons,
            "retrieved_doc_count": len(actual_doc_ids),
            "retrieved_file_count": len(actual_file_ids),
            "retrieved_context_count": len(actual_contexts),
            "graph_message_count": len(graph_messages),
            "graph_event_count": len(graph_events),
        },
    )
