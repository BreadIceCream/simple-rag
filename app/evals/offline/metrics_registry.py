from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetricsSelection:
    metrics: list[Any]
    metric_names: list[str]
    skipped: list[str]


def _get_metric_from_module(metric_module: Any, candidates: list[str]) -> Any | None:
    for name in candidates:
        if not hasattr(metric_module, name):
            continue
        metric_obj = getattr(metric_module, name)
        try:
            return metric_obj() if callable(metric_obj) else metric_obj
        except TypeError:
            return metric_obj
    return None


def select_ragas_metrics(capabilities: list[str]) -> MetricsSelection:
    try:
        from ragas import metrics as rm
    except ImportError as exc:
        raise ImportError("ragas is required. Install dependencies first.") from exc

    capability_set = set(capabilities or [])
    metrics: list[Any] = []
    metric_names: list[str] = []
    skipped: list[str] = []

    def add_metric(display_name: str, candidates: list[str], enabled: bool = True) -> None:
        if not enabled:
            skipped.append(display_name)
            return
        metric_obj = _get_metric_from_module(rm, candidates)
        if metric_obj is None:
            skipped.append(display_name)
            return
        metrics.append(metric_obj)
        metric_names.append(display_name)

    add_metric("faithfulness", ["Faithfulness", "faithfulness"])
    add_metric("answer_relevancy", ["ResponseRelevancy", "AnswerRelevancy", "response_relevancy", "answer_relevancy"])
    add_metric(
        "context_precision",
        [
            "LLMContextPrecisionWithoutReference",
            "LLMContextPrecisionWithReference",
            "ContextPrecision",
            "context_precision",
        ],
    )
    add_metric(
        "context_recall",
        ["LLMContextRecall", "ContextRecall", "context_recall"],
        enabled="has_reference_answer" in capability_set,
    )
    add_metric(
        "context_entities_recall",
        ["ContextEntityRecall", "ContextEntitiesRecall", "context_entity_recall", "context_entities_recall"],
        enabled="has_reference_contexts" in capability_set or "has_reference_answer" in capability_set,
    )

    if not metrics:
        raise RuntimeError("No compatible RAGAS metrics were found in current ragas version.")
    return MetricsSelection(metrics=metrics, metric_names=metric_names, skipped=skipped)


def default_retrieval_k_values() -> list[int]:
    return [1, 3, 5, 8]
