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
        if hasattr(metric_module, name):
            metric_obj = getattr(metric_module, name)
            try:
                return metric_obj() if callable(metric_obj) else metric_obj
            except TypeError:
                return metric_obj
    return None


def select_single_turn_metrics(has_reference: bool) -> MetricsSelection:
    """
    选择 P0 单轮评测指标。
    - 无 reference 时：Faithfulness / AnswerRelevancy / ContextPrecision(无参考变体)
    - 有 reference 时：额外加入 ContextRecall
    """
    try:
        from ragas import metrics as rm
    except ImportError as e:
        raise ImportError("ragas is required. Install dependencies first.") from e

    metrics: list[Any] = []
    metric_names: list[str] = []
    skipped: list[str] = []

    def add_metric(display_name: str, candidates: list[str]) -> None:
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

    if has_reference:
        add_metric("context_recall", ["LLMContextRecall", "ContextRecall", "context_recall"])
    else:
        skipped.append("context_recall (requires reference)")

    if not metrics:
        raise RuntimeError("No compatible RAGAS metrics were found in current ragas version.")

    return MetricsSelection(metrics=metrics, metric_names=metric_names, skipped=skipped)
