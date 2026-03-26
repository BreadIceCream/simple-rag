from __future__ import annotations

import math
from typing import Any

from app.evals.metrics_registry import default_retrieval_k_values
from app.evals.schema import EvalRunRecord


def _binary_relevance(actual_doc_ids: list[str], reference_doc_ids: list[str], k: int) -> list[int]:
    reference_set = set(reference_doc_ids)
    return [1 if doc_id in reference_set else 0 for doc_id in actual_doc_ids[:k]]


def _precision_at_k(actual_doc_ids: list[str], reference_doc_ids: list[str], k: int) -> float | None:
    if not reference_doc_ids:
        return None
    top_k = actual_doc_ids[:k]
    if not top_k:
        return 0.0
    hit_count = sum(1 for doc_id in top_k if doc_id in set(reference_doc_ids))
    return hit_count / len(top_k)


def _recall_at_k(actual_doc_ids: list[str], reference_doc_ids: list[str], k: int) -> float | None:
    if not reference_doc_ids:
        return None
    relevant = set(reference_doc_ids)
    hit_count = sum(1 for doc_id in actual_doc_ids[:k] if doc_id in relevant)
    return hit_count / len(relevant) if relevant else None


def _hit_at_k(actual_doc_ids: list[str], reference_doc_ids: list[str], k: int) -> float | None:
    if not reference_doc_ids:
        return None
    return 1.0 if any(doc_id in set(reference_doc_ids) for doc_id in actual_doc_ids[:k]) else 0.0


def _mrr_at_k(actual_doc_ids: list[str], reference_doc_ids: list[str], k: int) -> float | None:
    if not reference_doc_ids:
        return None
    relevant = set(reference_doc_ids)
    for index, doc_id in enumerate(actual_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(actual_doc_ids: list[str], reference_doc_ids: list[str], k: int) -> float | None:
    if not reference_doc_ids:
        return None
    gains = _binary_relevance(actual_doc_ids, reference_doc_ids, k)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = [1] * min(len(reference_doc_ids), k)
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    if idcg == 0:
        return None
    return dcg / idcg


def score_retrieval_metrics(records: list[EvalRunRecord], k_values: list[int] | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
    ks = k_values or default_retrieval_k_values()
    item_scores: list[dict[str, Any]] = []
    aggregate: dict[str, list[float]] = {}

    for record in records:
        row: dict[str, Any] = {"sample_id": record.sample_id}
        for k in ks:
            row[f"precision@{k}"] = _precision_at_k(record.actual_doc_ids, record.reference_doc_ids, k)
            row[f"recall@{k}"] = _recall_at_k(record.actual_doc_ids, record.reference_doc_ids, k)
            row[f"hit@{k}"] = _hit_at_k(record.actual_doc_ids, record.reference_doc_ids, k)
            row[f"mrr@{k}"] = _mrr_at_k(record.actual_doc_ids, record.reference_doc_ids, k)
            row[f"ndcg@{k}"] = _ndcg_at_k(record.actual_doc_ids, record.reference_doc_ids, k)
        item_scores.append(row)
        for key, value in row.items():
            if key == "sample_id" or value is None:
                continue
            aggregate.setdefault(key, []).append(float(value))

    summary = {key: sum(values) / len(values) for key, values in aggregate.items() if values}
    return item_scores, summary
