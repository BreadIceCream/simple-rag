from __future__ import annotations

import asyncio
from typing import Any

from langchain.chat_models import init_chat_model

from app.config.global_config import global_config
from app.evals.offline.metrics_registry import select_ragas_metrics
from app.evals.online.online_eval_schema import OnlineEvalRecord, OnlineEvalSummary
from app.evals.online.online_eval_store import append_online_eval_entry


def _record_capabilities(record: OnlineEvalRecord) -> list[str]:
    capabilities: list[str] = []
    if record.reference_answer:
        capabilities.append("has_reference_answer")
    if record.reference_contexts:
        capabilities.append("has_reference_contexts")
    if record.reference_context_ids:
        capabilities.append("has_reference_doc_ids")
    return capabilities


def _load_models(llm_name: str):
    from app.core.embeddings import EmbeddingModelFactory

    llm = init_chat_model(llm_name)
    embeddings = EmbeddingModelFactory.init_embedding_model()
    try:
        from ragas.llms import LangchainLLMWrapper

        llm_obj = LangchainLLMWrapper(llm)
    except Exception:
        llm_obj = llm
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        embeddings_obj = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        embeddings_obj = embeddings
    return llm_obj, embeddings_obj


def _record_to_ragas_payload(record: OnlineEvalRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_input": record.user_input,
        "response": record.actual_response,
        "retrieved_contexts": list(record.actual_contexts),
    }
    if record.reference_answer:
        payload["reference"] = record.reference_answer
    if record.reference_contexts:
        payload["reference_contexts"] = list(record.reference_contexts)
    if record.reference_context_ids:
        payload["reference_context_ids"] = list(record.reference_context_ids)
        payload["retrieved_context_ids"] = list(record.actual_doc_ids)
    return payload


def _build_single_turn_sample(record: OnlineEvalRecord):
    try:
        from ragas import SingleTurnSample
    except Exception:
        from ragas.dataset_schema import SingleTurnSample  # type: ignore

    return SingleTurnSample(**_record_to_ragas_payload(record))


def _bind_metric_dependencies(metric: Any, *, llm: Any, embeddings: Any) -> None:
    if hasattr(metric, "llm") and getattr(metric, "llm", None) is None:
        metric.llm = llm
    if hasattr(metric, "embeddings") and getattr(metric, "embeddings", None) is None:
        metric.embeddings = embeddings


async def _score_single_metric(metric: Any, sample: Any, *, llm: Any, embeddings: Any, timeout: float | None) -> float:
    _bind_metric_dependencies(metric, llm=llm, embeddings=embeddings)
    if hasattr(metric, "single_turn_ascore"):
        return float(await metric.single_turn_ascore(sample, timeout=timeout))
    if hasattr(metric, "single_turn_score"):
        return float(await asyncio.to_thread(metric.single_turn_score, sample))
    raise RuntimeError(f"Metric does not support single-turn scoring: {metric}")


async def score_online_rag_request(record: OnlineEvalRecord) -> OnlineEvalSummary:
    try:
        capabilities = _record_capabilities(record)
        metrics_selection = select_ragas_metrics(capabilities)
        sample = _build_single_turn_sample(record)
        chat_cfg = global_config.get("chat_model", {})
        llm_name = chat_cfg.get("light") or chat_cfg.get("default")
        ragas_llm, embeddings = _load_models(llm_name)
        eval_cfg = global_config.get("eval_online", {}) or {}
        timeout = eval_cfg.get("metric_timeout_seconds", 120)

        metrics: dict[str, float] = {}
        metric_failures: dict[str, str] = {}
        skipped_metrics = list(metrics_selection.skipped)

        for metric in metrics_selection.metrics:
            metric_name = str(getattr(metric, "name", metric.__class__.__name__) or metric.__class__.__name__)
            try:
                metrics[metric_name] = await _score_single_metric(
                    metric,
                    sample,
                    llm=ragas_llm,
                    embeddings=embeddings,
                    timeout=timeout,
                )
            except Exception as metric_exc:
                metric_failures[metric_name] = f"{metric_exc.__class__.__name__}: {metric_exc}"
                skipped_metrics.append(metric_name)

        status = "success" if metrics else "failed"
        error_message = None if metrics else "No online metric completed successfully."
        summary = OnlineEvalSummary(
            request_id=record.request_id,
            conversation_id=record.conversation_id,
            status=status,
            metrics=metrics,
            skipped_metrics=skipped_metrics,
            query_type=record.query_type,
            latency_ms=record.latency_ms,
            error_message=error_message,
            metadata={
                "metric_names": list(metrics_selection.metric_names),
                "metric_failures": metric_failures,
                "metric_timeout_seconds": timeout,
                "successful_metric_count": len(metrics),
            },
        )
    except Exception as exc:
        summary = OnlineEvalSummary(
            request_id=record.request_id,
            conversation_id=record.conversation_id,
            status="failed",
            metrics={},
            skipped_metrics=[],
            query_type=record.query_type,
            latency_ms=record.latency_ms,
            error_message=f"{exc.__class__.__name__}: {exc}",
        )
        print(f"ONLINE EVAL: request_id={record.request_id} status=failed error={summary.error_message}")
    else:
        print(
            f"ONLINE EVAL: request_id={record.request_id} status={summary.status} metrics={sorted(summary.metrics.keys())} query_type={record.query_type}"
        )

    append_online_eval_entry(record, summary)
    return summary


def schedule_online_rag_evaluation(record: OnlineEvalRecord) -> asyncio.Task[OnlineEvalSummary]:
    return asyncio.create_task(score_online_rag_request(record))
