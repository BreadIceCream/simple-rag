from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.config.global_config import global_config
from app.evals.offline.build_synthetic_dataset import (
    _build_ragas_run_config,
    _init_generation_llm,
    _iter_sub_batches,
    _load_generation_plan,
    _map_reference_doc_ids,
    _resolve_dynamic_chunk_limit,
    _resolve_testset_generator,
)
from app.evals.offline.dataset_builder import build_manifest, format_build_report, save_dataset
from app.evals.offline.querytype_synthesizers import (
    QUERY_TYPES,
    QueryTypeSynthesizerFacade,
    allocate_query_type_counts,
    classify_querytype_error,
    probe_available_query_types,
    reallocate_query_type_counts,
    resolve_query_distribution,
)
from app.evals.offline.querytype_validator import QueryTypeValidator
from app.evals.offline.runtime import close_eval_runtime, init_eval_runtime
from app.evals.offline.schema import EvalSample


@dataclass
class NonRetriableQueryTypeGenerationError(RuntimeError):
    classification_category: str
    classification_reason: str


@dataclass
class BatchPreparationResult:
    requested_counts: dict[str, int]
    effective_counts: dict[str, int]
    available_query_types: list[str]
    unavailable_query_types: dict[str, str]
    probe_signals: dict[str, Any]
    fallback_events: list[dict[str, Any]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build specialized query-type synthetic datasets.")
    parser.add_argument("--name", required=True, help="dataset name")
    parser.add_argument("--version", required=True, help="dataset version")
    parser.add_argument("--category", default="specialized", choices=["synthetic", "exploration", "specialized"], help="dataset category")
    parser.add_argument("--size", type=int, default=100, help="target sample count")
    parser.add_argument("--doc-limit", type=int, default=30, help="max source documents selected for generation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--recency-tau-days", type=float, default=30.0, help="recency decay factor for document sampling")
    parser.add_argument("--alloc-alpha", type=float, default=0.7, help="allocation exponent for chunk budget")
    parser.add_argument("--use-light-model", action="store_true", help="use configured light chat model")
    parser.add_argument("--difficulty", default="unknown", help="default difficulty label")
    parser.add_argument("--scenario", default="single_turn", help="default scenario label")
    parser.add_argument("--description", default="", help="dataset description")
    parser.add_argument("--output-dir", default="", help="optional explicit dataset directory")

    parser.add_argument("--query-distribution-profile", default="multihop_focus", choices=["balanced", "multihop_focus"], help="preset query distribution profile")
    parser.add_argument("--query-distribution-json", default="", help="optional inline json for query distribution")
    parser.add_argument("--query-distribution-file", default="", help="optional json file for query distribution")

    parser.add_argument("--validator-mode", default="warn", choices=["off", "warn", "strict"], help="query type validator mode")
    parser.add_argument("--min-hop-evidence", type=int, default=2, help="minimum evidence units required for multi-hop")
    parser.add_argument("--enable-multi-file", action="store_true", help="enable multi-file batch mode for querytype generation")

    parser.add_argument("--max-batch-retries", type=int, default=3, help="retry count for each generation batch")
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0, help="base backoff seconds between retries")
    parser.add_argument("--max-chunks-per-batch", type=int, default=0, help="optional hard cap for chunks sent to one generation batch")

    parser.add_argument("--ragas-max-workers", type=int, default=1, help="max concurrent workers used by ragas testset generation")
    parser.add_argument("--ragas-timeout", type=int, default=240, help="per-operation timeout in seconds passed to ragas RunConfig")
    parser.add_argument("--ragas-max-retries", type=int, default=8, help="max retries passed to ragas RunConfig")
    parser.add_argument("--ragas-max-wait", type=int, default=30, help="max wait seconds between ragas retries")
    parser.add_argument("--llm-timeout", type=int, default=240, help="timeout in seconds passed to init_chat_model")
    parser.add_argument("--llm-max-retries", type=int, default=6, help="max retries passed to init_chat_model")
    parser.add_argument("--llm-requests-per-second", type=float, default=0.5, help="optional rate limit for generation llm calls")
    return parser.parse_args()


def _load_distribution_file(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_model_name(use_light_model: bool) -> str:
    chat_cfg = global_config.get("chat_model", {})
    return chat_cfg.get("light") if use_light_model else chat_cfg.get("default")


def _derive_evidence_topology(reference_doc_ids: list[str], reference_file_ids: list[str]) -> str:
    if len(set(reference_file_ids)) >= 2:
        return "multi_file"
    if len(set(reference_doc_ids)) >= 2:
        return "multi_doc_same_file"
    return "single_doc"


def _build_reasoning_hops(query_type: str, reference_doc_ids: list[str], reference_file_ids: list[str]) -> list[dict[str, Any]]:
    if not query_type.startswith("multi_hop_"):
        return []

    hops: list[dict[str, Any]] = []
    for index, doc_id in enumerate(reference_doc_ids[:2], start=1):
        hops.append(
            {
                "hop_id": f"hop_{index}",
                "hop_intent": "evidence_link",
                "reference_doc_ids": [doc_id],
                "reference_file_ids": list(reference_file_ids[:1]),
            }
        )

    if not hops and len(reference_file_ids) >= 2:
        for index, file_id in enumerate(reference_file_ids[:2], start=1):
            hops.append(
                {
                    "hop_id": f"hop_{index}",
                    "hop_intent": "evidence_link",
                    "reference_doc_ids": [],
                    "reference_file_ids": [file_id],
                }
            )
    return hops


def _merge_chunk_lookup(sub_batches: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    merged: dict[str, list[dict[str, str]]] = {}
    for sub_batch in sub_batches:
        lookup = dict(sub_batch.get("chunk_lookup", {}) or {})
        for key, value in lookup.items():
            merged.setdefault(key, []).extend(list(value))
    return merged


def _build_multi_file_batch(sub_batches: list[dict[str, Any]], batch_index: int, batch_total: int) -> dict[str, Any]:
    merged_chunks: list[Any] = []
    merged_parent_ids: list[str] = []
    merged_char_lengths: list[int] = []
    source_file_ids: list[str] = []
    source_file_names: list[str] = []

    for sub_batch in sub_batches:
        merged_chunks.extend(list(sub_batch.get("chunks", [])))
        merged_parent_ids.extend(list(sub_batch.get("selected_parent_doc_ids", [])))
        merged_char_lengths.extend(list(sub_batch.get("chunk_char_lengths", [])))
        source_file_ids.append(str(sub_batch.get("file_id") or ""))
        source_file_names.append(str(sub_batch.get("file_name") or ""))

    total_chars = sum(merged_char_lengths)
    avg_chars = (total_chars / len(merged_char_lengths)) if merged_char_lengths else 0.0

    return {
        **sub_batches[0],
        "file_id": f"multi_file_batch_{batch_index}",
        "file_name": "MULTI_FILE_BATCH",
        "chunks": merged_chunks,
        "selected_parent_doc_ids": merged_parent_ids,
        "chunk_char_lengths": merged_char_lengths,
        "allocated_quota": len(merged_chunks),
        "total_chunk_chars": total_chars,
        "avg_chunk_chars": avg_chars,
        "max_chunk_chars": max(merged_char_lengths) if merged_char_lengths else 0,
        "chunk_lookup": _merge_chunk_lookup(sub_batches),
        "sub_batch_index": batch_index,
        "sub_batch_count": batch_total,
        "batch_mode": "multi_file",
        "source_file_ids": [item for item in source_file_ids if item],
        "source_file_names": [item for item in source_file_names if item],
    }


def _build_generation_batches(generation_plan: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sub_batches_all: list[dict[str, Any]] = []
    for plan_item in generation_plan:
        effective_chunk_limit, decision = _resolve_dynamic_chunk_limit(plan_item, generation_plan, args.max_chunks_per_batch)
        plan_item["effective_chunk_limit"] = effective_chunk_limit
        plan_item["dynamic_batch_decision"] = decision
        per_file_sub_batches = _iter_sub_batches(plan_item, effective_chunk_limit)
        for sub_batch in per_file_sub_batches:
            sub_batch["batch_mode"] = "single_file"
            sub_batch["source_file_ids"] = [sub_batch.get("file_id")]
            sub_batch["source_file_names"] = [sub_batch.get("file_name")]
        sub_batches_all.extend(per_file_sub_batches)

    if not args.enable_multi_file:
        return sub_batches_all, {
            "mode": "single_file",
            "requested": False,
            "fallback_reason": None,
            "input_sub_batch_count": len(sub_batches_all),
            "output_sub_batch_count": len(sub_batches_all),
        }

    if not sub_batches_all:
        return sub_batches_all, {
            "mode": "single_file",
            "requested": True,
            "fallback_reason": "no_sub_batches",
            "input_sub_batch_count": 0,
            "output_sub_batch_count": 0,
        }

    queue = list(sub_batches_all)
    merged_batches: list[dict[str, Any]] = []
    while queue:
        base = queue.pop(0)
        partner_index = -1
        for index, candidate in enumerate(queue):
            if candidate.get("file_id") != base.get("file_id"):
                partner_index = index
                break

        if partner_index < 0:
            base["batch_mode"] = "single_file_fallback"
            base["multi_file_fallback_reason"] = "no_partner_file_batch"
            merged_batches.append(base)
            continue

        partner = queue.pop(partner_index)
        merged_batches.append(_build_multi_file_batch([base, partner], len(merged_batches) + 1, 0))

    total_batches = len(merged_batches)
    for index, batch in enumerate(merged_batches, start=1):
        batch["sub_batch_index"] = index
        batch["sub_batch_count"] = total_batches

    return merged_batches, {
        "mode": "mixed_multi_file",
        "requested": True,
        "fallback_reason": None,
        "input_sub_batch_count": len(sub_batches_all),
        "output_sub_batch_count": len(merged_batches),
        "merged_multi_file_batch_count": sum(1 for item in merged_batches if item.get("batch_mode") == "multi_file"),
        "single_file_fallback_batch_count": sum(1 for item in merged_batches if item.get("batch_mode") == "single_file_fallback"),
    }


def _to_samples(
    *,
    rows: list[dict[str, Any]],
    plan_item: dict[str, Any],
    scope_file_ids: list[str],
    args: argparse.Namespace,
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for row in rows:
        user_input = row.get("user_input") or row.get("question") or row.get("query")
        reference = row.get("reference") or row.get("ground_truth") or row.get("answer")
        contexts = row.get("reference_contexts") or row.get("contexts") or row.get("retrieved_contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = [str(item) for item in contexts if str(item).strip()]
        if not user_input or not contexts:
            continue

        reference_doc_ids, reference_file_ids = _map_reference_doc_ids(
            contexts=contexts,
            chunk_lookup=plan_item["chunk_lookup"],
            fallback_parent_doc_ids=plan_item["selected_parent_doc_ids"],
        )
        query_type = str(row.get("query_type") or "unknown")
        query_type_source = str(row.get("query_type_source") or row.get("synthesizer_name") or "ragas_synthesizer")

        sample = EvalSample(
            user_input=str(user_input),
            reference_answer=str(reference).strip() if reference else None,
            reference_contexts=contexts,
            reference_doc_ids=reference_doc_ids,
            scope_file_ids=list(scope_file_ids),
            difficulty_level=args.difficulty,
            scenario_type=args.scenario,
            source_type="synthetic",
            query_type=query_type,
            query_type_source=query_type_source,
            evidence_topology=_derive_evidence_topology(reference_doc_ids, reference_file_ids),
            reasoning_hops=_build_reasoning_hops(query_type, reference_doc_ids, reference_file_ids),
            tags=[args.category, "synthetic", "query_type_specialized"],
            review_status="pending",
            metadata={
                "reference_file_ids": reference_file_ids,
                "source_file_id": plan_item["file_id"],
                "source_file_name": plan_item["file_name"],
                "source_file_ids": list(plan_item.get("source_file_ids") or [plan_item.get("file_id")]),
                "source_file_names": list(plan_item.get("source_file_names") or [plan_item.get("file_name")]),
                "batch_mode": plan_item.get("batch_mode", "single_file"),
                "selected_parent_doc_ids": list(plan_item["selected_parent_doc_ids"]),
                "allocated_quota": plan_item["allocated_quota"],
                "sub_batch_index": plan_item.get("sub_batch_index"),
                "sub_batch_count": plan_item.get("sub_batch_count"),
                "query_distribution_profile": args.query_distribution_profile,
            },
        )
        samples.append(sample)
    return samples


def _log_batch_event(*, file_id: str, file_name: str, sub_batch_label: str, payload: dict[str, Any]) -> None:
    fields = " ".join(f"{key}={payload[key]}" for key in sorted(payload.keys()))
    print(f"QUERYTYPE BATCH: file_id={file_id} file_name={file_name} sub_batch={sub_batch_label} {fields}")


def _prepare_counts_for_batch(
    *,
    allocated_quota: int,
    requested_distribution: dict[str, float],
    chunks: list[Any],
    enable_multi_file: bool,
) -> BatchPreparationResult:
    requested_counts = allocate_query_type_counts(allocated_quota, requested_distribution)
    availability = probe_available_query_types(chunks=chunks, enable_multi_file=enable_multi_file)
    reallocation = reallocate_query_type_counts(requested_counts, set(availability.available_query_types))
    fallback_events = [
        {
            "source_query_type": event.source_query_type,
            "target_query_type": event.target_query_type,
            "count": event.count,
            "reason": event.reason,
        }
        for event in reallocation.fallback_events
    ]
    return BatchPreparationResult(
        requested_counts=requested_counts,
        effective_counts=reallocation.effective_counts,
        available_query_types=availability.available_query_types,
        unavailable_query_types=availability.unavailable_query_types,
        probe_signals=availability.signals,
        fallback_events=fallback_events,
    )


def _generate_with_retry(
    *,
    facade: QueryTypeSynthesizerFacade,
    generator: Any,
    chunks: list[Any],
    run_config: Any,
    query_type_counts: dict[str, int],
    args: argparse.Namespace,
    file_id: str,
    file_name: str,
    sub_batch_label: str,
):
    max_attempts = max(1, args.max_batch_retries)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(
                    f"DATASET BUILD: retrying querytype batch file_id={file_id} file_name={file_name} sub_batch={sub_batch_label} attempt={attempt}/{max_attempts}"
                )
            return facade.generate_rows(
                generator=generator,
                chunks=chunks,
                run_config=run_config,
                query_type_counts=query_type_counts,
            )
        except Exception as exc:
            last_exc = exc
            classification = classify_querytype_error(exc)
            _log_batch_event(
                file_id=file_id,
                file_name=file_name,
                sub_batch_label=sub_batch_label,
                payload={
                    "attempt": f"{attempt}/{max_attempts}",
                    "event": "generation_error",
                    "error": exc.__class__.__name__,
                    "retriable": classification.retriable,
                    "category": classification.category,
                    "reason": classification.reason,
                },
            )
            if not classification.retriable:
                raise NonRetriableQueryTypeGenerationError(classification.category, classification.reason) from exc
            if attempt >= max_attempts:
                break
            sleep_seconds = max(args.retry_backoff_seconds, 0.0) * attempt
            print(
                f"DATASET BUILD: querytype batch failed file_id={file_id} file_name={file_name} sub_batch={sub_batch_label} attempt={attempt}/{max_attempts} error={exc.__class__.__name__}: {exc}. retry_in={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Querytype generation failed for file_id={file_id} file_name={file_name} sub_batch={sub_batch_label} after {max_attempts} attempts"
    ) from last_exc


def _distribution_from_counts(counter: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {query_type: 0.0 for query_type in QUERY_TYPES}
    return {query_type: round(counter.get(query_type, 0) / total, 6) for query_type in QUERY_TYPES}


def main() -> None:
    args = _parse_args()
    distribution_file_payload = _load_distribution_file(args.query_distribution_file)
    requested_distribution = resolve_query_distribution(
        profile=args.query_distribution_profile,
        distribution_json=args.query_distribution_json or None,
        distribution_file_payload=distribution_file_payload,
    )

    init_eval_runtime("dataset_synthetic")
    try:
        from app.core.embeddings import EmbeddingModelFactory

        model_name = _resolve_model_name(args.use_light_model)
        embeddings = EmbeddingModelFactory.init_embedding_model()
        llm = _init_generation_llm(model_name, args)
        generator = _resolve_testset_generator(llm, embeddings)
        run_config = _build_ragas_run_config(args)

        generation_plan, scope_file_ids, plan_summary = _load_generation_plan(
            doc_limit=args.doc_limit,
            size=args.size,
            seed=args.seed,
            recency_tau_days=args.recency_tau_days,
            alloc_alpha=args.alloc_alpha,
        )

        facade = QueryTypeSynthesizerFacade()
        generated_samples: list[EvalSample] = []
        requested_counts_total: Counter[str] = Counter()
        effective_counts_total: Counter[str] = Counter()
        generated_counts_total: Counter[str] = Counter()
        availability_reason_counter: Counter[str] = Counter()
        fallback_events_all: list[dict[str, Any]] = []
        non_retriable_failures: list[dict[str, Any]] = []

        generation_batches, batch_mode_summary = _build_generation_batches(generation_plan, args)
        print(f"QUERYTYPE BUILD: batch_mode_summary={batch_mode_summary}")

        for sub_batch in generation_batches:
            sub_batch_label = f"{sub_batch.get('sub_batch_index', 1)}/{sub_batch.get('sub_batch_count', 1)}"
            file_id = str(sub_batch.get("file_id") or "")
            file_name = str(sub_batch.get("file_name") or "")
            preparation = _prepare_counts_for_batch(
                allocated_quota=sub_batch["allocated_quota"],
                requested_distribution=requested_distribution,
                chunks=sub_batch["chunks"],
                enable_multi_file=bool(args.enable_multi_file),
            )

            requested_counts_total.update(preparation.requested_counts)
            effective_counts_total.update(preparation.effective_counts)
            for _, reason in preparation.unavailable_query_types.items():
                availability_reason_counter[reason] += 1
            fallback_events_all.extend(
                [
                    {
                        **event,
                        "file_id": file_id,
                        "file_name": file_name,
                        "sub_batch": sub_batch_label,
                    }
                    for event in preparation.fallback_events
                ]
            )

            _log_batch_event(
                file_id=file_id,
                file_name=file_name,
                sub_batch_label=sub_batch_label,
                payload={
                    "event": "pre_generate",
                    "batch_mode": sub_batch.get("batch_mode", "single_file"),
                    "chunk_count": len(sub_batch.get("chunks", [])),
                    "requested_counts": preparation.requested_counts,
                    "available_query_types": preparation.available_query_types,
                    "unavailable_query_types": preparation.unavailable_query_types,
                    "effective_counts": preparation.effective_counts,
                    "fallback_events": preparation.fallback_events,
                },
            )

            generation_counts = dict(preparation.effective_counts)
            try:
                synthesis_result = _generate_with_retry(
                    facade=facade,
                    generator=generator,
                    chunks=sub_batch["chunks"],
                    run_config=run_config,
                    query_type_counts=generation_counts,
                    args=args,
                    file_id=file_id,
                    file_name=file_name,
                    sub_batch_label=sub_batch_label,
                )
            except NonRetriableQueryTypeGenerationError as exc:
                _log_batch_event(
                    file_id=file_id,
                    file_name=file_name,
                    sub_batch_label=sub_batch_label,
                    payload={
                        "event": "non_retriable_path",
                        "classification_category": exc.classification_category,
                        "classification_reason": exc.classification_reason,
                        "action": "fallback_to_single_hop",
                    },
                )
                secondary = reallocate_query_type_counts(generation_counts, {"single_hop_specific", "single_hop_abstract"})
                generation_counts = secondary.effective_counts
                fallback_events_all.extend(
                    {
                        "source_query_type": event.source_query_type,
                        "target_query_type": event.target_query_type,
                        "count": event.count,
                        "reason": f"secondary_{event.reason}",
                        "file_id": file_id,
                        "file_name": file_name,
                        "sub_batch": sub_batch_label,
                    }
                    for event in secondary.fallback_events
                )
                _log_batch_event(
                    file_id=file_id,
                    file_name=file_name,
                    sub_batch_label=sub_batch_label,
                    payload={
                        "event": "secondary_fallback",
                        "secondary_effective_counts": generation_counts,
                        "secondary_fallback_events": [
                            {
                                "source_query_type": event.source_query_type,
                                "target_query_type": event.target_query_type,
                                "count": event.count,
                                "reason": event.reason,
                            }
                            for event in secondary.fallback_events
                        ],
                    },
                )
                try:
                    synthesis_result = _generate_with_retry(
                        facade=facade,
                        generator=generator,
                        chunks=sub_batch["chunks"],
                        run_config=run_config,
                        query_type_counts=generation_counts,
                        args=args,
                        file_id=file_id,
                        file_name=file_name,
                        sub_batch_label=sub_batch_label,
                    )
                except Exception as secondary_exc:
                    non_retriable_failures.append(
                        {
                            "file_id": file_id,
                            "file_name": file_name,
                            "sub_batch": sub_batch_label,
                            "classification_category": exc.classification_category,
                            "classification_reason": exc.classification_reason,
                            "final_error": f"{secondary_exc.__class__.__name__}: {secondary_exc}",
                        }
                    )
                    _log_batch_event(
                        file_id=file_id,
                        file_name=file_name,
                        sub_batch_label=sub_batch_label,
                        payload={
                            "event": "batch_skipped",
                            "reason": "secondary_fallback_failed",
                            "final_error": f"{secondary_exc.__class__.__name__}: {secondary_exc}",
                        },
                    )
                    continue

            generated_counts_total.update(synthesis_result.generated_counts)
            _log_batch_event(
                file_id=file_id,
                file_name=file_name,
                sub_batch_label=sub_batch_label,
                payload={
                    "event": "generation_done",
                    "generated_counts": synthesis_result.generated_counts,
                },
            )
            generated_samples.extend(
                _to_samples(
                    rows=synthesis_result.rows,
                    plan_item=sub_batch,
                    scope_file_ids=scope_file_ids,
                    args=args,
                )
            )

        if len(generated_samples) > args.size:
            rng = random.Random(args.seed)
            generated_samples = rng.sample(generated_samples, args.size)

        validator = QueryTypeValidator(mode=args.validator_mode, min_hop_evidence=args.min_hop_evidence)
        validated_samples, validation_summary = validator.validate_samples(generated_samples)

        query_type_counter = Counter(sample.query_type for sample in validated_samples)
        realized_distribution = _distribution_from_counts(query_type_counter, len(validated_samples))

        plan_summary["requested_sample_count"] = args.size
        plan_summary["pre_validation_sample_count"] = len(generated_samples)
        plan_summary["actual_sample_count"] = len(validated_samples)
        plan_summary["effective_chunk_limits"] = {
            item["file_id"]: item.get("effective_chunk_limit", item["allocated_quota"]) for item in generation_plan
        }
        plan_summary["dynamic_batch_decisions"] = {
            item["file_id"]: item.get("dynamic_batch_decision", {}) for item in generation_plan
        }

        fallback_summary_counter: Counter[str] = Counter()
        for event in fallback_events_all:
            fallback_summary_counter[f"{event.get('source_query_type')}->{event.get('target_query_type')}"] += int(event.get("count", 0))

        manifest = build_manifest(
            name=args.name,
            version=args.version,
            category=args.category,
            source_type="synthetic",
            description=args.description or "Specialized synthetic dataset with query-type distribution.",
            metadata={
                "model_name": model_name,
                "seed": args.seed,
                "doc_limit": args.doc_limit,
                "recency_tau_days": args.recency_tau_days,
                "alloc_alpha": args.alloc_alpha,
                "scope_file_ids": scope_file_ids,
                "generation_plan": plan_summary,
                "requested_distribution": requested_distribution,
                "realized_distribution": realized_distribution,
                "requested_query_type_counts": dict(requested_counts_total),
                "effective_query_type_counts": dict(effective_counts_total),
                "generated_query_type_counts": dict(generated_counts_total),
                "availability_probe_summary": {
                    "unavailable_reason_counts": dict(availability_reason_counter),
                    "batch_mode_summary": batch_mode_summary,
                },
                "fallback_event_summary": {
                    "event_count": len(fallback_events_all),
                    "counts_by_path": dict(fallback_summary_counter),
                    "events": fallback_events_all,
                },
                "non_retriable_failure_summary": {
                    "count": len(non_retriable_failures),
                    "items": non_retriable_failures,
                },
                "validation_summary": validation_summary,
                "experiment_config": {
                    "profile": args.query_distribution_profile,
                    "distribution_source": "json" if args.query_distribution_json else "file" if args.query_distribution_file else "profile",
                    "validator_mode": args.validator_mode,
                    "min_hop_evidence": args.min_hop_evidence,
                    "enable_multi_file": args.enable_multi_file,
                    "batch_mode": batch_mode_summary.get("mode"),
                },
                "migration_note": "Phase B introduces build_querytype_dataset.py while keeping build_synthetic_dataset.py backward compatible.",
                "review_note": "Specialized synthetic datasets require manual review before baseline/regression usage.",
            },
        )

        output_dir = save_dataset(manifest, validated_samples, output_dir=args.output_dir or None)
        print(f"DATASET BUILD: querytype dataset saved to {output_dir}")
        print(
            f"DATASET BUILD: requested_samples={args.size} pre_validation={len(generated_samples)} actual_samples={len(validated_samples)}"
        )
        print(f"DATASET BUILD: requested_distribution={requested_distribution}")
        print(f"DATASET BUILD: realized_distribution={realized_distribution}")
        print(f"DATASET BUILD: validation_summary={validation_summary}")
        for line in format_build_report(manifest, validated_samples):
            print(line)
    finally:
        close_eval_runtime()


if __name__ == "__main__":
    main()
