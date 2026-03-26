from __future__ import annotations

import argparse
import inspect
import math
import random
import time
from datetime import datetime
from typing import Any

from langchain.chat_models import init_chat_model

from app.config.db_config import DatabaseManager
from app.config.global_config import global_config
from app.evals.dataset_builder import build_manifest, format_build_report, save_dataset
from app.evals.runtime import close_eval_runtime, init_eval_runtime
from app.evals.schema import EvalSample
from app.models.schemas import EmbeddedDocument


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic datasets with RAGAS TestsetGenerator.")
    parser.add_argument("--name", required=True, help="dataset name")
    parser.add_argument("--version", required=True, help="dataset version")
    parser.add_argument("--category", default="synthetic", choices=["synthetic", "exploration", "specialized"], help="dataset category")
    parser.add_argument("--size", type=int, default=100, help="target sample count")
    parser.add_argument("--doc-limit", type=int, default=30, help="max source documents selected for generation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--recency-tau-days", type=float, default=30.0, help="recency decay factor for document sampling")
    parser.add_argument("--alloc-alpha", type=float, default=0.7, help="allocation exponent for chunk budget")
    parser.add_argument("--use-light-model", action="store_true", help="use the configured light chat model")
    parser.add_argument("--difficulty", default="unknown", help="default difficulty label")
    parser.add_argument("--scenario", default="single_turn", help="default scenario label")
    parser.add_argument("--description", default="", help="dataset description")
    parser.add_argument("--output-dir", default="", help="optional explicit dataset directory")
    parser.add_argument("--max-batch-retries", type=int, default=3, help="retry count for each per-file synthetic generation batch")
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0, help="base backoff seconds between batch retries")
    parser.add_argument("--max-chunks-per-batch", type=int, default=3, help="default maximum parent chunks sent to one ragas scenario-generation batch")
    parser.add_argument("--max-topup-rounds", type=int, default=5, help="maximum rounds used to top up synthetic samples when the first pass undershoots target size")
    return parser.parse_args()


def _safe_weighted_choice_index(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    cursor = rng.random() * total
    acc = 0.0
    for index, weight in enumerate(weights):
        acc += weight
        if cursor <= acc:
            return index
    return len(weights) - 1


def _weighted_sample_without_replacement(items: list[Any], weights: list[float], k: int, rng: random.Random) -> list[Any]:
    candidates = list(items)
    candidate_weights = list(weights)
    selected: list[Any] = []
    for _ in range(min(k, len(candidates))):
        idx = _safe_weighted_choice_index(candidate_weights, rng)
        selected.append(candidates.pop(idx))
        candidate_weights.pop(idx)
        if not candidates:
            break
    return selected


def _doc_sampling_weight(doc: EmbeddedDocument, now: datetime, recency_tau_days: float) -> float:
    created_at = getattr(doc, "created_at", None)
    if created_at is None:
        age_days = recency_tau_days
    else:
        age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
    tau = max(recency_tau_days, 1e-6)
    recency_weight = max(math.exp(-age_days / tau), 1e-8)
    parent_count = max(len(doc.parent_doc_ids or []), 1)
    count_weight = max(float(parent_count) ** 0.5, 1.0)
    return recency_weight * count_weight


def _allocate_chunk_budget(parent_counts: list[int], total_budget: int, alloc_alpha: float, rng: random.Random) -> list[int]:
    if not parent_counts:
        return []
    capped_budget = min(max(total_budget, 0), sum(parent_counts))
    quotas = [0] * len(parent_counts)
    if capped_budget >= len(parent_counts):
        for index, count in enumerate(parent_counts):
            if count > 0:
                quotas[index] = 1
    remaining_budget = capped_budget - sum(quotas)
    alpha = max(alloc_alpha, 0.0)
    while remaining_budget > 0:
        eligible = [index for index, count in enumerate(parent_counts) if quotas[index] < count]
        if not eligible:
            break
        weights = []
        for index in eligible:
            remaining_capacity = parent_counts[index] - quotas[index]
            weights.append(max(float(remaining_capacity) ** alpha, 1e-8))
        local_idx = _safe_weighted_choice_index(weights, rng)
        quotas[eligible[local_idx]] += 1
        remaining_budget -= 1
    return quotas


def _resolve_testset_generator(llm: Any, embeddings: Any):
    from ragas.testset import TestsetGenerator

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

    init_sig = inspect.signature(TestsetGenerator)
    kwargs = {}
    if "llm" in init_sig.parameters:
        kwargs["llm"] = llm_obj
    if "generator_llm" in init_sig.parameters:
        kwargs["generator_llm"] = llm_obj
    if "critic_llm" in init_sig.parameters:
        kwargs["critic_llm"] = llm_obj
    if "embeddings" in init_sig.parameters:
        kwargs["embeddings"] = embeddings_obj
    if "embedding_model" in init_sig.parameters:
        kwargs["embedding_model"] = embeddings_obj
    return TestsetGenerator(**kwargs) if kwargs else TestsetGenerator(llm_obj, embeddings_obj)


def _generate_with_chunks(generator: Any, chunks: list[Any], size: int):
    if not hasattr(generator, "generate_with_chunks"):
        raise RuntimeError("Current ragas version does not expose generate_with_chunks.")
    try:
        return generator.generate_with_chunks(chunks=chunks, testset_size=size)
    except TypeError:
        try:
            return generator.generate_with_chunks(chunks, size)
        except TypeError:
            return generator.generate_with_chunks(chunks=chunks, size=size)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_chunk_lookup(chunk_records: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    lookup: dict[str, list[dict[str, str]]] = {}
    for record in chunk_records:
        key = _normalize_text(record["page_content"])
        if not key:
            continue
        lookup.setdefault(key, []).append(
            {
                "parent_doc_id": record["parent_doc_id"],
                "file_id": record["file_id"],
                "file_name": record["file_name"],
            }
        )
    return lookup


def _load_generation_plan(doc_limit: int, size: int, seed: int, recency_tau_days: float, alloc_alpha: float) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    from app.core.vector_store import VectorStoreFactory

    with DatabaseManager.get_sync_db() as db:
        docs = db.query(EmbeddedDocument).all()
    docs = [doc for doc in docs if doc.parent_doc_ids]
    if not docs:
        raise ValueError("No documents with parent_doc_ids found in embedded_document.")

    rng = random.Random(seed)
    now = datetime.now()
    doc_weights = [_doc_sampling_weight(doc, now=now, recency_tau_days=recency_tau_days) for doc in docs]
    selected_doc_count = min(max(doc_limit, 1), len(docs), max(size, 1))
    selected_docs = _weighted_sample_without_replacement(docs, doc_weights, selected_doc_count, rng)
    parent_counts = [len(doc.parent_doc_ids or []) for doc in selected_docs]
    quotas = _allocate_chunk_budget(parent_counts, max(size, len(selected_docs)), alloc_alpha, rng)

    docstore = VectorStoreFactory.init_docstore()
    plan: list[dict[str, Any]] = []
    total_selected_parent_chunks = 0

    for doc, quota in zip(selected_docs, quotas):
        parent_ids = list(doc.parent_doc_ids or [])
        if quota <= 0 or not parent_ids:
            continue
        take_n = min(quota, len(parent_ids))
        selected_parent_ids = rng.sample(parent_ids, take_n) if take_n < len(parent_ids) else parent_ids
        parent_docs = docstore.mget(selected_parent_ids)

        chunk_records: list[dict[str, Any]] = []
        valid_parent_ids: list[str] = []
        chunks: list[Any] = []
        chunk_char_lengths: list[int] = []
        for index, (parent_id, parent_doc) in enumerate(zip(selected_parent_ids, parent_docs)):
            if parent_doc is None or not getattr(parent_doc, "page_content", ""):
                continue
            text = str(parent_doc.page_content)
            parent_doc.metadata = dict(parent_doc.metadata or {})
            parent_doc.metadata.update(
                {
                    "file_id": doc.id,
                    "file_name": doc.file_name,
                    "file_extension": doc.file_extension,
                    "source": "synthetic_parent_docs",
                    "parent_doc_id": parent_id,
                    "parent_rank_in_sample": index,
                    "sample_seed": seed,
                }
            )
            chunks.append(parent_doc)
            valid_parent_ids.append(parent_id)
            chunk_char_lengths.append(len(text))
            chunk_records.append(
                {
                    "parent_doc_id": parent_id,
                    "file_id": doc.id,
                    "file_name": str(doc.file_name or ""),
                    "page_content": text,
                }
            )

        if not chunks:
            continue

        total_selected_parent_chunks += len(chunks)
        total_chunk_chars = sum(chunk_char_lengths)
        avg_chunk_chars = total_chunk_chars / len(chunk_char_lengths)
        plan.append(
            {
                "file_id": doc.id,
                "file_name": str(doc.file_name or ""),
                "all_parent_doc_ids": parent_ids,
                "parent_count": len(parent_ids),
                "allocated_quota": quota,
                "selected_parent_doc_ids": valid_parent_ids,
                "used_parent_doc_ids": set(valid_parent_ids),
                "chunks": chunks,
                "chunk_lookup": _build_chunk_lookup(chunk_records),
                "chunk_char_lengths": chunk_char_lengths,
                "total_chunk_chars": total_chunk_chars,
                "avg_chunk_chars": avg_chunk_chars,
                "max_chunk_chars": max(chunk_char_lengths),
            }
        )

    if not plan:
        raise ValueError("No valid parent chunks found for synthetic dataset generation.")

    selected_file_ids = [item["file_id"] for item in plan]
    summary = {
        "selected_doc_count": len(plan),
        "selected_parent_chunk_count": total_selected_parent_chunks,
        "doc_parent_counts": {item["file_id"]: item["parent_count"] for item in plan},
        "doc_allocations": {item["file_id"]: item["allocated_quota"] for item in plan},
        "doc_avg_chunk_chars": {item["file_id"]: round(item["avg_chunk_chars"], 1) for item in plan},
        "doc_max_chunk_chars": {item["file_id"]: item["max_chunk_chars"] for item in plan},
    }
    return plan, selected_file_ids, summary


def _resolve_dynamic_chunk_limit(plan_item: dict[str, Any], full_plan: list[dict[str, Any]], default_limit: int) -> tuple[int, dict[str, Any]]:
    base_limit = max(1, default_limit)
    same_quota_items = [item for item in full_plan if item["allocated_quota"] == plan_item["allocated_quota"]]
    if not same_quota_items:
        return base_limit, {"reason": "default", "safe_avg_chunk_chars": None, "safe_batch_chars": None}

    safe_avg_chunk_chars = min(item["avg_chunk_chars"] for item in same_quota_items)
    safe_batch_chars = safe_avg_chunk_chars * base_limit
    current_avg = max(plan_item["avg_chunk_chars"], 1.0)
    effective_limit = max(1, min(base_limit, int(safe_batch_chars // current_avg) or 1))
    return effective_limit, {
        "reason": "dynamic_chars",
        "safe_avg_chunk_chars": round(safe_avg_chunk_chars, 1),
        "safe_batch_chars": round(safe_batch_chars, 1),
        "current_avg_chunk_chars": round(plan_item["avg_chunk_chars"], 1),
    }


def _to_rows(testset: Any) -> list[dict[str, Any]]:
    if hasattr(testset, "to_pandas"):
        return testset.to_pandas().to_dict(orient="records")
    rows = getattr(testset, "samples", testset)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            normalized.append(row.model_dump())
        elif isinstance(row, dict):
            normalized.append(row)
        else:
            normalized.append(row.__dict__)
    return normalized


def _map_reference_doc_ids(contexts: list[str], chunk_lookup: dict[str, list[dict[str, str]]], fallback_parent_doc_ids: list[str]) -> tuple[list[str], list[str]]:
    reference_doc_ids: list[str] = []
    reference_file_ids: list[str] = []
    for context in contexts:
        for match in chunk_lookup.get(_normalize_text(context), []):
            reference_doc_ids.append(match["parent_doc_id"])
            reference_file_ids.append(match["file_id"])
    reference_doc_ids = _ordered_unique(reference_doc_ids) or list(fallback_parent_doc_ids)
    reference_file_ids = _ordered_unique(reference_file_ids)
    return reference_doc_ids, reference_file_ids


def _iter_sub_batches(plan_item: dict[str, Any], effective_chunk_limit: int) -> list[dict[str, Any]]:
    chunk_limit = max(1, effective_chunk_limit)
    chunks = list(plan_item["chunks"])
    parent_ids = list(plan_item["selected_parent_doc_ids"])
    char_lengths = list(plan_item["chunk_char_lengths"])
    sub_batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(chunks), chunk_limit), start=1):
        batch_chunks = chunks[start:start + chunk_limit]
        batch_parent_ids = parent_ids[start:start + chunk_limit]
        batch_char_lengths = char_lengths[start:start + chunk_limit]
        sub_batches.append(
            {
                **plan_item,
                "chunks": batch_chunks,
                "selected_parent_doc_ids": batch_parent_ids,
                "chunk_char_lengths": batch_char_lengths,
                "allocated_quota": len(batch_chunks),
                "total_chunk_chars": sum(batch_char_lengths),
                "avg_chunk_chars": sum(batch_char_lengths) / len(batch_char_lengths),
                "max_chunk_chars": max(batch_char_lengths),
                "sub_batch_index": batch_index,
                "sub_batch_count": math.ceil(len(chunks) / chunk_limit),
            }
        )
    return sub_batches


def _generate_samples_for_plan_item(
    embeddings: Any,
    model_name: str,
    plan_item: dict[str, Any],
    args: argparse.Namespace,
    global_scope_file_ids: list[str],
) -> list[EvalSample]:
    llm = init_chat_model(model_name)
    generator = _resolve_testset_generator(llm, embeddings)
    testset = _generate_with_chunks(generator, plan_item["chunks"], plan_item["allocated_quota"])
    rows = _to_rows(testset)

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

        sample = EvalSample(
            user_input=str(user_input),
            reference_answer=str(reference).strip() if reference else None,
            reference_contexts=contexts,
            reference_doc_ids=reference_doc_ids,
            scope_file_ids=list(global_scope_file_ids),
            difficulty_level=args.difficulty,
            scenario_type=args.scenario,
            source_type="synthetic",
            tags=[args.category, "synthetic"],
            review_status="pending",
            metadata={
                "query_type": row.get("query_type") or row.get("evolution_type"),
                "persona": row.get("persona"),
                "synthesizer": row.get("synthesizer_name") or row.get("synthesizer"),
                "reference_file_ids": reference_file_ids or [plan_item["file_id"]],
                "source_file_id": plan_item["file_id"],
                "source_file_name": plan_item["file_name"],
                "selected_parent_doc_ids": list(plan_item["selected_parent_doc_ids"]),
                "allocated_quota": plan_item["allocated_quota"],
                "sub_batch_index": plan_item.get("sub_batch_index"),
                "sub_batch_count": plan_item.get("sub_batch_count"),
                "avg_chunk_chars": round(plan_item["avg_chunk_chars"], 1),
                "max_chunk_chars": plan_item["max_chunk_chars"],
                "topup_round": plan_item.get("topup_round"),
            },
        )
        samples.append(sample)
    return samples


def _generate_samples_with_retry(
    embeddings: Any,
    model_name: str,
    plan_item: dict[str, Any],
    args: argparse.Namespace,
    global_scope_file_ids: list[str],
) -> list[EvalSample]:
    max_attempts = max(1, args.max_batch_retries)
    last_exc: Exception | None = None
    batch_label = f"{plan_item.get('sub_batch_index', 1)}/{plan_item.get('sub_batch_count', 1)}"
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(
                    f"DATASET BUILD: retrying synthetic batch file_id={plan_item['file_id']} file_name={plan_item['file_name']} sub_batch={batch_label} attempt={attempt}/{max_attempts}"
                )
            return _generate_samples_for_plan_item(
                embeddings=embeddings,
                model_name=model_name,
                plan_item=plan_item,
                args=args,
                global_scope_file_ids=global_scope_file_ids,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = max(args.retry_backoff_seconds, 0.0) * attempt
            print(
                f"DATASET BUILD: synthetic batch failed file_id={plan_item['file_id']} file_name={plan_item['file_name']} sub_batch={batch_label} attempt={attempt}/{max_attempts} error={exc.__class__.__name__}: {exc}. retry_in={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"Synthetic generation failed for file_id={plan_item['file_id']} file_name={plan_item['file_name']} sub_batch={batch_label} quota={plan_item['allocated_quota']} after {max_attempts} attempts"
    ) from last_exc


def _initial_generate(
    generation_plan: list[dict[str, Any]],
    embeddings: Any,
    model_name: str,
    args: argparse.Namespace,
    global_scope_file_ids: list[str],
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for plan_item in generation_plan:
        effective_chunk_limit, decision = _resolve_dynamic_chunk_limit(plan_item, generation_plan, args.max_chunks_per_batch)
        plan_item["effective_chunk_limit"] = effective_chunk_limit
        plan_item["dynamic_batch_decision"] = decision
        sub_batches = _iter_sub_batches(plan_item, effective_chunk_limit)
        if len(sub_batches) > 1:
            print(
                f"DATASET BUILD: splitting file_id={plan_item['file_id']} file_name={plan_item['file_name']} quota={plan_item['allocated_quota']} into {len(sub_batches)} sub-batches (effective_chunk_limit={effective_chunk_limit}, avg_chunk_chars={plan_item['avg_chunk_chars']:.1f}, safe_avg_chunk_chars={decision.get('safe_avg_chunk_chars')})"
            )
        for sub_batch in sub_batches:
            samples.extend(
                _generate_samples_with_retry(
                    embeddings=embeddings,
                    model_name=model_name,
                    plan_item=sub_batch,
                    args=args,
                    global_scope_file_ids=global_scope_file_ids,
                )
            )
    return samples


def _allocate_topup_counts(generation_plan: list[dict[str, Any]], remaining: int) -> dict[str, int]:
    if remaining <= 0:
        return {}
    file_ids = [item["file_id"] for item in generation_plan]
    weights = [max(item["allocated_quota"], 1) for item in generation_plan]
    total = sum(weights)
    allocations = {file_id: 0 for file_id in file_ids}
    assigned = 0
    for item, weight in zip(generation_plan, weights):
        count = int(math.floor(remaining * weight / total))
        allocations[item["file_id"]] = count
        assigned += count
    order = sorted(generation_plan, key=lambda item: item["allocated_quota"], reverse=True)
    idx = 0
    while assigned < remaining and order:
        allocations[order[idx % len(order)]["file_id"]] += 1
        assigned += 1
        idx += 1
    return allocations


def _sample_additional_chunks(plan_item: dict[str, Any], count: int, seed: int) -> dict[str, Any] | None:
    from app.core.vector_store import VectorStoreFactory

    if count <= 0:
        return None

    rng = random.Random(seed)
    all_parent_ids = list(plan_item["all_parent_doc_ids"])
    used_parent_ids = set(plan_item["used_parent_doc_ids"])
    remaining_parent_ids = [parent_id for parent_id in all_parent_ids if parent_id not in used_parent_ids]
    if len(remaining_parent_ids) >= count:
        chosen_parent_ids = rng.sample(remaining_parent_ids, count)
    else:
        chosen_parent_ids = list(remaining_parent_ids)
        fallback_pool = list(all_parent_ids)
        while len(chosen_parent_ids) < count and fallback_pool:
            chosen_parent_ids.append(rng.choice(fallback_pool))

    if not chosen_parent_ids:
        return None

    docstore = VectorStoreFactory.init_docstore()
    parent_docs = docstore.mget(chosen_parent_ids)
    chunk_records: list[dict[str, Any]] = []
    valid_parent_ids: list[str] = []
    chunks: list[Any] = []
    chunk_char_lengths: list[int] = []
    for index, (parent_id, parent_doc) in enumerate(zip(chosen_parent_ids, parent_docs)):
        if parent_doc is None or not getattr(parent_doc, "page_content", ""):
            continue
        text = str(parent_doc.page_content)
        parent_doc.metadata = dict(parent_doc.metadata or {})
        parent_doc.metadata.update(
            {
                "file_id": plan_item["file_id"],
                "file_name": plan_item["file_name"],
                "source": "synthetic_parent_docs_topup",
                "parent_doc_id": parent_id,
                "parent_rank_in_sample": index,
                "sample_seed": seed,
            }
        )
        chunks.append(parent_doc)
        valid_parent_ids.append(parent_id)
        chunk_char_lengths.append(len(text))
        chunk_records.append(
            {
                "parent_doc_id": parent_id,
                "file_id": plan_item["file_id"],
                "file_name": plan_item["file_name"],
                "page_content": text,
            }
        )

    if not chunks:
        return None

    plan_item["used_parent_doc_ids"].update(valid_parent_ids)
    return {
        **plan_item,
        "selected_parent_doc_ids": valid_parent_ids,
        "chunks": chunks,
        "chunk_lookup": _build_chunk_lookup(chunk_records),
        "chunk_char_lengths": chunk_char_lengths,
        "allocated_quota": len(chunks),
        "total_chunk_chars": sum(chunk_char_lengths),
        "avg_chunk_chars": sum(chunk_char_lengths) / len(chunk_char_lengths),
        "max_chunk_chars": max(chunk_char_lengths),
    }


def _topup_samples(
    samples: list[EvalSample],
    generation_plan: list[dict[str, Any]],
    embeddings: Any,
    model_name: str,
    args: argparse.Namespace,
    global_scope_file_ids: list[str],
) -> list[EvalSample]:
    round_index = 0
    while len(samples) < args.size and round_index < max(1, args.max_topup_rounds):
        round_index += 1
        before_round = len(samples)
        remaining = args.size - len(samples)
        allocations = _allocate_topup_counts(generation_plan, remaining)
        print(f"DATASET BUILD: top-up round={round_index} current_samples={len(samples)} target={args.size} remaining={remaining} allocations={allocations}")
        for plan_item in generation_plan:
            topup_count = allocations.get(plan_item["file_id"], 0)
            if topup_count <= 0:
                continue
            extra_plan = _sample_additional_chunks(plan_item, topup_count, seed=args.seed + round_index * 1000 + topup_count)
            if extra_plan is None:
                continue
            effective_chunk_limit = plan_item.get("effective_chunk_limit", args.max_chunks_per_batch)
            sub_batches = _iter_sub_batches(extra_plan, effective_chunk_limit)
            for sub_batch in sub_batches:
                sub_batch["topup_round"] = round_index
                new_samples = _generate_samples_with_retry(
                    embeddings=embeddings,
                    model_name=model_name,
                    plan_item=sub_batch,
                    args=args,
                    global_scope_file_ids=global_scope_file_ids,
                )
                if not new_samples:
                    continue
                remaining_capacity = args.size - len(samples)
                samples.extend(new_samples[:remaining_capacity])
                if len(samples) >= args.size:
                    break
            if len(samples) >= args.size:
                break
        if len(samples) == before_round:
            break
    return samples


def main() -> None:
    args = _parse_args()
    init_eval_runtime("dataset_synthetic")
    try:
        from app.core.embeddings import EmbeddingModelFactory

        chat_cfg = global_config.get("chat_model", {})
        model_name = chat_cfg.get("light") if args.use_light_model else chat_cfg.get("default")
        embeddings = EmbeddingModelFactory.init_embedding_model()
        generation_plan, scope_file_ids, plan_summary = _load_generation_plan(
            doc_limit=args.doc_limit,
            size=args.size,
            seed=args.seed,
            recency_tau_days=args.recency_tau_days,
            alloc_alpha=args.alloc_alpha,
        )

        samples = _initial_generate(
            generation_plan=generation_plan,
            embeddings=embeddings,
            model_name=model_name,
            args=args,
            global_scope_file_ids=scope_file_ids,
        )

        if len(samples) < args.size:
            samples = _topup_samples(
                samples=samples,
                generation_plan=generation_plan,
                embeddings=embeddings,
                model_name=model_name,
                args=args,
                global_scope_file_ids=scope_file_ids,
            )

        if len(samples) > args.size:
            rng = random.Random(args.seed)
            samples = rng.sample(samples, args.size)

        plan_summary["requested_sample_count"] = args.size
        plan_summary["actual_sample_count"] = len(samples)
        plan_summary["undershot_sample_count"] = max(args.size - len(samples), 0)
        plan_summary["effective_chunk_limits"] = {item["file_id"]: item.get("effective_chunk_limit", args.max_chunks_per_batch) for item in generation_plan}
        plan_summary["dynamic_batch_decisions"] = {
            item["file_id"]: item.get("dynamic_batch_decision", {}) for item in generation_plan
        }

        manifest = build_manifest(
            name=args.name,
            version=args.version,
            category=args.category,
            source_type="synthetic",
            description=args.description or "Synthetic dataset built with RAGAS TestsetGenerator.",
            metadata={
                "model_name": model_name,
                "seed": args.seed,
                "doc_limit": args.doc_limit,
                "recency_tau_days": args.recency_tau_days,
                "alloc_alpha": args.alloc_alpha,
                "scope_file_ids": scope_file_ids,
                "generation_plan": plan_summary,
                "max_batch_retries": args.max_batch_retries,
                "retry_backoff_seconds": args.retry_backoff_seconds,
                "max_chunks_per_batch": args.max_chunks_per_batch,
                "max_topup_rounds": args.max_topup_rounds,
                "review_note": "Synthetic datasets require manual review before they are promoted into baseline or regression usage.",
            },
        )
        output_dir = save_dataset(manifest, samples, output_dir=args.output_dir or None)
        print(f"DATASET BUILD: synthetic dataset saved to {output_dir}")
        print(f"DATASET BUILD: selected_files={plan_summary['selected_doc_count']} selected_parent_chunks={plan_summary['selected_parent_chunk_count']} actual_samples={len(samples)} requested_samples={args.size}")
        if len(samples) < args.size:
            print(
                f"DATASET BUILD: warning target sample count not fully reached requested={args.size} actual={len(samples)}"
            )
        for line in format_build_report(manifest, samples):
            print(line)
        print("DATASET BUILD: synthetic datasets should be manually reviewed before any non-exploration use.")
    finally:
        close_eval_runtime()


if __name__ == "__main__":
    main()
