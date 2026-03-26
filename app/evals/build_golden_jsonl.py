from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.config.db_config import DatabaseManager
from app.config.global_config import global_config
from app.core.embeddings import EmbeddingModelFactory
from app.core.vector_store import VectorStoreFactory
from app.models.schemas import EmbeddedDocument

_RAG_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUT = _RAG_ROOT / "store" / "evals" / "datasets" / "v1.generated.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build golden JSONL dataset from parent_docs via ragas generate_with_chunks.")
    parser.add_argument("--size", type=int, default=100, help="target sample count")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT), help="output jsonl path")
    parser.add_argument("--doc-limit", type=int, default=30, help="max source documents selected for generation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible sampling")
    parser.add_argument(
        "--recency-tau-days",
        type=float,
        default=30.0,
        help="recency decay factor (days) for document weighted random sampling",
    )
    parser.add_argument(
        "--alloc-alpha",
        type=float,
        default=0.7,
        help="allocation exponent for chunk budget by parent_doc count (higher = more bias to large docs)",
    )
    parser.add_argument("--use-light-model", action="store_true", help="use chat_model.light as generator llm")
    parser.add_argument("--fill-response-with-reference", action="store_true", help="set response=reference for immediate P0 compatibility")
    return parser.parse_args()


def _init_runtime() -> None:
    global_config.load()
    DatabaseManager.init()


def _close_runtime() -> None:
    try:
        asyncio.run(DatabaseManager.close())
    except RuntimeError:
        pass


def _safe_weighted_choice_index(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def _weighted_sample_without_replacement(items: list[Any], weights: list[float], k: int, rng: random.Random) -> list[Any]:
    candidates = list(items)
    candidate_weights = list(weights)
    selected: list[Any] = []

    pick_n = min(k, len(candidates))
    for _ in range(pick_n):
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
    recency_weight = math.exp(-age_days / tau)
    return max(recency_weight, 1e-8)


def _allocate_chunk_budget(
    parent_counts: list[int],
    total_budget: int,
    alloc_alpha: float,
    rng: random.Random,
) -> list[int]:
    if not parent_counts:
        return []

    capped_budget = min(max(total_budget, 0), sum(parent_counts))
    if capped_budget <= 0:
        return [0] * len(parent_counts)

    quotas = [0] * len(parent_counts)

    # 预算足够时，先给每个参与文档至少1个父文档块。
    if capped_budget >= len(parent_counts):
        for i, c in enumerate(parent_counts):
            if c > 0:
                quotas[i] = 1

    remaining_budget = capped_budget - sum(quotas)
    alpha = max(alloc_alpha, 0.0)

    while remaining_budget > 0:
        eligible = [i for i, c in enumerate(parent_counts) if quotas[i] < c]
        if not eligible:
            break

        weights = []
        for i in eligible:
            remaining_capacity = parent_counts[i] - quotas[i]
            w = float(remaining_capacity) ** alpha
            weights.append(max(w, 1e-8))

        chosen_local_idx = _safe_weighted_choice_index(weights, rng)
        chosen_i = eligible[chosen_local_idx]
        quotas[chosen_i] += 1
        remaining_budget -= 1

    return quotas


def _load_parent_chunks(
    doc_limit: int,
    size: int,
    seed: int,
    recency_tau_days: float,
    alloc_alpha: float,
) -> list[Any]:
    with DatabaseManager.get_sync_db() as db:
        docs = db.execute(select(EmbeddedDocument)).scalars().all()

    docs = [doc for doc in docs if doc.parent_doc_ids]
    if not docs:
        raise ValueError("No documents with parent_doc_ids found in embedded_document.")

    rng = random.Random(seed)
    now = datetime.now()

    doc_weights = [_doc_sampling_weight(doc, now=now, recency_tau_days=recency_tau_days) for doc in docs]
    # doc-limit 是上限；为确保每个参与文档至少1个父文档块，参与文档数不超过 size。
    selected_doc_count = min(max(doc_limit, 1), len(docs), max(size, 1))
    selected_docs = _weighted_sample_without_replacement(
        items=docs,
        weights=doc_weights,
        k=selected_doc_count,
        rng=rng,
    )

    if not selected_docs:
        raise ValueError("Document weighted sampling returned empty result.")

    parent_counts = [len(doc.parent_doc_ids or []) for doc in selected_docs]
    # 使用 size 作为总块预算，并确保每个参与文档至少可分1个块。
    total_chunk_budget = max(size, len(selected_docs))
    quotas = _allocate_chunk_budget(
        parent_counts=parent_counts,
        total_budget=total_chunk_budget,
        alloc_alpha=alloc_alpha,
        rng=rng,
    )

    docstore = VectorStoreFactory.init_docstore()
    chunks = []

    for doc, quota in zip(selected_docs, quotas):
        parent_ids = list(doc.parent_doc_ids or [])
        if not parent_ids or quota <= 0:
            continue

        take_n = min(quota, len(parent_ids))
        selected_parent_ids = rng.sample(parent_ids, take_n) if take_n < len(parent_ids) else parent_ids

        parent_docs = docstore.mget(selected_parent_ids)
        for index, parent_doc in enumerate(parent_docs):
            if parent_doc is None or not getattr(parent_doc, "page_content", ""):
                continue
            parent_doc.metadata = dict(parent_doc.metadata or {})
            parent_doc.metadata.update(
                {
                    "file_id": doc.id,
                    "file_name": doc.file_name,
                    "file_extension": doc.file_extension,
                    "source": "parent_docs",
                    "parent_rank_in_sample": index,
                    "sample_seed": seed,
                }
            )
            chunks.append(parent_doc)

    if not chunks:
        raise ValueError("No valid parent chunks found after weighted random sampling.")

    print(
        f"BUILD GOLDEN: sampled_docs={len(selected_docs)}, total_chunk_budget={total_chunk_budget}, "
        f"sampled_chunks={len(chunks)}, seed={seed}, recency_tau_days={recency_tau_days}, alloc_alpha={alloc_alpha}"
    )

    return chunks


def _resolve_testset_generator(llm: Any, embeddings: Any):
    try:
        from ragas.testset import TestsetGenerator
    except ImportError as e:
        raise ImportError("Cannot import ragas.testset.TestsetGenerator.") from e

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

    if kwargs:
        return TestsetGenerator(**kwargs)

    return TestsetGenerator(llm_obj, embeddings_obj)


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


def _to_records(testset: Any, fill_response_with_reference: bool) -> list[dict[str, Any]]:
    if hasattr(testset, "to_pandas"):
        rows = testset.to_pandas().to_dict(orient="records")
    elif isinstance(testset, list):
        rows = testset
    else:
        rows = getattr(testset, "samples", None)
        if rows is None:
            raise RuntimeError("Unknown testset structure, cannot extract rows.")

    records: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            data = row.model_dump()
        elif isinstance(row, dict):
            data = row
        else:
            data = row.__dict__

        user_input = data.get("user_input") or data.get("question") or data.get("query")
        reference = data.get("reference") or data.get("ground_truth") or data.get("answer")
        contexts = (
            data.get("reference_contexts")
            or data.get("contexts")
            or data.get("retrieved_contexts")
            or []
        )

        if isinstance(contexts, str):
            contexts = [contexts]

        if not user_input or not contexts:
            continue

        record = {
            "user_input": str(user_input),
            "response": str(reference) if (fill_response_with_reference and reference) else "",
            "retrieved_contexts": [str(c) for c in contexts if str(c).strip()],
            "reference": str(reference) if reference else None,
            "metadata": {
                "source": "ragas_generate_with_chunks",
                "query_type": data.get("query_type") or data.get("evolution_type"),
                "persona": data.get("persona"),
                "synthesizer": data.get("synthesizer_name") or data.get("synthesizer"),
            },
        }

        if not record["retrieved_contexts"]:
            continue

        records.append(record)

    return records


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    _init_runtime()

    try:
        from langchain.chat_models import init_chat_model
    except ImportError as e:
        raise ImportError("Cannot import langchain.chat_models.init_chat_model") from e

    try:
        chat_cfg = global_config.get("chat_model", {})
        model_name = chat_cfg.get("light") if args.use_light_model else chat_cfg.get("default")
        llm = init_chat_model(model_name)
        embeddings = EmbeddingModelFactory.init_embedding_model()

        chunks = _load_parent_chunks(
            doc_limit=args.doc_limit,
            size=args.size,
            seed=args.seed,
            recency_tau_days=args.recency_tau_days,
            alloc_alpha=args.alloc_alpha,
        )
        generator = _resolve_testset_generator(llm=llm, embeddings=embeddings)

        print(f"BUILD GOLDEN: source_chunks={len(chunks)}, target_size={args.size}, model={model_name}")
        testset = _generate_with_chunks(generator, chunks, args.size)
        records = _to_records(testset, fill_response_with_reference=args.fill_response_with_reference)

        if not records:
            raise RuntimeError("generate_with_chunks returned no valid records.")

        output_path = Path(args.output)
        _write_jsonl(records, output_path)
        print(f"BUILD GOLDEN: done. output={output_path}, count={len(records)}")

    finally:
        _close_runtime()


if __name__ == "__main__":
    main()

