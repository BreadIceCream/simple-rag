from __future__ import annotations

import argparse
import random
import re
import time
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from sqlalchemy import select

from app.config.db_config import DatabaseManager
from app.config.global_config import global_config
from app.evals.offline.dataset_builder import build_manifest, format_build_report, save_dataset
from app.evals.offline.runtime import close_eval_runtime, init_eval_runtime
from app.evals.offline.schema import EvalSample
from app.models.schemas import ChatHistory

_REFERENCE_PROMPT = (
    "You are building a candidate evaluation reference answer for a RAG dataset. "
    "Given the user question and supporting contexts, produce a concise, factual answer in the same language as the user. "
    "Do not mention the contexts explicitly. If the contexts are insufficient, say you do not know.\n\n"
    "Question:\n{question}\n\n"
    "Contexts:\n{contexts}\n"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build replay datasets from chat_history.")
    parser.add_argument("--name", required=True, help="dataset name")
    parser.add_argument("--version", required=True, help="dataset version")
    parser.add_argument("--category", default="baseline", choices=["regression", "baseline", "exploration", "specialized"], help="dataset category")
    parser.add_argument("--limit", type=int, default=100, help="max number of samples")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible sampling")
    parser.add_argument("--reference-mode", choices=["history", "ai"], default="history", help="how to generate candidate reference answers")
    parser.add_argument("--difficulty", default="unknown", help="default difficulty label")
    parser.add_argument("--scenario", default="single_turn", help="default scenario label")
    parser.add_argument("--description", default="", help="dataset description")
    parser.add_argument("--output-dir", default="", help="optional explicit dataset directory")
    parser.add_argument("--llm-retries", type=int, default=3, help="retry count when generating AI reference answers")
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0, help="base backoff seconds between AI reference answer retries")
    parser.add_argument("--classify-query-type", action="store_true", help="classify replay samples into query types")
    parser.add_argument("--query-type-mode", choices=["heuristic"], default="heuristic", help="query type classifier mode")
    return parser.parse_args()


def _load_pairs() -> list[dict[str, Any]]:
    from app.core.retriever import EnhancedParentDocumentRetrieverFactory

    with DatabaseManager.get_sync_db() as db:
        rows = db.execute(
            select(ChatHistory).order_by(ChatHistory.conversation_id.asc(), ChatHistory.created_at.asc())
        ).scalars().all()

    pd_retriever = EnhancedParentDocumentRetrieverFactory.get_instance()
    grouped: dict[str, list[ChatHistory]] = {}
    for row in rows:
        grouped.setdefault(row.conversation_id, []).append(row)

    pairs: list[dict[str, Any]] = []
    for conversation_id, messages in grouped.items():
        pending_user: ChatHistory | None = None
        for message in messages:
            if message.role == "user":
                pending_user = message
                continue
            if message.role != "ai" or pending_user is None:
                continue
            parent_doc_ids = list(message.parent_doc_ids or [])
            if not parent_doc_ids:
                pending_user = None
                continue
            parent_docs = [doc for _, doc in pd_retriever.get_parent_docs(parent_doc_ids)]
            if not parent_docs:
                pending_user = None
                continue
            scope_file_ids: list[str] = []
            for doc in parent_docs:
                file_id = str(doc.metadata.get("file_id") or "").strip()
                if file_id and file_id not in scope_file_ids:
                    scope_file_ids.append(file_id)
            if not scope_file_ids:
                pending_user = None
                continue
            pairs.append(
                {
                    "conversation_id": conversation_id,
                    "user_message_id": pending_user.id,
                    "ai_message_id": message.id,
                    "user_input": pending_user.content,
                    "history_answer": message.content,
                    "reference_contexts": [doc.page_content for doc in parent_docs if getattr(doc, "page_content", "")],
                    "reference_doc_ids": parent_doc_ids,
                    "scope_file_ids": scope_file_ids,
                    "reference_file_ids": scope_file_ids,
                    "source_file_count": len(scope_file_ids),
                }
            )
            pending_user = None
    return pairs


def _generate_reference_answers(samples: list[EvalSample], retries: int, retry_backoff_seconds: float) -> None:
    if not samples:
        return
    chat_cfg = global_config.get("chat_model", {})
    llm = init_chat_model(chat_cfg.get("default"))
    max_attempts = max(1, retries)

    for sample in samples:
        prompt = _REFERENCE_PROMPT.format(
            question=sample.user_input,
            contexts="\n\n".join(sample.reference_contexts),
        )
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                sample.reference_answer = str(getattr(response, "content", "")).strip() or sample.reference_answer
                sample.metadata["reference_answer_source"] = "ai"
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Failed to generate replay reference answer for sample_id={sample.sample_id} after {max_attempts} attempts"
                    ) from exc
                sleep_seconds = max(retry_backoff_seconds, 0.0) * attempt
                print(
                    f"DATASET BUILD: replay reference answer generation failed sample_id={sample.sample_id} attempt={attempt}/{max_attempts} error={exc.__class__.__name__}: {exc}. retry_in={sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)


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


def _classify_query_type(args: argparse.Namespace, pair: dict[str, Any]) -> tuple[str, str, str, str, float, list[str]]:
    if not args.classify_query_type:
        return "unknown", "unknown", "unknown", "unknown", 0.0, []
    if args.query_type_mode != "heuristic":
        raise ValueError(f"Unsupported query type mode: {args.query_type_mode}")
    query_type, hop_count, abstraction_level, confidence, reasons = _detect_query_type_heuristic(
        user_input=str(pair.get("user_input") or ""),
        reference_doc_ids=list(pair.get("reference_doc_ids") or []),
        scope_file_ids=list(pair.get("scope_file_ids") or []),
    )
    return query_type, hop_count, abstraction_level, "heuristic", confidence, reasons


def main() -> None:
    args = _parse_args()
    init_eval_runtime("dataset_replay")
    try:
        pairs = _load_pairs()
        if not pairs:
            raise ValueError("No replay candidates with retrieval contexts were found in chat_history.")
        rng = random.Random(args.seed)
        if len(pairs) > args.limit:
            pairs = rng.sample(pairs, args.limit)

        samples: list[EvalSample] = []
        for pair in pairs:
            query_type, hop_count, abstraction_level, query_type_source, query_type_confidence, query_type_reasons = _classify_query_type(args, pair)
            sample = EvalSample(
                user_input=pair["user_input"],
                reference_answer=pair["history_answer"],
                reference_contexts=pair["reference_contexts"],
                reference_doc_ids=pair["reference_doc_ids"],
                scope_file_ids=pair["scope_file_ids"],
                difficulty_level=args.difficulty,
                scenario_type=args.scenario,
                source_type="replay",
                query_type=query_type,
                hop_count=hop_count,
                abstraction_level=abstraction_level,
                query_type_source=query_type_source,
                tags=[args.category, "replay"],
                review_status="pending",
                metadata={
                    "conversation_id": pair["conversation_id"],
                    "user_message_id": pair["user_message_id"],
                    "ai_message_id": pair["ai_message_id"],
                    "history_answer": pair["history_answer"],
                    "reference_answer_source": "history",
                    "reference_file_ids": pair["reference_file_ids"],
                    "source_file_count": pair["source_file_count"],
                    "source_file_id": pair["reference_file_ids"][0] if len(pair["reference_file_ids"]) == 1 else "",
                    "query_type_mode": args.query_type_mode if args.classify_query_type else "disabled",
                    "query_type_confidence": query_type_confidence,
                    "query_type_reasons": query_type_reasons,
                },
            )
            samples.append(sample)

        if args.reference_mode == "ai":
            _generate_reference_answers(samples, retries=args.llm_retries, retry_backoff_seconds=args.retry_backoff_seconds)

        manifest = build_manifest(
            name=args.name,
            version=args.version,
            category=args.category,
            source_type="replay",
            description=args.description or "Replay dataset built from chat_history.",
            metadata={
                "reference_mode": args.reference_mode,
                "sampling_seed": args.seed,
                "requested_limit": args.limit,
                "llm_retries": args.llm_retries,
                "retry_backoff_seconds": args.retry_backoff_seconds,
                "classify_query_type": args.classify_query_type,
                "query_type_mode": args.query_type_mode,
                "review_note": "Replay datasets require manual review because historical answers may contain errors.",
            },
        )
        output_dir = save_dataset(manifest, samples, output_dir=args.output_dir or None)
        print(f"DATASET BUILD: replay dataset saved to {output_dir}")
        for line in format_build_report(manifest, samples):
            print(line)
        print("DATASET BUILD: replay datasets should be manually reviewed before use in regression or baseline evaluation.")
    finally:
        close_eval_runtime()


if __name__ == "__main__":
    main()
