from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.evals.offline.dataset_builder import load_dataset
from app.evals.offline.reporter import new_run_dir, write_run_artifacts
from app.evals.offline.runtime import close_eval_runtime, get_documents_by_ids, init_eval_runtime
from app.evals.offline.schema import EvalRunRecord


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute real RAG runs for an evaluation dataset.")
    parser.add_argument("--dataset-dir", required=True, help="dataset directory containing manifest.json and samples.jsonl")
    parser.add_argument("--output-root", default="", help="optional experiment output root")
    parser.add_argument("--limit", type=int, default=0, help="optional max sample count")
    parser.add_argument(
        "--review-status",
        default="approved,pending",
        help="comma-separated allowed review statuses; default=approved,pending",
    )
    return parser.parse_args()


def _set_scope_file_ids(file_ids: list[str]) -> None:
    from app.core.retriever import EnhancedParentDocumentRetrieverFactory, HybridPDRetrieverFactory

    hybrid_retriever = HybridPDRetrieverFactory.get_instance()
    if not file_ids:
        hybrid_retriever.reset_file_ids(set(), [{}], None)
        return
    docs = get_documents_by_ids(file_ids)
    if not docs:
        raise ValueError("No documents found for scope_file_ids.")
    pd_retriever = EnhancedParentDocumentRetrieverFactory.get_instance()
    all_child_docs = []
    file_infos = []
    valid_file_ids: list[str] = []
    for doc in docs:
        if not doc.parent_doc_ids:
            continue
        valid_file_ids.append(doc.id)
        file_infos.append(
            {
                "file_id": doc.id,
                "file_name": doc.file_name if doc.file_name else doc.path,
                "file_summary": doc.file_summary,
            }
        )
        all_child_docs.extend(pd_retriever.get_child_docs(doc.parent_doc_ids))
    if not valid_file_ids or not all_child_docs:
        raise ValueError("scope_file_ids do not contain any retrievable documents.")
    hybrid_retriever.reset_file_ids(set(valid_file_ids), file_infos, all_child_docs)


def _run_single_sample(dataset_name: str, dataset_version: str, category: str, sample) -> EvalRunRecord:
    from app.core.graph import Graph

    graph = Graph.get_compiled_graph()
    thread_id = f"eval-{dataset_name}-{sample.sample_id}"
    config = {"configurable": {"thread_id": thread_id}}
    actual_doc_ids: list[str] = []
    actual_file_ids: list[str] = []
    actual_contexts: list[str] = []
    actual_response = ""
    latency_ms: float | None = None
    rewrite_count: int | None = None
    generate_count: int | None = None
    error_message: str | None = None
    status = "success"

    start = time.perf_counter()
    try:
        _set_scope_file_ids(sample.scope_file_ids)
        for mode, chunk in graph.stream(
            {"messages": [HumanMessage(content=sample.user_input)]},
            config,
            stream_mode=["updates"],
        ):
            if mode != "updates":
                continue
            for _, updates in chunk.items():
                if not isinstance(updates, dict):
                    continue
                for doc in updates.get("documents", []):
                    doc_id = str(getattr(doc, "id", "") or "").strip()
                    file_id = str((getattr(doc, "metadata", {}) or {}).get("file_id") or "").strip()
                    page_content = str(getattr(doc, "page_content", "") or "").strip()
                    if doc_id and doc_id not in actual_doc_ids:
                        actual_doc_ids.append(doc_id)
                    if file_id and file_id not in actual_file_ids:
                        actual_file_ids.append(file_id)
                    if page_content:
                        actual_contexts.append(page_content)
                for message in updates.get("messages", []):
                    if isinstance(message, AIMessage) and message.content:
                        actual_response = str(message.content)
        state_snapshot = graph.get_state(config)
        values = getattr(state_snapshot, "values", {}) or {}
        rewrite_count = values.get("rewrite_count")
        generate_count = values.get("generate_count")
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        try:
            Graph.delete_thread(thread_id)
        except Exception:
            pass

    return EvalRunRecord(
        sample_id=sample.sample_id,
        user_input=sample.user_input,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        category=category,
        source_type=sample.source_type,
        scope_file_ids=list(sample.scope_file_ids),
        query_type=sample.query_type,
        hop_count=sample.hop_count,
        abstraction_level=sample.abstraction_level,
        evidence_topology=sample.evidence_topology,
        reasoning_hops=[dict(item) for item in sample.reasoning_hops],
        query_type_source=sample.query_type_source,
        actual_response=actual_response,
        actual_contexts=actual_contexts,
        actual_doc_ids=actual_doc_ids,
        actual_file_ids=actual_file_ids,
        reference_answer=sample.reference_answer,
        reference_contexts=list(sample.reference_contexts),
        reference_doc_ids=list(sample.reference_doc_ids),
        capabilities=list(sample.capabilities),
        difficulty_level=sample.difficulty_level,
        scenario_type=sample.scenario_type,
        tags=list(sample.tags),
        review_status=sample.review_status,
        latency_ms=latency_ms,
        rewrite_count=rewrite_count,
        generate_count=generate_count,
        status=status,
        error_message=error_message,
        metadata=dict(sample.metadata),
    )


def main() -> None:
    args = _parse_args()
    init_eval_runtime()
    try:
        manifest, samples = load_dataset(args.dataset_dir)
        allowed_status = {item.strip() for item in args.review_status.split(",") if item.strip()}
        selected_samples = [sample for sample in samples if sample.review_status in allowed_status]
        if args.limit > 0:
            selected_samples = selected_samples[: args.limit]
        if not selected_samples:
            raise ValueError("No samples matched the requested review_status filter.")

        run_id, run_dir = new_run_dir(args.output_root or None)
        records: list[EvalRunRecord] = []
        for index, sample in enumerate(selected_samples, start=1):
            record = _run_single_sample(manifest.name, manifest.version, manifest.category, sample)
            records.append(record)
            print(
                f"LIVE RUN: processed {index}/{len(selected_samples)} sample_id={sample.sample_id} status={record.status} latency_ms={record.latency_ms}"
            )

        config_snapshot: dict[str, Any] = {
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "review_status": sorted(allowed_status),
            "selected_sample_count": len(selected_samples),
            "runner": "live_rag_runner",
        }
        write_run_artifacts(
            run_dir=run_dir,
            manifest=manifest,
            records=records,
            config_snapshot=config_snapshot,
            source_label="live_rag",
        )
        print(f"LIVE RUN: run_id={run_id}")
        print(f"LIVE RUN: artifacts={run_dir}")
    finally:
        close_eval_runtime()


if __name__ == "__main__":
    main()
