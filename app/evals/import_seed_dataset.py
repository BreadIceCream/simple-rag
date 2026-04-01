from __future__ import annotations

import argparse
from pathlib import Path

from app.evals.dataset_builder import build_manifest, format_build_report, save_dataset
from app.evals.schema import EvalSample, read_json, read_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import curated seed data into dataset format.")
    parser.add_argument("--input", required=True, help="input .json or .jsonl file")
    parser.add_argument("--name", required=True, help="dataset name")
    parser.add_argument("--version", required=True, help="dataset version")
    parser.add_argument("--category", default="baseline", choices=["regression", "baseline", "exploration", "specialized", "synthetic"], help="dataset category")
    parser.add_argument("--source-type", default="manual", help="source type label")
    parser.add_argument("--default-difficulty", default="unknown", help="default difficulty label")
    parser.add_argument("--default-scenario", default="single_turn", help="default scenario label")
    parser.add_argument("--default-scope", choices=["none", "all"], default="none", help="fill scope_file_ids when input does not provide them")
    parser.add_argument("--description", default="", help="dataset description")
    parser.add_argument("--output-dir", default="", help="optional explicit dataset directory")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return payload["samples"]
    raise ValueError(f"Unsupported seed payload format: {path}")


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _infer_scope_file_ids(reference_doc_ids: list[str]) -> list[str]:
    from app.evals.runtime import get_parent_chunks_by_ids

    file_ids: list[str] = []
    for _, parent_doc in get_parent_chunks_by_ids(reference_doc_ids):
        file_id = str((getattr(parent_doc, "metadata", {}) or {}).get("file_id") or "").strip()
        if file_id:
            file_ids.append(file_id)
    return _ordered_unique(file_ids)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    rows = _load_rows(input_path)

    needs_runtime = args.default_scope == "all"
    if not needs_runtime:
        for row in rows:
            reference_doc_ids = list(row.get("reference_doc_ids") or [])
            scope_file_ids = list(row.get("scope_file_ids") or [])
            if reference_doc_ids and not scope_file_ids:
                needs_runtime = True
                break

    default_scope: list[str] = []
    runtime_started = False
    if needs_runtime:
        from app.evals.runtime import close_eval_runtime, get_all_document_ids, init_eval_runtime

        init_eval_runtime("dataset_seed")
        runtime_started = True
        if args.default_scope == "all":
            default_scope = get_all_document_ids()

    try:
        samples: list[EvalSample] = []
        inferred_scope_count = 0
        for row in rows:
            reference_doc_ids = list(row.get("reference_doc_ids") or [])
            scope_file_ids = list(row.get("scope_file_ids") or [])
            inferred_scope_file_ids: list[str] = []
            if not scope_file_ids and reference_doc_ids:
                inferred_scope_file_ids = _infer_scope_file_ids(reference_doc_ids)
                scope_file_ids = inferred_scope_file_ids
                if inferred_scope_file_ids:
                    inferred_scope_count += 1
            if not scope_file_ids:
                scope_file_ids = list(default_scope)

            metadata = dict(row.get("metadata") or {})
            if inferred_scope_file_ids and not metadata.get("reference_file_ids"):
                metadata["reference_file_ids"] = list(inferred_scope_file_ids)
            if len(scope_file_ids) == 1 and not metadata.get("source_file_id"):
                metadata["source_file_id"] = scope_file_ids[0]
            if inferred_scope_file_ids:
                metadata["scope_inferred_from_reference_doc_ids"] = True

            sample = EvalSample.from_dict(
                {
                    "user_input": row.get("user_input") or row.get("question"),
                    "sample_id": row.get("sample_id") or row.get("id"),
                    "reference_answer": row.get("reference_answer") or row.get("reference"),
                    "reference_contexts": row.get("reference_contexts") or [],
                    "reference_doc_ids": reference_doc_ids,
                    "scope_file_ids": scope_file_ids,
                    "difficulty_level": row.get("difficulty_level") or args.default_difficulty,
                    "scenario_type": row.get("scenario_type") or args.default_scenario,
                    "source_type": row.get("source_type") or args.source_type,
                    "query_type": row.get("query_type"),
                    "hop_count": row.get("hop_count"),
                    "abstraction_level": row.get("abstraction_level"),
                    "evidence_topology": row.get("evidence_topology"),
                    "reasoning_hops": row.get("reasoning_hops") or [],
                    "query_type_source": row.get("query_type_source"),
                    "tags": row.get("tags") or [args.category, args.source_type],
                    "review_status": row.get("review_status") or "pending",
                    "review_notes": row.get("review_notes") or "",
                    "metadata": metadata,
                    "rubric": row.get("rubric"),
                    "reference_tool_calls": row.get("reference_tool_calls") or [],
                }
            )
            samples.append(sample)

        manifest = build_manifest(
            name=args.name,
            version=args.version,
            category=args.category,
            source_type=args.source_type,
            description=args.description or f"Imported dataset from {input_path.name}.",
            metadata={
                "input_path": str(input_path),
                "default_scope": args.default_scope,
                "inferred_scope_count": inferred_scope_count,
                "review_note": "Imported seed datasets should be reviewed before production use unless they come from a trusted annotated source.",
            },
        )
        output_dir = save_dataset(manifest, samples, output_dir=args.output_dir or None)
        print(f"DATASET BUILD: imported dataset saved to {output_dir}")
        for line in format_build_report(manifest, samples):
            print(line)
    finally:
        if runtime_started:
            close_eval_runtime()


if __name__ == "__main__":
    main()

