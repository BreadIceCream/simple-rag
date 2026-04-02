from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from app.evals.offline.schema import (
    EvalDatasetManifest,
    EvalSample,
    dataset_dir,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

_MANIFEST_FILE = "manifest.json"
_SAMPLES_FILE = "samples.jsonl"
_REVIEW_FILE = "review_sheet.csv"
_REVIEW_GUIDE_FILE = "review_guide.md"


def review_required_for_category(category: str) -> bool:
    return category in {"regression", "baseline", "specialized", "synthetic"}


def review_instructions_for(category: str, source_type: str) -> list[str]:
    instructions = [
        "Check whether the user_input is clear, answerable, and scoped to the selected files.",
        "Check whether reference_answer is correct, concise, and directly answers the question.",
        "If reference_contexts/reference_doc_ids are present, verify they really support the answer.",
        "Mark review_status as approved, rejected, or needs_revision before using the dataset in evaluation.",
    ]
    if source_type == "synthetic":
        instructions.extend(
            [
                "Synthetic samples must be reviewed before they are promoted into baseline or regression usage.",
                "Reject samples that look templated, ambiguous, trivially answerable, or distributionally unrealistic.",
            ]
        )
    if category == "regression":
        instructions.extend(
            [
                "Regression samples should represent real failures or critical business paths.",
                "Do not include speculative or weakly grounded samples in the regression set.",
            ]
        )
    if category == "baseline":
        instructions.append("Baseline samples should be representative of normal usage, not only edge cases.")
    if category == "exploration":
        instructions.append("Exploration samples can keep harder or more speculative cases, but still require factual sanity checks.")
    return instructions


def infer_manifest_capabilities(samples: list[EvalSample]) -> list[str]:
    if not samples:
        return []
    capability_sets = [set(sample.capabilities) for sample in samples]
    shared = set.intersection(*capability_sets) if capability_sets else set()
    return sorted(shared)


def validate_samples(samples: list[EvalSample]) -> list[EvalSample]:
    normalized: list[EvalSample] = []
    seen_ids: set[str] = set()
    for sample in samples:
        sample.normalize()
        if not sample.user_input.strip():
            raise ValueError("Sample user_input cannot be empty.")
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id detected: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        normalized.append(sample)
    return normalized


def _build_report_payload(manifest: EvalDatasetManifest, samples: list[EvalSample]) -> dict[str, Any]:
    review_counts = Counter(sample.review_status for sample in samples)
    source_counts = Counter(sample.source_type for sample in samples)
    scenario_counts = Counter(sample.scenario_type for sample in samples)
    difficulty_counts = Counter(sample.difficulty_level for sample in samples)

    unique_scope_file_ids: set[str] = set()
    unique_reference_doc_ids: set[str] = set()
    file_coverage = Counter()
    for sample in samples:
        unique_scope_file_ids.update(sample.scope_file_ids)
        unique_reference_doc_ids.update(sample.reference_doc_ids)

        reference_file_ids = sample.metadata.get("reference_file_ids") or []
        if isinstance(reference_file_ids, str):
            reference_file_ids = [reference_file_ids]
        reference_file_ids = [str(item).strip() for item in reference_file_ids if str(item).strip()]
        if not reference_file_ids:
            source_file_id = str(sample.metadata.get("source_file_id") or "").strip()
            if source_file_id:
                reference_file_ids = [source_file_id]
        if not reference_file_ids and len(sample.scope_file_ids) == 1:
            reference_file_ids = list(sample.scope_file_ids)
        for file_id in sorted(set(reference_file_ids)):
            file_coverage[file_id] += 1

    sample_count = len(samples)
    return {
        "dataset": f"{manifest.name}:{manifest.version}",
        "category": manifest.category,
        "source_type": manifest.source_type,
        "sample_count": sample_count,
        "capabilities": list(manifest.capabilities),
        "with_reference_answer": sum(1 for sample in samples if str(sample.reference_answer or "").strip()),
        "with_reference_contexts": sum(1 for sample in samples if sample.reference_contexts),
        "with_reference_doc_ids": sum(1 for sample in samples if sample.reference_doc_ids),
        "with_scope_file_ids": sum(1 for sample in samples if sample.scope_file_ids),
        "unique_scope_file_count": len(unique_scope_file_ids),
        "unique_reference_doc_count": len(unique_reference_doc_ids),
        "review_status_counts": dict(sorted(review_counts.items())),
        "source_type_counts": dict(sorted(source_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "top_file_coverage": file_coverage.most_common(10),
    }


def format_build_report(manifest: EvalDatasetManifest, samples: list[EvalSample]) -> list[str]:
    report = _build_report_payload(manifest, samples)
    sample_count = max(report["sample_count"], 1)

    def _ratio_line(name: str, count: int) -> str:
        return f"DATASET REPORT: {name}={count}/{report['sample_count']} ({count / sample_count:.0%})"

    lines = [
        f"DATASET REPORT: dataset={report['dataset']}",
        f"DATASET REPORT: category={report['category']} source_type={report['source_type']} samples={report['sample_count']}",
        f"DATASET REPORT: capabilities={','.join(report['capabilities']) if report['capabilities'] else '(none)'}",
        _ratio_line("reference_answer", report["with_reference_answer"]),
        _ratio_line("reference_contexts", report["with_reference_contexts"]),
        _ratio_line("reference_doc_ids", report["with_reference_doc_ids"]),
        _ratio_line("scope_file_ids", report["with_scope_file_ids"]),
        f"DATASET REPORT: unique_scope_files={report['unique_scope_file_count']} unique_reference_docs={report['unique_reference_doc_count']}",
        f"DATASET REPORT: review_status_counts={report['review_status_counts']}",
    ]
    if report["scenario_counts"]:
        lines.append(f"DATASET REPORT: scenario_counts={report['scenario_counts']}")
    if report["difficulty_counts"]:
        lines.append(f"DATASET REPORT: difficulty_counts={report['difficulty_counts']}")
    if report["top_file_coverage"]:
        formatted = ", ".join(f"{file_id}:{count}" for file_id, count in report["top_file_coverage"])
        lines.append(f"DATASET REPORT: top_file_coverage={formatted}")
    return lines


def save_dataset(manifest: EvalDatasetManifest, samples: list[EvalSample], output_dir: str | Path | None = None) -> Path:
    samples = validate_samples(samples)
    manifest.sample_count = len(samples)
    manifest.capabilities = infer_manifest_capabilities(samples)
    manifest.review_required = manifest.review_required or review_required_for_category(manifest.category)
    if not manifest.review_instructions:
        manifest.review_instructions = review_instructions_for(manifest.category, manifest.source_type)

    dataset_path = Path(output_dir) if output_dir else dataset_dir(manifest.category, manifest.name, manifest.version)
    dataset_path.mkdir(parents=True, exist_ok=True)
    write_json(dataset_path / _MANIFEST_FILE, manifest.to_dict())
    write_jsonl(dataset_path / _SAMPLES_FILE, [sample.to_dict() for sample in samples])
    export_review_sheet(dataset_path)
    export_review_guide(dataset_path)
    return dataset_path


def load_dataset(dataset_path: str | Path) -> tuple[EvalDatasetManifest, list[EvalSample]]:
    root = Path(dataset_path)
    manifest = EvalDatasetManifest.from_dict(read_json(root / _MANIFEST_FILE))
    samples = [EvalSample.from_dict(row) for row in read_jsonl(root / _SAMPLES_FILE)]
    return manifest, samples


def export_review_sheet(dataset_path: str | Path) -> Path:
    manifest, samples = load_dataset(dataset_path)
    review_path = Path(dataset_path) / _REVIEW_FILE
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "review_status",
                "review_notes",
                "difficulty_level",
                "scenario_type",
                "user_input",
                "reference_answer",
                "scope_file_ids",
                "reference_doc_ids",
                "tags",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "review_status": sample.review_status,
                    "review_notes": sample.review_notes,
                    "difficulty_level": sample.difficulty_level,
                    "scenario_type": sample.scenario_type,
                    "user_input": sample.user_input,
                    "reference_answer": sample.reference_answer or "",
                    "scope_file_ids": "|".join(sample.scope_file_ids),
                    "reference_doc_ids": "|".join(sample.reference_doc_ids),
                    "tags": "|".join(sample.tags),
                }
            )
    return review_path


def export_review_guide(dataset_path: str | Path) -> Path:
    manifest, _ = load_dataset(dataset_path)
    lines = [
        f"# Review Guide: {manifest.name} {manifest.version}",
        "",
        f"- category: `{manifest.category}`",
        f"- source_type: `{manifest.source_type}`",
        f"- review_required: `{manifest.review_required}`",
        "",
        "## Checklist",
    ]
    for item in manifest.review_instructions:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Review Status",
            "- `pending`: not reviewed yet.",
            "- `approved`: ready for evaluation use.",
            "- `needs_revision`: keep the sample, but edit before use.",
            "- `rejected`: remove from the dataset.",
            "",
            "## Suggested Workflow",
            "1. Read `review_sheet.csv` and inspect each sample.",
            "2. Update `review_status` and `review_notes`.",
            "3. Run `python -m app.evals.dataset_builder apply-review --dataset-dir <dir> --review-file <csv>`.",
            "4. Use only approved samples for regression or baseline runs.",
        ]
    )
    review_guide_path = Path(dataset_path) / _REVIEW_GUIDE_FILE
    review_guide_path.write_text("\n".join(lines), encoding="utf-8")
    return review_guide_path


def apply_review_sheet(dataset_path: str | Path, review_file: str | Path) -> Path:
    manifest, samples = load_dataset(dataset_path)
    sample_map = {sample.sample_id: sample for sample in samples}
    with Path(review_file).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = sample_map.get((row.get("sample_id") or "").strip())
            if sample is None:
                continue
            review_status = (row.get("review_status") or sample.review_status).strip() or sample.review_status
            sample.review_status = review_status
            sample.review_notes = (row.get("review_notes") or sample.review_notes).strip()
            sample.difficulty_level = (row.get("difficulty_level") or sample.difficulty_level).strip() or sample.difficulty_level
            sample.scenario_type = (row.get("scenario_type") or sample.scenario_type).strip() or sample.scenario_type
    approved_samples = [sample for sample in samples if sample.review_status != "rejected"]
    return save_dataset(manifest, approved_samples, output_dir=dataset_path)


def build_manifest(
    *,
    name: str,
    version: str,
    category: str,
    source_type: str,
    description: str,
    metadata: dict[str, Any] | None = None,
) -> EvalDatasetManifest:
    return EvalDatasetManifest(
        name=name,
        version=version,
        category=category,
        source_type=source_type,
        description=description,
        review_required=review_required_for_category(category),
        review_instructions=review_instructions_for(category, source_type),
        metadata=dict(metadata or {}),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset utilities for evals.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-review", help="Export a CSV review sheet for a dataset.")
    export_parser.add_argument("--dataset-dir", required=True, help="Dataset directory containing manifest.json and samples.jsonl")

    apply_parser = subparsers.add_parser("apply-review", help="Apply review statuses from CSV back into the dataset.")
    apply_parser.add_argument("--dataset-dir", required=True, help="Dataset directory containing manifest.json and samples.jsonl")
    apply_parser.add_argument("--review-file", required=True, help="CSV review sheet path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "export-review":
        review_path = export_review_sheet(args.dataset_dir)
        guide_path = export_review_guide(args.dataset_dir)
        print(f"REVIEW EXPORT: sheet={review_path}")
        print(f"REVIEW EXPORT: guide={guide_path}")
        return
    if args.command == "apply-review":
        dataset_path = apply_review_sheet(args.dataset_dir, args.review_file)
        print(f"REVIEW APPLY: updated dataset at {dataset_path}")
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
