from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evals.schema import EvalDatasetManifest, EvalRunRecord, read_jsonl, write_jsonl


def experiment_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "store" / "evals" / "experiments"


def new_run_dir(output_root: str | Path | None = None) -> tuple[str, Path]:
    root = Path(output_root) if output_root else experiment_root()
    run_id = uuid.uuid4().hex[:12]
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_run_artifacts(
    *,
    run_dir: str | Path,
    manifest: EvalDatasetManifest,
    records: list[EvalRunRecord],
    config_snapshot: dict[str, Any],
    source_label: str,
) -> Path:
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "source": source_label,
                "sample_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "records.jsonl", [record.to_dict() for record in records])
    return output_dir


def load_run_records(run_dir: str | Path) -> list[EvalRunRecord]:
    rows = read_jsonl(Path(run_dir) / "records.jsonl")
    return [EvalRunRecord.from_dict(row) for row in rows]


def write_summary(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(run_dir) / "summary.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_item_scores_csv(run_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = Path(run_dir) / "item_scores.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report_markdown(
    run_dir: str | Path,
    summary: dict[str, Any],
    item_rows: list[dict[str, Any]],
    records: list[EvalRunRecord],
) -> Path:
    output_path = Path(run_dir) / "report.md"
    dataset_name = records[0].dataset_name if records else "unknown"
    dataset_version = records[0].dataset_version if records else "unknown"
    category = records[0].category if records else "unknown"
    total_count = len(records)
    success_count = sum(1 for record in records if record.status == "success")
    failed_records = [record for record in records if record.status != "success"]

    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Run Dir: `{Path(run_dir)}`")
    lines.append(f"- Dataset: `{dataset_name}`")
    lines.append(f"- Version: `{dataset_version}`")
    lines.append(f"- Category: `{category}`")
    lines.append(f"- Total Samples: `{total_count}`")
    lines.append(f"- Successful Samples: `{success_count}`")
    lines.append(f"- Failed Samples: `{len(failed_records)}`")
    lines.append("")

    metrics = summary.get("metric_avg") or {}
    if metrics:
        lines.append("## Average Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key in sorted(metrics.keys()):
            lines.append(f"| `{key}` | {_format_metric(metrics[key])} |")
        lines.append("")

    retrieval_summary = summary.get("retrieval_summary") or {}
    if retrieval_summary:
        lines.append("## Retrieval Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key in sorted(retrieval_summary.keys()):
            lines.append(f"| `{key}` | {_format_metric(retrieval_summary[key])} |")
        lines.append("")

    correctness_summary = summary.get("correctness_summary") or {}
    if correctness_summary:
        lines.append("## Correctness Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key in sorted(correctness_summary.keys()):
            lines.append(f"| `{key}` | {_format_metric(correctness_summary[key])} |")
        lines.append("")

    skipped_metrics = summary.get("skipped_metrics") or []
    if skipped_metrics:
        lines.append("## Skipped Metrics")
        lines.append("")
        for item in skipped_metrics:
            lines.append(f"- `{item}`")
        lines.append("")

    if failed_records:
        lines.append("## Failed Samples")
        lines.append("")
        lines.append("| Sample ID | Error |")
        lines.append("| --- | --- |")
        for record in failed_records[:20]:
            error_message = (record.error_message or "").replace("\n", " ").strip() or "-"
            lines.append(f"| `{record.sample_id}` | {error_message} |")
        if len(failed_records) > 20:
            lines.append("")
            lines.append(f"Only the first 20 failed samples are shown here. Total failed samples: `{len(failed_records)}`.")
        lines.append("")

    if item_rows:
        lines.append("## Sample Scores")
        lines.append("")
        lines.append("See `item_scores.csv` for the full per-sample score table.")
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path
