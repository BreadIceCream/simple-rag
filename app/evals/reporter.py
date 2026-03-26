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
