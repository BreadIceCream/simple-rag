from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.evals.dataset_builder import EvalRecord


def _safe_result_dataframe(result: Any) -> pd.DataFrame:
    if hasattr(result, "to_pandas"):
        return result.to_pandas()

    if isinstance(result, dict):
        return pd.DataFrame([result])

    rows = getattr(result, "scores", None)
    if rows:
        return pd.DataFrame(rows)

    raise RuntimeError("Unsupported RAGAS result object: cannot convert to pandas dataframe.")


def _summary_from_df(df: pd.DataFrame) -> dict[str, float]:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}
    return {str(k): float(v) for k, v in numeric_df.mean(numeric_only=True).to_dict().items()}


def write_experiment_report(
    *,
    run_id: str,
    output_root: Path,
    records: list[EvalRecord],
    result: Any,
    metric_names: list[str],
    skipped_metrics: list[str],
    source: str,
) -> Path:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df = _safe_result_dataframe(result)
    summary = _summary_from_df(df)

    (run_dir / "item_scores.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": source,
                "sample_count": len(records),
                "metrics": metric_names,
                "skipped_metrics": skipped_metrics,
                "metric_avg": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return run_dir
