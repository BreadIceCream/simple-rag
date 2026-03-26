from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from app.config.db_config import DatabaseManager
from app.config.global_config import global_config
from app.evals.dataset_builder import (
    build_records_from_chat_history,
    default_dataset_path,
    load_records_from_jsonl,
    records_to_ragas_dataset,
)
from app.evals.metrics_registry import select_single_turn_metrics
from app.evals.reporter import write_experiment_report

_RAG_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUT_ROOT = _RAG_ROOT / "store" / "evals" / "experiments"


def _init_db_runtime() -> None:
    global_config.load()
    DatabaseManager.init()


def _close_db_runtime() -> None:
    try:
        asyncio.run(DatabaseManager.close())
    except RuntimeError:
        # no running loop / or db not initialized
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P0 RAGAS evaluation.")
    parser.add_argument("--dataset", default="v1", help="dataset name, default v1")
    parser.add_argument("--dataset-path", default="", help="explicit dataset jsonl path")
    parser.add_argument(
        "--source",
        choices=["auto", "jsonl", "history"],
        default="auto",
        help="auto: prefer jsonl if exists, else use chat_history",
    )
    parser.add_argument("--limit", type=int, default=100, help="max samples when reading from chat_history")
    parser.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT), help="report output directory")
    return parser.parse_args()


def _load_records(source: str, dataset: str, dataset_path: str, limit: int):
    path = Path(dataset_path) if dataset_path else default_dataset_path(dataset)

    if source == "jsonl":
        return load_records_from_jsonl(path), "jsonl"

    if source == "history":
        _init_db_runtime()
        try:
            return build_records_from_chat_history(limit=limit), "chat_history"
        finally:
            _close_db_runtime()

    # auto
    if path.exists():
        return load_records_from_jsonl(path), "jsonl"

    _init_db_runtime()
    try:
        return build_records_from_chat_history(limit=limit), "chat_history"
    finally:
        _close_db_runtime()


def main() -> None:
    args = _parse_args()

    try:
        from ragas import evaluate
    except ImportError as e:
        raise ImportError("ragas is not installed. Please install requirements first.") from e

    records, source = _load_records(args.source, args.dataset, args.dataset_path, args.limit)
    has_reference = all((record.reference is not None and record.reference.strip()) for record in records)

    metrics_selection = select_single_turn_metrics(has_reference=has_reference)
    dataset = records_to_ragas_dataset(records)

    print(
        f"EVAL RUNNER: loaded {len(records)} samples from {source}, "
        f"metrics={metrics_selection.metric_names}, skipped={metrics_selection.skipped}"
    )

    result = evaluate(dataset=dataset, metrics=metrics_selection.metrics)

    run_id = uuid.uuid4().hex[:12]
    run_dir = write_experiment_report(
        run_id=run_id,
        output_root=Path(args.output_root),
        records=records,
        result=result,
        metric_names=metrics_selection.metric_names,
        skipped_metrics=metrics_selection.skipped,
        source=source,
    )

    print(f"EVAL RUNNER: done. report_dir={run_dir}")


if __name__ == "__main__":
    main()
