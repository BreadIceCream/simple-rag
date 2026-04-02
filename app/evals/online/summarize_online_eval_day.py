from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.evals.online.online_eval_store import load_online_eval_entries, online_eval_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize daily online evaluation metric averages from JSONL.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_online_eval_entries(args.date)
    if not entries:
        print(f"ONLINE EVAL DAILY SUMMARY: no entries found for date={args.date}")
        return

    metric_values: dict[str, list[float]] = defaultdict(list)
    by_query_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    status_counts: dict[str, int] = defaultdict(int)

    for entry in entries:
        status = str(entry.get("status") or "unknown")
        status_counts[status] += 1
        query_type = str(entry.get("query_type") or "unknown")
        metrics = dict(entry.get("metrics") or {})
        for metric_name, metric_value in metrics.items():
            try:
                value = float(metric_value)
            except (TypeError, ValueError):
                continue
            metric_values[str(metric_name)].append(value)
            by_query_type[query_type][str(metric_name)].append(value)

    overall_averages = {
        key: round(sum(values) / len(values), 6)
        for key, values in sorted(metric_values.items())
        if values
    }
    query_type_averages = {
        query_type: {
            key: round(sum(values) / len(values), 6)
            for key, values in sorted(metric_map.items())
            if values
        }
        for query_type, metric_map in sorted(by_query_type.items())
    }

    payload = {
        "event_date": args.date,
        "record_count": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "metric_averages": overall_averages,
        "by_query_type": query_type_averages,
    }

    output_path = Path(online_eval_root()) / args.date / "daily_metric_averages.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"ONLINE EVAL DAILY SUMMARY: wrote {output_path}")


if __name__ == "__main__":
    main()
