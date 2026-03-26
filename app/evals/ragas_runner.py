from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper: execute live run then score it.")
    parser.add_argument("--dataset-dir", required=True, help="dataset directory")
    parser.add_argument("--output-root", default="", help="optional experiment output root")
    parser.add_argument("--limit", type=int, default=0, help="optional max sample count")
    parser.add_argument("--review-status", default="approved,pending", help="comma-separated allowed review statuses")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    live_cmd = [
        sys.executable,
        "-m",
        "app.evals.live_rag_runner",
        "--dataset-dir",
        args.dataset_dir,
        "--review-status",
        args.review_status,
    ]
    if args.output_root:
        live_cmd.extend(["--output-root", args.output_root])
    if args.limit > 0:
        live_cmd.extend(["--limit", str(args.limit)])
    completed = subprocess.run(live_cmd, check=True, capture_output=True, text=True)
    print(completed.stdout, end="")
    run_dir = None
    for line in completed.stdout.splitlines():
        if line.startswith("LIVE RUN: artifacts="):
            run_dir = line.split("=", 1)[1].strip()
            break
    if not run_dir:
        raise RuntimeError("Could not determine run_dir from live runner output.")
    score_cmd = [sys.executable, "-m", "app.evals.ragas_scorer", "--run-dir", run_dir]
    scored = subprocess.run(score_cmd, check=True, capture_output=True, text=True)
    print(scored.stdout, end="")
    print(f"RAGAS RUNNER: completed run_dir={Path(run_dir)}")


if __name__ == "__main__":
    main()
