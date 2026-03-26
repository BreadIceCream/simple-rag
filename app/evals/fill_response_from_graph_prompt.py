from __future__ import annotations

import sys


def main() -> None:
    message = (
        "fill_response_from_graph_prompt.py has been retired from the official evaluation pipeline.\n"
        "Reason: it only runs prompt-only generation and does not execute the real retrieval or full graph chain.\n"
        "Use:\n"
        "  python -m app.evals.live_rag_runner --dataset-dir <dataset_dir>\n"
        "then:\n"
        "  python -m app.evals.ragas_scorer --run-dir <run_dir>\n"
    )
    sys.stderr.write(message)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
