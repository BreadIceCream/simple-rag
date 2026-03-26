from __future__ import annotations

from app.evals.build_synthetic_dataset import main


if __name__ == "__main__":
    print("DEPRECATED: build_golden_jsonl is now an alias of build_synthetic_dataset.")
    print("DEPRECATED: synthetic datasets are not the primary baseline for real RAG evaluation.")
    main()
