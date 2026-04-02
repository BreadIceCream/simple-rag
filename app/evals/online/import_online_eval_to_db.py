from __future__ import annotations

import argparse
import json
from datetime import datetime

from app.config.db_config import DatabaseManager
from app.config.global_config import global_config
from app.crud.online_eval import upsert_online_eval_entries
from app.evals.online.online_eval_store import load_online_eval_entries, online_eval_root
from app.models.schemas import Base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import daily online evaluation JSONL entries into database.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for initial bulk upsert.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_online_eval_entries(args.date)
    if not entries:
        print(f"ONLINE EVAL IMPORT: no entries found for date={args.date}")
        return

    global_config.load()
    DatabaseManager.init()
    Base.metadata.create_all(DatabaseManager._sync_engine)
    db = DatabaseManager.get_sync_db()
    try:
        inserted, updated, failed_request_ids = upsert_online_eval_entries(
            db,
            entries,
            batch_size=args.batch_size,
        )
    finally:
        db.close()

    print(
        "ONLINE EVAL IMPORT: "
        f"date={args.date} rows={len(entries)} batch_size={args.batch_size} inserted={inserted} updated={updated} "
        f"failed={len(failed_request_ids)}"
    )
    if failed_request_ids:
        failure_path = online_eval_root() / args.date / "import_failures.json"
        failure_payload = {
            "event_date": args.date,
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "batch_size": args.batch_size,
            "row_count": len(entries),
            "failed_count": len(failed_request_ids),
            "failed_request_ids": failed_request_ids,
        }
        failure_path.write_text(
            json.dumps(failure_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"ONLINE EVAL IMPORT FAILED REQUEST IDS: {failed_request_ids}")
        print(f"ONLINE EVAL IMPORT FAILURE FILE: {failure_path}")


if __name__ == "__main__":
    main()
