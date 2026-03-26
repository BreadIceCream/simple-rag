from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.config.db_config import DatabaseManager
from app.models.schemas import ChatHistory

_RAG_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _RAG_ROOT / "store" / "evals" / "datasets"


@dataclass
class EvalRecord:
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str | None = None
    metadata: dict[str, Any] | None = None


def default_dataset_path(dataset_name: str) -> Path:
    return _DEFAULT_DATASET_DIR / f"{dataset_name}.jsonl"


def load_records_from_jsonl(path: str | Path) -> list[EvalRecord]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    records: list[EvalRecord] = []
    with dataset_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                EvalRecord(
                    user_input=data["user_input"],
                    response=data["response"],
                    retrieved_contexts=data.get("retrieved_contexts", []),
                    reference=data.get("reference"),
                    metadata=data.get("metadata", {}),
                )
            )
    if not records:
        raise ValueError(f"No valid samples in dataset: {dataset_path}")
    return records


def build_records_from_chat_history(limit: int | None = None) -> list[EvalRecord]:
    """
    从 chat_history 构建最小可用单轮评测集：
    user_input + response + retrieved_contexts(由 parent_doc_ids 回溯父文档内容)。
    """
    with DatabaseManager.get_sync_db() as db:
        rows = db.execute(
            select(ChatHistory).order_by(ChatHistory.conversation_id.asc(), ChatHistory.created_at.asc())
        ).scalars().all()

    if not rows:
        raise ValueError("chat_history is empty, cannot build evaluation records.")
    from app.core.vector_store import VectorStoreFactory
    docstore = VectorStoreFactory.init_docstore()
    records: list[EvalRecord] = []

    grouped: dict[str, list[ChatHistory]] = {}
    for row in rows:
        grouped.setdefault(row.conversation_id, []).append(row)

    for conversation_id, messages in grouped.items():
        pending_user: str | None = None
        for msg in messages:
            if msg.role == "user":
                pending_user = msg.content
                continue

            if msg.role != "ai" or not pending_user:
                continue

            parent_doc_ids = msg.parent_doc_ids or []
            if not parent_doc_ids:
                pending_user = None
                continue

            parent_docs = docstore.mget(parent_doc_ids)
            contexts = [doc.page_content for doc in parent_docs if doc is not None and getattr(doc, "page_content", "")]
            if not contexts:
                pending_user = None
                continue

            records.append(
                EvalRecord(
                    user_input=pending_user,
                    response=msg.content,
                    retrieved_contexts=contexts,
                    metadata={
                        "conversation_id": conversation_id,
                        "chat_history_id": msg.id,
                        "parent_doc_ids": parent_doc_ids,
                        "source": "chat_history",
                    },
                )
            )

            pending_user = None
            if limit is not None and len(records) >= limit:
                return records

    if not records:
        raise ValueError("No suitable chat_history samples with retrieval contexts were found.")
    return records


def records_to_ragas_dataset(records: list[EvalRecord]):
    if not records:
        raise ValueError("records is empty.")

    try:
        from ragas import EvaluationDataset
    except ImportError as e:
        raise ImportError("ragas is required. Install dependencies first.") from e

    sample_cls = None
    for candidate in (
        ("ragas", "SingleTurnSample"),
        ("ragas.dataset_schema", "SingleTurnSample"),
    ):
        module_name, attr = candidate
        try:
            module = __import__(module_name, fromlist=[attr])
            sample_cls = getattr(module, attr)
            break
        except Exception:
            continue

    if sample_cls is None:
        raise RuntimeError("Cannot find SingleTurnSample in ragas package.")

    samples = []
    for record in records:
        payload = {
            "user_input": record.user_input,
            "response": record.response,
            "retrieved_contexts": record.retrieved_contexts,
        }
        if record.reference:
            payload["reference"] = record.reference
        samples.append(sample_cls(**payload))

    return EvaluationDataset(samples=samples)


def dump_records_as_jsonl(records: list[EvalRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return output


