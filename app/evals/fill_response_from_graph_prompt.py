from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from app.config.global_config import global_config
from app.core.graph import GENERATE_ANSWER_PROMPT

_RAG_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RUN_DIR = _RAG_ROOT / "store" / "evals" / "datasets" / "runs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill response for golden jsonl by reusing graph GENERATE_ANSWER_PROMPT and LLM.invoke."
    )
    parser.add_argument("--input", required=True, help="input golden jsonl path")
    parser.add_argument("--output", default="", help="output run jsonl path; default auto under store/evals/datasets/runs/")
    parser.add_argument(
        "--context-mode",
        choices=["reference", "retrieved", "both"],
        default="retrieved",
        help="context source for prompt. default=retrieved",
    )
    parser.add_argument("--qps", type=int, default=10, help="requests per second, default 10")
    parser.add_argument("--concurrency", type=int, default=10, help="max concurrent in-flight requests, default 10")
    parser.add_argument("--model", default="", help="override chat model name")
    parser.add_argument("--skip-existing-response", action="store_true", help="skip records with non-empty response")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No valid records in {path}")
    return rows


def _auto_output_path(input_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_RUN_DIR / f"{input_path.stem}.run_{ts}.jsonl"


def _normalize_context_list(record: dict[str, Any]) -> list[str]:
    contexts = record.get("retrieved_contexts") or []
    if isinstance(contexts, str):
        return [contexts]
    if isinstance(contexts, list):
        return [str(c) for c in contexts if str(c).strip()]
    return []


def _build_context(record: dict[str, Any], mode: str) -> str:
    reference = (record.get("reference") or "").strip()
    contexts = _normalize_context_list(record)
    contexts_joined = "\n\n".join(contexts)

    if mode == "reference":
        return reference or contexts_joined
    if mode == "retrieved":
        return contexts_joined or reference

    parts = []
    if reference:
        parts.append(f"Reference:\n{reference}")
    if contexts_joined:
        parts.append(f"Retrieved Contexts:\n{contexts_joined}")
    return "\n\n".join(parts)


async def _invoke_once(llm: Any, question: str, context: str) -> str:
    prompt = GENERATE_ANSWER_PROMPT.format(question=question, context=context)
    response = await asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)])
    content = getattr(response, "content", "")
    return str(content).strip()


async def _fill_responses(
    *,
    rows: list[dict[str, Any]],
    llm: Any,
    context_mode: str,
    qps: int,
    concurrency: int,
    skip_existing_response: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    output_rows = [dict(row) for row in rows]

    async def run_one(idx: int) -> None:
        row = output_rows[idx]
        question = (row.get("user_input") or "").strip()
        if not question:
            row["response"] = row.get("response") or ""
            meta = dict(row.get("metadata") or {})
            meta["fill_response_status"] = "skipped_empty_user_input"
            row["metadata"] = meta
            return

        if skip_existing_response and str(row.get("response") or "").strip():
            meta = dict(row.get("metadata") or {})
            meta["fill_response_status"] = "skipped_existing_response"
            row["metadata"] = meta
            return

        context = _build_context(row, context_mode)
        if not context:
            meta = dict(row.get("metadata") or {})
            meta["fill_response_status"] = "skipped_empty_context"
            row["metadata"] = meta
            row["response"] = row.get("response") or ""
            return

        async with semaphore:
            answer = await _invoke_once(llm, question=question, context=context)

        row["response"] = answer
        meta = dict(row.get("metadata") or {})
        meta["fill_response_status"] = "filled"
        meta["fill_response_mode"] = context_mode
        meta["fill_response_at"] = datetime.now().isoformat(timespec="seconds")
        row["metadata"] = meta

    total = len(rows)
    for start in range(0, total, qps):
        end = min(start + qps, total)
        await asyncio.gather(*(run_one(i) for i in range(start, end)))
        print(f"FILL RESPONSE: processed {end}/{total}")
        if end < total:
            await asyncio.sleep(1)

    return output_rows


def _write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _auto_output_path(input_path)

    global_config.load()
    chat_cfg = global_config.get("chat_model", {})
    model_name = args.model or chat_cfg.get("default")
    if not model_name:
        raise ValueError("chat_model.default is empty, and --model is not provided.")

    rows = _load_jsonl(input_path)
    llm = init_chat_model(model_name)

    print(
        f"FILL RESPONSE: input={input_path}, output={output_path}, model={model_name}, "
        f"context_mode={args.context_mode}, qps={args.qps}, concurrency={args.concurrency}"
    )

    filled_rows = asyncio.run(
        _fill_responses(
            rows=rows,
            llm=llm,
            context_mode=args.context_mode,
            qps=max(1, args.qps),
            concurrency=max(1, args.concurrency),
            skip_existing_response=args.skip_existing_response,
        )
    )

    _write_jsonl(filled_rows, output_path)
    print(f"FILL RESPONSE: done. output={output_path}")
    print(f"FILL RESPONSE: input(golden) kept untouched at {input_path}")


if __name__ == "__main__":
    main()

