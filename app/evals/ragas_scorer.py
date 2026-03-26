from __future__ import annotations

import argparse
from statistics import mean
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.config.global_config import global_config
from app.evals.metrics_registry import select_ragas_metrics
from app.evals.reporter import load_run_records, write_item_scores_csv, write_summary
from app.evals.retrieval_scorer import score_retrieval_metrics
from app.evals.runtime import close_eval_runtime, init_eval_runtime


class CorrectnessVerdict(BaseModel):
    label: str = Field(description="correct, partial, or incorrect")
    passed: bool = Field(description="true when the answer should count as correct enough")


_CORRECTNESS_PROMPT = (
    "You are scoring the correctness of a RAG answer. Compare the user question, the expected reference answer, and the actual answer. "
    "Return whether the actual answer is correct enough.\n\n"
    "Question:\n{question}\n\n"
    "Reference Answer:\n{reference_answer}\n\n"
    "Actual Answer:\n{actual_answer}\n"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score live RAG run records with RAGAS and retrieval metrics.")
    parser.add_argument("--run-dir", required=True, help="run directory containing records.jsonl and dataset_manifest.json")
    return parser.parse_args()


def _load_models(llm_name: str):
    from app.core.embeddings import EmbeddingModelFactory

    llm = init_chat_model(llm_name)
    embeddings = EmbeddingModelFactory.init_embedding_model()
    try:
        from ragas.llms import LangchainLLMWrapper
        llm_obj = LangchainLLMWrapper(llm)
    except Exception:
        llm_obj = llm
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        embeddings_obj = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        embeddings_obj = embeddings
    return llm, llm_obj, embeddings_obj


def _record_to_ragas_sample(record):
    payload = {
        "user_input": record.user_input,
        "response": record.actual_response,
        "retrieved_contexts": record.actual_contexts,
    }
    if record.reference_answer:
        payload["reference"] = record.reference_answer
    return payload


def _records_to_ragas_dataset(records):
    try:
        from ragas import EvaluationDataset
    except ImportError as exc:
        raise ImportError("ragas is required. Install dependencies first.") from exc

    sample_cls = None
    for module_name, attr in (("ragas", "SingleTurnSample"), ("ragas.dataset_schema", "SingleTurnSample")):
        try:
            module = __import__(module_name, fromlist=[attr])
            sample_cls = getattr(module, attr)
            break
        except Exception:
            continue
    if sample_cls is None:
        raise RuntimeError("Cannot find SingleTurnSample in ragas package.")
    return EvaluationDataset(samples=[sample_cls(**_record_to_ragas_sample(record)) for record in records])


def _score_correctness(records, llm) -> tuple[list[dict[str, Any]], dict[str, float]]:
    judge = llm.with_structured_output(CorrectnessVerdict)
    rows: list[dict[str, Any]] = []
    passed_values: list[float] = []
    for record in records:
        if not record.reference_answer or not record.actual_response:
            rows.append({"sample_id": record.sample_id, "correctness_label": None, "correctness_pass": None})
            continue
        prompt = _CORRECTNESS_PROMPT.format(
            question=record.user_input,
            reference_answer=record.reference_answer,
            actual_answer=record.actual_response,
        )
        verdict = judge.invoke([HumanMessage(content=prompt)])
        row = {
            "sample_id": record.sample_id,
            "correctness_label": getattr(verdict, "label", None),
            "correctness_pass": 1.0 if getattr(verdict, "passed", False) else 0.0,
        }
        rows.append(row)
        if row["correctness_pass"] is not None:
            passed_values.append(row["correctness_pass"])
    summary = {"correctness_pass_rate": mean(passed_values)} if passed_values else {}
    return rows, summary


def main() -> None:
    args = _parse_args()
    init_eval_runtime()
    try:
        from ragas import evaluate

        records = load_run_records(args.run_dir)
        successful_records = [record for record in records if record.status == "success" and record.actual_response]
        if not successful_records:
            raise ValueError("No successful records with actual_response were found.")

        capability_set = sorted(set().union(*(record.capabilities for record in successful_records)))
        metrics_selection = select_ragas_metrics(capability_set)
        dataset = _records_to_ragas_dataset(successful_records)
        chat_cfg = global_config.get("chat_model", {})
        llm_name = chat_cfg.get("light") or chat_cfg.get("default")
        judge_llm, ragas_llm, embeddings = _load_models(llm_name)
        result = evaluate(dataset=dataset, metrics=metrics_selection.metrics, llm=ragas_llm, embeddings=embeddings)

        ragas_rows = result.to_pandas().to_dict(orient="records") if hasattr(result, "to_pandas") else []
        item_rows: list[dict[str, Any]] = []
        for record, ragas_row in zip(successful_records, ragas_rows):
            row = {"sample_id": record.sample_id}
            for key, value in ragas_row.items():
                if key in {"user_input", "response", "retrieved_contexts", "reference"}:
                    continue
                row[key] = value
            item_rows.append(row)

        retrieval_rows, retrieval_summary = score_retrieval_metrics(successful_records)
        retrieval_row_map = {row["sample_id"]: row for row in retrieval_rows}
        correctness_rows, correctness_summary = _score_correctness(successful_records, judge_llm)
        correctness_row_map = {row["sample_id"]: row for row in correctness_rows}

        for row in item_rows:
            row.update({k: v for k, v in retrieval_row_map.get(row["sample_id"], {}).items() if k != "sample_id"})
            row.update({k: v for k, v in correctness_row_map.get(row["sample_id"], {}).items() if k != "sample_id"})

        metric_avg = {}
        for row in item_rows:
            for key, value in row.items():
                if key == "sample_id" or value is None or isinstance(value, str):
                    continue
                metric_avg.setdefault(key, []).append(float(value))
        summary = {
            "run_dir": args.run_dir,
            "sample_count": len(successful_records),
            "metrics": metrics_selection.metric_names,
            "skipped_metrics": metrics_selection.skipped,
            "metric_avg": {key: mean(values) for key, values in metric_avg.items() if values},
            "retrieval_summary": retrieval_summary,
            "correctness_summary": correctness_summary,
        }
        write_item_scores_csv(args.run_dir, item_rows)
        write_summary(args.run_dir, summary)
        print(f"SCORER: wrote item scores and summary to {args.run_dir}")
    finally:
        close_eval_runtime()


if __name__ == "__main__":
    main()
