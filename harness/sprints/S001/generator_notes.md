# Sprint S001 Generator Notes

## Tasks Worked On

- T-001 implemented: schema query_type field support and backward compatibility in `app/evals/schema.py`.
- T-002 implemented: replay dataset query-type classification as explicit opt-in in `app/evals/build_replay_dataset.py`.
- T-003 implemented: query_type propagation to run records and by-query-type aggregation/reporting in `app/evals/live_rag_runner.py`, `app/evals/ragas_scorer.py`, `app/evals/reporter.py`.
- T-004 implemented: seed import query-type explicit field pass-through in `app/evals/import_seed_dataset.py`.

## Files Changed

- `app/evals/schema.py`
- `app/evals/build_replay_dataset.py`
- `app/evals/live_rag_runner.py`
- `app/evals/ragas_scorer.py`
- `app/evals/reporter.py`
- `app/evals/import_seed_dataset.py`
- `harness/backlog/todos.json`
- `harness/progress/progress.txt`
- `harness/runtime/state.json`
- `harness/sprints/S001/generator_notes.md`
- `harness/sprints/S001/handoff.txt`

## Design Decisions

- Kept query-type support backward compatible by normalizing unknown/missing fields and deriving `hop_count` + `abstraction_level` from `query_type` when possible.
- Replay classification is disabled by default and only runs when `--classify-query-type` is passed, preserving legacy behavior.
- By-query-type scoring aggregates numeric fields from per-sample item rows; no new multi-hop retrieval metrics were introduced in S001.
- Seed import remains pass-through only for query-type fields to stay within Phase A boundaries (no automatic seed classification).

## Verification Run

- `python -m py_compile app/evals/schema.py app/evals/build_replay_dataset.py app/evals/live_rag_runner.py app/evals/ragas_scorer.py app/evals/reporter.py app/evals/import_seed_dataset.py`
- `python -m app.evals.build_replay_dataset --help`
- `python -m app.evals.import_seed_dataset --input tmp_seed_qtype.json --name seed_qtype_smoke --version v1 --category exploration --source-type manual --output-dir store/evals/datasets/exploration/seed_qtype_smoke/v1`
- Run-record-shaped scorer/reporter smoke wrote:
  - `store/evals/experiments/s001_qtype_smoke/summary.json`
  - `store/evals/experiments/s001_qtype_smoke/report.md`

## Unresolved Limitations

- Replay query-type heuristic is intentionally lightweight and may misclassify edge cases.
- No full end-to-end live graph run + ragas evaluation execution was performed in this sprint pass; evaluator should validate with contract verification checks.

## Recommended Evaluator Focus

- Confirm old dataset/run-record compatibility with missing query-type fields.
- Validate replay opt-in classification output shape and metadata.
- Validate records-to-summary-to-report by-query-type path using run-record-shaped data.
- Validate seed import with and without query-type fields.

## Git Commit Hash

- 1363ced
