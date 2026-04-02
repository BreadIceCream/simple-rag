# Sprint S001 Evaluation Report

Conclusion: PASS

Passed checks:
- V-001: schema query_type fields serialize/deserialize correctly and old payloads remain compatible.
- V-002: replay query-type classification is explicit opt-in and CLI compatibility is preserved.
- V-003: query_type propagation into run-record path is present and by_query_type aggregation/report generation works on run-record-shaped data.
- V-004: seed import preserves explicit query-type fields and remains compatible with legacy inputs without those fields.

Failed checks:
- none

Evidence:
- Commit under evaluation: `1363ceded1a16c3e2f491c8685f6b2278ab7798c`
- `V001_SCHEMA_COMPAT_OK` via inline schema compatibility assertions.
- `python -m app.evals.build_replay_dataset --help` confirms `--classify-query-type` and `--query-type-mode` flags.
- `V002_REPLAY_CLASSIFIER_OK` via classifier unit smoke.
- Code inspection of `app/evals/live_rag_runner.py` confirms query-type fields are written into `EvalRunRecord`.
- `V003_QTYPE_REPORT_FLOW_OK` via run-record-shaped aggregation/report smoke.
- Artifacts generated: `.tmp_eval/s001_eval_qtype_check/summary.json`, `.tmp_eval/s001_eval_qtype_check/report.md`.
- `V004_SEED_IMPORT_QTYPE_OK` via seed import with and without query-type fields.

Required fixes:
- none

Decision:
- Sprint S001 satisfies Phase A scope in contract (`T-001`..`T-004`). Move to `passed` and hand back to `planner` for next-sprint decision.
