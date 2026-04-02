# Sprint S002 Generator Notes

## Tasks Worked On

- Implemented T-005: added `app/evals/build_querytype_dataset.py` with query distribution profile/json/file inputs and specialized synthetic dataset output.
- Implemented T-006: added `app/evals/querytype_synthesizers.py` wrapper layer for RAGAS query-type synthesis with project adapter for `single_hop_abstract`.
- Implemented T-007: added `app/evals/querytype_validator.py` and integrated validation (`off|warn|strict`) into dataset generation pipeline.
- Implemented T-008: updated `docs/Evals命令文档.md` with Phase B specialized experiment commands (balanced + multihop_focus), execution order, and migration guidance.

## Files Changed

- app/evals/build_querytype_dataset.py
- app/evals/querytype_synthesizers.py
- app/evals/querytype_validator.py
- docs/Evals命令文档.md
- harness/backlog/todos.json
- harness/runtime/state.json
- harness/progress/progress.txt
- harness/sprints/S002/handoff.txt
- harness/sprints/S002/generator_notes.md

## Design Decisions

- Kept `build_synthetic_dataset.py` unchanged to preserve backward compatibility and isolate Phase B into a dedicated entrypoint.
- Reused existing synthetic planning/runtime helpers to avoid divergence in document/chunk sampling behavior.
- Centralized query distribution and RAGAS synthesizer adaptation logic in `querytype_synthesizers.py`.
- Added project-side `single_hop_abstract` adapter strategy to avoid hard dependency on a dedicated upstream synthesizer class.
- Added validator summary and query distribution summaries directly into manifest metadata for auditability.

## Unresolved Limitations

- `single_hop_abstract` generation currently uses a lightweight project adapter, not a fully custom RAGAS synthesizer class.
- Full E2E generation validation was not executed in generator phase because it depends on local runtime services (DB/vector/docstore/LLM availability).

## Verification Executed by Generator

- `python -m app.evals.build_querytype_dataset --help`
- `python -c "from app.evals.querytype_synthesizers import QUERY_DISTRIBUTION_PROFILES, allocate_query_type_counts; from app.evals.querytype_validator import QueryTypeValidator; ..."`

## Recommended Evaluator Focus

- V-005: run small querytype dataset build and verify `requested_distribution`/`realized_distribution` in manifest metadata.
- V-006: verify wrapper can emit all four query types and that multihop_focus profile yields higher multi-hop share.
- V-007: validate `warn` vs `strict` behavior and check `validation_summary` + per-sample validation metadata.
- V-008: execute documented specialized experiment commands and confirm docs stay within Phase B scope.

## Git Commit Hash

- fed9af8


## Repair Loop (HR-001)

- Reworked `docs/Evals命令文档.md` into a coherent post-S002 command guide instead of an appended patchwork section.
- Explicitly added Phase B out-of-scope text: no retrieval scorer redesign, no ragas_scorer/reporter redesign, no Phase C metric redesign in this sprint.
- Merged S002 deliverables into the main doc flow:
  - `build_querytype_dataset.py` full parameter table
  - `querytype_synthesizers.py` and `querytype_validator.py` module responsibilities
  - specialized experiment commands (`balanced`, `multihop_focus`, custom JSON)
  - migration guidance from legacy `build_synthetic_dataset.py`
  - environment dependency note for runtime issues (including `asyncpg` as an example)

Repair scope remained in-scope to S002 (`T-005`, `T-008`) and did not modify scorer/reporter implementation.

- Repair loop commit hash: d107634


