# Sprint S003 Generator Notes

## Tasks Worked On

- Implemented T-009: added querytype batch capability probe, deterministic non-retriable cluster error classification, and deterministic query-type reallocation helpers.
- Implemented T-010: integrated probe -> effective counts -> generation path in querytype builder, added non-retriable handling path, and added per-sub-batch diagnostic console logs.
- Implemented T-011: made `--enable-multi-file` executable by building mixed multi-file batches where possible, with explicit single-file fallback logging/metadata when pairing is unavailable.
- Implemented T-012: updated Evals command documentation with S003 behavior, diagnostics, multi-file handling, and scope boundary notes.

## Files Changed

- app/evals/querytype_synthesizers.py
- app/evals/build_querytype_dataset.py
- docs/Evals命令文档.md
- harness/backlog/todos.json
- harness/progress/progress.txt
- harness/runtime/state.json
- harness/sprints/S003/generator_notes.md
- harness/sprints/S003/handoff.txt

## Design Decisions

- Kept all querytype control-plane policies in `querytype_synthesizers.py` to avoid leaking error-shape details across modules.
- Treated cluster/relationship absence as non-retriable and routed to deterministic fallback rather than increasing retry count.
- Added structured console diagnostics (`QUERYTYPE BATCH`) so evaluator can verify requested counts, probe result, fallback events, error classification, and generated counts directly from runtime logs.
- Implemented a minimal executable `--enable-multi-file` behavior by pairing sub-batches from different files into merged multi-file batches; if no partner exists, the batch explicitly falls back with reason metadata and logs.
- Preserved `build_synthetic_dataset.py` behavior and did not modify scorer/reporter code.

## Unresolved Limitations

- Availability probe is heuristic-first (chunk/parent/file signals), so some downstream RAGAS edge cases may still trigger fallback in runtime.
- In extreme sparse corpora, merged multi-file pairing can still leave fallback-only batches; this is now explicit and diagnosable, not silent.

## Verification Executed by Generator

- `python -m py_compile app/evals/querytype_synthesizers.py app/evals/build_querytype_dataset.py`
- `python -m app.evals.build_querytype_dataset --help`
- Inline function smoke check for:
  - `probe_available_query_types(...)`
  - `reallocate_query_type_counts(...)`
  - `classify_querytype_error(...)`

## Recommended Evaluator Focus

- V-009: verify deterministic behavior of probe/error classification/reallocation on controlled inputs.
- V-010: reproduce deterministic cluster-failure path and confirm no blind retry for non-retriable errors; validate required diagnostic log fields.
- V-011: validate `--enable-multi-file` runtime behavior and explicit fallback path when multi-file pairing cannot be formed.
- V-012: verify docs reflect S003 runtime behavior and keep Phase C out-of-scope boundaries.

## Git Commit Hash

- 642b386

