# Sprint S003 Evaluation Report

Conclusion: PASS

Passed checks:
- V-009 passed. Controlled probe inputs showed stable availability decisions, non-retriable cluster error classification, and deterministic query-type reallocation.
- V-010 passed. A controlled deterministic cluster-failure path raised `NonRetriableQueryTypeGenerationError` after one generation attempt, proving non-retriable cluster errors no longer trigger blind retry; required `QUERYTYPE BATCH` diagnostics and manifest summary fields are present.
- V-011 passed. `--enable-multi-file` now materially changes batch assembly: different-file sub-batches are merged into multi-file batches where possible, and otherwise surface explicit single-file fallback behavior instead of metadata-only no-op behavior.
- V-012 passed. `docs/Evals????.md` now documents S003 probe/fallback/non-retriable/diagnostic behavior, multi-file semantics, and Phase C scope boundaries consistently with the current CLI.

Failed checks:
- none.

Evidence:
- `python -m app.evals.build_querytype_dataset --help` succeeded and exposed the expected S003 CLI surface, including `--enable-multi-file`.
- `python -m app.evals.build_synthetic_dataset --help` succeeded, supporting compatibility of the legacy synthetic entrypoint.
- Controlled probe/classification/reallocation smoke confirmed:
  - single-file two-parent probe => multi_hop_specific available, multi_hop_abstract unavailable
  - same chunks with multi-file enabled => only single-hop types available
  - cluster-missing ValueError => retriable=False, category=cluster_missing
  - repeated reallocation => identical output
- Controlled `_generate_with_retry(...)` smoke using a fake facade confirmed one call only and no retry loop for deterministic cluster-missing errors; captured output included `QUERYTYPE BATCH` diagnostic fields.
- Controlled `_build_generation_batches(...)` smoke confirmed:
  - two different files => `multi_file` batch
  - one file only => `single_file_fallback` batch with observable fallback mode
- Controlled `_to_samples(...)` smoke confirmed batch_mode and source_file_ids/source_file_names survive into sample metadata.

Files and behaviors inspected:
- `app/evals/querytype_synthesizers.py`
- `app/evals/build_querytype_dataset.py`
- `app/evals/build_synthetic_dataset.py`
- `docs/Evals????.md`
- Sprint S003 harness artifacts

Required fixes:
- none within current sprint scope.

Decision:
- Close Sprint S003 as PASS and return control to planner for next-sprint or closure decision.
