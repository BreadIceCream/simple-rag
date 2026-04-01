# Sprint S002 Contract

## Goal

完成 Phase B 的 query-type 专项 synthetic 生成链路规划，交付：
- `build_querytype_dataset.py`
- `querytype_synthesizers.py`
- `querytype_validator.py`

并确保可以用 specialized 数据集做配置实验，重点覆盖 `multi_hop_specific` 与 `multi_hop_abstract`。

## In Scope

- `T-005` 新增 `build_querytype_dataset.py` 与 query distribution 参数体系
- `T-006` 新增专项 synthesizer 封装层（RAGAS 0.4.3 适配）
- `T-007` 新增 query type validator 并接入生成流程
- `T-008` 落地专项配置实验入口与文档（balanced / multihop_focus）

## Out of Scope

- Phase C 的 scorer/reporter 体系扩展
- retrieval multi-hop 新指标设计（`hop_hit@k` 等）
- full multi-hop scoring 改造
- 与 Phase B 无关的 evals 重构
- 改坏现有 `build_synthetic_dataset.py` 默认行为

## Implementation Rules

- 必须新增独立入口 `build_querytype_dataset.py`。
- RAGAS synthesizer 调用必须集中在 `querytype_synthesizers.py`，避免散落在多个脚本。
- validator 必须在主生成流程中被调用，不能只做孤立工具。
- 默认分布要偏重 multi-hop，并允许命令行覆盖。
- 需要明确迁移策略：旧 `build_synthetic_dataset.py` 保持兼容，新能力走新脚本。

## Delivery Metrics

- `build_querytype_dataset --help` 可见分布/多跳/validator 参数。
- specialized 数据集构建成功，且 manifest metadata 含：
  - `requested_distribution`
  - `realized_distribution`
  - `validation_summary`
  - `experiment_config`
- 在多跳偏重配置下，multi-hop 样本数量高于 single-hop。
- 支持至少两个实验配置：`balanced` 与 `multihop_focus`。

## Verification Metrics

- E2E 生成专项数据集并产物完整。
- 样本 `query_type` 合法且分布可追踪。
- validator 规则可执行且对异常样本有结果输出。
- 旧 `build_synthetic_dataset.py` 兼容性验证通过。

## Required Artifacts

- `app/evals/build_querytype_dataset.py`
- `app/evals/querytype_synthesizers.py`
- `app/evals/querytype_validator.py`
- `harness/backlog/todos.json`
- `harness/sprints/S002/generator_notes.md`
- `harness/sprints/S002/verification.json`
- `harness/progress/progress.txt`
- Git commit hash

## Exit Criteria

- `T-005..T-008` 完成实现并写入生成说明。
- evaluator 可按 `verification.json` 完整验收。
- 交接前 `validate_harness.ps1` 通过。
