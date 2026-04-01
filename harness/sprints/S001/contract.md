# Sprint S001 Contract

## Goal

完成 RAGAS 2.0 Phase A 的最小完整切片规划：让 query type 进入 `schema -> replay / seed dataset -> live run records -> scorer/reporter` 主链路，并能直接交付 `generator` 开始实现。

## In Scope

- `T-001` 扩展离线评估 schema 以承载 query type
- `T-002` 为 replay 数据集构建链路增加 query type 分类
- `T-003` 打通 query type 到 run record 并输出分桶评分报告
- `T-004` 为 seed import 提供 query type 字段接入

## Out of Scope

- 新的 synthetic query type builder
- multi-hop 专项检索指标
- seed import 自动分类
- Phase B 及以后工作

## Minimum-Slice Decision

Phase A 在方案文档中明确要求让现有数据集“可分型”，其中包含 replay 和 seed。基于这一点，S001 现在把 seed import 的显式 query type 字段接入纳入 scope。

同时，S001 仍保持最小切片：

- replay 在本 sprint 中支持显式开启的 query type 分类
- seed import 在本 sprint 中只支持显式字段导入与保留，不做自动分类
- synthetic query type builder 仍然明确留到 Phase B

这样做的原因是：

- 保证与 `docs/RAGAS集成方案2.0.md` Phase A 对齐
- 保持 S001 仍然只覆盖现有数据源可分型，不提前进入 Phase B 的 synthetic builder 工作
- 保证 by_query_type 报表具备从 dataset 到 run record 再到 scorer/reporter 的一条完整端到端链路

## Implementation Rules

- 不得扩展 `build_synthetic_dataset.py` 的 Phase B 功能
- schema 变更必须兼容旧数据集与旧 run records
- replay query type 分类必须是显式开启能力，默认行为保持兼容
- seed import 在 S001 中只做显式字段导入与保留，不做自动分类
- `live_rag_runner.py` 必须把样本中的 query type 信息透传到 `EvalRunRecord`，不能把 by_query_type 聚合建立在隐式假设上
- scorer/reporter 的目标是新增 `by_query_type` 聚合，不在本 sprint 引入新的多跳指标

## Delivery Metrics

- schema 支持 query type 相关字段
- replay 构建器具备 query type 分类入口
- seed import 能接入并保留 query type 字段
- records.jsonl、summary 和 report 支持分 query type 输出

## Verification Metrics

- 新旧样本 schema 兼容验证通过
- replay 构建输出包含 query type 字段
- seed import 保留 query type 字段且旧输入兼容
- 使用真实 run-record 形状数据验证 `summary.json` 和 `report.md` 含 `by_query_type` 结果

## Required Artifacts

- Updated `harness/backlog/todos.json`
- Updated `harness/progress/progress.txt`
- `harness/sprints/S001/generator_notes.md`
- `harness/sprints/S001/evaluator_report.md`
- Git commit hash

## Exit Criteria

- `T-001`、`T-002`、`T-003`、`T-004` 全部完成实现并进入 evaluator 验收
- generator 提交实现说明与 commit hash
- evaluator 依据 verification.json 给出正式结论
