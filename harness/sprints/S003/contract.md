# Sprint S003 Contract

## Goal

修复 querytype synthetic 生成在 deterministic multi-hop cluster/relationship 缺失时的鲁棒性问题，交付可用性探测、非重试错误分类、query type 降级重分配、多文件处理边界和控制台诊断日志。

## In Scope

- T-009：为 querytype synthesizers 增加可用性探测与非重试错误分类
- T-010：在 querytype builder 中接入非重试路径、fallback 与控制台诊断日志
- T-011：让 enable-multi-file 行为可执行且与合同边界一致
- T-012：更新命令与诊断文档以反映鲁棒性修复后的行为

## Out of Scope

- Phase C scorer/reporter/retrieval metric redesign
- 与 querytype synthetic 鲁棒性无关的 evals 重构
- 改写已通过的 S001/S002 目标或合同
- 扩展到线上评估链路的新指标体系

## Implementation Rules

- 必须先修复 deterministic cluster failure 的控制面逻辑，再考虑重试；禁止仅通过增加重试次数应对该问题。
- `querytype_synthesizers.py` 必须集中承载 availability probe、non-retriable cluster error 分类和 deterministic reallocation 策略。
- `build_querytype_dataset.py` 必须在每个 sub-batch 生成前执行 probe 与 effective query type counts 计算，并打印控制台诊断日志。
- cluster/relationship 缺失类错误不得进入盲重试；如需降级，必须记录 fallback reason、source/target query type 和 effective counts。
- 必须对 `--enable-multi-file` 给出可执行行为；若无法形成 multi-file batch，必须打印明确日志并记录 metadata，而不是静默写 metadata。
- `build_synthetic_dataset.py` 默认 CLI 与行为必须保持兼容；若抽取极小公共 helper，必须有明确必要性且不改变现有输出语义。

## Delivery Metrics

- `build_querytype_dataset.py` 能在 deterministic cluster failure 场景下收敛为 probe/fallback/skip 路径，而不是连续盲重试。
- 控制台可见每个 sub-batch 的 requested counts、available query types、fallback 事件、error classification 和 generated counts。
- manifest metadata 含 availability/fallback/effective counts/non-retriable diagnostics 摘要。
- `--enable-multi-file` 行为与合同一致，可被日志与 metadata 证明。

## Verification Metrics

- 可复现 cluster/relationship 缺失路径并证明不再进行盲重试。
- 可证明 fallback/reallocation 结果 deterministic。
- 可证明控制台日志输出满足排障字段要求。
- 可证明 CLI 兼容性保留，且 `--enable-multi-file` 行为与合同描述一致。

## Required Artifacts

- `app/evals/querytype_synthesizers.py`
- `app/evals/build_querytype_dataset.py`
- `docs/Evals命令文档.md`
- `harness/backlog/todos.json`
- `harness/sprints/S003/generator_notes.md`
- `harness/sprints/S003/verification.json`
- `harness/progress/progress.txt`
- Git commit hash

## Exit Criteria

- T-009..T-012 均达到 implemented 并由 generator 记录实现说明。
- verification.json 中 V-009..V-012 足以让 evaluator 在不发明新要求的情况下完成验收。
- `validate_harness.ps1` 通过后方可交接给 generator。
