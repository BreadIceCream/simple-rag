# Evals 命令文档（S002 后）

本文档基于当前 `app/evals` 最新代码，覆盖：
1. `app/evals` 下全部文件作用。
2. 所有可执行脚本参数说明。
3. 可直接执行的评估流程（包含四类查询类型）。
4. S002（Phase B）新增能力与边界。

适用目录：
- `app/evals`

## 1. 四类查询类型与当前能力

项目内四类 query type：
- `single_hop_specific`
- `single_hop_abstract`
- `multi_hop_specific`
- `multi_hop_abstract`

当前最稳妥的四类评估入口：
- `replay + classify-query-type`（快速现状）
- `seed + explicit query_type`（可控覆盖）
- `build_querytype_dataset.py`（S002 新增，专项 synthetic 配置实验）

## 2. app/evals 文件说明

### 2.1 核心模型与运行时

#### `app/evals/__init__.py`
- 作用：`app.evals` 包标记与模块说明。
- CLI 参数：无。

#### `app/evals/schema.py`
- 作用：定义 `EvalDatasetManifest`、`EvalSample`、`EvalRunRecord`。
- 重点字段：`query_type`、`hop_count`、`abstraction_level`、`evidence_topology`、`reasoning_hops`、`query_type_source`。
- CLI 参数：无。

#### `app/evals/runtime.py`
- 作用：初始化/关闭评估运行时依赖。
- 主要 profile：`dataset_seed`、`dataset_synthetic`、`dataset_replay`、`full`。
- CLI 参数：无。

### 2.2 数据集构建

#### `app/evals/dataset_builder.py`
- 作用：统一数据集落盘与审核流程。
- 产物：`manifest.json`、`samples.jsonl`、`review_sheet.csv`、`review_guide.md`。
- 子命令：
- `export-review`
- 参数：`--dataset-dir`（必填）
- `apply-review`
- 参数：`--dataset-dir`（必填）、`--review-file`（必填）

#### `app/evals/build_replay_dataset.py`
- 作用：从历史数据构建 replay 数据集；可启用 query type 分类。
- 参数：
- `--name`（必填）
- `--version`（必填）
- `--category`（默认 `baseline`，可选 `regression|baseline|exploration|specialized`）
- `--limit`（默认 `100`）
- `--seed`（默认 `42`）
- `--reference-mode`（默认 `history`，可选 `history|ai`）
- `--difficulty`（默认 `unknown`）
- `--scenario`（默认 `single_turn`）
- `--description`（默认空）
- `--output-dir`（默认自动目录）
- `--llm-retries`（默认 `3`）
- `--retry-backoff-seconds`（默认 `2.0`）
- `--classify-query-type`（flag）
- `--query-type-mode`（当前仅 `heuristic`）

#### `app/evals/import_seed_dataset.py`
- 作用：导入 seed 数据集；支持显式导入 query type 相关字段。
- 可导入字段：`query_type`、`hop_count`、`abstraction_level`、`evidence_topology`、`reasoning_hops`、`query_type_source`。
- 参数：
- `--input`（必填）
- `--name`（必填）
- `--version`（必填）
- `--category`（默认 `baseline`，可选 `regression|baseline|exploration|specialized|synthetic`）
- `--source-type`（默认 `manual`）
- `--default-difficulty`（默认 `unknown`）
- `--default-scenario`（默认 `single_turn`）
- `--default-scope`（默认 `none`，可选 `none|all`）
- `--description`（默认空）
- `--output-dir`（默认自动目录）

#### `app/evals/build_synthetic_dataset.py`
- 作用：旧 synthetic 构建入口（保留兼容）。
- 参数：
- `--name`（必填）
- `--version`（必填）
- `--category`（`synthetic|exploration|specialized`）
- `--size`
- `--doc-limit`
- `--seed`
- `--recency-tau-days`
- `--alloc-alpha`
- `--use-light-model`（flag）
- `--difficulty`
- `--scenario`
- `--description`
- `--output-dir`
- `--max-batch-retries`
- `--retry-backoff-seconds`
- `--max-chunks-per-batch`
- `--max-topup-rounds`
- `--ragas-max-workers`
- `--ragas-timeout`
- `--ragas-max-retries`
- `--ragas-max-wait`
- `--llm-timeout`
- `--llm-max-retries`
- `--llm-requests-per-second`

#### `app/evals/build_querytype_dataset.py`（S002 新增）
- 作用：Phase B 专项 synthetic 构建入口，面向 query-type 配置实验。
- 内置能力：
- query distribution（profile/json/file）
- multi-hop 偏重分布
- validator（`off|warn|strict`）
- manifest metadata 写入：`requested_distribution`、`realized_distribution`、`validation_summary`、`experiment_config`
- 参数：
- `--name`（必填）
- `--version`（必填）
- `--category`（默认 `specialized`，可选 `synthetic|exploration|specialized`）
- `--size`（默认 `100`）
- `--doc-limit`（默认 `30`）
- `--seed`（默认 `42`）
- `--recency-tau-days`（默认 `30.0`）
- `--alloc-alpha`（默认 `0.7`）
- `--use-light-model`（flag）
- `--difficulty`（默认 `unknown`）
- `--scenario`（默认 `single_turn`）
- `--description`（默认空）
- `--output-dir`（默认自动目录）
- `--query-distribution-profile`（默认 `multihop_focus`，可选 `balanced|multihop_focus`）
- `--query-distribution-json`（可选）
- `--query-distribution-file`（可选）
- `--validator-mode`（默认 `warn`，可选 `off|warn|strict`）
- `--min-hop-evidence`（默认 `2`）
- `--enable-multi-file`（flag）
- `--max-batch-retries`（默认 `3`）
- `--retry-backoff-seconds`（默认 `2.0`）
- `--max-chunks-per-batch`（默认 `0`）
- `--ragas-max-workers`（默认 `1`）
- `--ragas-timeout`（默认 `240`）
- `--ragas-max-retries`（默认 `8`）
- `--ragas-max-wait`（默认 `30`）
- `--llm-timeout`（默认 `240`）
- `--llm-max-retries`（默认 `6`）
- `--llm-requests-per-second`（默认 `0.5`）

#### `app/evals/querytype_synthesizers.py`（S002 新增）
- 作用：query-type synthesizer 封装层。
- 提供：
- `QUERY_DISTRIBUTION_PROFILES`
- `resolve_query_distribution(...)`
- `allocate_query_type_counts(...)`
- `QueryTypeSynthesizerFacade`
- 说明：`single_hop_abstract` 通过项目适配器生成。
- CLI 参数：无。

#### `app/evals/querytype_validator.py`（S002 新增）
- 作用：query type 规则校验。
- 支持：
- query type 合法性检查
- multi-hop/ single-hop 证据一致性检查
- `warn/strict/off` 三种模式
- 结果写入 `sample.metadata["validation"]`
- CLI 参数：无。

### 2.3 运行与评分

#### `app/evals/live_rag_runner.py`
- 作用：执行真实 RAG 推理并产出 run artifacts。
- 参数：
- `--dataset-dir`（必填）
- `--output-root`（可选）
- `--limit`（可选）
- `--review-status`（默认 `approved,pending`）

#### `app/evals/ragas_scorer.py`
- 作用：读取 run 目录并评分（RAGAS + retrieval + correctness）。
- 参数：
- `--run-dir`（必填）
- 产物：`summary.json`、`item_scores.csv`、`report.md`（含 `by_query_type` 聚合）。

#### `app/evals/ragas_runner.py`
- 作用：一键执行 `live_rag_runner + ragas_scorer`。
- 参数：
- `--dataset-dir`（必填）
- `--output-root`（可选）
- `--limit`（可选）
- `--review-status`（默认 `approved,pending`）

#### `app/evals/retrieval_scorer.py`
- 作用：检索指标计算（`precision/recall/hit/mrr/ndcg@k`）。
- CLI 参数：无。

#### `app/evals/reporter.py`
- 作用：统一报告落盘（`summary.json`、`report.md` 等）。
- CLI 参数：无。

#### `app/evals/metrics_registry.py`
- 作用：指标注册与默认 retrieval k 管理。
- CLI 参数：无。

## 3. 可直接执行的评估流程（含四类 query type）

### 流程 A：Replay 快速基线

1. 构建 replay 数据集（开启 query type 分类）
```powershell
python -m app.evals.build_replay_dataset `
  --name replay_qtype_baseline `
  --version v1 `
  --category baseline `
  --limit 100 `
  --reference-mode history `
  --classify-query-type `
  --query-type-mode heuristic
```

2. 审核导出/回填
```powershell
python -m app.evals.dataset_builder export-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1
```
```powershell
python -m app.evals.dataset_builder apply-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-file store/evals/datasets/baseline/replay_qtype_baseline/v1/review_sheet.csv
```

3. 真实运行
```powershell
python -m app.evals.live_rag_runner `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-status approved,pending
```

4. 评分
```powershell
python -m app.evals.ragas_scorer `
  --run-dir store/evals/runs/<your-run-dir>
```

### 流程 B：Seed 可控四类覆盖

1. 准备 `seeds_querytype.jsonl`（显式写入四类 `query_type`）。
2. 导入 seed 数据集
```powershell
python -m app.evals.import_seed_dataset `
  --input .\seeds_querytype.jsonl `
  --name seed_qtype_baseline `
  --version v1 `
  --category baseline `
  --source-type manual
```
3. 后续执行与流程 A 相同：`export-review -> apply-review -> live_rag_runner -> ragas_scorer`。

### 流程 C：S002 专项 synthetic 配置实验

#### C1. 平衡分布
```powershell
python -m app.evals.build_querytype_dataset `
  --name querytype_balanced_baseline `
  --version v1 `
  --category specialized `
  --size 80 `
  --doc-limit 30 `
  --query-distribution-profile balanced `
  --validator-mode warn
```

#### C2. 多跳偏重分布
```powershell
python -m app.evals.build_querytype_dataset `
  --name querytype_multihop_focus `
  --version v1 `
  --category specialized `
  --size 80 `
  --doc-limit 30 `
  --query-distribution-profile multihop_focus `
  --validator-mode strict `
  --min-hop-evidence 2 `
  --enable-multi-file
```

#### C3. 自定义分布
```powershell
python -m app.evals.build_querytype_dataset `
  --name querytype_custom_dist `
  --version v1 `
  --category specialized `
  --size 80 `
  --query-distribution-json '{"single_hop_specific":0.2,"single_hop_abstract":0.1,"multi_hop_specific":0.4,"multi_hop_abstract":0.3}' `
  --validator-mode warn
```

#### C4. 专项数据集评估顺序
1. `build_querytype_dataset`
2. `dataset_builder export-review`
3. `dataset_builder apply-review`
4. `live_rag_runner`
5. `ragas_scorer`

示例（以 multihop_focus 为例）：
```powershell
python -m app.evals.dataset_builder export-review `
  --dataset-dir store/evals/datasets/specialized/querytype_multihop_focus/v1
```
```powershell
python -m app.evals.dataset_builder apply-review `
  --dataset-dir store/evals/datasets/specialized/querytype_multihop_focus/v1 `
  --review-file store/evals/datasets/specialized/querytype_multihop_focus/v1/review_sheet.csv
```
```powershell
python -m app.evals.live_rag_runner `
  --dataset-dir store/evals/datasets/specialized/querytype_multihop_focus/v1 `
  --review-status approved,pending
```
```powershell
python -m app.evals.ragas_scorer `
  --run-dir store/evals/runs/<your-run-dir>
```

## 4. S002 迁移策略与边界

### 4.1 迁移策略
- `build_synthetic_dataset.py` 保持兼容，继续可用。
- `build_querytype_dataset.py` 是 Phase B 的专项入口，用于 query type 可控分布实验。
- 建议新实验逐步迁移到 `build_querytype_dataset.py`。

### 4.2 Phase B 明确 out-of-scope（本次修复补充）
- Phase B 不包含 `retrieval_scorer.py` 指标体系重构。
- Phase B 不包含 `ragas_scorer.py`/`reporter.py` 的评分类能力重构。
- Phase B 不包含 Phase C 的 multi-hop 检索新指标改造。

## 5. 结果查看

核心查看文件：
- `summary.json`
- `report.md`
- `item_scores.csv`

重点关注：
- `by_query_type` 的样本分布与均值指标
- `multi_hop_specific` 与 `multi_hop_abstract` 的表现差距

## 6. 环境注意事项

专项 synthetic 构建依赖运行时组件（数据库/向量库/模型配置等）。
如果执行 `build_querytype_dataset` 时出现依赖缺失（如 `asyncpg`），需要先补齐本地环境依赖后再做 E2E 验证。

## 7. S003 鲁棒性修复（deterministic cluster failure）

本节描述 S003 之后 `build_querytype_dataset.py` 的关键行为变化。

### 7.1 修复点
- 增加 per-batch query type 可用性探测（availability probe）。
- 增加 deterministic cluster/relationship 缺失错误分类（non-retriable）。
- 增加 query type 配额 deterministic fallback/reallocation。
- 增加每个 sub-batch 的控制台诊断日志。
- `--enable-multi-file` 不再是 metadata-only：会尝试构造 multi-file batch；若无法构造会显式输出 fallback 日志。

### 7.2 non-retriable 错误处理
当出现类似错误时：
- `No relationships match the provided condition. Cannot form clusters.`
- `No clusters found in the knowledge graph...`

系统会识别为 non-retriable，不再盲重试同一路径，而会进入 fallback（优先降级到可用 query types）。

### 7.3 诊断日志字段（控制台）
每个 sub-batch 至少会输出这些字段：
- `file_id`
- `file_name`
- `sub_batch`
- `chunk_count`
- `requested_counts`
- `available_query_types`
- `unavailable_query_types`
- `effective_counts`
- `fallback_events`
- `error classification`（retriable/category/reason）
- `generated_counts`

### 7.4 metadata 诊断摘要
`manifest.metadata` 中可查看：
- `requested_query_type_counts`
- `effective_query_type_counts`
- `generated_query_type_counts`
- `availability_probe_summary`
- `fallback_event_summary`
- `non_retriable_failure_summary`

### 7.5 multi-file 行为
开启 `--enable-multi-file` 后：
- 系统会优先尝试把不同文件的 sub-batch 合并成 multi-file batch。
- 若找不到可合并对象，会退回 single-file fallback，并输出明确日志和 metadata 标记。

### 7.6 观察修复行为的示例命令
```powershell
python -m app.evals.build_querytype_dataset `
  --name querytype_multihop_focus_s003 `
  --version v1 `
  --category specialized `
  --size 40 `
  --doc-limit 20 `
  --query-distribution-profile multihop_focus `
  --validator-mode warn `
  --enable-multi-file `
  --max-batch-retries 3
```

建议同时观察：
- 控制台 `QUERYTYPE BATCH:` 行
- 产物 `manifest.json` 中的 `availability_probe_summary` / `fallback_event_summary`

### 7.7 仍不在本阶段范围
S003 仍然不包含：
- `retrieval_scorer.py` 指标体系重构
- `ragas_scorer.py` / `reporter.py` 重构
- Phase C 的 multi-hop 评分指标体系改造
