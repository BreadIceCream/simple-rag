# Evals 命令文档

本文档基于当前 [app/evals](D:/Bread/College/AI/Code/RAG/app/evals) 最新代码，覆盖：

1. `app/evals` 下主要文件作用
2. 所有可执行脚本参数说明
3. 可直接执行的评估流程
4. querytype specialized synthetic 的当前行为、诊断与边界

## 1. 四类查询类型与当前能力

项目内四类 query type：

- `single_hop_specific`
- `single_hop_abstract`
- `multi_hop_specific`
- `multi_hop_abstract`

当前最稳妥的四类评估入口：

- `replay + classify-query-type`
- `seed + explicit query_type`
- `build_querytype_dataset.py`

## 2. app/evals 文件说明

### 2.1 核心模型与运行时

#### [app/evals/__init__.py](D:/Bread/College/AI/Code/RAG/app/evals/__init__.py)
- 作用：`app.evals` 包标记。
- CLI 参数：无。

#### [app/evals/schema.py](D:/Bread/College/AI/Code/RAG/app/evals/schema.py)
- 作用：定义 `EvalDatasetManifest`、`EvalSample`、`EvalRunRecord`。
- 重点字段：`query_type`、`hop_count`、`abstraction_level`、`evidence_topology`、`reasoning_hops`、`query_type_source`。
- CLI 参数：无。

#### [app/evals/runtime.py](D:/Bread/College/AI/Code/RAG/app/evals/runtime.py)
- 作用：初始化和关闭评估运行时依赖。
- 主要 profile：`dataset_seed`、`dataset_synthetic`、`dataset_replay`、`full`。
- CLI 参数：无。

### 2.2 数据集构建

#### [app/evals/dataset_builder.py](D:/Bread/College/AI/Code/RAG/app/evals/dataset_builder.py)
- 作用：统一数据集落盘与审核流程。
- 产物：`manifest.json`、`samples.jsonl`、`review_sheet.csv`、`review_guide.md`。
- 子命令：
- `export-review`
  - `--dataset-dir` 必填
- `apply-review`
  - `--dataset-dir` 必填
  - `--review-file` 必填

#### [app/evals/build_replay_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_replay_dataset.py)
- 作用：从历史数据构建 replay 数据集，可启用 query type 分类。
- 参数：
- `--name` 必填
- `--version` 必填
- `--category` 默认 `baseline`
- `--limit` 默认 `100`
- `--seed` 默认 `42`
- `--reference-mode` 默认 `history`
- `--difficulty` 默认 `unknown`
- `--scenario` 默认 `single_turn`
- `--description`
- `--output-dir`
- `--llm-retries` 默认 `3`
- `--retry-backoff-seconds` 默认 `2.0`
- `--classify-query-type`
- `--query-type-mode` 当前仅 `heuristic`

#### [app/evals/import_seed_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/import_seed_dataset.py)
- 作用：导入 seed 数据集，支持显式导入 query type 字段。
- 可导入字段：`query_type`、`hop_count`、`abstraction_level`、`evidence_topology`、`reasoning_hops`、`query_type_source`。
- 参数：
- `--input` 必填
- `--name` 必填
- `--version` 必填
- `--category` 默认 `baseline`
- `--source-type` 默认 `manual`
- `--default-difficulty` 默认 `unknown`
- `--default-scenario` 默认 `single_turn`
- `--default-scope` 默认 `none`
- `--description`
- `--output-dir`

#### [app/evals/build_synthetic_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_synthetic_dataset.py)
- 作用：旧 synthetic 构建入口，继续保留兼容。
- 参数：
- `--name` 必填
- `--version` 必填
- `--category`
- `--size`
- `--doc-limit`
- `--seed`
- `--recency-tau-days`
- `--alloc-alpha`
- `--use-light-model`
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

#### [app/evals/build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py)
- 作用：四类 query type 的专项 synthetic 构建入口，面向 query-type 配置实验与 multi-hop 压测。
- 当前能力：
- query distribution：`profile/json/file`
- validator：`off|warn|strict`
- per-batch availability probe
- deterministic non-retriable cluster error handling
- deterministic fallback / reallocation
- `QUERYTYPE BATCH` 控制台诊断日志
- 真正生效的 `--enable-multi-file`
- manifest metadata 中的分布与诊断摘要
- 参数：
- `--name` 必填
- `--version` 必填
- `--category` 默认 `specialized`
- `--size` 默认 `100`
- `--doc-limit` 默认 `30`
- `--seed` 默认 `42`
- `--recency-tau-days` 默认 `30.0`
- `--alloc-alpha` 默认 `0.7`
- `--use-light-model`
- `--difficulty` 默认 `unknown`
- `--scenario` 默认 `single_turn`
- `--description`
- `--output-dir`
- `--query-distribution-profile` 默认 `multihop_focus`
- `--query-distribution-json`
- `--query-distribution-file`
- `--validator-mode` 默认 `warn`
- `--min-hop-evidence` 默认 `2`
- `--enable-multi-file`
- `--max-batch-retries` 默认 `3`
- `--retry-backoff-seconds` 默认 `2.0`
- `--max-chunks-per-batch` 默认 `0`
- `--ragas-max-workers` 默认 `1`
- `--ragas-timeout` 默认 `240`
- `--ragas-max-retries` 默认 `8`
- `--ragas-max-wait` 默认 `30`
- `--llm-timeout` 默认 `240`
- `--llm-max-retries` 默认 `6`
- `--llm-requests-per-second` 默认 `0.5`

#### [app/evals/querytype_synthesizers.py](D:/Bread/College/AI/Code/RAG/app/evals/querytype_synthesizers.py)
- 作用：query-type synthesizer 封装层。
- 提供：
- `QUERY_DISTRIBUTION_PROFILES`
- `resolve_query_distribution(...)`
- `allocate_query_type_counts(...)`
- `probe_available_query_types(...)`
- `classify_querytype_error(...)`
- `reallocate_query_type_counts(...)`
- `QueryTypeSynthesizerFacade`
- 说明：`single_hop_abstract` 通过项目适配器生成。

#### [app/evals/querytype_validator.py](D:/Bread/College/AI/Code/RAG/app/evals/querytype_validator.py)
- 作用：query type 规则校验。
- 支持：
- query type 合法性检查
- multi-hop / single-hop 证据一致性检查
- `warn|strict|off`
- 结果写入 `sample.metadata["validation"]`

### 2.3 运行与评分

#### [app/evals/live_rag_runner.py](D:/Bread/College/AI/Code/RAG/app/evals/live_rag_runner.py)
- 作用：执行真实 RAG 推理并产出 run artifacts。
- 参数：
- `--dataset-dir` 必填
- `--output-root`
- `--limit`
- `--review-status` 默认 `approved,pending`

#### [app/evals/ragas_scorer.py](D:/Bread/College/AI/Code/RAG/app/evals/ragas_scorer.py)
- 作用：读取 run 目录并评分。
- 参数：
- `--run-dir` 必填
- 产物：`summary.json`、`item_scores.csv`、`report.md`

#### [app/evals/ragas_runner.py](D:/Bread/College/AI/Code/RAG/app/evals/ragas_runner.py)
- 作用：一键执行 `live_rag_runner + ragas_scorer`。
- 参数：
- `--dataset-dir` 必填
- `--output-root`
- `--limit`
- `--review-status` 默认 `approved,pending`

#### [app/evals/retrieval_scorer.py](D:/Bread/College/AI/Code/RAG/app/evals/retrieval_scorer.py)
- 作用：检索指标计算。

#### [app/evals/reporter.py](D:/Bread/College/AI/Code/RAG/app/evals/reporter.py)
- 作用：统一报告落盘。

#### [app/evals/metrics_registry.py](D:/Bread/College/AI/Code/RAG/app/evals/metrics_registry.py)
- 作用：指标注册与默认 retrieval k 管理。

## 3. 可直接执行的评估流程

### 流程 A：Replay 快速基线

1. 构建 replay 数据集
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

2. 审核导出与回填
```powershell
python -m app.evals.dataset_builder export-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1
```

```powershell
python -m app.evals.dataset_builder apply-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-file store/evals/datasets/baseline/replay_qtype_baseline/v1/review_sheet.csv
```

3. 执行真实 RAG
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

1. 准备显式带 `query_type` 的 `seeds_querytype.jsonl`
2. 导入 seed 数据集
```powershell
python -m app.evals.import_seed_dataset `
  --input .\seeds_querytype.jsonl `
  --name seed_qtype_baseline `
  --version v1 `
  --category baseline `
  --source-type manual
```

3. 后续执行与流程 A 相同：
- `dataset_builder export-review`
- `dataset_builder apply-review`
- `live_rag_runner`
- `ragas_scorer`

### 流程 C：专项 synthetic 配置实验

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

#### C4. 后续执行顺序

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

## 4. querytype synthetic 的当前运行特征

### 4.1 multi-hop 不可用时的行为
如果某个 batch 缺少足够的关系图信号，控制台可能出现：

```text
QUERYTYPE BATCH: ... category=cluster_missing ... retriable=False
QUERYTYPE BATCH: ... action=fallback_to_single_hop ...
QUERYTYPE BATCH: ... event=secondary_fallback ...
```

这表示：

1. 当前 batch 不适合生成 multi-hop
2. 系统已将该错误识别为 non-retriable
3. 脚本不会对同一路径盲重试
4. 配额会被重分配到可用的 single-hop 类型

只有后续出现 `event=batch_skipped` 或最终样本数明显不足，才说明需要继续排查。

### 4.2 manifest 中的诊断信息
`manifest.json` 的 `metadata` 中可查看：

- `requested_query_type_counts`
- `effective_query_type_counts`
- `generated_query_type_counts`
- `availability_probe_summary`
- `fallback_event_summary`
- `non_retriable_failure_summary`

### 4.3 multi-file 行为
开启 `--enable-multi-file` 后：

1. 构建器会优先尝试把不同文件的 sub-batch 合并成 multi-file batch
2. 如果找不到可配对的文件，会退回 `single_file_fallback`
3. 该回退会出现在控制台日志和 manifest metadata 中

## 5. 迁移策略与边界

### 5.1 迁移策略
- [build_synthetic_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_synthetic_dataset.py) 保持兼容，继续可用。
- [build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py) 是四类 query type 的专项入口。
- 新的 query-type 实验建议优先使用 `build_querytype_dataset.py`。

### 5.2 当前 out-of-scope
- 不包含 [retrieval_scorer.py](D:/Bread/College/AI/Code/RAG/app/evals/retrieval_scorer.py) 指标体系重构
- 不包含 [ragas_scorer.py](D:/Bread/College/AI/Code/RAG/app/evals/ragas_scorer.py) / [reporter.py](D:/Bread/College/AI/Code/RAG/app/evals/reporter.py) 大改
- 不包含 Phase C 的 multi-hop 检索新指标改造

## 6. 结果查看

核心文件：

- `summary.json`
- `report.md`
- `item_scores.csv`

重点关注：

- `by_query_type` 的样本分布
- `multi_hop_specific` 与 `multi_hop_abstract` 的结果差距
- specialized synthetic 的实际生成分布与请求分布是否偏差过大

## 7. 环境注意事项

专项 synthetic 构建依赖本地运行时组件。
如果执行 `build_querytype_dataset` 时出现依赖缺失，例如 `asyncpg`，需要先补齐本地环境后再做真实 E2E 验证。
