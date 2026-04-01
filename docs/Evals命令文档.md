# Evals 命令文档

本文档基于当前仓库最新 `app/evals` 代码整理，目标是说明：

1. `app/evals` 目录下每个文件的作用。
2. 所有可执行脚本的参数及含义。
3. 当前可直接执行的评估流程，尤其是如何围绕四类查询类型开展评估。

适用目录：

- [`app/evals`](D:/Bread/College/AI/Code/RAG/app/evals)

## 1. 总体说明

当前 `app/evals` 的评估链路可以分成 3 段：

1. 构建数据集
2. 运行真实 RAG，生成 run artifacts
3. 对 run artifacts 进行评分并出报告

在当前实现里，四类查询类型支持最完整的是这两条链路：

1. `replay dataset`
   通过 `build_replay_dataset.py --classify-query-type` 对现有历史样本做 query type 分类。
2. `seed dataset`
   通过 `import_seed_dataset.py` 显式导入 `query_type` 等字段。

当前 `synthetic dataset` 仍然主要沿用旧逻辑，尚未形成完整的“四类查询专项生成器”实现，因此如果你的目标是“立刻评估当前系统在四类查询上的现状”，优先建议使用：

1. `replay + classify-query-type`
2. `seed + explicit query_type`

当前项目内部采用的四类查询类型为：

- `single_hop_specific`
- `single_hop_abstract`
- `multi_hop_specific`
- `multi_hop_abstract`

## 2. 目录文件说明

### 2.1 包与数据模型

#### `app/evals/__init__.py`

作用：

- `app.evals` 包标记文件。
- 提供模块级说明，表明该目录用于“数据集构建、线上 RAG 执行、评分”。

参数：

- 无。

#### `app/evals/schema.py`

作用：

- 定义评估体系的核心数据结构。
- 目前最重要的三个结构：
  - `EvalDatasetManifest`
  - `EvalSample`
  - `EvalRunRecord`
- 负责 query type 相关字段的归一化、兼容旧数据、默认值处理。

当前 query type 相关字段包括：

- `query_type`
- `hop_count`
- `abstraction_level`
- `evidence_topology`
- `reasoning_hops`
- `query_type_source`

参数：

- 无 CLI 参数。

#### `app/evals/runtime.py`

作用：

- 构建评估运行时依赖。
- 根据 profile 初始化数据库、向量检索、LLM、embedding 等运行时对象。
- 供 dataset builder、runner、scorer 复用。

当前 profile：

- `dataset_seed`
- `dataset_synthetic`
- `dataset_replay`
- `full`

参数：

- 无独立 CLI 参数。

### 2.2 数据集构建与审核

#### `app/evals/dataset_builder.py`

作用：

- 提供数据集通用读写能力。
- 生成和维护：
  - `manifest.json`
  - `samples.jsonl`
  - `review_sheet.csv`
  - `review_guide.md`
- 支持导出审核表和回填审核结果。

子命令：

1. `export-review`
2. `apply-review`

##### `export-review` 参数

- `--dataset-dir`
  - 必填。
  - 数据集目录，例如 `store/evals/datasets/baseline/replay_qtype_baseline/v1`。

##### `apply-review` 参数

- `--dataset-dir`
  - 必填。
  - 目标数据集目录。
- `--review-file`
  - 必填。
  - 要回填的审核 CSV 文件路径。

#### `app/evals/build_replay_dataset.py`

作用：

- 从历史问答/对话记录中抽取样本，构建 replay 数据集。
- 是当前“快速得到四类 query type 分布”的主入口。
- 支持启用 query type 分类，把 replay 样本补齐为四类标签。

参数：

- `--name`
  - 必填。
  - 数据集名称。
- `--version`
  - 必填。
  - 数据集版本。
- `--category`
  - 可选，默认 `baseline`。
  - 可选值：`regression`、`baseline`、`exploration`、`specialized`。
- `--limit`
  - 可选，默认 `100`。
  - 抽样样本上限。
- `--seed`
  - 可选，默认 `42`。
  - 抽样随机种子。
- `--reference-mode`
  - 可选，默认 `history`。
  - 可选值：`history`、`ai`。
  - `history` 通常表示从历史上下文推断参考证据。
- `--difficulty`
  - 可选，默认 `unknown`。
  - 写入样本难度标签。
- `--scenario`
  - 可选，默认 `single_turn`。
  - 写入样本场景标签。
- `--description`
  - 可选，默认空字符串。
  - 数据集描述。
- `--output-dir`
  - 可选，默认自动生成。
  - 指定输出目录。
- `--llm-retries`
  - 可选，默认 `3`。
  - LLM 辅助步骤的最大重试次数。
- `--retry-backoff-seconds`
  - 可选，默认 `2.0`。
  - 失败重试间隔。
- `--classify-query-type`
  - 可选 flag。
  - 启用 query type 分类。
- `--query-type-mode`
  - 可选，当前只支持 `heuristic`，默认 `heuristic`。
  - 使用启发式规则对 replay 样本做四分类。

说明：

- 这是当前最方便的四类查询评估入口。
- 但分类是启发式的，不保证四类数量均衡。

#### `app/evals/import_seed_dataset.py`

作用：

- 导入人工整理或外部准备好的 seed 数据集。
- 当前支持显式导入 query type 相关字段，是“保证四类查询全覆盖”的推荐入口。

支持从输入样本中读取的 query type 字段：

- `query_type`
- `hop_count`
- `abstraction_level`
- `evidence_topology`
- `reasoning_hops`
- `query_type_source`

参数：

- `--input`
  - 必填。
  - 输入文件路径，通常为 `.jsonl`。
- `--name`
  - 必填。
  - 数据集名称。
- `--version`
  - 必填。
  - 数据集版本。
- `--category`
  - 可选，默认 `baseline`。
  - 可选值：`regression`、`baseline`、`exploration`、`specialized`、`synthetic`。
- `--source-type`
  - 可选，默认 `manual`。
  - 数据来源类型说明。
- `--default-difficulty`
  - 可选，默认 `unknown`。
  - 当输入样本未提供难度时使用。
- `--default-scenario`
  - 可选，默认 `single_turn`。
  - 当输入样本未提供场景时使用。
- `--default-scope`
  - 可选，默认 `none`。
  - 可选值：`none`、`all`。
  - 为未提供 scope 的样本设置默认行为。
- `--description`
  - 可选，默认空字符串。
  - 数据集说明。
- `--output-dir`
  - 可选，默认自动生成。
  - 输出目录。

说明：

- 如果你希望四类 query type 都有明确样本，优先推荐 seed 导入而不是 replay 自动分类。

#### `app/evals/build_synthetic_dataset.py`

作用：

- 基于当前 synthetic 生成逻辑构建合成数据集。
- 仍可用于生成探索性测试集。
- 但目前尚不是“四类查询类型专项评估”的最佳入口。

参数：

- `--name`
  - 必填。
- `--version`
  - 必填。
- `--category`
  - 可选。
  - 可选值：`synthetic`、`exploration`、`specialized`。
- `--size`
  - 可选。
  - 目标样本数量。
- `--doc-limit`
  - 可选。
  - 限制参与构建的文档数。
- `--seed`
  - 可选。
  - 随机种子。
- `--recency-tau-days`
  - 可选。
  - 控制文档时间衰减参数。
- `--alloc-alpha`
  - 可选。
  - 控制采样分配策略。
- `--use-light-model`
  - 可选 flag。
  - 使用较轻量模型生成。
- `--difficulty`
  - 可选。
- `--scenario`
  - 可选。
- `--description`
  - 可选。
- `--output-dir`
  - 可选。
- `--max-batch-retries`
  - 可选。
- `--retry-backoff-seconds`
  - 可选。
- `--max-chunks-per-batch`
  - 可选。
- `--max-topup-rounds`
  - 可选。
- `--ragas-max-workers`
  - 可选。
- `--ragas-timeout`
  - 可选。
- `--ragas-max-retries`
  - 可选。
- `--ragas-max-wait`
  - 可选。
- `--llm-timeout`
  - 可选。
- `--llm-max-retries`
  - 可选。
- `--llm-requests-per-second`
  - 可选。

说明：

- 当前脚本还不是 RAGAS 集成方案 2.0 中“专项四类查询生成器”的最终形态。
- 如果目标是评估四类 query type，当前优先级低于 `replay` 和 `seed`。

### 2.3 真实运行与评分

#### `app/evals/live_rag_runner.py`

作用：

- 用真实 RAG 系统对数据集逐条运行。
- 生成评估 run artifacts。
- 当前已支持把 query type 字段从样本透传到 `EvalRunRecord`。

参数：

- `--dataset-dir`
  - 必填。
  - 数据集目录。
- `--output-root`
  - 可选。
  - run artifacts 的输出根目录。
- `--limit`
  - 可选。
  - 限制执行样本数。
- `--review-status`
  - 可选，默认 `approved,pending`。
  - 用逗号分隔的审核状态白名单。

说明：

- 这是生成 `records.jsonl` 的核心脚本。
- 如果要做 `by_query_type` 报表，必须经过这个脚本让 query type 写入 run record。

#### `app/evals/ragas_scorer.py`

作用：

- 读取 `live_rag_runner.py` 生成的 run 目录。
- 运行 RAGAS 指标、检索指标、答案正确性判断。
- 输出总体 summary、逐样本结果和按 query type 分桶统计。

参数：

- `--run-dir`
  - 必填。
  - 由 `live_rag_runner.py` 生成的 run 目录。

当前输出重点：

- `summary.json`
- `item_scores.csv`
- `report.md`

当前已支持：

- `by_query_type` 聚合
- `report.md` 中的 `Query Type Breakdown`
- `report.md` 中的 `Metrics by Query Type`

#### `app/evals/ragas_runner.py`

作用：

- 一键执行：
  - `live_rag_runner`
  - `ragas_scorer`
- 适合快速跑通完整链路。

参数：

- `--dataset-dir`
  - 必填。
- `--output-root`
  - 可选。
- `--limit`
  - 可选。
- `--review-status`
  - 可选，默认 `approved,pending`。

说明：

- 如果你不想手动分两步跑，可以直接使用该脚本。
- 但在排障时，仍建议先跑 `live_rag_runner`，确认 run artifacts 正常，再跑 `ragas_scorer`。

### 2.4 评分与输出支持模块

#### `app/evals/retrieval_scorer.py`

作用：

- 计算传统检索指标。
- 当前主要输出：
  - `precision@k`
  - `recall@k`
  - `hit@k`
  - `mrr@k`
  - `ndcg@k`

参数：

- 无独立 CLI 参数。

说明：

- 当前 Phase A 的重点是“按 query type 分桶统计这些指标”，而不是新增 multi-hop 专项检索指标。

#### `app/evals/reporter.py`

作用：

- 将 scoring 结果落盘为可读 artifacts。
- 负责写出：
  - `dataset_manifest.json`
  - `config.json`
  - `run_meta.json`
  - `records.jsonl`
  - `summary.json`
  - `item_scores.csv`
  - `report.md`

参数：

- 无独立 CLI 参数。

#### `app/evals/metrics_registry.py`

作用：

- 统一定义和注册当前启用的评估指标。
- 根据能力选择 RAGAS metrics。

当前主要包含：

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`
- `context_entities_recall`

以及默认 retrieval `k` 值：

- `1`
- `3`
- `5`
- `8`

参数：

- 无独立 CLI 参数。

## 3. 当前推荐的评估方式

当前如果你的目标是：

1. 评估现有系统在四类查询上的真实表现。
2. 不改 synthetic 主逻辑。
3. 立刻拿到第一版 query-type 画像。

推荐顺序如下：

1. 优先跑 `replay dataset + query type classification`
2. 如果想确保四类样本都存在，再补一套 `seed dataset`
3. 对通过审核的数据集运行 `live_rag_runner`
4. 再用 `ragas_scorer` 出报告

## 4. 可直接执行的评估流程

下面给出两套流程：

1. 快速基线流程：`replay`
2. 稳定覆盖四类流程：`seed`

### 4.1 流程一：Replay 基线评估

适用场景：

- 你想先看“当前线上/历史真实问题”在四类 query type 上的表现。
- 接受 query type 来自启发式分类，不强求四类样本数量均衡。

#### 第一步：构建 replay 数据集，并启用 query type 分类

```powershell
python -m app.evals.build_replay_dataset `
  --name replay_qtype_baseline `
  --version v1 `
  --category baseline `
  --limit 100 `
  --reference-mode history `
  --difficulty unknown `
  --scenario single_turn `
  --classify-query-type `
  --query-type-mode heuristic
```

说明：

- 该命令会生成 replay 数据集。
- `--classify-query-type` 会为样本补齐四类 query type 字段。
- `--query-type-mode heuristic` 目前是唯一支持模式。

预期结果：

- 生成一个数据集目录，通常位于：
  - `store/evals/datasets/baseline/replay_qtype_baseline/v1`

#### 第二步：导出审核表

```powershell
python -m app.evals.dataset_builder export-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1
```

说明：

- 该步骤会生成 `review_sheet.csv`。
- 你可以人工审核样本、修正文案、过滤不合格样本。

#### 第三步：回填审核结果

```powershell
python -m app.evals.dataset_builder apply-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-file store/evals/datasets/baseline/replay_qtype_baseline/v1/review_sheet.csv
```

说明：

- 如果你已经完成审核，就执行该命令回填。
- 如果只是快速试跑，也可以直接让后续 runner 读取 `approved,pending`。

#### 第四步：运行真实 RAG，生成 run artifacts

```powershell
python -m app.evals.live_rag_runner `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-status approved,pending
```

说明：

- 该步骤会真正调用当前 RAG 系统。
- query type 会随着 sample 一起写入 `records.jsonl`。

预期结果：

- 控制台会打印类似：

```text
LIVE RUN: artifacts=store/evals/runs/...
```

- 记下这个 `run_dir`，下一步要用。

#### 第五步：评分并输出分 query type 报告

```powershell
python -m app.evals.ragas_scorer `
  --run-dir store/evals/runs/<你的run目录>
```

说明：

- 该步骤会计算：
  - RAGAS 指标
  - 检索指标
  - 正确性判断
  - `by_query_type` 聚合

重点产物：

- `summary.json`
- `item_scores.csv`
- `report.md`

推荐查看：

- `summary.json`
  - 看机器可读 summary 和 `by_query_type`
- `report.md`
  - 看人工可读总结

### 4.2 流程二：Seed 数据集评估，确保四类查询都覆盖

适用场景：

- 你希望四类查询类型都明确存在。
- 你希望更可控地观察：
  - `single_hop_specific`
  - `single_hop_abstract`
  - `multi_hop_specific`
  - `multi_hop_abstract`

#### 第一步：准备 seed 输入文件

建议准备一个 `.jsonl` 文件，例如 `seeds_querytype.jsonl`，每条样本显式包含 `query_type`。

示例字段建议：

- `user_input`
- `reference_answer`
- `reference_doc_ids`
- `scope_document_ids`
- `query_type`
- `hop_count`
- `abstraction_level`
- `evidence_topology`
- `reasoning_hops`
- `query_type_source`

四类 query type 建议至少各准备若干条：

- `single_hop_specific`
- `single_hop_abstract`
- `multi_hop_specific`
- `multi_hop_abstract`

#### 第二步：导入 seed 数据集

```powershell
python -m app.evals.import_seed_dataset `
  --input .\seeds_querytype.jsonl `
  --name seed_qtype_baseline `
  --version v1 `
  --category baseline `
  --source-type manual `
  --default-difficulty unknown `
  --default-scenario single_turn `
  --default-scope none
```

说明：

- 该命令会保留输入样本中显式提供的 query type 字段。
- 当前 seed import 不会自动分类，只会导入你提供的数据。

#### 第三步：导出审核表

```powershell
python -m app.evals.dataset_builder export-review `
  --dataset-dir store/evals/datasets/baseline/seed_qtype_baseline/v1
```

#### 第四步：回填审核结果

```powershell
python -m app.evals.dataset_builder apply-review `
  --dataset-dir store/evals/datasets/baseline/seed_qtype_baseline/v1 `
  --review-file store/evals/datasets/baseline/seed_qtype_baseline/v1/review_sheet.csv
```

#### 第五步：运行真实 RAG

```powershell
python -m app.evals.live_rag_runner `
  --dataset-dir store/evals/datasets/baseline/seed_qtype_baseline/v1 `
  --review-status approved,pending
```

#### 第六步：评分

```powershell
python -m app.evals.ragas_scorer `
  --run-dir store/evals/runs/<你的run目录>
```

说明：

- 这条流程最大的价值是：
  - 四类 query type 是你明确指定的
  - 不依赖 replay 启发式分类
  - 更适合专项分析和回归测试

### 4.3 流程三：一键执行完整评估

如果你已经有一个准备好的数据集目录，也可以直接一键执行：

```powershell
python -m app.evals.ragas_runner `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-status approved,pending
```

说明：

- 该命令会自动执行：
  - `live_rag_runner`
  - `ragas_scorer`

适合：

- 快速验证流程是否跑通。

不太适合：

- 你需要单独排查 live run 阶段或 scorer 阶段的问题。

## 5. 如何阅读评估结果

运行完成后，重点看以下文件：

### `summary.json`

主要用途：

- 机器可读汇总结果。
- 当前最关键的是查看 `by_query_type`。

你可以重点关注：

- 每个 query type 的样本量
- 每个 query type 的平均指标
- 哪一类 query type 的检索表现最差

### `report.md`

主要用途：

- 人工可读报告。

当前你应重点看：

- `Query Type Breakdown`
- `Metrics by Query Type`

### `item_scores.csv`

主要用途：

- 逐样本排查问题。

适合回答的问题：

- 哪些 `multi_hop_abstract` 样本失败最多
- 哪些 `single_hop_specific` 样本 recall 低
- 某类 query type 是否存在明显误分型

## 6. 当前限制与注意事项

### 6.1 replay 分类是启发式，不是严格标注

`build_replay_dataset.py --classify-query-type` 当前使用启发式规则。

含义：

- 可以快速得到第一版 query type 画像。
- 但不能把它等同于“严格人工标注数据集”。

建议：

- 先用 replay 看现状。
- 再用 seed 做更稳定的专项验证。

### 6.2 current synthetic 还不是四类查询专项生成器

`build_synthetic_dataset.py` 目前仍是旧的 synthetic 构建链路，并非 RAGAS 集成方案 2.0 里规划的“四类专项合成器”。

含义：

- 你可以继续用它做探索性测试。
- 但如果目的是严格评估四类 query type，不应优先依赖它。

### 6.3 by-query-type 报表依赖 run record 透传

当前代码已经把 query type 从 sample 透传到 run record，因此 `by_query_type` 报表是可用的。

这也意味着：

- 如果原始 dataset 没有 query type，最终 report 里也不会自动长出四类分桶结果。

## 7. 推荐执行顺序

如果你现在就要开始评估，推荐按下面顺序：

1. 先跑一版 replay 基线

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

2. 审核并回填

```powershell
python -m app.evals.dataset_builder export-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1
```

```powershell
python -m app.evals.dataset_builder apply-review `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-file store/evals/datasets/baseline/replay_qtype_baseline/v1/review_sheet.csv
```

3. 运行真实 RAG

```powershell
python -m app.evals.live_rag_runner `
  --dataset-dir store/evals/datasets/baseline/replay_qtype_baseline/v1 `
  --review-status approved,pending
```

4. 评分

```powershell
python -m app.evals.ragas_scorer `
  --run-dir store/evals/runs/<你的run目录>
```

5. 如果 replay 的四类分布不够均衡，再补一套 seed 数据集，确保四类 query type 都有覆盖。

## 8. 结论

基于当前最新代码，最实用的评估结论是：

1. 现在已经可以评估四类查询类型。
2. 最成熟的方式不是 synthetic，而是：
  - `replay + classify-query-type`
  - `seed + explicit query_type`
3. 真正出结果的关键文件是：
  - `build_replay_dataset.py`
  - `import_seed_dataset.py`
  - `live_rag_runner.py`
  - `ragas_scorer.py`
  - `ragas_runner.py`

如果后续要继续推进 RAGAS 集成方案 2.0，下一阶段重点应放在：

1. 四类查询专项 synthetic builder
2. multi-hop 专项 retrieval metrics
3. 更严格的 query type 标注与审核规则

## 9. Phase B 专项 Synthetic（S002）

### 9.1 新入口

Phase B 新增入口：

- `python -m app.evals.build_querytype_dataset`

该入口面向 query-type 专项集构建，支持：

- query distribution profile/json/file
- `multi_hop_specific` 与 `multi_hop_abstract` 偏重分布
- validator `off|warn|strict`
- manifest metadata 落盘：`requested_distribution`、`realized_distribution`、`validation_summary`、`experiment_config`

### 9.2 参数示例

#### A. 平衡分布（balanced）

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

#### B. 多跳偏重（multihop_focus）

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

#### C. 自定义分布（json）

```powershell
python -m app.evals.build_querytype_dataset `
  --name querytype_custom_dist `
  --version v1 `
  --category specialized `
  --size 80 `
  --query-distribution-json '{"single_hop_specific":0.2,"single_hop_abstract":0.1,"multi_hop_specific":0.4,"multi_hop_abstract":0.3}' `
  --validator-mode warn
```

### 9.3 专项评估执行顺序

1. 构建 query-type 专项数据集（上面三种任一）
2. 导出/回填审核表

```powershell
python -m app.evals.dataset_builder export-review `
  --dataset-dir store/evals/datasets/specialized/querytype_multihop_focus/v1
```

```powershell
python -m app.evals.dataset_builder apply-review `
  --dataset-dir store/evals/datasets/specialized/querytype_multihop_focus/v1 `
  --review-file store/evals/datasets/specialized/querytype_multihop_focus/v1/review_sheet.csv
```

3. 运行真实 RAG

```powershell
python -m app.evals.live_rag_runner `
  --dataset-dir store/evals/datasets/specialized/querytype_multihop_focus/v1 `
  --review-status approved,pending
```

4. 评分与报告

```powershell
python -m app.evals.ragas_scorer `
  --run-dir store/evals/runs/<你的run目录>
```

### 9.4 迁移策略（与旧 synthetic 兼容）

- `build_synthetic_dataset.py` 保持原有 CLI 和行为，不作为 query-type 专项入口。
- 新增 `build_querytype_dataset.py` 作为 Phase B 专项入口，便于配置实验和 query type 可控分布。
- 现有依赖旧命令的脚本无需改动；新实验脚本建议逐步迁移到 `build_querytype_dataset.py`。
