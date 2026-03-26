# RAGAS 真实 RAG 评测集成方案

## 1. 文档目标

本文档说明当前项目的 RAG 评测集成方案与现状实现，重点是：

1. 用真实 RAG 系统执行评测，而不是对合成四元组做静态自评。
2. 将“数据集构建”和“测评执行”彻底解耦。
3. 以统一 schema 支持多种数据集来源。
4. 让文档与当前 `app/evals` 脚本实现保持一致。

当前方案的核心结论：

1. 旧的 `build_golden_jsonl.py -> fill_response_from_graph_prompt.py -> ragas_runner.py` 不能代表真实 RAG 能力。
2. 主评测链路必须是：`数据集构建 -> 人工审核（按需） -> 真实 RAG 执行 -> RAGAS/检索指标评分 -> 结果分析`。
3. synthetic 数据集只适合作为冷启动、探索和 smoke test，不直接代表真实线上能力。

---

## 2. 设计原则

### 2.1 数据集与测评执行解耦

1. 数据集负责回答“测什么问题”。
2. 测评执行器负责回答“如何真实运行系统并如何打分”。
3. 测评执行器不应因为数据集类别不同而切换主流程。
4. 指标选择由样本字段完整性和 `capabilities` 决定，而不是由数据集类别决定。

### 2.2 真实评测优先

所有正式评测都应遵守：

1. `user_input` 由数据集提供。
2. `actual_response`、`actual_contexts`、`actual_doc_ids` 必须由真实 RAG 链路运行得到。
3. RAGAS 只负责对真实运行结果打分，不负责替代真实运行。

### 2.3 synthetic 的正确定位

synthetic 数据集保留，但定位为：

1. 冷启动评测。
2. 长尾问题覆盖。
3. 探索型实验。
4. 失败模式探索。
5. 链路 smoke test。

synthetic 默认不直接承担：

1. 正式 baseline。
2. regression 门禁。
3. 发布质量结论。

---

## 3. 当前已落地的脚本结构

### 3.1 数据集构建层

1. `app/evals/build_replay_dataset.py`
   - 从 `chat_history` 与历史检索结果构建 replay 数据集。
2. `app/evals/build_synthetic_dataset.py`
   - 基于 RAGAS `TestsetGenerator` 从知识库父文档块生成 synthetic / exploration 数据集。
3. `app/evals/import_seed_dataset.py`
   - 导入人工整理、业务侧提供或外部系统导出的 seed 数据集。
4. `app/evals/dataset_builder.py`
   - 统一负责 schema 校验、保存数据集、导出审核表、回填审核结果、生成构建报告。
5. `app/evals/schema.py`
   - 定义 `EvalDatasetManifest`、`EvalSample`、`EvalRunRecord`。
6. `app/evals/runtime.py`
   - 提供按 profile 的最小化运行时初始化。

### 3.2 真实执行与评分层

1. `app/evals/live_rag_runner.py`
   - 真实运行 Graph + Retriever，记录真实回答、真实上下文、真实命中文档和轨迹统计。
2. `app/evals/ragas_scorer.py`
   - 对真实 run 结果计算 RAGAS 指标、检索指标和 correctness。
3. `app/evals/ragas_runner.py`
   - 兼容入口，串联执行 `live_rag_runner` 与 `ragas_scorer`。
4. `app/evals/retrieval_scorer.py`
   - 负责传统检索指标。
5. `app/evals/metrics_registry.py`
   - 根据 `capabilities` 选择可用指标。
6. `app/evals/reporter.py`
   - 统一写入 run 产物。

### 3.3 兼容与退役入口

1. `app/evals/build_golden_jsonl.py`
   - 已退化为 `build_synthetic_dataset` 的兼容入口。
2. `app/evals/fill_response_from_graph_prompt.py`
   - 已退役，不再用于真实评测链路。

---

## 4. 运行时初始化策略

当前 `runtime.py` 已经改为按 profile 初始化最小依赖，而不是所有脚本都初始化完整 Graph 运行时。

支持的 profile：

1. `dataset_seed`
   - 仅初始化基础配置与数据库。
2. `dataset_synthetic`
   - 初始化基础配置、数据库、embedding、vector store、docstore。
3. `dataset_replay`
   - 在 `dataset_synthetic` 基础上增加 parent retriever。
4. `full`
   - 初始化完整 Graph / 检索 / rerank / checkpointer 运行时，供真实评测执行使用。

当前实现上的意义：

1. 数据集构建脚本不再误初始化 Graph / checkpointer / 完整检索链路。
2. 减少无关资源在异常时的清理干扰。
3. 降低 Windows 下 async 资源清理带来的问题暴露面。

另外，`runtime.py` 在 Windows 下会显式切换到 `WindowsSelectorEventLoopPolicy`，用于降低 `httpx / anyio / OpenAI async client` 在连接异常时触发 `Event loop is closed` 的概率。

---

## 5. 数据集模型

### 5.1 数据集分类

当前方案使用以下分类：

1. `regression`
   - 小规模、稳定、强约束，用于回归和门禁。
2. `baseline`
   - 覆盖主业务场景，用于版本比较。
3. `exploration`
   - 用于探索边界、长尾、困难样本和 smoke test。
4. `synthetic`
   - 由 AI 自动生成，主要用于冷启动和覆盖补充。
5. `specialized`
   - 某个专项能力的数据集，例如拒答、多跳、代码文档问答等。

### 5.2 capabilities

每个样本和数据集都由字段自动推导 `capabilities`，常见包括：

1. `has_reference_answer`
2. `has_reference_contexts`
3. `has_reference_doc_ids`
4. `has_rubric`
5. `has_reference_tool_calls`

### 5.3 样本最小字段

每条样本至少应尽量具备：

1. `sample_id`
2. `user_input`
3. `reference_answer`
4. `reference_contexts`
5. `reference_doc_ids`
6. `scope_file_ids`
7. `difficulty_level`
8. `scenario_type`
9. `source_type`
10. `tags`
11. `metadata`

---

## 6. 三种当前落地的数据集形态

### 6.1 replay

1. 来源于真实历史会话与历史检索结果。
2. 最接近真实用户问题分布。
3. 适合沉淀为 `baseline` 和 `regression`。

### 6.2 synthetic

1. 来源于知识库父文档块，经 RAGAS `TestsetGenerator` 自动生成。
2. 构建成本最低，适合冷启动、探索集和 smoke test。
3. 不应直接作为正式质量结论。

### 6.3 seed import

1. 来源于人工整理、业务侧提供、外部系统导出或已有题库文件。
2. 最灵活，适合作为专项集、回归集初稿或历史沉淀数据导入入口。

### 6.4 区别

1. 问题真实性
   - `replay` 最高。
   - `seed import` 取决于输入来源。
   - `synthetic` 最弱。
2. 构建成本
   - `synthetic` 最低。
   - `seed import` 取决于已有数据质量。
   - `replay` 依赖历史会话与检索记录。
3. 参考答案可信度
   - `replay` 中等，历史回答可能本身有误。
   - `seed import` 取决于标注来源。
   - `synthetic` 依赖 AI 生成，默认只作探索用途。
4. 最适合用途
   - `replay`：真实回放、baseline、regression 候选集。
   - `synthetic`：smoke test、探索实验、冷启动覆盖。
   - `seed import`：专项集、人工题库接入、外部样本迁移。

---

## 7. 当前三种数据集构建方式

### 7.1 replay 数据集

入口：`app/evals/build_replay_dataset.py`

当前逻辑：

1. 从 `chat_history` 中提取 `user -> ai` 对。
2. 使用历史 `parent_doc_ids` 回溯对应父文档块。
3. 生成 `reference_contexts`、`reference_doc_ids`、`scope_file_ids`。
4. `reference_answer` 默认使用历史回答。
5. 也可用 `--reference-mode ai` 让模型基于上下文重写候选参考答案。
6. 当启用 `--reference-mode ai` 时，支持 LLM 重试与退避。

当前状态：

1. replay 链路不存在 synthetic 那种 `reference_doc_ids` 全空的问题。
2. 已补充 `metadata.reference_file_ids`、`metadata.source_file_count` 和构建报告。
3. replay 数据集仍然需要人工审核，因为历史回答可能本身就有错误。
4. `--limit` 的语义是“最多抽多少条”，不是“必须交付多少条”，因此不存在 synthetic 那种 top-up 问题。

### 7.2 synthetic 数据集

入口：`app/evals/build_synthetic_dataset.py`

当前逻辑：

1. 随机选择参与构建的文件。
2. 文件选择权重同时考虑文档新旧和父文档块数量。
3. 按文件的父文档块数量动态分配样本预算，父块越多，分到的 quota 越多。
4. 只要文件参与构建，就至少保证 1 个父文档块进入生成流程。
5. 根据 `reference_contexts` 回填 `reference_doc_ids`，保证样本可追溯。
6. 记录每个文件的父块数、quota、chunk 字符长度统计和分批决策。

#### 7.2.1 动态 batch 控制

为了降低大文件在 `Generating Scenarios` 阶段的失败概率，synthetic 构建加入了动态 batch 控制：

1. 每个文件统计本次被抽中父块的 `avg_chunk_chars`、`max_chunk_chars`、`total_chunk_chars`。
2. 对同一 `quota` 的文件，系统以“能正常处理的较轻文件”的平均 chunk 体量作为安全参考。
3. 如果当前文件的平均 chunk 体量明显更大，则自动缩小该文件的实际 `effective_chunk_limit`。
4. 该缩小只影响“每次提交给 RAGAS 的父块数量”，不会截断父文档块内容。
5. 最终采用的 batch 上限会写入 `generation_plan.effective_chunk_limits`。

#### 7.2.2 synthetic 的 `--size` 语义

当前 `--size` 表示“目标样本数”，而不是简单的首轮预算：

1. 首轮生成后，如果样本数不足 `--size`，会自动进入 top-up。
2. top-up 会按各文件原始 `quota` 比例继续抽取新的父文档块。
3. top-up 优先使用当前文件尚未使用过的父块；如果父块不足，才回退到已有父块池。
4. 若经过多轮 top-up 后仍未达到目标样本数，脚本不会报错退出。
5. 会保留已完成的数据集，并在 metadata 与日志中记录欠交数量。

#### 7.2.3 synthetic 的当前状态

1. 解决了 `reference_doc_ids` 为空的问题。
2. 解决了参与构建文件没有实际样本覆盖的问题。
3. 降低了大文件在 `Generating Scenarios` 阶段因单批上下文过重导致的失败概率。
4. 将 `--size` 从“预算目标”提升为“尽量交付的目标样本数”，并支持按比例 top-up。
5. 即使最终样本数低于目标值，也会保留已完成数据集并给出明确告警。

#### 7.2.4 边界说明

1. 这套逻辑主要解决“文件覆盖”“样本追溯”“大文件批次稳定性”和“样本数不足自动补齐”问题。
2. 它不会自动产生真正的跨文件多跳样本。
3. 如果需要多文件问题，后续应单独设计 `specialized` multi-hop 数据集生成器。

### 7.3 seed 数据集导入

入口：`app/evals/import_seed_dataset.py`

当前逻辑：

1. 导入 `.json` 或 `.jsonl` 外部样本。
2. 标准化为统一 schema。
3. 若输入缺少 `scope_file_ids` 但提供了 `reference_doc_ids`，会尝试自动根据父块反推 `scope_file_ids`。
4. 若显式传入 `--default-scope all`，则会为缺少范围的样本填充全库范围。

当前状态：

1. 不存在 synthetic 的配额和 top-up 问题，因为它不负责生成样本，只负责导入样本。
2. 当前版本已支持自动反推 `scope_file_ids`，并在 `metadata` 中记录是否进行了推断。

### 7.4 seed 导入格式要求

`import_seed_dataset.py` 支持输入 `.json` 或 `.jsonl`。

支持的顶层格式：

1. `jsonl`
   - 每行一个样本对象。
2. `json`
   - 顶层是样本数组；或
   - 顶层是对象，且包含 `samples` 数组字段。

每条 seed 样本建议字段：

1. `user_input`
   - 推荐必填。
   - 也兼容 `question`。
2. `sample_id`
   - 可选。
   - 也兼容 `id`。
3. `reference_answer`
   - 推荐填写。
   - 也兼容 `reference`。
4. `reference_contexts`
   - 可选，字符串数组。
5. `reference_doc_ids`
   - 可选，字符串数组。
6. `scope_file_ids`
   - 可选，字符串数组。
7. `difficulty_level`
   - 可选。
8. `scenario_type`
   - 可选。
9. `source_type`
   - 可选。
10. `tags`
   - 可选，字符串数组。
11. `review_status`
   - 可选，默认 `pending`。
12. `review_notes`
   - 可选。
13. `metadata`
   - 可选，对象。
14. `rubric`
   - 可选，对象。
15. `reference_tool_calls`
   - 可选，数组。

最小可用示例：

```json
[
  {
    "user_input": "什么是父文档检索？",
    "reference_answer": "父文档检索是先检索子块，再回溯父块作为回答上下文的策略。"
  }
]
```

推荐完整示例：

```json
[
  {
    "sample_id": "seed-001",
    "user_input": "什么是父文档检索？",
    "reference_answer": "父文档检索是先检索子块，再回溯父块作为回答上下文的策略。",
    "reference_contexts": [
      "父文档检索通常先对更细粒度的子块做向量召回，再映射回父块。"
    ],
    "reference_doc_ids": ["parent-doc-id-1"],
    "scope_file_ids": ["file-id-1"],
    "difficulty_level": "L0",
    "scenario_type": "definition",
    "source_type": "manual",
    "tags": ["baseline", "retrieval"],
    "metadata": {
      "owner": "eval-team"
    }
  }
]
```

导入时自动补全行为：

1. 若样本缺少 `scope_file_ids` 但提供了 `reference_doc_ids`，系统会尝试自动反推 `scope_file_ids`。
2. 若样本同时缺少 `scope_file_ids` 且命令传入 `--default-scope all`，则会补成全库范围。
3. 若样本缺少 `difficulty_level` 或 `scenario_type`，会使用命令行默认值补齐。
4. 若样本缺少 `tags`，会自动补为 `[category, source_type]`。

---

## 8. 构建报告与产物

### 8.1 数据集产物

1. `manifest.json`
   - 数据集名称、版本、类别、来源、capabilities、审核要求、构建元数据。
2. `samples.jsonl`
   - 标准化 `EvalSample` 列表。
3. `review_sheet.csv`
   - 人工审核表。
4. `review_guide.md`
   - 审核说明。

### 8.2 统一构建报告

三种数据集构建脚本在完成构建后，都会打印统一格式的构建报告。报告由 `dataset_builder.format_build_report(...)` 生成，主要包含：

1. 数据集名称、版本、分类、来源。
2. 样本总数。
3. `reference_answer` 覆盖率。
4. `reference_contexts` 覆盖率。
5. `reference_doc_ids` 覆盖率。
6. `scope_file_ids` 覆盖率。
7. 唯一 `scope_file_ids` 数量。
8. 唯一 `reference_doc_ids` 数量。
9. `review_status` 分布。
10. 场景标签、难度标签分布。
11. 文件覆盖 Top N。

对于 synthetic 数据集，`manifest.metadata.generation_plan` 还会额外记录：

1. `requested_sample_count` 与 `actual_sample_count`。
2. `undershot_sample_count`。
3. 每个文件的 `doc_allocations`。
4. 每个文件的 `doc_avg_chunk_chars` / `doc_max_chunk_chars`。
5. 每个文件最终采用的 `effective_chunk_limits`。
6. 动态 batch 决策明细 `dynamic_batch_decisions`。

---

## 9. 人工审核策略

### 9.1 哪些数据集必须人工审核

正式使用前必须人工审核的数据集：

1. `regression`
2. `baseline`
3. `specialized`
4. `replay`
5. 准备升格为正式基线的 `synthetic`

可以先自动构建、后抽检的数据集：

1. `exploration`
2. 普通 `synthetic`

### 9.2 审核重点

1. `user_input` 是否真实、清晰、无歧义。
2. `reference_answer` 是否正确。
3. `reference_contexts` / `reference_doc_ids` 是否真的支撑答案。
4. `scope_file_ids` 是否合理。
5. 标签是否正确，例如 `difficulty_level`、`scenario_type`、`tags`。

建议采用：`AI 初稿 + 人工复核`。

审核细则见：[Evals数据集审核说明](./Evals数据集审核说明.md)

---

## 10. 真实测评链路

真实测评流程：

1. 从数据集读取样本。
2. `live_rag_runner.py` 逐条调用真实 Graph + Retriever。
3. 记录真实 `actual_response`、`actual_contexts`、`actual_doc_ids`、`actual_file_ids`、耗时和轨迹统计。
4. `ragas_scorer.py` 根据样本 capabilities 自动选择可计算指标。
5. 输出样本级结果和实验级汇总。

当前 runner 约束：

1. 可按 `review_status` 过滤样本。
2. 可限制执行样本数做 smoke test。
3. 同一 runner 支持 replay、synthetic、seed 等多种来源的数据集。

---

## 11. 指标体系

### 11.1 检索层指标

在样本具备 `reference_doc_ids` 时，优先计算：

1. `Recall@k`
2. `Precision@k`
3. `Hit@k`
4. `MRR@k`
5. `nDCG@k`

### 11.2 RAGAS 指标

在样本具备相应能力时，按需启用：

1. `faithfulness`
2. `answer_relevancy`
3. `context_precision`
4. `context_recall`

### 11.3 回答正确性

当前实现中，`ragas_scorer.py` 还会基于 LLM judge 计算 correctness。

### 11.4 后续可扩展指标

1. 拒答正确率。
2. 抗噪能力指标。
3. rubric-based 业务评分。
4. Agent 工具调用准确率。

---

## 12. 当前可用命令

### 12.1 构建 replay 数据集

```bash
python -m app.evals.build_replay_dataset --name replay_baseline --version v1 --category baseline
```

参数说明：

1. `--name`：必填，数据集名称。
2. `--version`：必填，数据集版本。
3. `--category`：可选，`regression`、`baseline`、`exploration`、`specialized`。
4. `--limit`：可选，最大样本数。
5. `--seed`：可选，抽样随机种子。
6. `--reference-mode`：可选，`history` 或 `ai`。
7. `--difficulty`：可选，默认难度标签。
8. `--scenario`：可选，默认场景标签。
9. `--description`：可选，数据集描述。
10. `--output-dir`：可选，显式输出目录。
11. `--llm-retries`：可选，AI 生成参考答案时的重试次数。
12. `--retry-backoff-seconds`：可选，AI 参考答案生成的退避秒数。

### 12.2 构建 synthetic 数据集

```bash
python -m app.evals.build_synthetic_dataset --name synthetic_smoke --version v1 --category exploration --size 20 --doc-limit 10 --use-light-model
```

参数说明：

1. `--name`：必填，数据集名称。
2. `--version`：必填，数据集版本。
3. `--category`：可选，`synthetic`、`exploration`、`specialized`。
4. `--size`：可选，目标样本数；若首轮不足，会自动进入 top-up。
5. `--doc-limit`：可选，参与构建的文件上限。
6. `--seed`：可选，随机种子。
7. `--recency-tau-days`：可选，文档时间衰减参数。
8. `--alloc-alpha`：可选，父块预算分配指数。
9. `--use-light-model`：可选，使用轻量模型。
10. `--difficulty`：可选，默认难度标签。
11. `--scenario`：可选，默认场景标签。
12. `--description`：可选，数据集描述。
13. `--output-dir`：可选，显式输出目录。
14. `--max-batch-retries`：可选，每个 synthetic 子批次的最大重试次数。
15. `--retry-backoff-seconds`：可选，子批次重试的退避秒数。
16. `--max-chunks-per-batch`：可选，默认的每批父块上限；实际执行时会按文件 chunk 体量动态缩小。
17. `--max-topup-rounds`：可选，样本不足时允许进行的 top-up 轮数。

补充说明：

1. synthetic smoke test 默认允许首轮不足后继续 top-up。
2. 若某些大文件的父块平均内容明显更重，脚本会自动缩小这些文件的实际 batch 大小，无需手动截断父块内容。
3. 若最终样本数仍低于 `--size`，脚本会保留已完成的数据集，并在 `manifest.json` 中记录欠交数量。

### 12.3 导入 seed 数据集

```bash
python -m app.evals.import_seed_dataset --input seeds.jsonl --name seed_smoke --version v1 --category exploration --source-type manual --default-scope all
```

参数说明：

1. `--input`：必填，输入 `.json` 或 `.jsonl` 文件。
2. `--name`：必填，数据集名称。
3. `--version`：必填，数据集版本。
4. `--category`：可选，`regression`、`baseline`、`exploration`、`specialized`、`synthetic`。
5. `--source-type`：可选，来源标签。
6. `--default-difficulty`：可选，默认难度标签。
7. `--default-scenario`：可选，默认场景标签。
8. `--default-scope`：可选，`none` 或 `all`。
9. `--description`：可选，数据集描述。
10. `--output-dir`：可选，显式输出目录。

### 12.4 导出和回填审核表

导出：

```bash
python -m app.evals.dataset_builder export-review --dataset-dir <dataset_dir>
```

回填：

```bash
python -m app.evals.dataset_builder apply-review --dataset-dir <dataset_dir> --review-file <dataset_dir>/review_sheet.csv
```

### 12.5 真实运行与评分

真实执行：

```bash
python -m app.evals.live_rag_runner --dataset-dir <dataset_dir> --review-status approved
```

评分：

```bash
python -m app.evals.ragas_scorer --run-dir <run_dir>
```

兼容串联入口：

```bash
python -m app.evals.ragas_runner --dataset-dir <dataset_dir> --review-status approved
```

---

## 13. 推荐命令组合

### 13.1 最低人工成本的 smoke test

```bash
python -m app.evals.build_synthetic_dataset --name synthetic_smoke --version v1 --category exploration --size 20 --doc-limit 10 --use-light-model
python -m app.evals.ragas_runner --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --limit 10 --review-status pending,approved
```

适用场景：

1. 先验证整条链路能否跑通。
2. 不急于得出正式质量结论。

### 13.2 更接近真实问题分布的 quick check

```bash
python -m app.evals.build_replay_dataset --name replay_quickcheck --version v1 --category exploration --limit 30 --reference-mode ai
python -m app.evals.ragas_runner --dataset-dir store/evals/datasets/exploration/replay_quickcheck/v1 --limit 10 --review-status pending,approved
```

适用场景：

1. 先用真实历史问题验证评测链路。
2. 后续再把高质量样本筛进 `baseline` 或 `regression`。

### 13.3 已有外部样本的快速接入

```bash
python -m app.evals.import_seed_dataset --input seeds.jsonl --name seed_smoke --version v1 --category exploration --source-type manual --default-scope all
python -m app.evals.ragas_runner --dataset-dir store/evals/datasets/exploration/seed_smoke/v1 --limit 10 --review-status pending,approved
```

适用场景：

1. 你手里已经有一批问题。
2. 想先把它们纳入统一评测框架。

---

## 14. 后续阶段路线图

### 14.1 Phase 2

1. 引入更稳定的数据集治理流程。
2. 沉淀失败样本为 `regression`。
3. 为高价值样本补齐 `reference_doc_ids` 和 `reference_contexts`。
4. 增加专项实验集与更细的报表。

### 14.2 Phase 3

1. 扩展到 Agent 真实轨迹评测。
2. 评估工具调用与多轮链路。

### 14.3 Phase 4

1. 将小规模 `regression` 集接入 CI 门禁。
2. 为 `baseline` 建立版本对照和阈值管理。

---

## 15. 参考资料

1. [RAGAS Evaluate and Improve a RAG App](https://docs.ragas.io/en/stable/howtos/applications/evaluate-and-improve-rag/)
2. [RAGAS Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
3. [RAGAS TestsetGenerator](https://docs.ragas.io/en/stable/references/generate/)
