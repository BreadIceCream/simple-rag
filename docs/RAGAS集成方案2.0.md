# RAGAS 集成方案 2.0

## 1. 文档目标

本文档定义当前项目离线评估链路的 RAGAS 集成方案 2.0，目标是：

1. 在现有真实 RAG 评测链路基础上，支持四类查询类型：
   - `single_hop_specific`
   - `single_hop_abstract`
   - `multi_hop_specific`
   - `multi_hop_abstract`
2. 能够评估当前 RAG 系统对这四类查询的检索性能差异，而不是只给出整体平均值。
3. 保持“数据集构建”和“真实执行评分”解耦，不把 query type 逻辑硬编码进 runner。
4. 在不破坏现有 `replay / synthetic / seed import` 主链路的前提下，扩展一条面向 query type 的专项评测能力。

本方案基于以下事实：

1. 当前项目已经具备真实 RAG 执行与评分能力，主链路是：
   `数据集构建 -> 人工审核 -> live_rag_runner -> ragas_scorer -> report`
2. 当前单跳限制主要出现在 synthetic 数据集生成方式，而不在 runner/scorer 主链路。
3. 项目目标不是“只生成四种标签”，而是“构建能真实区分四类查询难度和检索行为的数据集，并给出分类型评估结果”。

---

## 2. 背景与当前问题

### 2.1 RAGAS 中的四类查询

根据 RAGAS 官方关于 RAG testset generation 的定义，RAG 中的查询类型可以分为四类：

1. `single-hop specific query`
   - 单跳、事实定位型问题。
   - 通常可由单个文档或单个证据点直接回答。
2. `single-hop abstract query`
   - 单跳、抽象解释型问题。
   - 仍主要依赖单个主题范围，但需要归纳、解释或概括。
3. `multi-hop specific query`
   - 多跳、事实连接型问题。
   - 需要从两个或以上证据点拼接出明确答案。
4. `multi-hop abstract query`
   - 多跳、抽象综合型问题。
   - 需要跨多个证据点做比较、演化解释、综合分析或高层概括。

这四类问题的主要区别，不只是问题长短，而是：

1. 检索所需证据数量不同。
2. 证据之间的关联方式不同。
3. 回答所需的推理和归纳强度不同。

### 2.2 当前项目的能力边界

结合当前代码与既有文档，现状如下：

1. [`app/evals/live_rag_runner.py`](../app/evals/live_rag_runner.py)
   - 已能真实运行当前 Graph + Retriever。
   - 已能记录 `actual_response`、`actual_contexts`、`actual_doc_ids`、`actual_file_ids`。
2. [`app/evals/retrieval_scorer.py`](../app/evals/retrieval_scorer.py)
   - 已能对 `reference_doc_ids` 做传统检索指标计算。
   - 当前只按“命中参考父块集合”计算，不区分 hop。
3. [`app/evals/ragas_scorer.py`](../app/evals/ragas_scorer.py)
   - 已能对真实运行结果做 RAGAS 评分和 correctness judge。
   - 当前只输出整体汇总，不支持按 query type 分桶。
4. [`app/evals/build_synthetic_dataset.py`](../app/evals/build_synthetic_dataset.py)
   - 当前基于父文档块批量调用 RAGAS `TestsetGenerator`。
   - 已解决覆盖、top-up、动态 batch、reference_doc_ids 回填等问题。
   - 但当前生成策略本质仍偏单跳，无法稳定产生真正的 multi-hop 样本。

### 2.3 当前方案的核心缺口

当前离线评估只能支持 single-hop 的根因，不是评分链路不能跑，而是缺少“显式建模 query type 的数据集层”。具体缺口如下：

1. schema 中没有 query type 的一等字段。
2. synthetic 构建器没有基于知识图谱的多跳场景构造能力。
3. replay / seed 数据集无法统一做四分类治理。
4. retrieval scorer 无法回答“多跳问题里每一跳是否都召回到了”。
5. report 无法展示“哪一类查询表现最差”。

因此，2.0 的重点应是：

1. 升级数据模型。
2. 升级数据集生成与导入逻辑。
3. 升级多跳检索指标。
4. 升级分类型报表。

而不是只在样本 metadata 里额外写一个 `query_type` 字段了事。

---

## 3. 2.0 设计原则

### 3.1 query type 是数据集能力，不是 runner 分支

1. query type 的定义和校验属于数据集层。
2. `live_rag_runner` 仍只负责真实执行。
3. `ragas_scorer` 和 `retrieval_scorer` 根据样本字段决定如何统计，而不是切换执行主流程。

### 3.2 真实评估优先

无论样本来源是 replay、synthetic 还是 seed：

1. `user_input` 必须来自数据集。
2. `actual_response`、`actual_contexts`、`actual_doc_ids` 必须来自真实 RAG 系统运行。
3. 结论必须基于真实运行结果，而不是 synthetic 四元组自评。

### 3.3 四类查询必须可审计

每个样本不仅要有一个 `query_type` 标签，还必须能解释：

1. 为什么它是 single-hop 或 multi-hop。
2. 为什么它是 specific 或 abstract。
3. 如果是 multi-hop，每一跳的参考证据分别是什么。

### 3.4 报表必须支持分桶对比

2.0 的目标不是只让四类样本“能跑”，而是能回答下面这些问题：

1. 当前系统在四类查询上的 Recall@k 分别是多少？
2. 当前系统在哪一类问题上最容易漏召回？
3. 当前系统在 multi-hop specific 和 multi-hop abstract 上的差距有多大？
4. 当前 rerank / final_k 配置对哪一类问题影响最大？

---

## 4. 总体方案

2.0 方案采用“统一 schema + 混合数据源 + 分类型评分”的结构。

### 4.1 总体链路

```text
replay / synthetic / seed import
    -> 统一 query type schema
    -> 人工审核 / 自动校验
    -> live_rag_runner 真实执行
    -> ragas_scorer + retrieval_scorer
    -> 按 query_type 输出 summary/report
```

### 4.2 数据源策略

建议采用双轨数据治理：

1. `replay`
   - 用于反映真实用户问题分布。
   - 作为当前系统真实能力 baseline。
2. `query-type specialized synthetic`
   - 用于补齐四类查询配额，尤其是多跳问题。
   - 用于专项能力分析和配置实验。
3. `seed import`
   - 用于接入人工构建的高价值专项题集。
   - 用于补齐 synthetic 难以稳定生成的 hard cases。

### 4.3 为什么不用纯 synthetic

如果只依赖 synthetic：

1. 容易失真，尤其是 abstract 问题容易模板化。
2. 容易高估系统表现。
3. 无法反映真实用户语言分布。

因此正式结论应以：

1. `replay baseline` 为主。
2. `specialized query-type set` 为辅。

---

## 5. 数据模型升级

2.0 建议在 [`app/evals/schema.py`](../app/evals/schema.py) 中扩展 `EvalSample` 和 `EvalRunRecord`。

### 5.1 新增字段

建议新增以下字段：

1. `query_type: str`
   - 取值：
     - `single_hop_specific`
     - `single_hop_abstract`
     - `multi_hop_specific`
     - `multi_hop_abstract`
2. `hop_count: str`
   - 取值：`single` / `multi`
3. `abstraction_level: str`
   - 取值：`specific` / `abstract`
4. `evidence_topology: str`
   - 取值建议：
     - `single_doc`
     - `multi_doc_same_file`
     - `multi_file`
5. `reasoning_hops: list[dict]`
   - 用于描述多跳样本每一跳的证据结构。
6. `query_type_source: str`
   - 取值建议：
     - `manual`
     - `heuristic`
     - `llm`
     - `ragas_synthesizer`

### 5.2 字段约束

建议约束如下：

1. `query_type` 为一等字段，必须显式存在。
2. `hop_count` 和 `abstraction_level` 可以由 `query_type` 自动派生，但仍建议落盘，便于报表和筛选。
3. `single-hop` 样本允许 `reasoning_hops` 为空或只包含 1 跳。
4. `multi-hop` 样本必须提供 `reasoning_hops`。
5. `multi-hop` 样本必须至少包含 2 个 hop。

### 5.3 `reasoning_hops` 建议结构

每个 hop 建议包含：

```json
{
  "hop_id": "hop_1",
  "hop_intent": "identify_influencer",
  "reference_doc_ids": ["parent-doc-a"],
  "reference_file_ids": ["file-a"],
  "notes": "Find the scientist that influenced Einstein."
}
```

对于 multi-hop 样本，最终答案对应的证据应拆分到不同 hop，而不是只在 `reference_doc_ids` 中放一个并集。

### 5.4 与现有字段的关系

1. `reference_doc_ids`
   - 保留，仍表示样本最终参考证据的并集。
2. `reference_contexts`
   - 保留，仍供 RAGAS 和审核使用。
3. `scenario_type`
   - 保留，用于业务场景标签，例如 `definition`、`comparison`、`timeline`。
   - 不再承担 `single-hop / multi-hop` 的职责。

---

## 6. 四类查询的项目内定义标准

为保证生成、审核、统计一致，2.0 需要给出项目内标准，而不是只引用 RAGAS 概念。

### 6.1 `single_hop_specific`

定义标准：

1. 问题是明确事实型。
2. 答案主要依赖一个证据点或一个父文档块主题。
3. 回答不要求跨证据综合。

典型问题：

1. 定义是什么？
2. 某步骤是什么？
3. 某配置项的默认值是什么？

### 6.2 `single_hop_abstract`

定义标准：

1. 问题仍围绕一个主题范围。
2. 但回答需要解释、总结或概括。
3. 不要求跨来源显式拼接两条独立事实。

典型问题：

1. 某机制为什么这样设计？
2. 某模块的整体作用是什么？
3. 某流程的核心思路是什么？

### 6.3 `multi_hop_specific`

定义标准：

1. 答案必须连接两个或以上独立证据点。
2. 每个证据点都相对明确。
3. 最终答案通常仍是事实型或结构化结论。

典型问题：

1. 哪个模块依赖另一个模块，以及它们通过什么字段关联？
2. 哪个配置会影响哪个执行阶段，默认值是多少？

### 6.4 `multi_hop_abstract`

定义标准：

1. 答案依赖两个或以上证据点。
2. 回答需要综合、比较、演化解释或高层归纳。
3. 相比 multi-hop specific，更依赖组织与综合，而不是简单拼接。

典型问题：

1. 当前评测链路为什么只能支持 single-hop，根因分布在哪几层？
2. 当前检索设计和评测设计之间有哪些耦合点，它们如何影响多跳能力？

---

## 7. 数据集构建方案

### 7.1 replay 数据集扩展

入口仍为 [`app/evals/build_replay_dataset.py`](../app/evals/build_replay_dataset.py)。

新增能力：

1. 对抽取出的历史问题自动做 query type 分类。
2. 将分类结果写入 `query_type`、`hop_count`、`abstraction_level`。
3. 在 metadata 中写入分类依据与置信度。

建议做法：

1. 先做 heuristic 初判：
   - 是否存在显式比较、关联、前因后果、跨模块问题特征。
   - 是否存在解释、总结、为什么、如何演化等抽象特征。
2. 再用 LLM 进行结构化判定。
3. 最后将低置信样本标记为 `needs_revision`，进入人工复核。

建议新增参数：

1. `--classify-query-type`
2. `--query-type-mode heuristic|llm|hybrid`
3. `--query-type-threshold`

### 7.2 seed import 扩展

入口仍为 [`app/evals/import_seed_dataset.py`](../app/evals/import_seed_dataset.py)。

新增能力：

1. 支持导入外部显式提供的 `query_type`。
2. 若外部未提供，则可选择自动分类。
3. 对 multi-hop 样本支持导入 `reasoning_hops`。

建议新增兼容字段：

1. `query_type`
2. `hop_count`
3. `abstraction_level`
4. `evidence_topology`
5. `reasoning_hops`

### 7.3 synthetic 数据集升级思路

当前 [`app/evals/build_synthetic_dataset.py`](../app/evals/build_synthetic_dataset.py) 更适合保留为：

1. 通用 synthetic 冷启动构建器。
2. 单跳 smoke test 构建器。

不建议直接在这个脚本里强行塞满四类 query type 逻辑。推荐新增一条专项构建链路。

### 7.4 新增专项构建器

建议新增：

1. `app/evals/build_querytype_dataset.py`
2. `app/evals/querytype_synthesizers.py`
3. `app/evals/querytype_validator.py`

职责如下：

#### 7.4.1 `build_querytype_dataset.py`

负责：

1. 读取知识库父块。
2. 构造 query type 专项知识图谱输入。
3. 按 query distribution 调用官方或自定义 synthesizer。
4. 统一标准化样本并保存。

#### 7.4.2 `querytype_synthesizers.py`

负责：

1. 封装 RAGAS 0.4.3 可用 synthesizer。
2. 补齐项目侧的 `single_hop_abstract` 适配能力。
3. 将 synthesizer 结果统一映射到项目 schema。

#### 7.4.3 `querytype_validator.py`

负责：

1. 校验 single-hop 样本是否误用了多跳证据。
2. 校验 multi-hop 样本是否真的需要两跳以上证据。
3. 校验 specific / abstract 标签是否合理。
4. 为可疑样本输出审核告警。

---

## 8. RAGAS 0.4.3 集成设计

### 8.1 0.4.3 可直接利用的能力

基于 RAGAS 0.4.3，建议优先使用：

1. `SingleHopSpecificQuerySynthesizer`
2. `MultiHopSpecificQuerySynthesizer`
3. `MultiHopAbstractQuerySynthesizer`

这些能力适合直接映射到：

1. `single_hop_specific`
2. `multi_hop_specific`
3. `multi_hop_abstract`

### 8.2 `single_hop_abstract` 的处理策略

在本项目中，`single_hop_abstract` 不建议等待默认分布自动支持，而应由项目侧显式补足。推荐两种实现方式：

#### 方案 A：轻定制 prompt 的单跳抽象生成器

1. 选单个节点或单个文档主题范围。
2. 使用抽象型 prompt 生成“解释 / 总结 / 概括”类问题。
3. 产物打上 `single_hop_abstract` 标签。

优点：

1. 与现有代码耦合最小。
2. 可快速落地。

缺点：

1. 对 prompt 质量依赖较强。

#### 方案 B：完全自定义 synthesizer

1. 对 RAGAS 的单跳场景类做扩展。
2. 定义新的抽象型 query generation prompt。
3. 完整接入 query distribution。

优点：

1. 结构更干净。
2. 与 multi-hop 体系更对称。

缺点：

1. 实现成本更高。
2. 对本地 RAGAS 版本 API 更敏感。

2.0 推荐先落地方案 A。

### 8.3 query distribution 设计

建议专项数据集支持命令行配置四类查询比例，例如：

```json
{
  "single_hop_specific": 0.25,
  "single_hop_abstract": 0.25,
  "multi_hop_specific": 0.25,
  "multi_hop_abstract": 0.25
}
```

同时允许：

1. 偏重多跳实验。
2. 偏重抽象问题实验。
3. 偏重真实线上分布回放。

建议新增参数：

1. `--query-distribution-json`
2. `--enable-multi-file`
3. `--max-hops`
4. `--min-hop-evidence`

---

## 9. 样本生成与校验策略

### 9.1 single-hop 样本生成

#### 9.1.1 `single_hop_specific`

要求：

1. 单个主题节点。
2. 问题可明确定位到某个父块或主题块。
3. `reference_doc_ids` 可以只有一个，也可以是同一主题下的少量父块，但不应体现跨主题拼接。

#### 9.1.2 `single_hop_abstract`

要求：

1. 单个主题节点或同主题父块集合。
2. 问题需要抽象解释。
3. 不允许依赖两个独立事实链路。

建议校验：

1. LLM 判断“若去掉任一外部证据，答案是否仍可成立”。
2. 若必须依赖多个独立证据点，则应改标为 multi-hop。

### 9.2 multi-hop 样本生成

#### 9.2.1 `multi_hop_specific`

要求：

1. 至少两个 hop。
2. 每一跳都对应明确证据。
3. hop 之间存在可解释的关系，例如实体重合、流程前后、字段映射、模块依赖。

#### 9.2.2 `multi_hop_abstract`

要求：

1. 至少两个 hop。
2. 问题要求比较、综合、演化说明或高层解释。
3. 最终回答不是简单事实拼接，而是归纳后的解释性结论。

### 9.3 自动校验规则

建议每个 synthetic 样本在落盘前经过以下校验：

1. 是否具备 `query_type`。
2. 若为 `multi-hop`，是否具备 `reasoning_hops`。
3. `reasoning_hops` 数量是否 >= 2。
4. `reference_doc_ids` 是否覆盖所有 hop 的证据并集。
5. `scope_file_ids` 是否合理覆盖样本证据来源。
6. `specific` 问题是否被生成成解释型泛问法。
7. `abstract` 问题是否只是伪装成长句的事实题。

### 9.4 人工审核重点

在 query type 数据集里，审核表需要新增以下检查点：

1. `query_type` 是否标对。
2. `multi-hop` 是否真的需要多跳。
3. `abstract` 是否真的需要抽象综合。
4. `reasoning_hops` 是否能解释该题的证据结构。

---

## 10. 真实执行层设计

### 10.1 `live_rag_runner.py` 的定位保持不变

[`app/evals/live_rag_runner.py`](../app/evals/live_rag_runner.py) 不需要因 query type 改主流程。

它仍只负责：

1. 从数据集读取样本。
2. 设置 `scope_file_ids`。
3. 真实运行 Graph + Retriever。
4. 记录真实命中的 doc/file/context。

### 10.2 run record 需要透传 query type

建议在 `EvalRunRecord` 中保留并透传：

1. `query_type`
2. `hop_count`
3. `abstraction_level`
4. `evidence_topology`
5. `reasoning_hops`

这样 scorer 和 reporter 不需要回读原始 dataset 才能分桶统计。

---

## 11. 检索指标升级方案

当前 [`app/evals/retrieval_scorer.py`](../app/evals/retrieval_scorer.py) 只基于 `reference_doc_ids` 并集计算：

1. `Precision@k`
2. `Recall@k`
3. `Hit@k`
4. `MRR@k`
5. `nDCG@k`

这些指标对 single-hop 够用，但对 multi-hop 不够。

### 11.1 保留现有指标

这些指标仍然保留，因为：

1. 它们能反映整体父块命中情况。
2. 可以继续横向对比历史实验。

### 11.2 新增 multi-hop 指标

建议新增：

1. `hop_hit@k`
   - 每个 hop 是否至少命中一个参考父块。
2. `all_hops_hit@k`
   - 所有 hop 是否都命中。
3. `min_hop_recall@k`
   - 各 hop recall 的最小值。
4. `avg_hop_recall@k`
   - 各 hop recall 的平均值。
5. `file_recall@k`
   - 以 `reference_file_ids` 为准的文件级召回。
6. `cross_file_hit@k`
   - 对 `multi_file` 样本，是否覆盖所有关键文件。

### 11.3 为什么要新增这些指标

举例：

1. 一个 multi-hop 问题总共依赖 A、B 两跳。
2. 系统只召回了 A，没召回 B。
3. 当前 `Recall@k` 可能并不算太差，因为命中了部分并集。
4. 但从业务角度看，这题本质上已经答不对了。

`all_hops_hit@k` 和 `min_hop_recall@k` 正是为了解决这个问题。

---

## 12. RAGAS 评分与 correctness 方案

### 12.1 RAGAS 指标层保持兼容

[`app/evals/ragas_scorer.py`](../app/evals/ragas_scorer.py) 当前支持：

1. `faithfulness`
2. `answer_relevancy`
3. `context_precision`
4. `context_recall`
5. `context_entities_recall`

这些指标在 2.0 中仍可继续使用。

### 12.2 2.0 中的使用方式

2.0 不改变这些指标的计算方式，而改变它们的观察方式：

1. 不只看全量平均值。
2. 必须按 `query_type` 分桶统计。
3. 必须支持对比：
   - `single_hop_specific` vs `single_hop_abstract`
   - `multi_hop_specific` vs `multi_hop_abstract`

### 12.3 correctness judge 保持保留

当前 correctness judge 对参考答案和实际答案做结构化判定，这一能力仍需保留。

因为：

1. retrieval 指标只能反映检索。
2. RAGAS 指标更偏生成相关性与支撑性。
3. correctness 能给出更直观的最终“答对没答对”的判断。

---

## 13. 报表升级方案

### 13.1 汇总结果必须分桶

[`app/evals/reporter.py`](../app/evals/reporter.py) 当前主要输出：

1. `summary.json`
2. `item_scores.csv`
3. `report.md`

2.0 建议新增按 query type 聚合结果，例如：

```json
{
  "by_query_type": {
    "single_hop_specific": {
      "sample_count": 40,
      "metric_avg": {},
      "retrieval_summary": {},
      "correctness_summary": {}
    }
  }
}
```

### 13.2 Markdown 报表建议新增章节

建议在 `report.md` 中新增：

1. `Query Type Breakdown`
2. `Retrieval Metrics by Query Type`
3. `Correctness by Query Type`
4. `Worst Samples by Query Type`

### 13.3 关键展示维度

必须至少支持以下维度：

1. 按 `query_type`
2. 按 `hop_count`
3. 按 `abstraction_level`
4. 按 `evidence_topology`

这样可以回答：

1. 系统是被多跳拖垮，还是被抽象问题拖垮？
2. 真正困难的是跨文件多跳，还是同文件多跳？

---

## 14. 命令行设计建议

### 14.1 构建专项 query type 数据集

建议新增命令：

```bash
python -m app.evals.build_querytype_dataset \
  --name querytype_baseline \
  --version v1 \
  --category specialized \
  --size 80 \
  --query-distribution-json query_distribution.json \
  --doc-limit 30 \
  --enable-multi-file
```

### 14.2 对 replay 数据集做 query type 分类

建议扩展命令：

```bash
python -m app.evals.build_replay_dataset \
  --name replay_qtype \
  --version v1 \
  --category baseline \
  --limit 100 \
  --reference-mode ai \
  --classify-query-type \
  --query-type-mode hybrid
```

### 14.3 导入人工四类专项集

建议扩展命令：

```bash
python -m app.evals.import_seed_dataset \
  --input seeds_querytype.jsonl \
  --name manual_querytype \
  --version v1 \
  --category specialized \
  --source-type manual
```

### 14.4 真实执行与评分

主命令不需要变化：

```bash
python -m app.evals.ragas_runner --dataset-dir <dataset_dir> --review-status approved
```

只是输出中需要新增按 query type 的汇总。

---

## 15. 分阶段实施计划

### 15.1 Phase A：先让现有数据集“可分型”

目标：

1. 先不动 synthetic 生成主逻辑。
2. 给 replay / seed 数据集补齐 query type 字段。
3. 给 scorer / reporter 加上分桶统计能力。

交付内容：

1. schema 扩展。
2. replay 分类器。
3. scorer/reporter 分 query type 输出。

收益：

1. 立刻能评估当前系统在四类问题上的现状。
2. 先拿到第一版真实差距画像。

### 15.2 Phase B：新增专项 synthetic 生成器

目标：

1. 引入 `build_querytype_dataset.py`。
2. 重点补齐 `multi_hop_specific` 与 `multi_hop_abstract`。
3. 以专项集方式做配置实验。

交付内容：

1. query distribution。
2. 专项 synthesizer 封装。
3. query type validator。

### 15.3 Phase C：沉淀正式 baseline / regression

目标：

1. 从 replay 中筛正式 baseline。
2. 从 hard cases 中沉淀 regression。
3. 建立四类问题的平衡专项集。

交付内容：

1. `baseline`
2. `regression`
3. `specialized query-type`

### 15.4 Phase D：门禁化

目标：

1. 对高价值小集合引入阈值管理。
2. 在 CI 或离线发布检查中纳入 query type 维度。

---

## 16. 风险与注意事项

### 16.1 风险一：伪 multi-hop

风险：

1. 看起来是多跳问题。
2. 实际上单个父块就能回答。

对策：

1. 引入 `querytype_validator.py`。
2. 对 multi-hop 样本要求显式 `reasoning_hops`。
3. 多跳专项集必须人工抽检。

### 16.2 风险二：伪 abstract

风险：

1. 问题写得长。
2. 本质仍是事实题。

对策：

1. 抽象问题必须要求解释 / 综合 / 总结型参考答案。
2. 对 `abstract` 标签增加人工审核要求。

### 16.3 风险三：分布失真

风险：

1. synthetic 四类比例平衡，但和真实线上分布不一致。

对策：

1. query type specialized 数据集只用于专项能力评估。
2. 真实效果结论仍以 replay baseline 为主。

### 16.4 风险四：RAGAS API 版本差异

风险：

1. RAGAS 0.4.3 的 testset API 与旧版本不同。
2. 某些 synthesizer 名称或导入路径可能变动。

对策：

1. 项目内新增一层 `querytype_synthesizers.py` 做版本适配。
2. 不把 RAGAS 内部类路径散落在多个脚本里。

---

## 17. 方案结论

RAGAS 集成方案 2.0 的核心，不是简单让数据集多一个 `query_type` 标签，而是完成四个层面的升级：

1. **数据模型升级**
   - 把 query type 和 hop 结构纳入统一 schema。
2. **数据集构建升级**
   - replay 可分型，seed 可导入，synthetic 可专项补齐四类查询。
3. **检索指标升级**
   - 从“整体命中”升级到“每一跳是否命中”。
4. **报表升级**
   - 从“全量平均值”升级到“按 query type 可比较”。

这样，系统才能真正回答：

1. 当前 RAG 系统是否擅长 single-hop specific？
2. 当前系统对 single-hop abstract 的检索是否已经开始退化？
3. 当前系统在 multi-hop specific 中是漏哪一跳？
4. 当前系统为什么在 multi-hop abstract 上表现更差？

这也是 2.0 相比当前方案最重要的价值。

---

## 18. 参考资料

1. [RAGAS: 为 RAG 生成测试集](https://docs.ragas.org.cn/en/stable/concepts/test_data_generation/rag/)
2. [RAGAS: Single-hop Query Testset](https://docs.ragas.io/en/v0.3.3/howtos/applications/singlehop_testset_gen/)
3. [RAGAS: Synthesizers / default_query_distribution](https://docs.ragas.io/en/v0.3.3/references/synthesizers/)
