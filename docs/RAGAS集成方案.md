# RAGAS 集成方案

本文档是当前项目 RAGAS 评估与集成的唯一主文档。

它整合并吸收了以下历史文档中的有效内容：

- [RAGAS集成方案2.0.md](D:/Bread/College/AI/Code/RAG/docs/RAGAS集成方案2.0.md)
- [QueryTypeSynthetic鲁棒性修复方案.md](D:/Bread/College/AI/Code/RAG/docs/QueryTypeSynthetic鲁棒性修复方案.md)

后续如无特殊说明，请始终以本文档为准。

## 1. 目标

本文档统一回答以下问题：

1. 当前项目的离线评估主链路是什么
2. 当前如何支持四类 query type
3. replay / seed / synthetic / querytype synthetic 各自的定位是什么
4. S001、S002、S003 已经带来了哪些能力
5. 当前推荐如何运行评估、审核数据和排查问题

## 2. 总体原则

### 2.1 真实评测优先

正式评估结论必须来自：

`数据集 -> 真实 RAG 执行 -> scorer -> 报告`

也就是说：

1. `user_input` 来自数据集
2. `actual_response`、`actual_contexts`、`actual_doc_ids` 必须来自真实系统运行
3. RAGAS 负责打分，不替代真实系统执行

### 2.2 数据集构建与评分解耦

当前评估链路按职责拆分：

1. 数据集构建器负责定义“测什么”
2. runner 负责定义“如何真实执行”
3. scorer 负责定义“如何打分”
4. query type 是样本 schema 能力，不是 runner 的主流程分支

### 2.3 synthetic 主要用于专项实验

synthetic 数据集适合：

1. cold start
2. smoke test
3. query-type 专项实验
4. multi-hop 压测

synthetic 数据集默认不直接承担：

1. 正式 baseline 结论
2. 关键 regression 门禁
3. 对线上真实能力的最终判断

## 3. 当前评估主链路

当前 [`app/evals`](D:/Bread/College/AI/Code/RAG/app/evals) 下已经形成如下主链路：

```text
replay / seed / synthetic / querytype synthetic
  -> dataset_builder 导出与审核
  -> live_rag_runner 真实执行
  -> ragas_scorer + retrieval_scorer
  -> reporter 输出 summary/report/item_scores
```

关键入口：

1. [build_replay_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_replay_dataset.py)
2. [import_seed_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/import_seed_dataset.py)
3. [build_synthetic_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_synthetic_dataset.py)
4. [build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py)
5. [live_rag_runner.py](D:/Bread/College/AI/Code/RAG/app/evals/live_rag_runner.py)
6. [ragas_scorer.py](D:/Bread/College/AI/Code/RAG/app/evals/ragas_scorer.py)

## 4. 四类 Query Type 设计

当前项目统一采用以下四类 query type：

1. `single_hop_specific`
2. `single_hop_abstract`
3. `multi_hop_specific`
4. `multi_hop_abstract`

### 4.1 项目内定义

#### `single_hop_specific`

1. 事实定位型问题
2. 主要依赖单个主题范围或单条证据链
3. 不要求跨证据拼接

#### `single_hop_abstract`

1. 单主题范围内的解释、总结、概括型问题
2. 不要求跨多个独立证据跳转
3. 需要抽象表达，但仍属单跳

#### `multi_hop_specific`

1. 至少依赖两条独立证据
2. 最终答案仍偏事实或结构化结论
3. 强调“连接多个证据点”

#### `multi_hop_abstract`

1. 至少依赖两条独立证据
2. 回答需要综合、比较、归纳或演化解释
3. 是当前最难的一类

## 5. Schema 与样本层支持

当前 schema 已支持 query type 一等字段，核心字段包括：

1. `query_type`
2. `hop_count`
3. `abstraction_level`
4. `evidence_topology`
5. `reasoning_hops`
6. `query_type_source`

相关实现位于：

- [schema.py](D:/Bread/College/AI/Code/RAG/app/evals/schema.py)

当前 sample 和 run record 都可透传这些字段，因此：

1. replay 可分型
2. seed 可显式导入
3. specialized synthetic 可显式生成
4. scorer / reporter 可按 query type 聚合

## 6. 数据集来源与定位

### 6.1 replay

入口：

- [build_replay_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_replay_dataset.py)

定位：

1. 最接近真实用户问题分布
2. 适合 baseline / regression 候选
3. 可快速得到四类 query type 的现状画像

### 6.2 seed

入口：

- [import_seed_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/import_seed_dataset.py)

定位：

1. 最可控的四类 query type 覆盖方式
2. 适合高价值人工专项集
3. 适合 regression 和 hard cases

### 6.3 通用 synthetic

入口：

- [build_synthetic_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_synthetic_dataset.py)

定位：

1. 保留兼容
2. 适合 cold start、smoke、探索性实验
3. 不再作为 query type 专项主入口

### 6.4 query-type specialized synthetic

入口：

- [build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py)

定位：

1. 四类 query type 的专项 synthetic 主入口
2. 支持 `balanced`、`multihop_focus` 等配置实验
3. 用于专项分析、多跳压测和配置对比

## 7. 已完成的演进阶段

### 7.1 Phase A 已完成

已落地：

1. schema 扩展
2. replay query type 分类
3. seed query type 字段接入
4. scorer / reporter 按 query type 分桶

结果：

1. 当前系统已经可以按四类 query type 输出评估结果
2. `summary.json` / `report.md` 已支持按 query type 聚合

### 7.2 Phase B 已完成

已落地：

1. [build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py)
2. [querytype_synthesizers.py](D:/Bread/College/AI/Code/RAG/app/evals/querytype_synthesizers.py)
3. [querytype_validator.py](D:/Bread/College/AI/Code/RAG/app/evals/querytype_validator.py)
4. `balanced` / `multihop_focus`
5. validator `off|warn|strict`

结果：

1. 已具备四类 query type 的专项 synthetic 生成能力
2. 已具备 query distribution 配置实验能力

### 7.3 S003 鲁棒性修复已完成

本轮修复解决的是 `build_querytype_dataset.py` 在 multi-hop cluster / relationship 缺失场景下的确定性失败与盲重试问题。

已落地能力：

1. per-batch availability probe
2. deterministic / non-retriable cluster error classification
3. deterministic fallback / reallocation
4. `QUERYTYPE BATCH` 控制台诊断日志
5. `--enable-multi-file` 具备真实行为
6. manifest metadata 增加诊断摘要

相关实现位于：

1. [querytype_synthesizers.py](D:/Bread/College/AI/Code/RAG/app/evals/querytype_synthesizers.py)
2. [build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py)

## 8. 当前 querytype synthetic 的关键行为

### 8.1 分布配置

当前支持：

1. `balanced`
2. `multihop_focus`
3. `--query-distribution-json`
4. `--query-distribution-file`

### 8.2 validator

当前支持：

1. `off`
2. `warn`
3. `strict`

由 [querytype_validator.py](D:/Bread/College/AI/Code/RAG/app/evals/querytype_validator.py) 执行。

### 8.3 multi-hop 失败处理

当前遇到类似错误时：

- `No relationships match the provided condition. Cannot form clusters.`
- `No clusters found in the knowledge graph`

系统会：

1. 将其识别为 `non-retriable`
2. 停止同一路径的盲重试
3. 进入 fallback / reallocation
4. 必要时降级到 single-hop
5. 打印控制台诊断日志

### 8.4 控制台日志

当前 `build_querytype_dataset.py` 会输出 `QUERYTYPE BATCH` 日志，至少包括：

1. `file_id`
2. `file_name`
3. `sub_batch`
4. `chunk_count`
5. `requested_counts`
6. `available_query_types`
7. `effective_counts`
8. `fallback_events`
9. `error classification`
10. `generated_counts`

### 8.5 multi-file 行为

当前开启 `--enable-multi-file` 后：

1. 系统会尝试构造真实 multi-file batch
2. 如果无法形成 multi-file pairing，会显式打印 fallback 日志
3. 不再是 silent metadata-only 行为

## 9. 指标与报表

当前已落地：

1. retrieval 指标
   - `precision@k`
   - `recall@k`
   - `hit@k`
   - `mrr@k`
   - `ndcg@k`
2. RAGAS 指标
   - `faithfulness`
   - `answer_relevancy`
   - `context_precision`
   - `context_recall`
   - `context_entities_recall`
3. correctness judge
4. by-query-type summary/report

当前仍不在范围内：

1. Phase C 的 multi-hop retrieval 新指标
2. [retrieval_scorer.py](D:/Bread/College/AI/Code/RAG/app/evals/retrieval_scorer.py) 指标体系重构
3. [ragas_scorer.py](D:/Bread/College/AI/Code/RAG/app/evals/ragas_scorer.py) / [reporter.py](D:/Bread/College/AI/Code/RAG/app/evals/reporter.py) 大改

## 10. 当前推荐评估方式

### 10.1 看真实分布

使用 replay：

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

### 10.2 做可控四类专项评估

使用 seed：

```powershell
python -m app.evals.import_seed_dataset `
  --input .\seeds_querytype.jsonl `
  --name seed_qtype_baseline `
  --version v1 `
  --category baseline `
  --source-type manual
```

### 10.3 做四类专项 synthetic 实验

平衡分布：

```powershell
python -m app.evals.build_querytype_dataset `
  --name querytype_balanced_eval `
  --version v1 `
  --category specialized `
  --size 80 `
  --doc-limit 30 `
  --query-distribution-profile balanced `
  --validator-mode warn
```

多跳偏重：

```powershell
python -m app.evals.build_querytype_dataset `
  --name querytype_multihop_eval `
  --version v1 `
  --category specialized `
  --size 80 `
  --doc-limit 30 `
  --query-distribution-profile multihop_focus `
  --validator-mode strict `
  --min-hop-evidence 2 `
  --enable-multi-file
```

后续统一执行：

1. `dataset_builder export-review`
2. `dataset_builder apply-review`
3. `live_rag_runner`
4. `ragas_scorer`

## 11. 审核与治理

正式使用前建议以下类型都经过人工审核：

1. `baseline`
2. `regression`
3. `specialized`
4. `replay`
5. 准备升级为正式基线的 `synthetic`

审核细则见：

- [Evals数据集审核说明.md](D:/Bread/College/AI/Code/RAG/docs/Evals数据集审核说明.md)

## 12. 后续路线

后续推荐继续推进：

1. Phase C：沉淀正式 `baseline/regression`
2. 为多跳专项集增加更细粒度统计
3. 在 CI 中逐步引入高价值小规模回归门禁

## 13. 关联文档

1. 运行命令与参数总表
   - [Evals命令文档.md](D:/Bread/College/AI/Code/RAG/docs/Evals命令文档.md)
2. 数据集审核说明
   - [Evals数据集审核说明.md](D:/Bread/College/AI/Code/RAG/docs/Evals数据集审核说明.md)
3. 历史索引文档
   - [RAGAS集成方案2.0.md](D:/Bread/College/AI/Code/RAG/docs/RAGAS集成方案2.0.md)
   - [QueryTypeSynthetic鲁棒性修复方案.md](D:/Bread/College/AI/Code/RAG/docs/QueryTypeSynthetic鲁棒性修复方案.md)

## 14. 结论

当前项目的 RAGAS 集成已经从“只能做 single-hop synthetic + 统一评分”演进为：

1. 支持四类 query type
2. 支持 replay / seed / specialized synthetic 三类主数据源
3. 支持按 query type 分桶评估
4. 支持 multi-hop 偏重配置实验
5. 在 multi-hop cluster 缺失时具备鲁棒降级与可观测性

当前最重要的实践原则是：

1. 正式结论以真实 RAG 执行为准
2. replay 看现状，seed 看可控样本，querytype synthetic 看专项能力
3. 出现多跳 batch 降级时，优先看 `QUERYTYPE BATCH` 日志和 manifest 诊断摘要，而不是简单增加重试次数
