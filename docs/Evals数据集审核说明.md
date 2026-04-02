# Evals 数据集审核说明

## 1. 哪些数据集必须人工审核

必须人工审核：

1. `regression`
2. `baseline`
3. `specialized`
4. 任何准备进入 `baseline/regression` 的 `synthetic`
5. 任何由历史对话回放生成的 `replay` 数据集

原因：

1. 历史 AI 回答本身可能是错的
2. 合成问题和合成参考答案可能存在分布偏差
3. `reference_contexts/reference_doc_ids` 一旦错误，会直接污染 retrieval 指标

可以弱审核或抽检：

1. 纯 `exploration` 数据集
2. 只用于研究和压力测试的 `synthetic` 数据集

## 2. 审核重点

每条样本至少检查：

1. `user_input` 是否清晰、单义、可答
2. `scope_file_ids` 是否与问题范围一致
3. `reference_answer` 是否正确、简洁、直接回答问题
4. `reference_contexts/reference_doc_ids` 是否真的支持参考答案
5. 样本是否属于合适的数据集分类
6. `difficulty_level` 和 `scenario_type` 是否标对
7. 如果带有 `query_type`，该类型是否与问题和证据形态一致
8. 如果带有 `reasoning_hops`，每一跳是否真的对应独立证据

### 2.1 query type 专项审核

对带 `query_type` 的样本，建议额外按下列规则判断：

1. `single_hop_specific`
   - 应是事实定位型问题
   - 不应依赖两个以上独立证据点
2. `single_hop_abstract`
   - 应是单主题内的解释、概括、总结
   - 不应只是换个说法的事实问答
3. `multi_hop_specific`
   - 应至少依赖两条独立证据
   - 最终答案仍应偏事实或结构化结论
4. `multi_hop_abstract`
   - 应至少依赖两条独立证据
   - 最终答案应体现综合、比较、归纳或演化解释

### 2.2 specialized synthetic 额外检查

对 [build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py) 生成的专项数据集，建议同时检查：

1. `manifest.json` 中的 `requested_distribution` 与 `realized_distribution` 是否偏差过大
2. `fallback_event_summary` 是否显示大量 multi-hop 被降级到 single-hop
3. `non_retriable_failure_summary` 是否出现异常高的跳过比例
4. 如果开启了 `--enable-multi-file`，是否真的出现 multi-file batch，而不是全部退回 single-file fallback

## 3. 审核状态建议

1. `pending`: 尚未审核
2. `approved`: 可进入正式评测
3. `needs_revision`: 保留，但要修改后再用
4. `rejected`: 删除，不进入正式评测

## 4. 推荐审核流程

1. 运行构建脚本生成数据集
2. 打开数据集目录中的 `review_sheet.csv` 和 `review_guide.md`
3. 在 `review_sheet.csv` 中逐条填写 `review_status` 和 `review_notes`
4. 如果是 query-type 数据集，优先抽检四类 query type 是否都成立
5. 如果是 specialized synthetic，先看 `manifest.json` 中的诊断摘要，再决定是否需要大范围返工
6. 执行：

```bash
python -m app.evals.dataset_builder export-review --dataset-dir <dataset_dir>
python -m app.evals.dataset_builder apply-review --dataset-dir <dataset_dir> --review-file <dataset_dir>/review_sheet.csv
```

7. 对 `regression` 和 `baseline`，只使用 `approved` 样本执行真实评测

## 5. 审核时的常见处理建议

1. 如果 `reference_answer` 正确，但 `query_type` 明显标错，优先改标签，不必直接拒绝
2. 如果 `multi_hop_*` 实际只需要单条证据，应降级为相应 single-hop 类型或标记 `needs_revision`
3. 如果 `reference_contexts` 无法支撑答案，即使问题本身合理，也不应标记为 `approved`
4. 如果 specialized synthetic 中大量样本都来自 fallback 降级，说明该批数据更适合探索或压测，而不是正式 baseline

## 6. AI 辅助能做什么

AI 可以辅助：

1. 从历史对话中抽取候选问题
2. 从上下文中生成候选 `reference_answer`
3. 给样本打初始标签，如单跳、多跳、拒答、易混淆
4. 预提取候选 `reference_contexts`
5. 做去重、聚类和脱敏

AI 不应直接替代人工确认：

1. `reference_answer` 的最终定稿
2. `reference_doc_ids` 的最终确认
3. `regression` 样本是否真的代表关键失败路径
4. 不可答题是否真的不可答
