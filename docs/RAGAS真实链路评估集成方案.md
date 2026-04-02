# RAGAS真实链路评估集成方案

## 1. 文档目标

本文档说明如何将 RAGAS 评估集成到当前真实的 RAG 链路中，使系统在每一次真实请求之后都能沉淀可追踪、可聚合、可分析的评估数据。

这里的“每一次 RAG 链路都有真实评估指标”需要分两层理解：

1. 每一次真实请求都可以得到在线可计算指标。
2. 只有在存在 reference answer / reference contexts / reference ids 时，才能计算 reference-aware 指标。

## 2. 结论

当前推荐架构已经从“每请求一个目录、多个文件”收敛为“按日期聚合、单 jsonl 主日志、可批量入库”。

推荐数据流：

```text
User Request
  -> conversation.py / graph.stream()
  -> 收集真实执行事实
  -> 返回答案给用户
  -> 后台异步提交 online evaluation
  -> RAGAS 单样本打分
  -> 追加写入 store/evals/online/YYYY-MM-DD/online_eval.jsonl
  -> 批量导入数据库 / 计算当日均值
```

这样设计的原因：

1. 不阻塞真实对话响应。
2. 不再为每次请求生成单独目录和 3 个文件。
3. 同一天的数据天然聚合，便于日级导入和日级统计。
4. `request_id` 可以作为数据库幂等主键，支持重复导入时更新。

## 3. RAGAS 官方依据

### 3.1 Evaluation Sample

RAGAS 官方文档给出了 `SingleTurnSample` 和 `MultiTurnSample`。

对当前项目而言：

1. 单次问答型 RAG 请求适合映射为 `SingleTurnSample`。
2. 包含 Human / AI / Tool 消息流的 agent trace 可以在后续映射为 `MultiTurnSample`。

参考：

- [Evaluation Sample](https://docs.ragas.io/en/stable/concepts/components/eval_sample/)
- [LangGraph Integration](https://docs.ragas.io/en/stable/howtos/integrations/_langgraph_agent_evaluation/)

### 3.2 在线指标与 reference-aware 指标

RAGAS 指标存在明显的数据依赖差异：

1. `Faithfulness`、`Response Relevancy` 这类指标可直接基于真实请求事实计算。
2. `ContextPrecision`、`ContextRecall`、`IDBasedContextPrecision`、`IDBasedContextRecall` 需要 reference 信息时才真正有意义。

参考：

- [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)
- [Context Precision](https://docs.ragas.io/en/v0.4.0/concepts/metrics/available_metrics/context_precision/)
- [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)

### 3.3 `evaluate()` / `aevaluate()` 的变化

RAGAS 官方 reference 已将 `evaluate()` / `aevaluate()` 标记为 deprecated。

因此当前真实链路已改为：

1. 将单次真实请求映射为 `SingleTurnSample`
2. 对每个 metric 分别执行 `single_turn_ascore()` 或 `single_turn_score()`
3. 单指标失败时局部容错，不拖垮整次在线评估

参考：

- [evaluate() Reference](https://docs.ragas.io/en/stable/references/evaluate/)
- [Metrics Reference](https://docs.ragas.io/en/stable/references/metrics/)

## 4. 当前项目中的最佳接入点

### 4.1 为什么不是 `graph.py`

[`graph.py`](D:/Bread/College/AI/Code/RAG/app/core/graph.py) 是业务执行引擎，负责 graph 节点调度与状态更新，适合作为评估事实来源，不适合作为同步评估执行器。

### 4.2 为什么是 `conversation.py`

[`conversation.py`](D:/Bread/College/AI/Code/RAG/app/routers/conversation.py) 在 `event_stream()` 中已经能拿到：

1. `user_input`
2. `final_answer`
3. `parent_doc_ids`
4. `file_ids`
5. `conversation_id`
6. `rewrite_count`
7. `generate_count`
8. `graph_messages`
9. `graph_events`

这正是在线评估所需的最小运行事实集合。

## 5. 已落地实现

当前项目已经完成真实链路最小可用接入，新增模块如下：

1. [`online_eval_schema.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/online_eval_schema.py)
2. [`online_eval_store.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/online_eval_store.py)
3. [`online_graph_capture.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/online_graph_capture.py)
4. [`online_ragas_service.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/online_ragas_service.py)
5. [`import_online_eval_to_db.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/import_online_eval_to_db.py)
6. [`summarize_online_eval_day.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/summarize_online_eval_day.py)
7. [`online_eval.py`](D:/Bread/College/AI/Code/RAG/app/crud/online_eval.py)

真实聊天路由接入点：

- [`conversation.py`](D:/Bread/College/AI/Code/RAG/app/routers/conversation.py)

数据库模型：

- [`schemas.py`](D:/Bread/College/AI/Code/RAG/app/models/schemas.py) 中新增 `OnlineEvalRun`

## 6. 当前在线评估存储设计

### 6.1 存储目录

当前按日期聚合到：

- `store/evals/online/YYYY-MM-DD/online_eval.jsonl`

同一天所有真实请求的最终评估结果都会追加写入这个文件。

### 6.2 为什么使用单 jsonl

相比“每请求一个目录 + record/summary/report 三文件”，单 jsonl 的优势是：

1. 更适合追加写入。
2. 更适合按日期批量导入数据库。
3. 更适合按日期做均值计算、失败率统计和 query type 聚合。
4. 更利于日志式审计和离线处理。

### 6.3 每条 jsonl 记录的字段设计

当前每条记录是“请求事实 + 评估结果”的合并对象，字段包括：

1. 主键与时间
   - `request_id`
   - `conversation_id`
   - `thread_id`
   - `event_date`
   - `request_created_at`
   - `evaluation_created_at`
2. 请求事实
   - `user_input`
   - `actual_response`
3. 检索事实
   - `actual_contexts`
   - `actual_doc_ids`
   - `actual_file_ids`
   - `retrieved_doc_count`
   - `retrieved_file_count`
   - `retrieved_context_count`
4. 查询类型
   - `query_type`
   - `hop_count`
   - `abstraction_level`
   - `query_type_source`
   - `query_type_confidence`
   - `query_type_reasons`
5. 运行指标
   - `latency_ms`
   - `rewrite_count`
   - `generate_count`
6. trace 摘要
   - `graph_messages`
   - `graph_events`
   - `graph_message_count`
   - `graph_event_count`
7. reference 扩展
   - `reference_answer`
   - `reference_contexts`
   - `reference_context_ids`
8. 评估结果
   - `status`
   - `metrics`
   - `skipped_metrics`
   - `error_message`
   - `metric_names`
   - `metric_failures`
   - `metric_timeout_seconds`
   - `successful_metric_count`
9. 扩展元数据
   - `metadata`

该结构已经由 [`online_eval_store.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/online_eval_store.py) 统一物化。

## 7. 当前数据库设计

### 7.1 表结构

当前新增表：`online_eval_run`

该表由 [`schemas.py`](D:/Bread/College/AI/Code/RAG/app/models/schemas.py) 中的 `OnlineEvalRun` 定义。

设计原则：

1. 高频筛选字段单独列出
2. 长文本保留为 `Text`
3. 列表与复杂结构保留为 `JSON`
4. `request_id` 作为主键，确保重复导入幂等

### 7.2 主要列映射

jsonl 和数据库字段一一对应，核心映射如下：

1. 标量列
   - `request_id`
   - `conversation_id`
   - `thread_id`
   - `event_date`
   - `request_created_at`
   - `evaluation_created_at`
   - `query_type`
   - `hop_count`
   - `abstraction_level`
   - `query_type_source`
   - `status`
   - `latency_ms`
   - `rewrite_count`
   - `generate_count`
   - `retrieved_doc_count`
   - `retrieved_file_count`
   - `retrieved_context_count`
   - `graph_message_count`
   - `graph_event_count`
   - `query_type_confidence`
   - `successful_metric_count`
   - `metric_timeout_seconds`
   - `error_message`
2. 文本列
   - `user_input`
   - `actual_response`
   - `reference_answer`
3. JSON 列
   - `actual_contexts`
   - `actual_doc_ids`
   - `actual_file_ids`
   - `graph_messages`
   - `graph_events`
   - `reference_contexts`
   - `reference_context_ids`
   - `metrics`
   - `skipped_metrics`
   - `query_type_reasons`
   - `metric_names`
   - `metric_failures`
   - `metadata_json`

## 8. 批量入库能力

当前新增脚本：

- [`import_online_eval_to_db.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/import_online_eval_to_db.py)

功能：

1. 读取某个日期对应的 `online_eval.jsonl`
2. 解析每一条请求评估记录
3. 按 `request_id` 对数据库做幂等 upsert
4. 先按批次 upsert，失败批次延后处理`r`n5. 所有批次完成后，对失败批次逐条重试`r`n6. 输出导入条数、插入数、更新数和最终失败数`r`n7. 若存在最终失败 request_id，则额外写入当日 `import_failures.json`

对应 CRUD：

- [`online_eval.py`](D:/Bread/College/AI/Code/RAG/app/crud/online_eval.py)

执行方式：

```powershell
python -m app.evals.online.import_online_eval_to_db --date 2026-04-02
```

## 9. 当日指标均值计算能力

当前新增脚本：

- [`summarize_online_eval_day.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/summarize_online_eval_day.py)

功能：

1. 读取某个日期的 `online_eval.jsonl`
2. 提取每条记录中的 `metrics`
3. 计算所有请求的当日指标平均值
4. 额外按 `query_type` 分组计算指标平均值
5. 输出到：
   - `store/evals/online/YYYY-MM-DD/daily_metric_averages.json`

执行方式：

```powershell
python -m app.evals.online.summarize_online_eval_day --date 2026-04-02
```

输出内容包括：

1. `record_count`
2. `status_counts`
3. `metric_averages`
4. `by_query_type`

## 10. 当前在线指标策略

在线评估仍采用“能力驱动”的指标选择方式：

1. 无 reference 时，只选择可在真实请求上直接计算的指标
2. 有 `reference_answer` / `reference_contexts` / `reference_context_ids` 时，再补充 reference-aware 指标

这一策略由：

- [`metrics_registry.py`](D:/Bread/College/AI/Code/RAG/app/evals/offline/metrics_registry.py)
- [`online_ragas_service.py`](D:/Bread/College/AI/Code/RAG/app/evals/online/online_ragas_service.py)

共同完成。

## 11. 当前实现边界

当前版本已经能实现：

1. 每次真实请求异步在线打分
2. 每次真实请求将结果追加写入当日 `online_eval.jsonl`
3. 按日批量导入数据库
4. 按日计算整体与按 query type 的指标平均值

但当前还没有完成：

1. 自动补全线上请求的 reference 信息
2. `MultiTurnSample` 的 agent trace 专项评估
3. 在线聚合查询 API
4. 更高级的日 / 周 / 月趋势报表

## 12. 如何验证当前实现

### 12.1 在线写入验证

启动服务并发起一次真实聊天请求后，检查：

```powershell
Get-Content -Tail 5 store/evals/online/2026-04-02/online_eval.jsonl
```

预期结果：

1. 当天目录存在
2. `online_eval.jsonl` 存在
3. 每次请求追加一条完整 JSON 记录

### 12.2 批量入库验证

```powershell
python -m app.evals.online.import_online_eval_to_db --date 2026-04-02
```

预期结果：

1. 脚本输出读取条数
2. 首次导入产生 `inserted > 0`
3. 重复导入产生 `updated > 0` 或保持幂等`r`n4. 若存在最终单条失败，则生成 `store/evals/online/YYYY-MM-DD/import_failures.json`

### 12.3 当日均值验证

```powershell
python -m app.evals.online.summarize_online_eval_day --date 2026-04-02
```

预期结果：

1. 控制台输出当日均值 JSON
2. 生成 `daily_metric_averages.json`
3. 文件中包含整体均值和 `by_query_type` 均值

## 13. 后续建议

下一阶段建议继续做：

1. Phase 2：给线上请求补充 reference-aware 指标
2. Phase 3：引入 `MultiTurnSample` 做 agent trace 专项评估
3. 增加数据库层的查询接口和运营报表
4. 增加定时任务，对每日 jsonl 自动汇总并自动入库

## 14. 参考资料

1. [Evaluation Sample](https://docs.ragas.io/en/stable/concepts/components/eval_sample/)
2. [LangGraph Integration](https://docs.ragas.io/en/stable/howtos/integrations/_langgraph_agent_evaluation/)
3. [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
4. [Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)
5. [Context Precision](https://docs.ragas.io/en/v0.4.0/concepts/metrics/available_metrics/context_precision/)
6. [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
7. [Metrics Reference](https://docs.ragas.io/en/stable/references/metrics/)
8. [evaluate() Reference](https://docs.ragas.io/en/stable/references/evaluate/)


