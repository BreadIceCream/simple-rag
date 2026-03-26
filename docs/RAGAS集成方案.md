# RAGAS 分阶段集成方案（面向当前 Agentic RAG 项目）

## 1. 文档目标与范围

本文档给出一个可落地、分阶段的 RAGAS 集成方案，用于将当前项目从“运行时自评”升级为“可复现、可回归、可治理”的专业评测体系。  
方案覆盖：

1. 现状评估与差距分析（基于现有代码）。
2. 多阶段集成路线（P0/P1/P2/P3）。
3. 指标体系（RAG 常见指标 + Agent 指标 + 工程指标）。
4. 数据建模、接口设计、任务调度、结果治理。
5. 每阶段交付物、验收标准、风险与回滚。

不在本方案范围：

1. 前端看板的完整 UI 设计实现。
2. 数据标注平台建设（仅给出最小可行流程）。
3. 多租户评测隔离。

---

## 2. 当前系统现状（与代码映射）

### 2.1 已有能力（可复用）

1. **检索与问答主链路完整**
   - 混合检索 + RRF + 可选 rerank：`app/core/retriever.py`
   - Agentic 对话图：`app/core/graph.py`

2. **运行时评估节点已存在（在线自评）**
   - 检索相关性判断：`handle_retrieve_result`（`graph.py`）
   - 幻觉判断：`check_hallucination`（`graph.py`）
   - 有用性判断：`check_usefulness`（`graph.py`）
   - 重试与上限：`max_rewrite_time`、`max_generate_time`

3. **可回溯会话数据已持久化**
   - 对话与消息：`conversation`、`chat_history`
   - 每条 AI 回答对应 `parent_doc_ids`：`app/models/schemas.py -> ChatHistory.parent_doc_ids`
   - SSE 输出可提取回答与参考片段：`app/routers/conversation.py`

4. **检索测试入口存在**
   - `/api/retrieval/query`、`/api/retrieval/references`（`app/routers/retriever.py`）

### 2.2 当前缺口

1. 无统一离线评测数据集（无法版本横向比较）。
2. 无结构化评测结果存储（无法趋势分析）。
3. 缺少标准检索排序指标（Recall@k/MRR/nDCG）。
4. 运行时“LLM判断”与离线“基准评测”未解耦。
5. 缺少发布门禁（CI/CD 质量阈值）。

---

## 3. 目标架构（双轨评估）

### 3.1 总体原则

1. **在线评估保留**：现有 Graph 节点继续服务实时质量控制。
2. **离线评估新增**：引入 RAGAS 批评测，作为版本治理标准。
3. **统一结果中心**：所有评测产出落库，支持趋势与回归门禁。

### 3.2 评测双轨

1. **Track A（Online Guardrail）**
   - 使用 `graph.py` 现有节点即时判断并驱动重试。
   - 指标偏“控制逻辑”。

2. **Track B（Offline Benchmark）**
   - 对固定样本集批跑 RAGAS 指标与检索排序指标。
   - 指标偏“质量基准与回归比较”。

---

## 4. 指标体系设计

## 4.1 第一层：RAG 核心指标（首批必须）

1. **Context Precision（检索精度）**
2. **Context Recall（检索召回）**
3. **Faithfulness（基于上下文的事实一致性）**
4. **Answer Relevancy（回答相关性）**

### 4.2 第二层：Agent/工具能力指标

1. **Tool Call Accuracy**
2. **Agent Goal Accuracy**
3. （可选）Tool Call F1

### 4.3 第三层：传统检索排序指标（必须补齐）

1. Recall@k
2. MRR@k
3. nDCG@k

### 4.4 第四层：工程与稳定性指标

1. P50/P95 响应延迟
2. 首 token 延迟（SSE）
3. 单问答 token 消耗 / 费用估算
4. 重写次数分布（rewrite_count）
5. 生成重试次数分布（generate_count）
6. 异常率（含 checkpoint 恢复率）

---

## 5. 数据与结果模型（新增）

## 5.1 新增表建议

### 5.1.1 `eval_dataset`

1. `id` (uuid, pk)
2. `name` (varchar)
3. `version` (varchar)
4. `description` (text)
5. `created_at`

### 5.1.2 `eval_sample`

1. `id` (uuid, pk)
2. `dataset_id` (fk)
3. `sample_type` (`single_turn` / `multi_turn_agent`)
4. `user_input` (text)
5. `reference_answer` (text, nullable)
6. `reference_contexts` (json, nullable)
7. `reference_tool_calls` (json, nullable)
8. `metadata` (json)
9. `created_at`

### 5.1.3 `eval_run`

1. `id` (uuid, pk)
2. `dataset_id`
3. `code_version` (commit sha / tag)
4. `config_snapshot` (json)
5. `status` (`running/success/failed`)
6. `started_at`
7. `finished_at`
8. `trigger_source` (`manual/ci/nightly`)

### 5.1.4 `eval_result_item`

1. `id` (uuid, pk)
2. `run_id`
3. `sample_id`
4. `question`
5. `response`
6. `retrieved_contexts` (json)
7. `metric_scores` (json)
8. `pass_fail` (bool, nullable)
9. `error_message` (text, nullable)

### 5.1.5 `eval_result_summary`

1. `run_id` (pk/fk)
2. `metric_avg` (json)
3. `metric_p50` (json)
4. `metric_p95` (json)
5. `regression_vs_baseline` (json)
6. `gate_passed` (bool)

---

## 6. 分阶段实施方案

## 6.1 P0（准备阶段）：评测基础能力打底

### 目标

建立最小可运行的离线评测链路，先跑通，再扩展。

### 任务

1. 依赖接入
   - 在 `requirements.txt` 增加 `ragas`（建议锁定稳定版本）。
2. 目录规划
   - 新建 `app/evals/`：
     - `dataset_builder.py`
     - `ragas_runner.py`
     - `metrics_registry.py`
     - `reporter.py`
3. 数据抽取器
   - 从 `parent_docs` 构建测试样本：按文档带权随机抽样，再按父文档数量动态配额抽块。
4. 建立首批黄金集
   - 50~100 条高价值问答（覆盖 PDF/HTML/Markdown/代码文档场景）。
5. 输出首份基线报告
   - CSV + JSON：包含每条样本分数与总体均值。

### 交付物

1. 可执行脚本：`python -m app.evals.ragas_runner --dataset v1`
2. 报告文件：`store/evals/experiments/<run_id>/`
3. 首版阈值建议文档（初始无门禁，仅观测）。

### 验收标准

1. 评测任务可稳定跑完（成功率 > 95%）。
2. 输出至少 4 个核心 RAG 指标。
3. 可定位低分样本与对应上下文。

### P0 严格执行步骤（按顺序）

> 本节为当前项目的“标准 P0 操作流程”。  
> 流程固定为：`parent_docs -> generate_with_chunks -> 用 retrieved_contexts 生成 response -> RAGAS 评测`。

1. 安装依赖（包含 `ragas`）

```bash
pip install -r requirements.txt
```

2. 使用 `parent_docs` + `generate_with_chunks` 构建黄金集（不填 response）

```bash
python -m app.evals.build_golden_jsonl --size 300 --doc-limit 30 --output store/evals/datasets/v1.generated.jsonl
```

3. 基于黄金集生成“当次系统回答文件”（保留原黄金集文件不变）  
   默认使用 `retrieved_contexts` 作为上下文（当前脚本默认值）。

```bash
python -m app.evals.fill_response_from_graph_prompt --input store/evals/datasets/v1.generated.jsonl
```

4. 运行 RAGAS 评测（输入第 3 步生成的 run 文件）

```bash
python -m app.evals.ragas_runner --source jsonl --dataset-path store/evals/datasets/runs/v1.generated.run_YYYYMMDD_HHMMSS.jsonl
```

5. 查看评测产物

```text
store/evals/experiments/<run_id>/
  ├─ item_scores.csv
  ├─ summary.json
  └─ records.jsonl
```

### P0 脚本参数说明

#### A. `app.evals.build_golden_jsonl`（黄金集构建）

命令示例：

```bash
python -m app.evals.build_golden_jsonl --size 100 --output store/evals/datasets/v1.generated.jsonl
```

参数：

1. `--size`  
   目标样本数。默认 `100`。
2. `--output`  
   输出 jsonl 路径。默认：`store/evals/datasets/v1.generated.jsonl`。
3. `--doc-limit`  
   参与构建的源文档数量上限。默认 `30`。
4. `--seed`  
   随机种子。默认 `42`（保证可复现）。
5. `--recency-tau-days`  
   文档时间衰减系数（天）。默认 `30.0`。控制“新文档优先”的强弱。
6. `--alloc-alpha`  
   父文档块动态分配指数。默认 `0.7`。控制块预算向“大文档”倾斜程度。
7. `--use-light-model`  
   开启后使用 `chat_model.light` 作为生成模型；默认使用 `chat_model.default`。
8. `--fill-response-with-reference`  
   开启后会把 `response` 直接填为 `reference`（用于快速打通，不建议作为正式评测输入）。

作用：

1. 基于 `parent_docs` 直接构建黄金集，不依赖 `chat_history`。
2. 自动调用 RAGAS 的 `generate_with_chunks` 生成测试样本。
3. 将结果标准化为当前项目可评测的 jsonl 格式。

原理：

1. 文档抽样：对 `embedded_document` 做“带权随机不放回采样”，时间越近权重越高（由 `recency_tau_days` 控制），最多抽 `doc-limit` 个。
2. 动态配额：以 `size` 作为总块预算，按每个文档 `parent_doc_ids` 数量，以 `alloc_alpha` 做预算分配，多文档多取、少文档少取。
3. 参与约束：参与构建的源文档保证至少抽到 1 个父文档块（在 `size` 足够时）。
4. 块内抽样：每个选中文档内部再随机抽取父文档块，保持随机性和覆盖面。
5. 样本生成：将抽到的父文档块作为 `chunks` 输入 `generate_with_chunks`，产出问题 `user_input`、参考答案 `reference` 和参考上下文 `retrieved_contexts`。

#### B. `app.evals.fill_response_from_graph_prompt`（填充 response）

命令示例：

```bash
python -m app.evals.fill_response_from_graph_prompt --input store/evals/datasets/v1.generated.jsonl
```

参数：

1. `--input`  
   输入黄金集 jsonl（必填）。
2. `--output`  
   输出 run 文件路径。默认自动生成到：`store/evals/datasets/runs/<input_stem>.run_<timestamp>.jsonl`。
3. `--context-mode`  
   上下文来源，支持：`retrieved` / `reference` / `both`。默认 `retrieved`（推荐）。
4. `--qps`  
   每秒请求数上限。默认 `10`。
5. `--concurrency`  
   最大并发 in-flight 请求数。默认 `10`。
6. `--model`  
   覆盖默认模型名；不传则使用 `chat_model.default`。
7. `--skip-existing-response`  
   开启后跳过已有非空 `response` 的样本。

说明：

1. 该脚本复用 `app/core/graph.py` 的 `GENERATE_ANSWER_PROMPT`。
2. 该脚本不会走完整 LangGraph 流程；仅执行“拼 prompt + `llm.invoke`”。
3. 输入黄金集文件不会被覆盖，输出为新 run 文件（符合“两份文件”策略）。

作用：

1. 为黄金集补齐“系统当次回答（response）”。
2. 形成可用于回归评测的 run 文件，不污染原始黄金集。

原理：

1. 逐条读取 `user_input`，按 `context-mode` 选择上下文（默认 `retrieved_contexts`）。
2. 套用 `GENERATE_ANSWER_PROMPT` 构造提示词，调用 `llm.invoke` 生成回答。
3. 通过 `qps + concurrency` 做吞吐与并发控制，最终写入新 jsonl。

#### C. `app.evals.ragas_runner`（离线评测）

命令示例：

```bash
python -m app.evals.ragas_runner --source jsonl --dataset-path store/evals/datasets/runs/v1.generated.run_YYYYMMDD_HHMMSS.jsonl
```

参数：

1. `--dataset`  
   数据集名称，默认 `v1`。当 `--dataset-path` 未给定时用于解析 `store/evals/datasets/<dataset>.jsonl`。
2. `--dataset-path`  
   直接指定 jsonl 文件路径。优先级高于 `--dataset`。
3. `--source`  
   数据来源：`auto` / `jsonl` / `history`。  
   当前严格 P0 流程建议使用 `jsonl`。
4. `--limit`  
   当 `--source history` 时，最多抽样条数。默认 `100`。
5. `--output-root`  
   评测输出根目录。默认 `store/evals/experiments`。

说明：

1. P0 严格流程里推荐固定 `--source jsonl`，保证可复现。
2. 报告中 `summary.json` 为均值汇总，`item_scores.csv` 为样本级分数。

作用：

1. 统一执行 RAGAS 离线评测并导出报告。
2. 输出样本级和汇总级结果，支持后续回归对比。

原理：

1. 将 jsonl 转成 RAGAS `EvaluationDataset`（SingleTurnSample）。
2. 自动装配单轮指标（如 faithfulness、answer_relevancy、context_precision、context_recall*）。
3. 运行 `evaluate(...)` 计算样本分数，并对数值列聚合均值输出。  
   注：`context_recall` 需要样本中存在 `reference`。

---

## 6.2 P1（核心阶段）：标准指标 + 持久化 + 接口化

### 目标

把离线评测变成可追踪系统能力，而非临时脚本。

### 任务

1. 数据库存储
   - 新增第 5 节数据表。
2. API 接口
   - 新增 `app/routers/evaluation.py`：
     - `POST /api/evaluation/runs`：触发评测
     - `GET /api/evaluation/runs/{id}`：查看 run 状态
     - `GET /api/evaluation/runs/{id}/summary`
     - `GET /api/evaluation/runs/{id}/items`
3. 指标注册中心
   - `metrics_registry.py` 支持按样本类型装配指标集合：
     - `single_turn_metrics`
     - `multi_turn_agent_metrics`
4. 报告聚合器
   - 产出 `eval_result_summary`，包含均值、分位数、失败样本 TopN。

### 交付物

1. 评测结果落库可追踪。
2. API 可查每次评测详情。
3. 支持比较两个 run 的指标差值（baseline vs current）。

### 验收标准

1. 任意一次 run 可完整回放到样本级别。
2. 支持按数据集版本查看历史趋势。
3. 支持导出报告（CSV/JSON）。

---

## 6.3 P2（Agent阶段）：LangGraph 轨迹评测

### 目标

把 Agent 行为正确性纳入系统评估。

### 任务

1. 轨迹采集
   - 在 `graph.stream(..., stream_mode=["custom","updates","messages"])` 基础上，保存消息轨迹快照。
2. 轨迹转换
   - 将 LangGraph/LangChain 消息转换为 RAGAS 多轮样本（`MultiTurnSample`）。
3. Agent 指标接入
   - Tool Call Accuracy
   - Agent Goal Accuracy
4. 失败归因
   - 增加错误标签：
     - 错误工具选择
     - 工具参数错误
     - 工具调用时机错误
     - 目标未完成

### 交付物

1. 多轮 Agent 评测报告。
2. Agent 失败样本库（可用于回归集扩充）。

### 验收标准

1. 多轮评测结果可落库并可检索。
2. Tool/Goal 指标可稳定计算。

---

## 6.4 P3（治理阶段）：CI 门禁 + 夜间巡检 + 迭代闭环

### 目标

形成工程化质量闭环。

### 任务

1. CI/CD 门禁
   - PR 或发布前自动跑小样本回归集。
   - 关键指标低于阈值则失败。
2. 夜间全量评测
   - 定时跑全量基准集，更新趋势。
3. 阈值治理策略
   - 采用“基线相对下降阈值 + 绝对最低阈值”双阈值。
4. 自动生成改进建议
   - 低分样本按问题类型聚类（检索不足、上下文污染、回答偏题等）。

### 交付物

1. CI 集成配置（如 GitHub Actions/Jenkins）。
2. 每日评测报表。
3. 指标告警机制（邮件/IM webhook）。

### 验收标准

1. 回归异常能在发布前被拦截。
2. 指标波动可解释、可追溯。

---

## 7. 代码改造清单（按模块）

## 7.1 新增模块

1. `app/evals/dataset_builder.py`
   - 构建 `SingleTurnSample` / `MultiTurnSample`
2. `app/evals/metrics_registry.py`
   - 指标实例化与分组
3. `app/evals/ragas_runner.py`
   - 执行评测、异常重试、并发控制
4. `app/evals/reporter.py`
   - 持久化与报告导出
5. `app/routers/evaluation.py`
   - 评测 API
6. `app/models/eval_schemas.py`
   - 评测 ORM 表
7. `app/crud/evaluation.py`
   - 评测 CRUD

## 7.2 现有模块改造点

1. `app/main.py`
   - 注册 `evaluation` 路由。
2. `app/models/common.py`
   - 增加评测 VO/DTO。
3. `app/routers/conversation.py`
   - 可选：保存必要轨迹元信息，支持多轮评测构建。
4. `requirements.txt`
   - 增加 `ragas` 依赖。

---

## 8. 运行流程设计

## 8.1 离线评测流程（单轮）

1. 读取数据集样本。
2. 调用现有检索/问答链路生成 `response` 与 `retrieved_contexts`。
3. 组装 RAGAS `EvaluationDataset`。
4. 计算核心指标。
5. 落库样本结果与汇总结果。
6. 输出报告文件。

## 8.2 离线评测流程（多轮 Agent）

1. 执行目标问题，捕获 LangGraph 消息轨迹。
2. 转为多轮评测样本。
3. 计算 Tool/Goal 类指标。
4. 归因并落库。

---

## 9. 阈值与门禁策略（建议初始值）

> 说明：以下阈值仅作为初始建议，需基于 P0 基线跑数后校准。

1. Faithfulness >= 0.75
2. Answer Relevancy >= 0.75
3. Context Precision >= 0.65
4. Context Recall >= 0.70
5. Tool Call Accuracy >= 0.85（Agent集）
6. Agent Goal Accuracy >= 0.80（Agent集）

门禁策略：

1. 任一关键指标低于“绝对阈值” -> fail。
2. 相比 baseline 下降超过 5% -> fail。
3. 新增高危失败样本数超过阈值 -> fail。

---

## 10. 数据集建设策略

## 10.1 样本来源

1. 生产高频问题回放（脱敏后）。
2. 典型失败案例沉淀。
3. 新功能专项样本（如代码文档问答、跨段聚合问答）。

## 10.2 样本分层

1. L0：事实检索题（单跳）。
2. L1：多段聚合题。
3. L2：易混淆题（近义概念/相似实体）。
4. L3：Agent 工具调用题。

## 10.3 版本治理

1. 数据集采用 `name + version`。
2. 每次扩样保留旧版本可回放。
3. 回归集与探索集分离：
   - 回归集：小而稳定，CI 必跑。
   - 探索集：大而多样，夜间跑。

---

## 11. 性能与成本控制

1. 评测并发限制（避免模型限流）。
2. 缓存可复用评分（相同样本+相同响应可跳过）。
3. 失败重试（指数退避，限定次数）。
4. 大样本分批执行，支持断点续跑。
5. 分层运行：
   - PR：小样本 + 核心指标
   - Nightly：全量 + 扩展指标

---

## 12. 风险与应对

1. **风险：评测抖动（LLM judge 非确定性）**
   - 应对：固定模型、温度、随机种子；关键样本多次采样取均值。

2. **风险：样本质量不足**
   - 应对：先从生产失败样本构建高价值集；每周审查样本质量。

3. **风险：评测耗时过长**
   - 应对：分层数据集、分批并发、缓存。

4. **风险：指标与业务目标脱节**
   - 应对：增加业务 KPI 映射（命中率、人工反馈、工单解决率）。

---

## 13. 里程碑计划（建议）

1. **第 1 周（P0）**
   - 跑通离线评测脚本 + 50~100 样本基线。
2. **第 2 周（P1）**
   - 评测落库 + API + 汇总报告。
3. **第 3 周（P2）**
   - Agent 多轮评测 + Tool/Goal 指标。
4. **第 4 周（P3）**
   - CI 门禁 + 夜间任务 + 阈值治理。

---

## 14. 最小实施清单（可直接执行）

1. 在 `requirements.txt` 添加 `ragas`。
2. 新建 `app/evals/` 四个核心文件：`dataset_builder.py / metrics_registry.py / ragas_runner.py / reporter.py`。
3. 完成单轮四指标评测并导出报告。
4. 新增评测 ORM 表与 `evaluation` 路由。
5. 在 CI 增加“回归集评测”步骤并配置阈值。

---

## 15. 与当前项目的对应关系总结

1. 现有 `graph.py` 的相关性/幻觉/有用性节点继续保留，不替代。
2. 新增 RAGAS 离线评测作为“质量基线层”。
3. 复用 `chat_history.parent_doc_ids` 与文档库，减少采集成本。
4. 以“分阶段”推进，先可用再完备，避免一次性大改导致风险扩大。

---

## 16. 参考资料

1. RAGAS 官方文档：https://docs.ragas.io/
2. RAGAS 指标（RAG）：Context Precision / Context Recall / Faithfulness / Response Relevancy
3. RAGAS LangGraph 集成（Agent 指标）：Tool Call Accuracy / Agent Goal Accuracy
4. RAGAS 迁移说明（v0.3 -> v0.4）
5. RAGAS 论文（EACL 2024 Demo）
