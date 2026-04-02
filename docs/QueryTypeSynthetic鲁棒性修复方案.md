# QueryType Synthetic 鲁棒性修复方案

本文档用于说明 `build_querytype_dataset.py` 在生成四类 query type synthetic 数据集时，遇到 RAGAS multi-hop cluster/relationship 缺失错误的根因、影响范围，以及完整修复方案。

适用代码：

- [`app/evals/build_querytype_dataset.py`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py)
- [`app/evals/querytype_synthesizers.py`](D:/Bread/College/AI/Code/RAG/app/evals/querytype_synthesizers.py)
- [`app/evals/build_synthetic_dataset.py`](D:/Bread/College/AI/Code/RAG/app/evals/build_synthetic_dataset.py)
- [`app/evals/querytype_validator.py`](D:/Bread/College/AI/Code/RAG/app/evals/querytype_validator.py)

## 1. 问题描述

在本地执行 `build_querytype_dataset.py` 时，出现如下错误：

```text
querytype batch failed file_id=... file_name=... sub_batch=1/1 attempt=2/3 error=ValueError: No relationships match the provided condition. Cannot form clusters.. retry_in=4.0s
```

该错误常出现在：

- `multi_hop_specific`
- `multi_hop_abstract`

对应的 RAGAS multi-hop synthesizer 生成阶段。

当前现象是：

1. 某个 batch 本身无法形成 multi-hop cluster。
2. 代码仍然继续重试。
3. 重试后通常仍然失败。
4. 导致数据集构建效率低、日志噪声大、问题难以排查。

## 2. 调研结论

### 2.1 错误来源

该错误不是项目自身主动抛出的业务异常，而是 RAGAS 在知识图谱聚类阶段抛出的确定性错误。

已确认的官方资料方向：

1. RAGAS 的 multi-hop synthesizer 依赖知识图谱中的关系簇生成查询。  
   参考：[RAGAS Synthesizers](https://docs.ragas.io/en/v0.3.4/references/synthesizers/)

2. RAGAS 图谱层本身提供了基于关系条件寻找 cluster 的能力；当关系不满足条件时，无法形成 cluster。  
   参考：[RAGAS Graph Reference](https://docs.ragas.io/en/stable/references/graph/)

3. 官方 issue 中存在同类问题：`MultiHopAbstractQuerySynthesizer` 在知识图谱无可用 cluster 时会失败。  
   参考：[GitHub Issue #1696](https://github.com/explodinggradients/ragas/issues/1696)

### 2.2 性质判断

这个错误是：

- 确定性错误
- 数据/图谱条件不满足错误
- 非瞬时网络错误
- 非短暂服务端错误

因此它不应该被当作普通重试错误处理。

## 3. 当前代码中的问题

### 3.1 当前 batch 分配策略会强行请求 multi-hop

[`app/evals/build_querytype_dataset.py`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py) 当前逻辑会在每个 `sub_batch` 上直接根据 distribution 计算 `query_type_counts`：

- [`app/evals/build_querytype_dataset.py#L259`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py#L259)
- [`app/evals/build_querytype_dataset.py#L261`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py#L261)

问题在于：

- 它没有先判断当前 batch 是否真的支持 `multi_hop_specific` / `multi_hop_abstract`
- 只要 distribution 要求 multi-hop，就会请求 multi-hop

### 3.2 retry 逻辑把确定性 cluster 错误当作可重试错误

[`app/evals/build_querytype_dataset.py#L174`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py#L174) 的 `_generate_with_retry(...)` 当前对异常统一重试。

问题在于：

- `No relationships match the provided condition. Cannot form clusters.`
- `No clusters found in the knowledge graph...`

这类错误不是重试能解决的。

当前后果：

1. 同一 batch 反复失败
2. 增加等待时间
3. 隐藏真正原因
4. 影响整体数据集构建效率

### 3.3 `querytype_synthesizers.py` 缺少“可用性探测”和“配额降级”

[`app/evals/querytype_synthesizers.py`](D:/Bread/College/AI/Code/RAG/app/evals/querytype_synthesizers.py) 当前能力：

- 能按 query type 调用对应 synthesizer
- 能按 distribution 分配 query type 数量

但缺少：

- 当前 batch 的 query type 可用性探测
- cluster 错误分类
- 不可用 query type 的降级与重分配

### 3.4 `--enable-multi-file` 当前只进 metadata，没有进真实 batch 逻辑

[`app/evals/build_querytype_dataset.py#L54`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py#L54) 定义了 `--enable-multi-file`。

但目前实际效果只有：

- 写入 `experiment_config.enable_multi_file`

见：

- [`app/evals/build_querytype_dataset.py#L323`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py#L323)

当前没有：

- 真实多文件 batch 构造
- 跨文件 chunk 合并
- 为 multi-hop 提供更丰富的关系图谱输入

这意味着该 flag 目前是“宣称式参数”，不是“功能性参数”。

## 4. 根因分析

根因不是单点，而是 4 个问题叠加：

1. batch 过早固定了 query type 配额  
   当前先分配配额，再生成，而不是先判断 batch 能力。

2. multi-hop 生成依赖关系簇  
   某些文件或某些 sub-batch 的 chunk 本身就无法形成足够关系。

3. 失败后没有降级策略  
   不能把失败的 `multi_hop_abstract` 转移给别的 query type。

4. 控制面可观测性不足  
   当前日志只能看到“失败+重试”，看不到：
   - 本批请求了什么
   - 当前批次支持什么
   - 为什么判定不支持
   - fallback 是否发生

## 5. 修复目标

本次修复应满足以下目标：

1. 真正解决 deterministic cluster failure 导致的盲重试。
2. 在 batch 级别先判断 query type 可用性。
3. 当 multi-hop 不可用时，自动进行 deterministic fallback / reallocation。
4. 在控制台打印足够诊断日志，帮助定位问题。
5. 保持 CLI 兼容，不破坏已有 `build_querytype_dataset.py` 和 `build_synthetic_dataset.py` 的基本使用方式。
6. 对 `--enable-multi-file` 做明确处理：
   - 要么真实实现
   - 要么显式限制并写入文档与 contract

## 6. 总体修复思路

总体思路是把当前流程从：

```text
先分配 query type -> 直接调用 synthesizer -> 失败后统一重试
```

改为：

```text
先探测 batch 能力 -> 生成可用 query type 集合 -> 对请求配额做重分配 -> 调用 synthesizer -> 对非重试错误直接降级/记录 -> 输出诊断日志
```

## 7. 详细修复方案

### 7.1 `querytype_synthesizers.py` 增加 3 类核心能力

#### 7.1.1 增加 query type 可用性探测

建议新增函数：

```python
probe_available_query_types(
    *,
    chunks: list[Any],
    enable_multi_file: bool = False,
) -> QueryTypeAvailabilityResult
```

建议返回结构：

```python
{
  "available_query_types": [
    "single_hop_specific",
    "single_hop_abstract",
    "multi_hop_specific"
  ],
  "unavailable_query_types": {
    "multi_hop_abstract": "insufficient_relationship_clusters"
  },
  "signals": {
    "chunk_count": 5,
    "file_count": 1,
    "estimated_multihop_evidence": 1
  }
}
```

实现原则：

1. `single_hop_specific` 默认可用
2. `single_hop_abstract` 默认可用
3. `multi_hop_specific` 与 `multi_hop_abstract` 需要满足最低多跳条件

最低判断策略可分两层：

1. 轻量启发式前置判断  
   例如：
   - chunk 数量至少 >= 2
   - parent doc 数量至少 >= 2
   - 如启用 multi-file，则 file 数量至少 >= 2

2. 可选 guarded probe  
   如果需要更稳，可通过一个 very-small dry-run 或 guarded synthesizer 调用来判断 graph cluster 是否存在。  
   但不能让 probe 自己变成昂贵主流程。

#### 7.1.2 增加 non-retriable cluster error 分类

建议新增函数：

```python
is_non_retriable_cluster_error(exc: BaseException) -> bool
```

要匹配的典型错误文案包括：

- `No relationships match the provided condition. Cannot form clusters.`
- `No clusters found in the knowledge graph`

同时兼容大小写和嵌套异常链。

输出要求：

- 返回 `True` 表示不可重试
- 返回 `False` 表示仍可按当前 retry 规则处理

#### 7.1.3 增加 query type 配额重分配

建议新增函数：

```python
reallocate_query_type_counts(
    requested_counts: dict[str, int],
    available_query_types: set[str],
) -> QueryTypeReallocationResult
```

建议返回：

```python
{
  "effective_counts": {...},
  "fallback_events": [
    {
      "from": "multi_hop_abstract",
      "to": "multi_hop_specific",
      "count": 2,
      "reason": "unavailable_for_batch"
    }
  ]
}
```

推荐 fallback 策略：

1. `multi_hop_abstract` 不可用
   - 优先 `multi_hop_specific`
   - 否则 `single_hop_abstract`
   - 否则 `single_hop_specific`

2. `multi_hop_specific` 不可用
   - 优先 `single_hop_specific`
   - 否则 `single_hop_abstract`

3. `single_hop_abstract` 不可用
   - 优先 `single_hop_specific`

4. 如果多个类型都不可用
   - 按上述顺序递归/迭代重分配，直到所有配额都分配到 available 集合

要求：

- deterministic
- 不依赖随机数
- 必须落日志与 metadata

### 7.2 `build_querytype_dataset.py` 主流程改造

#### 7.2.1 在每个 sub-batch 生成前先 probe

当前逻辑：

1. 先 `allocate_query_type_counts`
2. 直接 `_generate_with_retry`

应改为：

1. 计算 `requested_counts`
2. `probe_available_query_types(...)`
3. `reallocate_query_type_counts(...)`
4. 使用 `effective_counts` 进入 generation

#### 7.2.2 非重试错误不再盲重试

当前 `_generate_with_retry(...)` 需要改造为：

1. 对异常先分类
2. 如果是 non-retriable cluster error：
   - 立即停止当前 query type 的 retry
   - 尝试 fallback 或批次降级
   - 输出日志

3. 如果是普通连接错误、超时错误：
   - 保持当前 retry 策略

即：

```text
cluster missing -> no retry -> fallback
timeout/network -> retry
```

#### 7.2.3 增加 per-batch 诊断日志

建议新增统一日志函数，例如：

```python
_log_querytype_batch_event(...)
```

建议至少打印这些字段：

- `file_id`
- `file_name`
- `sub_batch`
- `chunk_count`
- `selected_parent_doc_ids_count`
- `requested_query_type_counts`
- `available_query_types`
- `unavailable_query_types`
- `effective_query_type_counts`
- `fallback_events`
- `error_classification`
- `generated_query_type_counts`

推荐日志示例：

```text
QUERYTYPE BATCH: file_id=... file_name=... sub_batch=1/2 chunk_count=4 requested={'single_hop_specific':1,'single_hop_abstract':1,'multi_hop_specific':2,'multi_hop_abstract':2}
QUERYTYPE BATCH: availability available=['single_hop_specific','single_hop_abstract','multi_hop_specific'] unavailable={'multi_hop_abstract':'insufficient_relationship_clusters'}
QUERYTYPE BATCH: fallback from=multi_hop_abstract to=multi_hop_specific count=2 reason=unavailable_for_batch
QUERYTYPE BATCH: effective_counts={'single_hop_specific':1,'single_hop_abstract':1,'multi_hop_specific':4,'multi_hop_abstract':0}
QUERYTYPE BATCH: generation_done generated_counts={'single_hop_specific':1,'single_hop_abstract':1,'multi_hop_specific':3,'multi_hop_abstract':0}
```

#### 7.2.4 增加 manifest metadata 诊断摘要

建议新增：

- `availability_probe_summary`
- `fallback_event_summary`
- `non_retriable_failure_summary`
- `effective_query_type_counts`

至少保证最终 manifest 中能回答：

1. 本次原始请求分布是什么
2. 实际可用分布是什么
3. 哪些 batch 被降级
4. 哪些 query type 最常不可用

### 7.3 `--enable-multi-file` 的处理

这是当前实现里最容易误导用户的点。

建议做法有两种，只能选一种并明确写进后续实现计划：

#### 方案 A：本轮真正实现 multi-file batch

做法：

1. 在 generation plan 中允许一个 batch 合并多个 file 的 chunk
2. 当 `--enable-multi-file` 开启时：
   - 不再严格按单文件 sub-batch 生成
   - 允许跨文件 chunk 组合进入同一个 querytype batch

优点：

- 更符合 multi-hop 尤其 multi-file multi-hop 的目标

缺点：

- 改动较大
- 需要重新考虑 `file_id/file_name` 级日志和 metadata 结构

#### 方案 B：本轮只做“最小功能化”

做法：

1. 不立刻实现真正跨文件 batch
2. 但必须：
   - 在 CLI/help/doc 中说明该 flag 当前仅影响规划层或后续阶段
   - 或在代码里直接禁止它静默无效

例如：

- 若开启 `--enable-multi-file`，打印明确日志：
  - `QUERYTYPE BUILD: enable_multi_file requested but current sprint only supports single-file batching; falling back to single-file mode.`

如果希望这轮真正“解决问题”，更推荐 **方案 A**；如果考虑 sprint 尺寸，可用 **方案 B** 过渡，但必须诚实地暴露限制。

## 8. 建议的数据结构

### 8.1 可用性探测结果

```python
@dataclass
class QueryTypeAvailabilityResult:
    available_query_types: list[str]
    unavailable_query_types: dict[str, str]
    signals: dict[str, Any]
```

### 8.2 配额重分配结果

```python
@dataclass
class QueryTypeReallocationEvent:
    source_query_type: str
    target_query_type: str
    count: int
    reason: str

@dataclass
class QueryTypeReallocationResult:
    effective_counts: dict[str, int]
    fallback_events: list[QueryTypeReallocationEvent]
```

### 8.3 错误分类结果

```python
@dataclass
class QueryTypeErrorClassification:
    retriable: bool
    category: str
    reason: str
```

## 9. 建议的实现顺序

### Step 1

在 [`app/evals/querytype_synthesizers.py`](D:/Bread/College/AI/Code/RAG/app/evals/querytype_synthesizers.py) 加入：

- `probe_available_query_types`
- `is_non_retriable_cluster_error`
- `reallocate_query_type_counts`

### Step 2

在 [`app/evals/build_querytype_dataset.py`](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py) 接入：

- probe
- reallocation
- non-retriable 分类
- 日志打印

### Step 3

补 manifest metadata 诊断摘要。

### Step 4

落实 `--enable-multi-file` 的真实行为或显式限制。

### Step 5

补文档说明与命令示例。

## 10. 验证方案

这次修复不能只看 `--help` 或单元测试，必须验证“以前会失败的路径现在如何收敛”。

### 10.1 单元/集成验证

1. 给定一个没有 multi-hop cluster 的 batch
   - probe 识别 multi-hop 不可用
   - reallocation 生效
   - 不发生盲重试

2. 给定一个 cluster 缺失类异常
   - `is_non_retriable_cluster_error(...) == True`

3. 给定一个连接超时错误
   - 仍走 retry

4. fallback 是 deterministic 的
   - 同样输入必须得到同样输出

### 10.2 端到端验证

1. 构造一个关系稀疏文件 batch
   - 不再连续盲重试
   - 日志能说明为什么降级

2. 构造一个 multi-hop 条件充足 batch
   - multi-hop 仍能正常生成

3. 验证 manifest metadata
   - 存在 requested/effective/generated/fallback/availability 相关字段

### 10.3 日志验证

控制台必须能看到至少：

- batch 基本信息
- requested counts
- available query types
- fallback 决策
- error classification
- generated counts

## 11. 风险与注意事项

### 11.1 不能把所有 multi-hop 错误都简单降级

如果 batch 本来支持 multi-hop，但只是偶发 API/网络失败，仍应允许 retry。

所以分类必须区分：

- cluster/relationship absence
- network/timeout
- 其他未知错误

### 11.2 fallback 不能悄悄改变数据集统计语义

如果请求的是 `multihop_focus`，最后大面积降级成 single-hop，就必须在 manifest 和日志里可见。

否则会导致：

- 数据集名义分布和实际分布不一致
- 评估结论失真

### 11.3 诊断日志不能只写 metadata

本问题的核心之一是现场排障困难，所以必须打印到控制台，而不仅是写到 manifest。

## 12. 结论

这次问题的本质不是“某个 batch 偶发失败”，而是：

1. 当前 querytype synthetic 生成逻辑缺少 batch 能力判断
2. 把 deterministic cluster failure 当成了 retry 场景
3. 缺少 query type 降级与重分配
4. 缺少控制面诊断日志

真正的修复方向不是增加重试次数，而是把生成流程改造成：

```text
先探测 -> 再分配 -> 不可用则降级 -> 非重试错误直接收敛 -> 全程打印诊断日志
```

如果按本文档执行，能够真正解决你现在遇到的：

```text
No relationships match the provided condition. Cannot form clusters.
```

这类问题，并让 `build_querytype_dataset.py` 在 multi-hop 场景下更稳、更可观测、更容易排障。
