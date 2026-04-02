# QueryType Synthetic 鲁棒性修复方案（历史索引）

本文件原用于描述 `build_querytype_dataset.py` 在 multi-hop cluster / relationship 缺失场景下的鲁棒性修复设计。

当前其结论已经并入主文档：

- [RAGAS集成方案.md](D:/Bread/College/AI/Code/RAG/docs/RAGAS集成方案.md)

主文档已覆盖以下内容：

1. per-batch availability probe
2. non-retriable cluster error classification
3. deterministic fallback / reallocation
4. `QUERYTYPE BATCH` 控制台诊断日志
5. `--enable-multi-file` 的真实行为
6. manifest metadata 中的诊断摘要

若需查看当前实现，请直接参考：

- [build_querytype_dataset.py](D:/Bread/College/AI/Code/RAG/app/evals/build_querytype_dataset.py)
- [querytype_synthesizers.py](D:/Bread/College/AI/Code/RAG/app/evals/querytype_synthesizers.py)
- [Evals命令文档.md](D:/Bread/College/AI/Code/RAG/docs/Evals命令文档.md)

保留本文件仅用于历史索引与旧链接兼容。
