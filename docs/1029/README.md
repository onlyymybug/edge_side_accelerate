# 手机端 Agent 与本地 LLM 推理：汇报入口

**当前汇报稿：[手机上的 Agent 与本地大模型：别人做了什么，我们还能研究什么](08_mobile_agent_inference_research_briefing_20261029.md)。** 更新至 2026 年 10 月 29 日，已整体改为通俗表述，并保留原章节编号，便于对应此前讨论。

正文先用一个手机助手例子解释 Agent 和推理服务的分工，再介绍已有工作、最近方向、可以研究的问题，以及可用的数据和代码。每项工作说明遇到了什么问题、用了什么办法、实验做到了哪一步；1.3 补充模型与开源情况，1.4 补充非 GUI 任务。

4.1 用两个任务排队的例子解释：模型计算总量相同，改变执行顺序也可能让任务提前完成。随后说明真实系统为什么更难、应与哪些简单方法比较，以及怎样用实验决定是否值得继续。4.2、4.3 分别用内存不足和持续发热的例子解释。本文没有新增实验，也没有运行所列基准。

手机相关原始链接与证据边界见[sources_phone_only_20261029.json](sources_phone_only_20261029.json)。

## 历史调研留档

以下资料保留此前较宽的检索范围，尚未按本轮手机范围重写。汇报请使用当前稿，不沿用旧方向优先级。

- [01：原研究版图](01_landscape_and_prior_art.md)
- [02：原候选方向](02_research_directions.md)
- [03：原基准与实验协议](03_benchmarks_and_validation.md)
- [04：原98条注释目录](04_annotated_sources.md)
- [05：原检索说明](05_search_scope_and_evidence.md)
- [06：原中断数据集补查](06_interruption_datasets_update_20261028.md)
- [07：已被替代的旧汇报稿](07_mobile_agent_interruptible_inference_briefing_20261028.md)

历史机器可读材料：[原98条来源](sources.json)、[原链接检查](source_link_audit.json)、[此前补查记录](sources_supplement_20261029.json)。它们不作为当前稿的手机来源清单。
