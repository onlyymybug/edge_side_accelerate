# 用户中断、意图变化与过期推理：数据集补查

> **2026-10-29更正：** 数据集说明仍可参考，但“进一步限定到手机取消／状态释放”不自动构成新问题。[Retire](https://arxiv.org/abs/2610.01160)已有直接的版本化切换和状态继承方案；研究边界已在[新版汇报稿](08_mobile_agent_inference_research_briefing_20261029.md)重新评估。

补查日期：2026-10-28。本文件补充原调研，不表示已运行任何benchmark。**InterruptBench是此前报告漏列的直接相关工作**，因此“用户会中途改口”不能作为新问题；候选研究必须进一步限定到手机推理执行期间的取消、合并、资源释放及其实际成本。

## 1. 最直接的现有资源

| 数据集／基准 | 原始资料 | 本轮已核查内容 | 对手机推理研究的边界 |
|---|---|---|---|
| InterruptBench（2026-04） | [论文](https://arxiv.org/abs/2604.00892)、[代码与数据](https://github.com/HenryPengZou/InterruptBench) | 任务过程中追加、修订、撤回要求；公开raw JSON、注入配置与WebArena环境说明 | 是Web任务闭环；注入按动作位置／轨迹进度控制，不等于模型prefill/decode运行中取消或真实用户墙钟记录 |
| HammerBench（2024-12预印本；ACL Findings 2025） | [论文](https://aclanthology.org/2025.findings-acl.175/)、[数据](https://huggingface.co/datasets/MadeAgents/HammerBench)、[代码](https://github.com/MadeAgents/HammerBench) | 中英文多轮函数调用，包含Intent Shifts和Argument Shifts；README列出IS、SO-case1、SO-case2、mSv等类型 | 最贴近现有文本工具调用接口；原数据是对话／函数调用快照，不自带实际工具执行、用户到达时间和手机能耗 |
| Async-RBench（2026-10） | [论文](https://arxiv.org/abs/2610.32429)、[代码](https://github.com/pope8zdt/Async-RBench) | 200个公开任务实例，研究异步子Agent结果抵达后更新、保留、禁止及重验要求 | 更新来自子Agent结果，不等同用户改口；执行与评分依赖需另行验收，不能直接称手机benchmark |
| AmbiBench（2026-02） | [论文](https://arxiv.org/abs/2602.11750) | 手机GUI任务、不同指令清晰度、用户模拟与交互评估 | 核心是通过澄清识别真实意图，不等于用户在推理运行中改变意图；适合补充移动交互语义 |

证据：本轮读取InterruptBench和AmbiBench论文相关方法段，以及前三个项目README；并直接检查InterruptBench的公开JSON和中文运行指南。未安装WebArena、未跑模型或验证全部任务环境。

## 2. InterruptBench已经公开到什么程度

论文为 *When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation*。数据在仓库 `Eval/interrupt_config/raw/`，可见 `1update.json`、`2update.json`、`2modification.json`、`1retraction.json`、`2retraction.json` 和 `3mixed.json` 等文件。[官方仓库](https://github.com/HenryPengZou/InterruptBench)。

本轮直接读取了 `2modification.json`，该文件有165条记录；**这是该文件的记录数，不是所有类型相加后的独立任务总数**。条目包含 `task_id`、原任务 `intent`、修改后的初始目标 `transformed_initial_intent` 和后续 `updates`。这些信息可复用来构造明确的目标版本变化，保留原任务终态评估。

论文§2.3的协议是先生成初始目标轨迹，记录动作数量，再按轨迹进度选择注入点并继续执行。仓库配置可设置 `interrupt_at_action`；中文指南也说明 `interrupt_at_pct` 按保存的动作数量百分比指定中断点。[论文方法](https://arxiv.org/html/2604.00892v1)、[中文运行指南](https://github.com/HenryPengZou/InterruptBench/blob/main/Eval/RUN_PARALLEL_EVAL_ZH.md)。

因此它适合评估“Agent能否根据更新修复任务轨迹”。如果要研究“更新抵达时，NPU仍在生成旧请求”，还要扩展服务层事件协议，记录墙钟时间、入队／执行阶段、取消请求与生效时间。扩展后的实验不能冒充其原始基准协议。

论文的中断场景由合成流程构建并经人工核查；不能拿它估计真实手机用户每分钟改口多少次。按轨迹进度注入也不是自然用户到达分布。[论文§2.2—2.3](https://arxiv.org/html/2604.00892v1)。

## 3. 现有平台从哪一个开始

对你现有的QNN文本工具调用runner，**HammerBench的参数覆盖／意图变化子集更适合小规模接口验证**。可以复用对话、工具schema及正确调用标签，再添加可审计的任务版本与更新事件。需要注意，一段数据可能是同一对话不同轮次的快照，不能将全部记录视为独立任务，也不能把参考历史回放当作模型闭环成功率。[官方数据说明](https://github.com/MadeAgents/HammerBench)。

若要验证长任务已经执行部分动作后如何调整，再接InterruptBench。它有可执行Web环境和终态检查，但接入比文本工具调用重；模型在手机、Web环境在PC时，报告必须明确两者的位置，不能据此声称已测手机本地工具资源争用。

最小系统扩展是：

1. 保留原有目标更新内容和正确性判据。
2. 为请求增加 `task_id` 与 `intent_epoch`，独立记录更新抵达时间。
3. 比较旧请求完成后再处理、队列中最新版本覆盖、运行中取消等策略。
4. 报告更新后的正确性、有效任务响应、无效计算和取消成本；新增时间协议明确标为实验扩展。

## 4. 哪一类数据本轮仍未找到

本轮没有核实到一个可直接使用的数据集，同时提供以下四项：**真实手机用户自然改口／取消事件、事件墙钟时间、模型prefill/decode执行状态，以及同步资源或能量记录**。

这表示系统测量层仍需补充，不表示“用户中断”没人研究。受控注入可以验证机制和适用区间；自然发生频率与实际收益，需要后续真实交互数据支持。

对于第二个方向——手机本地工具与LLM争资源——上述数据集只能提供任务语义。是否发生统一内存／带宽争用，仍须在同一手机上实际运行相关工具和推理；不能用对话数据或人工等待替代物理负载。
