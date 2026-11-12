# 检索范围、版本核查与结论可信度

> **2026-10-29补充：** 本文保留原检索审计；本次进一步核查Retire、Agentix正式发表页、Leyline、Step-level Optimization、Stop Means Stop、AgentSysBench与AgentChangeBench，见[新增证据记录](sources_supplement_20261029.json)及[新版汇报稿](08_mobile_agent_inference_research_briefing_20261029.md)。

## 1. 时间、范围和工作方式

检索截止为**2026-10-27（Asia/Shanghai）**。以2024—2026为重点，补充PagedAttention、SGLang、LLMCad等2023基础工作。报告覆盖手机推理、Agent服务与模型服务协同、动态GUI、工具调用、状态与恢复、持续功耗／热、基准及开源运行时。

检索使用公开网页搜索，并回到arXiv摘要与HTML/PDF、USENIX与ACM等出版入口、作者页面、官方GitHub及平台文档核查。对直接影响选题的工作取得全文并检查相关方法、评测硬件、任务来源或讨论段；对背景文献保留摘要／元数据证据等级。没有以第三方自动生成的论文解读代替原始技术依据。

此外查看了你的[2026-10-26周记](../weeks/1026.md)、[AppWorld验证记录](../reports/appworld_agent_kv_next_validation_20261026.md)和[已有slack研究计划](../reports/mobile_llm_slack_reclamation_research_plan.md)，用于判断现有平台能力与选题衔接。既有实验结论均标成既有记录；没有新跑模型、手机、App或benchmark，也没有改动现有实验代码。

这是一轮针对选题的广泛检索和重点核查，**不是穷尽全部数据库的系统综述，也没有做引文计量**。未使用搜索结果数推导研究热度；成熟度判断来自是否已有多类方法、直接系统实现和相近评测证据。

## 2. 检索的覆盖轴

| 检索轴 | 代表性检索式／入口 | 重点追查的内容 |
|---|---|---|
| 近期手机推理 | `mobile on-device LLM agent inference system 2025 2026`；MobiSys/MobiCom/OSDI 2026目录 | NPU、异构、统一内存、真实硬件、持续负载 |
| Agent与服务跨层 | `Parrot semantic variable`、`Autellix`、`InferCept`、`Teola`、`Ayo`、`Pie` | 应用语义暴露、程序调度、接口与工作流优化 |
| Agent特有推理 | `Agent.xpu`、`Agent-X`、`EdgeAgent`、`SchemaWeaver`、`Tandem`、`PastForward` | 提示结构、工具schema、短输出、混合模型状态与空闲期 |
| 工具与资源 | `agent tool co-scheduling inference memory`、`mobile agent admission control`、`MARS` | 工具CPU/内存、准入、等待与可恢复状态 |
| 缓存的反向查重 | `agent cache retention scheduling`、`KV cache time-to-live`、`on-device LLMaaS` | 避免把工具等待、close和TTL重新包装 |
| 动态有效性 | `GUI agent stale observations`、`GUI agent latency-aware`、`GUI agent asynchronous real-time` | TOCTOU、动作重绑定、连续观察、过期动作与预编译策略 |
| 生命周期与恢复 | `on-device agent checkpoint recovery`、`mobile LLM process death`、`agent inference exactly-once` | 移动OS、工具效果、二进制模型状态和普通持久化先例 |
| 热与任务级效率 | `mobile agent thermal-aware`、`agent energy budget`、`AgentStop`、`PELM` | 现有DVFS／推测／早停、失败重试、成功任务成本 |
| 基准与工程 | AndroidWorld、AppWorld、ToolSandbox、BFCL、HammerBench及各官方仓库 | 可执行性、在线／离线、实际工具位置、可用checker |
| 综述的展望与引用链 | [S01](https://link.springer.com/article/10.1007/s10462-026-11538-1)、[S02](https://arxiv.org/abs/2609.02309)、[S03](https://www.sciencedirect.com/science/article/pii/S1566253526005075)、官方论文列表 | 从展望找线索后，再检索具体先例；不把展望直接当空白证明 |

检索时排除了大量使用传统“mobile agents”含义、无线网络调度、车联网、无人机或边缘服务器优化却没有直接手机LLM证据的结果；只在它们提供最接近方法时保留，例如TimelyLLM。硬件模拟/PIM设计也没有与可立即运行的商业手机系统混为一类。

## 3. 版本和出版状态的易错点

| 条目 | 本轮核查结果 | 报告如何处理 |
|---|---|---|
| [HeteroInfer／HeteroLLM，I02](https://arxiv.org/abs/2501.14794) | 旧检索结果题名为HeteroLLM；摘要页v2题名为Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference，系统名HeteroInfer | 合并为一个研究，保留旧名供检索 |
| [Ayo／Teola，C02](https://github.com/ttaliacc/Ayo) | 早期arXiv题名Teola，正式项目引用为Ayo | 不当成两项独立成果 |
| [TimelyLLM，C15](https://www.anuragkhandelwal.com/papers/timelyllm/) | 旧2024预印本为robotic applications题名；MobiSys 2026版题名和作者列表更新 | 以作者会议版页面和PDF为准；代码 https://github.com/Neawhen/TimelyLLM |
| [Continuum，C06](https://arxiv.org/abs/2511.02230) | 摘要页截至本轮为2026-09的v7；HTML正文有Continnum拼写 | 核查新版本，不将拼写误判为另一系统 |
| [MobileLLM-Flash，I17](https://arxiv.org/abs/2603.15954) | 摘要页正式题名包含Industry Scale Deployment | 目录保留完整题名 |
| [Tandem，C12](https://arxiv.org/abs/2610.26660) | 首次提交2026-10-19，距离检索很近 | 明确纳入，不把其检查点、短输出与power vote机制当空白 |
| [S03综述](https://www.sciencedirect.com/science/article/pii/S1566253526005075) | 公开页面已可检索，卷期标为2027-01；直连全文受限 | 只引用截至当前可见摘要，不声称查阅了未来已完成工作或全文展望 |
| [NPUGen，I16](https://www.sigmobile.org/mobicom/2026/program.html) | 已在MobiCom 2026目录找到题名与DOI；本轮未取得可核查全文 | 证据标M，只作最新工作提醒，不据此作具体机制或创新排除结论 |

MobiCom 2026会期为10月26—30日，检索日处在会期中。报告依据已公开目录／论文，不预言后续报告、代码发布或最终修订。

## 4. 来源目录、抓取记录和未解决的访问限制

[sources.json](sources.json)和[注释目录](04_annotated_sources.md)共98条记录。证据等级区分：F＝全文相关段核查；A＝摘要／出版页；R＝官方仓库／文档；M＝目录／DOI。它们都不代表安装成功、复现完成或手机实际兼容。

[source_link_audit.json](source_link_audit.json)保存逐条主链接的HTTP状态、重定向和页面题名，用于发现错链和版本变化。主链接可达不等于其所有二级代码／模型链接可达。部分出版站直连返回403，但公开搜索服务能读到原出版页内容；这些情况保留在记录中，不伪装成已取得全文。

特别是端侧LLMaaS和NPUGen，对推荐方向有潜在影响，但本轮取得的证据等级有限。正式选题前应优先补取全文；在此之前，报告只提出条件性机会，不给“现有工作都没做”的排他性结论。

正文和目录中不收录搜索工具内部引用编号；所有技术来源均使用可直接打开的原始URL。临时取得的全文只用于核查，报告目录没有复制整篇论文。

## 5. 哪些判断比较可靠，哪些仍是推断

| 判断 | 证据强度 | 原因 |
|---|---|---|
| KV复用、任务级调度、工具等待时管理状态已有大量先例 | 较强 | 存在多项原始系统论文与实现，不依赖单一综述 |
| 泛化的“Agent和Inference协同”已不足以立题 | 较强的选题判断 | 多条跨层路线已有直接方法；不意味着无法继续研究 |
| 一些on-device系统只在PC／Mac／Jetson／服务器验证 | 较强 | 核查具体论文实验硬件，不能从题名替代实验设置 |
| Android上观察与动作之间的状态不一致是实际风险 | 较强 | 有直接Android研究；但自然发生率没有由攻击实验确定 |
| 有效性传播进入手机推理执行层可能是机会 | 中等／假设 | 已有相近动态观察、安全和时序工作，需新实验区分并补查最新先例 |
| 手机本地工具与NPU需要不同于服务器的联合准入 | 中等／假设 | 资源机制合理，相关单项证据存在，但缺少你当前负载的直接测量 |
| 状态分离恢复和跨步骤热预算值得立题 | 待判别 | 已有重叠多；只有简单机制失效且收益足够时才保留 |
| 任一推荐方向完全没人做过 | **不作此结论** | 搜索无法证明不存在，尤其2026年新稿与代码变化快 |

## 6. 补充查重结果与选定方向前的补查清单

1. 对照方向一，本轮进一步取得并核查了[Temporary Authority, Permanent Effects](https://arxiv.org/abs/2607.10487)相关全文段：它已研究持久动作提交时的epoch、绑定、分支取消与授权证据。报告据此明确排除“提出版本契约／提交检查”作为新贡献；候选差异仅保留手机推理队列与执行层的取消、合并和重启成本，仍需对照实现。
2. 对照方向二补看：端侧LLMaaS、NPUGen及MARS源码，确认真实内存回收粒度、工具资源信息和可控制的执行点；也应关注[Online Context Scheduling under Cache Expiry and Human Approval Delays](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7621058)，其摘要讨论计算、证据和批准的不同有效期，本轮尚未完成全文与实现可信度核查。
3. 对照方向三，本轮也取得并核查了[Agentic Transaction](https://arxiv.org/abs/2608.13900)和[Forgetting Without Restarting](https://arxiv.org/abs/2609.04875)相关全文段：前者以语义ACID讨论Agent事务，后者通过溯源、状态裁剪与净化重放处理执行态遗忘。这些先例进一步排除了泛化的“首次做Agent一致性／恢复／跨层失效契约”。
4. 对照方向四补看：[Energy-Efficient Small Language Model Inference for Mobile Agents via DVFS](https://scholars.cityu.edu.hk/en/publications/energy-efficient-small-language-model-inference-for-mobile-agents/)已直接研究移动Agent的DVFS节能；这是MobiSys Workshops工作，不能误写成主会论文。再与PELM、AgentStop和手机NPU测试时扩展逐项比较。

这些补充来源也已纳入注释目录。没有将摘要线索冒充已取得全文或复现系统。正式选题时应重新跑一次截至当日的定向检索；较窄的差异化主张需要比综述级描述更严格的源码与全文对照。
