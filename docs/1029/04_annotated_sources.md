# 论文、项目与基准注释目录

> **2026-10-29补充：** 以下为原98条目录，不包含本次新增的Retire、Leyline、AgentSysBench等核查。新增资料及其证据范围见[补查记录](sources_supplement_20261029.json)，选题结论以[新版汇报稿](08_mobile_agent_inference_research_briefing_20261029.md)为准。

检索截止：2026-10-27（Asia/Shanghai）。每条含一句话介绍、原始链接和使用边界。日期优先区分首次预印本、正式发表和修订；未确认的接收状态不作推定。

证据等级：**F**＝已取得全文并核查相关方法/实验/讨论段；**A**＝原始摘要或出版页；**R**＝官方README、文档或项目页；**M**＝会议目录/DOI元数据。均不代表已复现。F条目也不意味着逐行通读全部附录。

## 综述

| 编号 | 论文／项目与时间 | 一句话介绍 | 证据与边界 |
|---|---|---|---|
| S01 | [On-device large language models: a survey of model compression and system optimization](https://link.springer.com/article/10.1007/s10462-026-11538-1)<br>2026-05 | 系统梳理端侧模型压缩、运行时优化、评测与未来挑战。 | **F**：全文第7节支持软硬件与压缩联合设计；不是手机Agent专项结论。 |
| S02 | [Efficient GUI Agents: A Systems Survey of Observation, Memory, Action, and Runtime Optimization](https://arxiv.org/abs/2609.02309)<br>2026-09 | 按观察、上下文、动作和运行时组织GUI Agent效率研究。 | **F**：全文第7节强调额外解析器、验证器和编排成本的完整核算。 |
| S03 | [Agentic mobile AI systems: From post-hoc optimization to runtime-elastic realization](https://www.sciencedirect.com/science/article/pii/S1566253526005075)<br>截至检索日可见；卷期2027-01 | 从表示、推理能力、反馈动作、状态保留和移动执行五类运行时决策讨论移动Agent。 | **A**：仅核查出版页摘要；卷期晚于截止日，不将其写成已完成的2027年会议工作。 |
| S04 | [A Review on Edge Large Language Models: Design, Execution, and Applications](https://doi.org/10.1145/3719664)<br>2025 | 按模型设计、部署与执行梳理边缘LLM全生命周期。 | **A**：边缘包含服务器、嵌入式和手机，不能直接据此判断手机空白。 |

## 推理与测量

| 编号 | 论文／项目与时间 | 一句话介绍 | 证据与边界 |
|---|---|---|---|
| I01 | [Fast On-device LLM Inference with NPUs (llm.npu)](https://arxiv.org/abs/2407.05858)<br>2024预印本；ASPLOS 2025 | 通过提示分块、异常值处理和异构执行降低手机NPU上的LLM预填充开销。 | **A**：手机NPU基础系统；代码关联mllm。 |
| I02 | [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference (HeteroInfer；早期HeteroLLM)](https://arxiv.org/abs/2501.14794)<br>2025-01 | 研究手机SoC异构处理器特性并联合层级与张量级划分。 | **A**：v2为2025-10，系统名HeteroInfer；初稿题名和系统名为HeteroLLM。 |
| I03 | [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](https://arxiv.org/abs/2406.06282)<br>2024-06 | 利用神经元簇和存储调度在手机上运行超出内存容量的LLM。 | **A**：不能假定对任意新模型和激活模式均适用。 |
| I04 | [LLM in a Flash: Efficient Large Language Model Inference with Limited Memory](https://machinelearning.apple.com/research/efficient-large-language)<br>2023预印本；ACL 2024 | 利用稀疏访问和闪存加载减少运行大模型所需的DRAM。 | **A**：存储与权重访问背景；不等于完整手机Agent系统。 |
| I05 | [Transformer-Lite: High-efficiency Deployment of Large Language Models on Mobile Phone GPUs](https://arxiv.org/abs/2403.20041)<br>2024-03 | 针对移动GPU优化算子、量化和执行以加速LLM预填充与解码。 | **A**：覆盖高通和联发科手机GPU。 |
| I06 | [LLMCad / EdgeLLM: Fast On-Device LLM Inference With Speculative Decoding](https://arxiv.org/abs/2309.04255)<br>2023预印本；EdgeLLM TMC 2025 | 用小模型草拟和大模型校验构建端侧级联推理。 | **A**：正式系列见 https://ieeexplore.ieee.org/document/10812936/ ；按版本核对方法增量，不重复计算其共同的端侧推测思路。 |
| I07 | [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379)<br>2024-11 | 为注意力与递归混合模型设计考虑复用收益和状态大小的前缀缓存策略。 | **A**：不是手机专用，但Tandem已将其作为检查点策略基线。 |
| I08 | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)<br>2023 | 以分页管理KV状态，减少碎片并支持共享和高并发服务。 | **A**：数据中心背景；普通分页KV不宜作为新选题。 |
| I09 | [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)<br>2023预印本；2024 | 联合结构化程序接口、RadixAttention和受约束输出优化多次模型调用。 | **A**：早已有程序语义与推理运行时联合设计。 |
| I10 | [MELTing point: Mobile Evaluation of Language Transformers](https://arxiv.org/abs/2403.12844)<br>2024 | 通过MELT在Android、iOS和Jetson上自动测量LLM质量、时延、内存及能耗。 | **F**：用于手机测量方法；项目见 https://github.com/brave-experiments/MELT-public 。 |
| I11 | [Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference](https://arxiv.org/abs/2607.05475)<br>2026-07；v2 2026-10-16 | 利用PowerBench分析手机CPU、GPU、NPU的性能及后端能量归因与协同开销。 | **F**：四部高通手机；第9—10节明确配置、并发和可推广性限制。 |
| I12 | [LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load](https://arxiv.org/abs/2603.23640)<br>2026-03；v2 2026-06 | 比较手机、NPU板卡与笔记本在持续LLM负载下的性能、功耗和热变化。 | **F**：单一主要模型，平台与后端不完全相同，不能归因全部差异于硬件。 |
| I13 | [PELM: Power Efficient On-Device LLM Inference with Speculative Decoding and Dynamic Voltage Frequency Scaling](https://arxiv.org/abs/2609.09662)<br>SenSys 2026；预印本2026-09 | 联合DVFS、自推测解码与可变深度控制持续推理功耗和温度。 | **F**：实验为两种Jetson；项目 https://github.com/imec-nu/PELM 。 |
| I14 | [An Efficient Context Management System for On-Device LLMaaS (Libra)](https://doi.org/10.1145/3774906.3800479)<br>2024早期预印本；SenSys 2026-05 | 解耦应用与模型上下文内存，以灵活分块及加载回收支持端侧共享LLM服务。 | **A**：正式版出版页可读；早期预印本 https://arxiv.org/abs/2403.11805 名为LLM as a System Service on Mobile Devices、系统名LLMS；正式版细节须单独核对。 |
| I15 | [Inference in the Shadows: Taming Memory Bandwidth Contention in Mobile LLM Inference with Sereno](https://www.usenix.org/conference/osdi26/presentation/xin)<br>OSDI 2026-07 | 揭示手机NPU带宽优先级引发的前台卡顿，并用可让出的推测执行缓解争用。 | **F**：真实商业手机；PDF https://www.usenix.org/system/files/osdi26-xin.pdf 。 |
| I16 | [NPUGen: NPU as KV Generator for Stable Long-Context Mobile LLM Inference](https://doi.org/10.1145/3795866.3844480)<br>MobiCom 2026-10 | 提出让NPU生成所需KV以支持稳定的长上下文手机推理。 | **M**：已核对大会目录和DOI；未取得可审阅全文，机制细节不作强结论。 |
| I17 | [MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale Deployment](https://arxiv.org/abs/2603.15954)<br>2026-03；ACL Industry 2026 | 借助真实时延模型搜索移动CPU上质量与速度兼顾的小语言模型。 | **A**：350M、650M、1.4B；模型高分不等于闭环Agent可靠。 |
| I18 | [MobileMoE: Scaling On-Device Mixture of Experts](https://arxiv.org/abs/2605.27358)<br>2026-05 | 研究小于十亿激活参数的端侧MoE模型及其部署权衡。 | **A**：激活参数少不代表总权重小，仍需验证手机内存与工具能力。 |
| I19 | [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324)<br>2025-09；EuroSys 2026 | 利用手机NPU空闲矩阵计算能力并行扩展小模型测试时计算。 | **A**：推理预算与硬件协同已有手机先例；需在Agent任务而非仅模型质量上证明新差异。 |
| I20 | [Energy-Efficient Small Language Model Inference for Mobile Agents via DVFS (DVFSLM)](https://scholars.cityu.edu.hk/en/publications/energy-efficient-small-language-model-inference-for-mobile-agents/)<br>MobiSys Workshops 2026-06 | 通过工作负载功耗与时延估计动态调整处理器频率以满足token时限并降低能耗。 | **A**：已核查作者机构页摘要；是Workshop论文，不是MobiSys主会；不可声称逐请求节能从未研究。 |

## Agent与推理协同

| 编号 | 论文／项目与时间 | 一句话介绍 | 证据与边界 |
|---|---|---|---|
| C01 | [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://arxiv.org/abs/2405.19888)<br>OSDI 2024 | 用Semantic Variable把应用调用关系暴露给服务层以联合优化工作流。 | **F**：服务器系统；“向下传Agent语义”已有先例。 |
| C02 | [Towards End-to-End Optimization of LLM-based Applications with Ayo (早期Teola)](https://arxiv.org/abs/2407.00326)<br>2024预印本；ASPLOS 2025 | 将LLM与非LLM流程细化为数据流原语，进行跨模块优化和调度。 | **A**：同一研究的题名演变；项目 https://github.com/ttaliacc/Ayo 。 |
| C03 | [InferCept: Efficient Intercept Support for Augmented Large Language Model Inference](https://arxiv.org/abs/2402.01869)<br>2024 | 针对工具、环境等中断联合选择保存、换出或重算推理状态。 | **F**：服务器GPU；工具等待时管理KV不是新问题。 |
| C04 | [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965)<br>2025-02 | 按Agent程序而非孤立请求安排服务，利用执行历史与程序结构改善完成时间。 | **F**：不能仅把请求优先级换成任务优先级作为创新。 |
| C05 | [Pie: A Programmable Serving System for Emerging LLM Applications](https://arxiv.org/abs/2510.24051)<br>2025-10 | 通过inferlet让应用编程控制模型前向、状态及外部交互。 | **F**：接口本身已有研究；项目 https://github.com/pie-project/pie 。 |
| C06 | [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230)<br>2025-11；v7 2026-09 | 结合工具返回时间、KV保留TTL和程序级排队改善多轮Agent完成时间。 | **F**：全文HTML出现Continnum拼写；检索和引用统一用摘要页题名。 |
| C07 | [Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC](https://arxiv.org/abs/2506.24045)<br>2025-06；v2 2026-01 | 联合异构执行图、前后台流优先级、抢占与空隙填充服务个人Agent。 | **F**：实测Intel Core Ultra迷你PC、64GB；项目 https://github.com/xinming-wei/LLM.xpu 。 |
| C08 | [Agent-X: Full Pipeline Acceleration of On-device AI Agents](https://arxiv.org/abs/2605.10380)<br>2026-05 | 通过提示重构和示例驱动推测解码加速TinyAgent流水线。 | **F**：实测Mac mini M4 Pro、64GB，不是手机验证。 |
| C09 | [EdgeAgent: Orchestrating On-Device LLM inference for End-User Multi-Agent Systems on CPU-GPU Unified Memory Architectures](https://arxiv.org/abs/2610.03394)<br>2026-10-02 | 联合UMA张量执行、草拟预算与工具等待时让出机制加速多Agent。 | **F**：实测Apple M4/M4 Pro，不能当作Android NPU已解决。 |
| C10 | [SchemaWeaver: Accelerating On-Device Mobile Agentic AI via Schema-Guided Parallel LLM Decoding](https://arxiv.org/abs/2610.17436)<br>2026-10-12 | 利用工具schema并行生成参数、减少手机工具调用的串行解码轮数。 | **F**：真实手机；正确性代价需检查，不能称通用无损等价解码。 |
| C11 | [PastForward: Faster On-Device GUI Agents via Computational Experience Reuse](https://arxiv.org/abs/2609.32166)<br>2026-09-26 | 用历史输出作可校验草稿并预测下一屏以重叠动作执行与推理。 | **F**：含Galaxy S26 Ultra、Xiaomi 15和Jetson，收益依赖重复与预测机会。 |
| C12 | [Tandem: Serving Hybrid Linear-Attention GUI Agents on Mobile NPUs](https://arxiv.org/abs/2610.26660)<br>2026-10-19 | 联合混合模型检查点、短输出推测执行和NPU空闲功耗管理。 | **F**：OnePlus 13T/15；主机部分时延实验用主动散热；项目 https://github.com/flyingdoog/tandem 。 |
| C13 | [AgentStop: Terminating Local AI Agents Early to Save Energy in Consumer Devices](https://arxiv.org/abs/2605.15206)<br>ACM CAIS 2026；2026-05 | 用低成本轨迹信号提前终止低成功概率的本地Agent以节能。 | **F**：主要Apple M1 Max和Jetson AGX Orin、64GB；项目 https://github.com/brave-experiments/AgentStop 。 |
| C14 | [MARS: Efficient, Adaptive Co-Scheduling for Heterogeneous Agentic Systems](https://arxiv.org/abs/2604.26963)<br>2026-04；v2 2026-06 | 统一观察GPU推理与CPU工具，联合准入、续跑优先级和KV保留。 | **F**：服务器GPU-CPU/OpenHands；预览代码 https://github.com/Afterglow231/MARS_preview 。 |
| C15 | [TimelyLLM: Time-sensitive LLM Serving System for Physical-I/O Limited Agents](https://www.anuragkhandelwal.com/papers/timelyllm/)<br>MobiSys 2026 | 按Agent执行时间分段生成和调度，利用生成与物理执行的速度差。 | **F**：机器人与物理I/O场景；旧预印本题名与作者列表不同，本文以会议版为准。 |
| C16 | [LOCAL: Enabling Learning On-device Contiguously for Agent LLMs](https://arxiv.org/abs/2608.15241)<br>2026-08 | 联合前后台执行、适配器版本和KV有效性协调Agent推理与学习。 | **F**：主要RTX 3090级24GB GPU；“on-device”不能直接理解成手机。 |
| C17 | [Not All AI Agents Are Equal: Characterizing Resource and Performance Dynamics](https://arxiv.org/abs/2609.19947)<br>2026-09 | 刻画Agent任务在工具CPU、内存和I/O上的资源异质性及并发瓶颈。 | **F**：主要远端模型API加本地容器；支持测工具资源，不能证明手机上瓶颈必然相同。 |
| C18 | [Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore for Low-Latency, Small-Batch, On-Device Physical-AI Serving](https://arxiv.org/abs/2606.20537)<br>2026-06 | 用图绑定状态胶囊保存KV、递归及其他运行状态以支持恢复与分叉。 | **F**：CUDA/RTX 5090、Jetson Thor、DGX Spark；不是Android进程死亡后工具一致性的完整方案。 |
| C19 | [Why Are GUI Agents Correct but Late? Decode on the Decision-Time Critical Path, Tested with Pre-Compiled Policy Trees](https://arxiv.org/abs/2607.28399)<br>2026-07 | 用预编译条件策略树减少瞬时GUI事件关键路径上的生成延迟。 | **F**：适用于可预枚举动作；未揭示值和临场定位是其边界。 |
| C20 | [Agent-Computer Observation Interfaces Enable Dynamic Computer Use](https://arxiv.org/abs/2606.29472)<br>2026-06 | 以连续关键帧、门控音频和文本叙述分离动态观察与离散动作。 | **F**：DynaCU-Bench为浏览器任务；项目 https://github.com/19PINE-AI/aoi 。 |
| C21 | [REBASE: Device-Cloud Experience Coherence for GUI Agents Across App Updates](https://arxiv.org/abs/2609.32130)<br>2026-09 | 检测并修复App更新导致的设备与云端经验不一致。 | **F**：两种Jetson；“经验会过期”已有直接先例。 |
| C22 | [SAIL: Redesigning Collaborative Language Inference with a Single Server-to-Mobile Handoff](https://snu.elsevierpure.com/en/publications/sail-redesigning-collaborative-language-inference-with-a-single-s/)<br>MobiSys 2026 | 通过一次服务器到手机的交接重设计协同生成以兼顾质量和响应。 | **A**：已核对作者机构页摘要和会议目录，未逐节核查全文；不是全本地推理方案。 |
| C23 | [Online Context Scheduling under Cache Expiry and Human Approval Delays](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7621058)<br>SSRN预印本2026-10-16 | 讨论计算缓存、业务证据与批准的不同有效期及上下文保留与准入。 | **A**：原始SSRN摘要搜索可见，但PDF及直连受限；未核查实现和评测，不作高可信成熟系统依据。 |
| C24 | [Elastic On-Device LLM Service (ElastiLM)](https://xumengwei.github.io/files/MobiCom25-ElastiLM.pdf)<br>MobiCom 2025；早期ELMS 2024 | 用可弹性切换的模型和提示规模满足手机不同应用请求的时延SLO。 | **F**：真实商业手机、移动Agent数据与合成端到端轨迹；代码 https://github.com/yinwangsong/ElastiLM ；不能泛称按任务调模型能力是新问题。 |

## Agent模型与机制

| 编号 | 论文／项目与时间 | 一句话介绍 | 证据与边界 |
|---|---|---|---|
| A01 | [TinyAgent: Function Calling at the Edge](https://arxiv.org/abs/2409.00608)<br>2024 | 结合工具检索与小模型训练实现本地工具调用Agent。 | **A**：主要macOS应用工具场景；不能以小模型部署本身主张新颖性。 |
| A02 | [Beyond Training: Enabling Self-Evolution of Agents with MOBIMEM](https://arxiv.org/abs/2512.15784)<br>2025-12 | 用画像、经验和动作记忆及AgentRR降低模型调用并支持执行恢复。 | **F**：已经涉及调度、动作重用和用户中断，不能把这些词直接当新方向。 |
| A03 | [AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management](https://arxiv.org/abs/2512.10371)<br>2025-12；MobiSys 2026 | 把长任务历史表示成程序和变量，保留有用信息并控制上下文。 | **A**：项目 https://github.com/MobileLLM/AgentProg ；语义上下文压缩已拥挤。 |
| A04 | [MobileExplorer: Accelerating On-Device Inference for Mobile GUI Agents via Online Exploration](https://arxiv.org/abs/2605.26546)<br>2026-05 | 利用VLM推理等待时间在线探索UI并保留可复用探索记录。 | **F**：与“利用Agent空闲做探索”直接重叠；不要与AgentExplorer题名不加核对地重复计数。 |
| A05 | [PalmClaw: A Native On-Device Agent Framework for Mobile Phones](https://arxiv.org/abs/2607.13027)<br>2026-07；EMNLP Demo 2026项目标注 | 将Agent循环、会话与设备工具放入手机原生框架。 | **F**：“循环在手机”与“模型全部在手机”须按配置区分；项目 https://github.com/ModalityDance/PalmClaw 。 |
| A06 | [Does Chain-of-Thought Reasoning Help Mobile GUI Agent? An Empirical Study](https://arxiv.org/abs/2503.16788)<br>2025预印本；ACL Findings 2026 | 实测推理链对GUI任务成功率、输出量和延迟的影响。 | **A**：更长推理并非稳定增益；不能据某一模型推广到全部SLM。 |
| A07 | [FunctionGemma model card](https://ai.google.dev/gemma/docs/functiongemma/model_card)<br>2025-12起 | 提供面向端侧工具调用的270M模型、微调方法与Mobile Actions示例。 | **R**：官方明确面向具体任务微调；示例能力不等于开放域长任务能力。 |
| A08 | [EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](https://ojs.aaai.org/index.php/AAAI/article/view/40230)<br>AAAI 2026 | 用端侧观察与验证反馈配合云端规划，降低通信和响应成本。 | **A**：端云协同路线已有直接AndroidWorld先例，摘要中的隐私表述不等于形式保证。 |
| A09 | [AIOS: LLM Agent Operating System](https://arxiv.org/abs/2403.16971)<br>2024 | 提供Agent调度、上下文快照、工具和资源管理的操作系统式架构。 | **A**：多Agent OS抽象早有先例；不等同于移动OS的进程生存能力。 |
| A10 | [ClawVM: Harness-Managed Virtual Memory for Stateful Tool-Using LLM Agents](https://arxiv.org/abs/2604.10352)<br>2026-04 | 用类型化页和运行时管理改善上下文压缩与重置下的状态持久性。 | **F**：语义状态持久化不是空白，但与NPU快照和移动进程事件有不同边界。 |
| A11 | [ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore](https://arxiv.org/abs/2603.20625)<br>2026-03 | 研究Agent恢复后重试工具时的语义回滚风险。 | **F**：仅增加检查点和重试无法自动保证工具副作用一致。 |
| A12 | [Temporal UI State Inconsistency in Desktop GUI Agents: Formalizing and Defending Against TOCTOU Attacks on Computer-Use Agents](https://arxiv.org/abs/2604.18860)<br>2026-04 | 形式化截图到动作之间的UI变化攻击并研究执行前检查。 | **F**：桌面安全研究；刷新截图作为强基线必须加入。 |
| A13 | [Mind the Gap: Action Rebinding Attacks against Android GUI Agents](https://arxiv.org/abs/2601.12349)<br>2026-01；v3 2026-07 | 展示Android观察到动作间隙可被前台切换利用而重定向动作。 | **F**：证明时序问题在Android存在，但攻击成功率不是自然场景发生率。 |
| A14 | [When Is Smaller Greener? Right-Sizing Language Models for Agentic AI](https://arxiv.org/abs/2610.19345)<br>2026-10-14 | 从任务长度和可靠性分析小模型在整条Agent轨迹上的能量、时延及成本交叉点。 | **F**：选择小模型不能只比较单token能耗；手机实测仍需补充。 |
| A15 | [AutoDroid-V2: Boosting SLM-based GUI Agents via Code Generation](https://arxiv.org/abs/2412.18116)<br>2024-12起 | 借助应用知识与代码生成降低小模型逐步操作GUI的能力要求。 | **A**：说明编译与脚本化路线已有；部署和泛化范围须按原文设置核查。 |
| A16 | [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326)<br>2025-01起 | 以截图输入和原生动作空间训练端到端GUI控制模型。 | **A**：native指Agent建模方式，不代表所有模型推理都在手机执行。 |
| A17 | [Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents](https://arxiv.org/abs/2602.16855)<br>2026-02 | 提供GUI-Owl-1.5多尺寸GUI模型与跨平台Agent能力。 | **A**：前代论文 https://arxiv.org/abs/2508.15144 ；代码 https://github.com/X-PLUG/MobileAgent ；手机运行需按模型、量化与后端单独验收。 |
| A18 | [Temporary Authority, Permanent Effects: Commit-Time Authorization for LLM Agents](https://arxiv.org/abs/2607.10487)<br>2026-07 | 在持久化动作提交边界重新检验早期授权、绑定和执行路径的有效性。 | **F**：已包含epoch、取消分支和依赖证据；提交前检查与版本契约本身不能作为新贡献。 |
| A19 | [Forgetting Without Restarting: Execution-State Unlearning for Stateful LLM Agents](https://arxiv.org/abs/2609.04875)<br>2026-09 | 通过溯源、检查点裁剪和净化重放清除Agent执行态中已撤销信息的影响。 | **F**：已有跨prompt、记忆与KV的契约；面向遗忘正确性，不能当成Android主动取消调度已解决。 |
| A20 | [Agentic Transaction: Towards ACID-Compliant Agent Systems](https://arxiv.org/abs/2608.13900)<br>2026-08 | 以语义原子性、一致性、隔离性和耐久性组织Agent事务框架。 | **F**：包含工具效果与恢复元数据等先例；手机有限预算下的实现仍须独立验证。 |

## Benchmark

| 编号 | 论文／项目与时间 | 一句话介绍 | 证据与边界 |
|---|---|---|---|
| B01 | [AndroidWorld](https://github.com/google-research/android_world)<br>2024发布；后续论文2025 | 在真实Android模拟器上提供20个App、116类可参数化任务和状态评估。 | **R**：适合闭环；工具或App在模拟器不能被写成全部在手机执行。 |
| B02 | [AndroidLab](https://github.com/THUDM/Android-Lab)<br>2024 | 提供Android交互、训练数据和系统性任务评测框架。 | **R**：适合第二环境；App版本与初始化过程要冻结。 |
| B03 | [AppWorld](https://github.com/StonyBrookNLP/appworld)<br>ACL 2024 | 以可执行应用API和数据库任务构成可控、多步Agent评测环境。 | **R**：是API/代码交互世界，不是Android GUI或真实手机工具负载。 |
| B04 | [ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox)<br>2024预印本；后续2025 | 提供有状态工具、隐式依赖、用户模拟器及中间里程碑评测。 | **R**：适合电话设置/联系人等语义，但执行器是模拟环境。 |
| B05 | [Berkeley Function Calling Leaderboard V4](https://gorilla.cs.berkeley.edu/leaderboard)<br>V4 2025；持续更新 | 覆盖函数调用、多轮、长上下文和Agent能力的评测集合。 | **R**：不同子集评测方式不同；静态参考输出回放不能证明闭环成功。 |
| B06 | [HammerBench: Fine-Grained Function-Calling Evaluation in Real Mobile Device Scenarios](https://github.com/MadeAgents/HammerBench)<br>2024-12预印本；ACL Findings 2025 | 围绕手机应用功能构造槽位填充、意图变化与多轮工具调用评测。 | **R**：适合意图变化语义；不是天然具有真实工具耗时和并发到达的系统负载。 |
| B07 | [MobileWorld](https://tongyi-mai.github.io/MobileWorld/)<br>2025-12；ACL 2026 | 加入跨App长任务、用户交互和MCP工具的移动Agent评测。 | **R**：比AndroidWorld更丰富；用户模拟和MCP服务位置要计入成本。 |
| B08 | [AndroidControl / On the Effects of Data Scale on UI Control Agents](https://arxiv.org/abs/2406.03679)<br>NeurIPS 2024 | 发布15,283条Android任务演示以研究数据规模与UI控制能力。 | **A**：主要离线示范，不能直接评估动态闭环、重试或系统成功率。 |
| B09 | [MobileAIBench](https://github.com/SalesforceAIResearch/MobileAIBench)<br>2024 | 对端侧语言及多模态模型的质量和资源开销提供评测。 | **R**：论文 https://arxiv.org/abs/2406.10290 ；适合模型部署筛选，不替代多步Agent评测。 |
| B10 | [MLPerf Mobile v6.0](https://mlcommons.org/2026/06/mlperf-mobile-v6/)<br>v6.0 2026-06 | 将标准化端侧生成式LLM测试引入MLPerf Mobile。 | **R**：用于基础性能对照；不直接提供多任务Agent、恢复和交互语义。 |
| B11 | [DynamicGUIBench / Benchmarking and Improving GUI Agents in High-Dynamic Environments](https://arxiv.org/abs/2604.25380)<br>2026-04 | 构造高动态GUI交互并以视频感知和反馈改善Agent表现。 | **F**：动态GUI先例；迁移到Android需确认支持平台和动作接口。 |
| B12 | [LivingScreen](https://github.com/BITHLP/LivingScreen)<br>2026-06 | 以持续播放短视频的浏览器环境评测观察时机、持续时间和任务正确性。 | **R**：公开代码数据；不能原封不动称为原生手机benchmark。 |
| B13 | [APPSim-Bench / AppSim](https://github.com/Acrab-Agentic-Labs/AppSim)<br>2026-09 | 用保留任务逻辑的可控模拟应用支撑可复现GUI评测。 | **R**：论文 https://arxiv.org/abs/2609.07712 ；模拟应用的系统资源负载需另行验证。 |
| B14 | [AndroidReality](https://github.com/Xiao0o0o/AndroidReality)<br>截至2026-10可见 | 提供Android界面状态、动作执行和转移扰动及恢复评测工具。 | **R**：早期仓库；已核查README，尚未安装验证，不能假定所有扰动都可直接复用。 |
| B15 | [MemGUI-Bench: Benchmarking Memory of Mobile GUI Agents in Dynamic Environments](https://arxiv.org/abs/2602.06075)<br>2026-02 | 评估短期记忆、跨会话学习和跨App动态任务中的记忆能力。 | **A**：语义记忆benchmark，不直接测物理KV有效性或OS进程恢复。 |
| B16 | [OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents](https://arxiv.org/abs/2506.16042)<br>2025-06 | 以人类与Agent对比刻画桌面任务完成时间及效率瓶颈。 | **A**：支持使用任务完成时延；不是手机任务集。 |
| B17 | [Async-RBench](https://github.com/pope8zdt/Async-RBench)<br>2026-10 | 用200个可执行任务测试异步子Agent返回后更新、保留和重验已有工作的能力。 | **R**：主要异步工作语义，不能直接当手机推理并发测试；未运行验收。 |

## 开源与平台

| 编号 | 论文／项目与时间 | 一句话介绍 | 证据与边界 |
|---|---|---|---|
| P01 | [llama.cpp](https://github.com/ggml-org/llama.cpp)<br>持续更新；核查2026-10 | 提供本地量化推理、服务、语法约束及多种CPU/GPU/NPU后端。 | **R**：Android/Hexagon支持须按设备与模型验收；不保证任意模型任意状态快照。 |
| P02 | [ExecuTorch / Qualcomm backend](https://github.com/pytorch/executorch)<br>持续更新；核查2026-10 | 提供PyTorch端侧导出、量化、运行时和Qualcomm QNN等加速后端。 | **R**：与你现有runner最接近；示例接口及多会话能力随版本变化。 |
| P03 | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)<br>持续更新；核查2026-10 | 在LiteRT上提供端侧LLM会话、流输出、工具调用及硬件后端封装。 | **R**：支持矩阵按具体版本核对；有工具API不代表已解决任务资源控制。 |
| P04 | [MNN](https://github.com/alibaba/MNN)<br>持续更新；核查2026-10 | 提供移动推理引擎、LLM与多模态运行及Android示例。 | **R**：不同后端可能回退CPU，必须记录实际执行路径。 |
| P05 | [MLC-LLM](https://github.com/mlc-ai/mlc-llm)<br>持续更新；核查2026-10 | 用编译和统一引擎将LLM部署到Android、iOS及多类GPU。 | **R**：适合移动GPU对照，编译和模型支持受版本约束。 |
| P06 | [mllm](https://github.com/UbiquitousLearning/mllm)<br>持续更新；核查2026-10 | 提供面向手机的轻量多模态LLM运行时及CPU/QNN等研究实现。 | **R**：llm.npu相关基础；旧分支和新版特性不能混为一谈。 |
| P07 | [MobiAgent / MobiInfer / AgentRR](https://github.com/IPADS-SAI/MobiAgent)<br>2025—2026 | 整合移动Agent模型、动作记录复用、评测及Android/HarmonyOS端侧推理组件。 | **R**：研究Agent与推理结合时需核查这些现成组件，不能只比较通用聊天引擎。 |
| P08 | [Android AppFunctions](https://developer.android.com/ai/appfunctions)<br>2026页面；实验预览 | 让应用暴露可被助手发现并执行的设备端函数。 | **R**：实验预览与权限限制；适合作为真实工具方向，不能假定所有商业App均已接入。 |
| P09 | [Apple Foundation Models framework](https://developer.apple.com/documentation/foundationmodels/expanding-generation-with-tool-calling)<br>2025—2026 | 提供系统端侧模型的结构化生成与工具调用接口。 | **R**：适合部署参照；不开放任意底层调度/KV控制，设备和地区可用性有条件。 |
| P10 | [Android service bindings and process states](https://developer.android.com/topic/performance/memory/guide/service-bindings)<br>Android现行官方文档 | 解释服务绑定、进程重要性和内存压力如何影响应用生命周期。 | **R**：支持研究问题的OS依据；不能假定进程死亡前总能收到回调。 |
| P11 | [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)<br>持续更新；核查2026-10 | 通过持久化检查点支持Agent执行恢复、回溯和长任务。 | **R**：普通检查点机制已有工程基线；不自动保证外部工具的恰好一次执行。 |
| P13 | [PowerServe](https://github.com/powerserve-project/PowerServe)<br>2025起；核查2026-10 | 提供面向Android与HarmonyOS的本地LLM服务及NPU推测推理实现。 | **R**：是Sereno等工作的部署对照之一；README的速度声明不作为跨系统公平性能结论。 |

## 检索入口

| 编号 | 论文／项目与时间 | 一句话介绍 | 证据与边界 |
|---|---|---|---|
| P12 | [Awesome Mobile LLMs / GUI Agents Paper List](https://github.com/stevelaskaridis/awesome-mobile-llm)<br>持续更新 | 提供移动模型、部署系统与相关论文线索，辅助交叉检索。 | **R**：另见 https://github.com/OSU-NLP-Group/GUI-Agents-Paper-List ；列表仅用于发现，技术结论回查原文。 |

共 **98 条目录记录**。同一条可同时包含论文与代码；这是注释目录条数，不是独立论文总数，也不以数量替代覆盖质量。
