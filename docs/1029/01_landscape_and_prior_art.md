# 研究版图：手机端侧Agent与推理服务之间还有什么空间

> **2026-10-29更正：** 本文保留原检索截止范围。补查发现[Retire](https://arxiv.org/abs/2610.01160)已直接研究版本化撤销、有效状态继承与连续修订，不能再将通用取消传播视为未解决主线。按现有工作重新推导的结论见[新版汇报稿](08_mobile_agent_inference_research_briefing_20261029.md)。

截止2026-10-27。编号对应[注释目录](04_annotated_sources.md)，文中直接链接原始论文、项目或官方文档。下面的“较多／较少”是本次检索的相对判断，未作完整文献计量；“没找到直接覆盖”不表示不存在。

## 1. 先划清三个容易混淆的范围

**手机操作Agent不一定在手机推理。** AndroidWorld上的Agent可以从服务器调用云模型操作模拟器；这证明操作能力，不能证明手机时延、功耗和内存可行。相反，模型在手机运行而AppWorld工具在PC运行，也只证明“手机推理＋主机工具”的闭环。[AndroidWorld，B01](https://github.com/google-research/android_world)、[AppWorld，B03](https://github.com/StonyBrookNLP/appworld)。

**On-device不一定指手机。** Agent.xpu实测Intel Core Ultra迷你PC，Agent-X实测Mac mini，EdgeAgent实测Apple M4系列，LOCAL主要实测RTX 3090级GPU。这些方法是必须尊重的先例，但不能直接代替Android上对NPU接口、统一内存、后台限制与热环境的验证。[C07](https://arxiv.org/abs/2506.24045)、[C08](https://arxiv.org/abs/2605.10380)、[C09](https://arxiv.org/abs/2610.03394)、[C16](https://arxiv.org/abs/2608.15241)。

**语义记忆与推理缓存是不同状态。** 历史摘要、工具结果、用户偏好、可复用动作属于Agent状态；KV、递归状态、位置与采样状态属于执行加速状态。文本相同、经验仍有用、KV可正确恢复是三个不同条件。Marconi、Tandem和Execution-State Capsules已经说明混合模型下“只保存KV”并不充分。[I07](https://arxiv.org/abs/2411.19379)、[C12](https://arxiv.org/abs/2610.26660)、[C18](https://arxiv.org/abs/2606.20537)。

## 2. 已经研究较多的部分

| 主题 | 代表证据 | 已经覆盖到哪里 | 对选题的含义 |
|---|---|---|---|
| 量化、小模型、压缩与硬件感知模型设计 | [S01](https://link.springer.com/article/10.1007/s10462-026-11538-1)、[MobileLLM-Flash](https://arxiv.org/abs/2603.15954)、[MobileMoE](https://arxiv.org/abs/2605.27358) | 从权重压缩走向模型结构、时延和系统联合设计 | “把模型压小并放到手机”通常不足以构成新问题 |
| 手机CPU/GPU/NPU推理优化 | [llm.npu](https://arxiv.org/abs/2407.05858)、[HeteroInfer／早期HeteroLLM](https://arxiv.org/abs/2501.14794)、[Transformer-Lite](https://arxiv.org/abs/2403.20041) | 分块预填充、异构映射、量化算子与执行调度 | 仍有难题，但通用异构加速不是稀少方向 |
| 容量不足时的权重与状态管理 | [PowerInfer-2](https://arxiv.org/abs/2406.06282)、[LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language)、[端侧LLMaaS](https://doi.org/10.1145/3774906.3800479) | 稀疏访问、闪存流送、上下文分块与应用解耦 | 不应把全部手机内存问题简化为KV，也不宜再提泛化的“换出重算” |
| 前缀共享和结构化程序运行 | [PagedAttention](https://arxiv.org/abs/2309.06180)、[SGLang](https://arxiv.org/abs/2312.07104)、[Marconi](https://arxiv.org/abs/2411.19379) | 分页、Radix复用、受约束输出、混合模型检查点 | 你的“KV研究很多”判断成立，但不能由此说所有状态问题已解决 |
| 任务级而非请求级服务 | [Parrot](https://arxiv.org/abs/2405.19888)、[Ayo/Teola](https://arxiv.org/abs/2407.00326)、[Autellix](https://arxiv.org/abs/2502.13965)、[Pie](https://arxiv.org/abs/2510.24051) | 语义依赖、程序执行历史、细粒度工作流和可编程前向接口 | “跨层传信息”及“程序优先调度”已有直接先例 |
| 工具等待与KV联合管理 | [InferCept](https://arxiv.org/abs/2402.01869)、[Continuum](https://arxiv.org/abs/2511.02230)、[MARS](https://arxiv.org/abs/2604.26963) | 中断时保存／换出／重算、返回分布与TTL、工具CPU和GPU准入 | 单独利用ready/waiting/close信号的新颖性很弱 |
| 小模型工具调用与工具检索 | [TinyAgent](https://arxiv.org/abs/2409.00608)、[FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma/model_card) | 专用训练、工具选择、手机动作示例 | 更重要的是领域外工具、长任务与实体绑定是否可靠 |
| 按应用SLO调整模型与提示规模 | [ElastiLM](https://xumengwei.github.io/files/MobiCom25-ElastiLM.pdf) | 在手机上联合模型与提示弹性，满足多类请求时延需求 | 按任务调模型能力也已有真实手机先例，不能仅与固定最大模型比较 |
| GUI规划、记忆、动作重用 | [MOBIMEM](https://arxiv.org/abs/2512.15784)、[AgentProg](https://arxiv.org/abs/2512.10371)、[S02综述](https://arxiv.org/abs/2609.02309) | 程序化上下文、画像／经验／动作记忆与恢复 | 不是“编排不像科研”，而是必须提出可检验的机制和失败边界 |

这里的“研究较多”不等于已经没有值得发表的工作。例如手机NPU的框架协同开销、后端能量归因和前台干扰在2026年仍不断出现新证据。但这些都已有明确先例，后续工作需要提出更细的问题。[PowerBench，I11](https://arxiv.org/abs/2607.05475)、[Sereno，I15](https://www.usenix.org/conference/osdi26/presentation/xin)。

## 3. 2025—2026最直接的交叉层工作

下表优先展示会直接改变你选题边界的工作，而非按性能数字排名。不同硬件、模型、任务及基线下的加速比不可横向比较。

| 工作 | 真正使用的跨层信息／机制 | 实际验证范围 | 尚不能据其认定已解决的问题 |
|---|---|---|---|
| [ElastiLM，C24](https://xumengwei.github.io/files/MobiCom25-ElastiLM.pdf) | 请求SLO、模型与提示的弹性配置 | 商业手机、NLP／移动Agent数据与合成服务轨迹 | 不是持续变化环境、取消传播或外部工具恢复的完整验证 |
| [Agent.xpu，C07](https://arxiv.org/abs/2506.24045) | reactive/proactive流、算子亲和性、阶段弹性、抢占和slack填充 | Intel Core Ultra 5 125H，64GB；个人Agent混合负载 | Android NPU、移动OS事件与真实手机长期运行 |
| [Agent-X，C08](https://arxiv.org/abs/2605.10380) | Agent提示结构、示例输出的可预测性 | Mac mini M4 Pro，TinyAgent文本工具流水线 | 真实手机GUI、动态环境与中途失效 |
| [EdgeAgent，C09](https://arxiv.org/abs/2610.03394) | UMA执行、序列草拟难度、工具停顿时让出 | Apple M4/M4 Pro；部分工具时延控制实验 | Android工具的真实资源开销、NPU与前台交互 |
| [SchemaWeaver，C10](https://arxiv.org/abs/2610.17436) | 工具schema中的参数结构，减少串行前向次数 | 两类手机SoC和多个量化模型，含BFCL工具调用 | 语法正确不保证工具语义正确；独立并行参数可能有质量损失 |
| [PastForward，C11](https://arxiv.org/abs/2609.32166) | 历史输出草拟、下一屏预测、验证后保留计算 | AndroidWorld派生负载；手机及Jetson等 | 对动态不可预测屏幕、低重复负载的收益不能预设 |
| [Tandem，C12](https://arxiv.org/abs/2610.26660) | 改写历史时的递归检查点、短输出校验、空闲power vote | OnePlus 13T/15，混合线性注意力GUI模型 | 主设备部分时延实验有主动散热；不能直接推出无散热长期热收益 |
| [Sereno，I15](https://www.usenix.org/conference/osdi26/presentation/xin) | 前台帧质量、NPU带宽非对称争用、细粒度可让出执行 | 商业手机与25个前台App | Agent任务语义、自然取消和工具副作用恢复 |
| [MARS，C14](https://arxiv.org/abs/2604.26963) | GPU块池、CPU工具开销、准入、程序续跑和KV收益 | H100/H200服务器、OpenHands | 手机端工具与NPU共享LPDDR且受权限限制的执行方式 |
| [TimelyLLM，C15](https://www.anuragkhandelwal.com/papers/timelyllm/) | Agent执行段耗时、生成与物理执行的速度差 | 机器人／物理I/O受限Agent | 不是手机GUI失效与NPU热预算的现成答案；但“工具间隙有slack”已非新发现 |
| [AgentStop，C13](https://arxiv.org/abs/2605.15206) | 轨迹成功概率信号与早停 | Mac M1 Max、Jetson AGX Orin；问答与代码任务 | 手机场景迁移、停止后恢复、错误早停的成本仍需验证 |
| [LOCAL，C16](https://arxiv.org/abs/2608.15241) | 适配器版本、前后台优先级和KV有效性 | RTX 3090级GPU、tau-bench | 手机在线训练可行性；仅说“版本会让缓存失效”已被覆盖 |
| [Execution-State Capsules，C18](https://arxiv.org/abs/2606.20537) | 图绑定的完整可恢复缓冲区及边界元数据 | CUDA、RTX 5090、Jetson Thor、DGX Spark | Android进程死亡、外部工具状态与本地推理状态的联合恢复 |

有两个特别容易误读的结果。SchemaWeaver全文报告，某Qwen3-1.7B设置中受约束解码的whole-call correctness为76%，其方法为74%；这说明它不能被泛称为“保证原输出分布不变的无损方法”。Tandem的主要时延对照含主动散热，而第二设备采用被动散热；评估长期热收益时要单独设计无散热测试。[C10全文§7.1](https://arxiv.org/html/2610.17436v1)、[C12全文§6.1](https://arxiv.org/html/2610.26660v1)。

## 4. 做得相对少的，是完整问题链条，而非某个关键词

### 4.1 请求还没结束，其依据已经失效

Android动作重绑定攻击已经证明观察到动作的间隙是真实存在的；桌面TOCTOU、防过期执行检查、动态观察接口和预编译策略树也都已有研究。因此，“屏幕会变”“先检查再点击”“动态GUI要快”均不能当新问题。[A13](https://arxiv.org/abs/2601.12349)、[A12](https://arxiv.org/abs/2604.18860)、[C20](https://arxiv.org/abs/2606.29472)、[C19](https://arxiv.org/abs/2607.28399)。

本次未找到充分覆盖以下整个组合的手机研究：**依据失效事件进入推理队列和执行层，决定是否取消正在运行的计算、如何保留仍有效的部分，并在变化频繁时避免持续重启。** 这只是候选缺口。安全攻击的发生率不能代表自然App变化率；如果真实任务很少出现该问题，或执行前刷新已经解决，就不应继续。

### 4.2 工具不再只是一个等待时长，而是同一台手机上的资源消费者

MARS和资源动态测量已明确研究CPU工具，因此不能声称“过去把所有工具都当黑盒”。相对缺乏的是手机闭环中对**NPU模型服务、本地OCR／图片／语音／检索工具、前台App共享内存与带宽**的联合证据，以及应用权限内可用的控制方式。[C14](https://arxiv.org/abs/2604.26963)、[C17](https://arxiv.org/abs/2609.19947)、[I15](https://www.usenix.org/conference/osdi26/presentation/xin)。

先决条件是任务真的调用这些本地工具，而且串行、固定并发上限或简单错峰不能充分解决。若所有工具只是PC上的轻量API响应，这个问题在当前平台上不成立。

### 4.3 移动生命周期下，哪些状态值得保存、哪些必须重验

MOBIMEM已经包含中断恢复，LangGraph已有持久化，ClawVM管理状态驻留与耐久性，Execution-State Capsules保存完整模型状态。剩余可检查的手机切片是：**被OS回收或重建服务后，语义进度、工具效果和二进制推理状态具有不同有效期，如何用有限存储恢复到可继续且不重复副作用的边界。**[A02](https://arxiv.org/abs/2512.15784)、[P11](https://docs.langchain.com/oss/python/langgraph/persistence)、[A10](https://arxiv.org/abs/2604.10352)、[C18](https://arxiv.org/abs/2606.20537)。

这更像执行正确性与恢复成本问题，不宜再包装成KV缓存命中率。若工具支持幂等、文本日志很短且冷重建便宜，普通持久化方案就可能足够。

### 4.4 从“最省token”到“完成有用任务所需的总成本”

持续负载热测量、PELM、AgentStop与模型规模交叉点分析均已有工作；端侧Agent的热管理不能被称为空白。可以检查的是**热状态是否改变后续步骤、验证、重试和可完成任务数，从而使逐请求最优策略失效**。[I12](https://arxiv.org/abs/2603.23640)、[I13](https://arxiv.org/abs/2609.09662)、[C13](https://arxiv.org/abs/2605.15206)、[A14](https://arxiv.org/abs/2610.19345)。

只有跨步骤预算比固定低功耗、固定上限、deadline降频等强简单基线有稳定额外价值时，这个方向才成立。不要先把bandit或RL作为贡献。

## 5. 从综述展望能借鉴什么，不能据此推出什么

端侧LLM综述提出量化、KV和算子融合需要联合设计，GUI效率综述强调解析器、验证器、检索和编排的新增开销应统一核算；移动Agent综述摘要则强调表示、推理、动作、状态与执行随运行时共同变化。这些是选择研究视角的依据，**不是“某个具体问题没人做”的证据**。[S01 §7](https://link.springer.com/article/10.1007/s10462-026-11538-1)、[S02 §7](https://arxiv.org/abs/2609.02309)、[S03摘要](https://www.sciencedirect.com/science/article/pii/S1566253526005075)。

对你最实用的启发是把优化单位从token或单次请求提升为**有外部状态、有截止或失效条件、有可验证终态的任务**；同时让推理层只接收少量可以可靠获得的契约字段。下一篇报告会把这些字段落到候选问题，而不把一个大而全的“Agent OS”作为目标。
