# MemoRoute-Agent：面向端侧 Agent 的记忆驱动、离线优先、异构执行统一基础设施

## 1. 核心问题

面向端侧 Agent（本地日程助手、文件/相册/系统控制 Agent、弱网/断网环境下的个人知识助理等），系统同时面临四类约束：

1. **Agent 任务是多阶段的**  
   这些阶段的计算特性不同：长上下文阶段更偏 **prefill-heavy**，逐步生成阶段更偏 **decode-heavy**。
   
2. **很多请求不应该一上来就调用大模型全量推理**  
   很多任务是重复的、低风险的、模板化的，或者可由小模型/规则系统解决。需要一个 **离线优先（offline-first）的分层任务路由器**，先判断是否能直接复用经验或本地轻量处理，再决定是否升级到完整 Agent 运行时。
   
3. **端侧资源受限且异构**  
   手机/边缘 SoC 上同时存在 CPU、GPU、NPU、DRAM 与 UFS/Flash。  
   不同执行阶段的瓶颈不同：prefill 更偏算力，decoding 更偏内存带宽；稀疏执行、I/O、GPU-NPU 同步都可能成为瓶颈。

4. **经验记忆不能只服务回答内容，还应服务系统决策**  
   端侧 Agent 的经验记忆不仅要帮助模型“记住用户”，还应帮助系统判断：
   - 是否可直接复用历史成功轨迹
   - 是否需要调用完整大模型
   - 下一个 phase 大概率是什么
   - 哪些上下文块、参数块、图执行计划值得提前预热

## 2. 现有工作的做法

### 2.1 PowerInfer：PC 端稀疏激活推理与 GPU-CPU 协同

**主要做法**：
- 观察到神经元激活呈幂律分布，存在少量 **hot neurons** 和大量 **cold neurons**
- 将 hot neurons 预放到 GPU，cold neurons 留在 CPU
- 使用在线 predictor 预测当前会激活哪些神经元
- 设计 neuron-aware sparse operator 和离线 solver 决定放置策略

### 2.2 PowerInfer-2：手机端 NPU/CPU 协同与 I/O 感知执行

**主要做法**：

- 提出 **neuron cluster** 作为手机端推理的执行基本单位
- prefill 阶段采用 **NPU-centric** dense 执行
- decoding 阶段采用 **CPU-NPU hybrid** 稀疏执行
- 设计 neuron cache、flexible loading、cluster-level pipeline 来隐藏 UFS I/O

### 2.3 EdgeMoE：MoE 模型的 expert 外存管理

**主要做法**：
- 针对 MoE 模型，观察到 expert 权重大、占内存，但每次只激活少数 expert
- 将 non-expert 权重常驻内存，将 expert 作为“冷数据”放在外存
- 设计 **expert-wise bitwidth adaptation**
- 设计基于 activation path 的 **expert preload**
- 设计面向 expert buffer 的 **cache eviction policy**

### 2.4 HeteroInfer：手机 SoC 上 GPU-NPU 的异构执行与同步优化

**主要做法**：

- 提出 **layer-level + tensor-level GPU-NPU parallelism**
- 提出 **weight-centric / activation-centric / hybrid** 三种 partition
- 设计基于 UMA 的 **fast synchronization**
- 建立 **profiler + solver + inference engine** 的完整框架


