# MemoRoute-Agent：面向端侧 Agent 的记忆驱动、离线优先、异构执行统一基础设施设计报告

## 1. 设计的核心技术

### 1.1 Memory-Guided Hierarchical Task Routing

#### 核心思想
让经验记忆直接参与任务路由，而不是只参与 prompt augmentation。

- **Level 0：Direct Memory Replay**  
  高相似、低风险任务直接复用历史成功模板
- **Level 1：Lightweight Local Executor**  
  用规则 / 小模型 / FSM 处理简单任务
- **Level 2：Small-LLM Planner**  
  用小模型完成工具选择、参数草拟、检索查询改写
- **Level 3：Full Agent Runtime**  
  复杂任务才进入完整大模型 Agent 流程

#### 创新点
记忆不再只是“生成时的附加上下文”，而是直接控制：
- 是否需要调用大模型
- 是否需要进入完整 Agent 规划
- 是否可复用成功轨迹

---

### 1.2 Phase-Conditioned Predict-then-Prepare Runtime

#### 核心思想
将一次 Agent 交互拆成多个 phase，并预测下一 phase，从而提前准备执行资源。

#### 划分的 phase 如：
1. **Intent / Plan phase**：长 prompt、长工具说明，prefill-heavy  
2. **Memory fusion / Retrieve phase**：插入记忆块、tool schema、历史摘要，仍偏长上下文  
3. **Action generation phase**：生成 tool call / 参数，decode-heavy  
4. **Tool-return resume phase**：工具结果回来后恢复上下文，shape 突变明显  
5. **Branch / Reflection phase**：Best-of-N、自一致性、多候选动作、失败重规划

#### 创新点
不是只预测下一层 expert 或同步等待时间，而是预测 **agent 层的 phase 转移**，把任务逻辑预测与底层资源准备统一起来。

---

### 1.3 Hierarchical State & Parameter Cache

#### 核心思想
统一管理三类缓存对象：

1. **Parameter Cache**
   
2. **Agent State Cache**
   
3. **Execution Artifact Cache**

#### 统一优先级打分
```text
score = α * reuse + β * next_phase_prob + γ * reload_cost + δ * latency_criticality
```

其中：
- `reuse`：历史复用率
- `next_phase_prob`：由 phase predictor 给出
- `reload_cost`：从 UFS / graph rebuild / retrieval rebuild 的代价
- `latency_criticality`：对 TTFT / TPOT 的影响程度

#### 创新点
现有工作多数只缓存模型参数；这里进一步把 **Agent 状态块** 和 **执行工件** 统一纳入缓存体系，形成面向 Agent OS / runtime 的统一资源管理层。

---

## 2. 底层运行时设计

### 2.1 端侧异构执行调度
借鉴 HeteroInfer：
- prefill：**NPU 主导，GPU 辅助**
- decoding：**GPU 主导，NPU 辅助**，重点是吃满 SoC 带宽

### 2.2 稀疏 / I/O 感知执行
借鉴 PowerInfer-2：
- neuron cluster hot/cold 管理
- 根据 batch / branch 数动态调整 hot/cold region

## 3. 评测数据集与任务设计

### 3.1 基础 LLM 任务集
用于验证 prefill / decode 的底层性能：
- **BELLE_multiturn_chat**
- **GSM8K**
- **LongBench / TriviaQA**

### 3.2 手机端真实工作负载
用于验证不同类型 agent workload 的一致性：
- role-play
- multi-turn dialog
- math
- coding

### 3.3 MoE / 总结类任务
用于测试 expert cache / preload / mixed precision：
- **XSum**
- **SAMSum**
- **Wikipedia-jp**

## 4. 评测指标

### 4.1 基础推理指标
- TTFT（Time To First Token）
- TPOT（Time Per Output Token）
- prefill throughput

### 4.2 Agent 任务级指标
- task completion latency
- offline completion ratio

### 4.3 资源指标
- cache hit rate（parameter / state / graph）
- energy per task

### 4.4 用户体验与干扰
- 与游戏 / 视频 / 相机并发时的 FPS 与时延变化
- foreground interference 下的 Agent latency 抖动
