# 图表说明与主要结论

## 1. Fig01：GSM8K 上配置对系统指标的影响

### 1.1 Prefill throughput

![Fig01a：Prefill throughput](assets/paper_v2_20260902/fig01a_prefill.png)

该图比较 Qwen2.5-3B、GSM8K、output budget 512 下，不同 QNN mode 和 T-MAC thread 的 prefill throughput。

主要结论：

- QNN 的 prefill 明显快于 T-MAC。以 Balanced/T6-512 为例，OnePlus 为 585.1 对 23.6 token/s，Redmi 为 565.6 对 22.7 token/s，约相差 25 倍。
- QNN 中 Burst 的 prefill 最高：OnePlus 766.0 token/s、Redmi 602.3 token/s；LowPower 最慢。
- T-MAC 增加线程可以提高 prefill，T8 在两部手机上均为三种 thread 中最高，但这一优势没有自动转化为更快 decode。
- 该图说明 prefill 是 QNN 相对 T-MAC 最稳定、最明显的系统优势之一。

### 1.2 Decode throughput

![Fig01b：Decode throughput](assets/paper_v2_20260902/fig01b_decode.png)

主要结论：

- 与 prefill 不同，两种 runtime 的 decode 位于相近量级，主要落在 10–17 token/s。
- QNN Balanced 在两部手机上都是三种 QNN mode 中最快的 decode 配置：OnePlus 14.93 token/s，Redmi 16.87 token/s。
- QNN Burst 并没有获得最高持续 decode；LowPower 则明显变慢。
- T-MAC T6 是三种 thread 中 decode 最均衡的配置。T8 虽然 prefill 更快，但 decode 在 OnePlus 和 Redmi 上均低于 T6，说明线程数不是越多越好。
- Redmi QNN Balanced-512 比 OnePlus 快约 13%，但这不能单独推出 Redmi 更节能。

### 1.3 TTFT

![Fig01c：TTFT](assets/paper_v2_20260902/fig01c_ttft.png)

主要结论：

- QNN 的 inference-only TTFT 约为 0.20–0.55 秒；T-MAC 约为 5.4–7.7 秒，两者差距超过一个数量级。
- 该差距主要与 prefill 路径有关，而不是 decode，因为两种 runtime 的 decode throughput 接近。
- QNN Burst 的 TTFT 最低，但 Burst 的能耗和 EDP 并不好，因此最低 TTFT 不等于最佳综合配置。
- T-MAC T8 的 TTFT 比 T4/T6 更低，但同时 decode 退化，表明 prefill 和 decode 对线程数的响应不同。

### 1.4 J/token

![Fig01d：J/token](assets/paper_v2_20260902/fig01d_energy.png)

主要结论：

- OnePlus 上 QNN Balanced-512 为 0.0265 J/token，明显低于 Burst 的 0.0711 J/token；LowPower 虽然更慢，却没有比 Balanced 更省每个 token 的能量。
- Redmi 上 QNN Balanced-512 同样是三种 mode 中最低，为 0.0291 J/token。
- T-MAC 的能效随设备和 thread 呈非单调变化：OnePlus T8 最差，而 Redmi T8 的本批次 J/token 最低。

综合 Fig01，Balanced/T6 比 Burst/T4 或 LowPower/T8 更接近稳定的性能折中点；单独追求某一阶段速度容易牺牲其他阶段或能耗。

## 2. Fig02：不同任务的 latency composition

Fig02 左图只看 inference 内部的 prefill/decode 占比；右图将模型加载加入，表示每题重新启动进程时的 cold-start latency。`N/A` 表示当前没有完整正式结果，不代表测得性能为零。

### 2.1 GSM8K

![Fig02a：GSM8K latency](assets/paper_v2_20260902/fig02a_gsm8k_latency.png)

- QNN 的 inference 时间主要由 decode 构成，prefill 仅占约 2.3%–2.6%；cold-start 总时长约为 OnePlus 13.9 秒、Redmi 12.2 秒。
- T-MAC 的 prefill 占 inference 的约 35.8%–36.9%，再叠加 10.6–13.0 秒模型加载，cold-start 总时长达到约 27.8–30.3 秒。
- 因而 GSM8K 的跨 runtime 差距不能只用 decode tokens/s 解释；模型加载和 prefill 是 T-MAC 的主要额外成本。

### 2.2 MATH-500

![Fig02b：MATH-500 latency](assets/paper_v2_20260902/fig02b_math_500_latency.png)

- MATH-500 输出较长，两种 runtime 的 inference 都由 decode 主导。
- OnePlus QNN 的平均 cold-start latency 约为 30.3 秒；T-MAC 约为 49.2 秒。
- T-MAC 的 decode 时长与 QNN 接近，但额外的模型加载和约 5.2 秒 prefill 将总等待时间显著拉长。
- 任务输出越长，固定的模型加载/prefill 成本占比越低；但它们仍会显著影响用户看到首 token 的时间。

### 2.3 HammerBench single-turn

![Fig02c：HammerBench latency](assets/paper_v2_20260902/fig02c_hammer_st_latency.png)

- QNN OnePlus/Redmi 的 cold-start latency 分别约为 3.72/3.27 秒，prefill 占 inference 的约 24%/31%。
- Redmi T-MAC 的 prefill 占 inference 的 92.4%，cold-start latency 约为 33.2 秒，瓶颈明显是长工具描述 prompt 的 prefill，而非短函数调用输出的 decode。
- OnePlus T-MAC 当前没有完整 HammerBench single-turn 结果，因此保留空位。
- 该结果说明任务结构会改变主要瓶颈：HammerBench 是“长输入、短输出”，比 GSM8K/MATH-500 更依赖 prefill 能力。

### 2.4 BFCL

![Fig02d：BFCL latency](assets/paper_v2_20260902/fig02d_bfcl_latency.png)

- 当前只有 OnePlus QNN 完整结果，平均 cold-start latency 约为 4.97 秒。
- BFCL inference 主要由 decode 构成，prefill 约占 8.7%。
- T-MAC 结果尚未完成，因此不能从当前图中进行 BFCL 跨 runtime 比较。

综合 Fig02，不同任务不能用同一个“总体 tokens/s”概括：长输入任务更敏感于 prefill，长输出任务更敏感于 decode，而每题重启的 runtime 还必须单独报告 model load。

## 3. Fig03：代理指标和任务指标排名

![Fig03：指标和排名表](assets/paper_v2_20260902/fig03_proxy_task_ranks.png)

每个格子给出指标值及同设备十种配置中的排名。Decode 越高越好，其余能耗/EDP 指标越低越好。

主要结论：

- OnePlus 上 T-MAC T6-256 的 decode 排名第一，但 EDP 仅第五、10/20/30 秒的任务能耗分别为第五/第四/第五，说明最快 decode 不等于最低任务成本。
- OnePlus QNN Balanced-no-budget 的 J/token 和 20/30 秒任务能耗排名第一，但 EDP 不如 Balanced-256；不同能耗代理会选出不同配置。
- Redmi 上 QNN Balanced-256 同时获得 decode、EDP 和各 deadline 任务能耗第一；Balanced-512 则获得 J/token 第一。
- Burst 的低 TTFT/preﬁll 优势没有转化为 decode 或 deadline 能效优势。
- 该表直接支持 Finding 1：tokens/s、J/token、EDP 与 energy per correct task under deadline 并不总给出相同排名，因此不能只报告一个代理指标。

## 4. Fig04：deadline-response curve

![Fig04：Deadline-response curves](assets/paper_v2_20260902/fig04_operating_boundary.png)

横轴是 deadline，纵轴是能在该 deadline 内完成且答案正确的题目比例。曲线最终高度同时受到最终正确率上限和推理时延影响。

主要结论：

- 5 秒 deadline 下，只有 QNN 能产生有意义的及时正确结果；T-MAC 几乎为零。
- 10 秒时 QNN Balanced-256 最好：OnePlus 31.7%，Redmi 40.2%。
- 20 秒时 T-MAC T6-256 开始反超，OnePlus/Redmi 分别达到 64.0%/62.3%。较短的 256 budget 在此时优于 T6-512，因为它更容易在 deadline 前完成。
- 30 秒时 T-MAC 的最终质量优势充分显现，T6-512 达到 OnePlus 75.4%、Redmi 75.6%；OnePlus T6-no-budget 略高至 76.0%，但其能耗和时延并不更优。
- QNN LowPower 在短 deadline 下最差，随着 deadline 放宽才逐步接近其最终正确率上限。
- 该图支持 Finding 2：所谓“最佳配置”随 deadline 和 budget 改变，不存在对所有服务目标都最优的固定点。

## 5. Fig05：MATH-500 difficulty

![Fig05：MATH-500 difficulty](assets/paper_v2_20260902/fig05_math500_difficulty.png)

左图给出 Level 1–5 的正确率，右图给出每个正确答案的平均能耗。

主要结论：

- 三种部署的正确率均随难度单调下降，任务难度是比单一总体准确率更有解释力的维度。
- QNN Qwen 从 Level 1 的 74.4% 降至 Level 5 的 10.4%；T-MAC Qwen 从 76.7% 降至 17.2%；QNN Llama 从 76.7% 降至 12.0%。
- T-MAC Qwen 在 Level 2–5 的正确率最高，但其每个正确答案能耗也高于 QNN Qwen。
- 难度提升会同时增加单题生成成本并降低成功概率，因此 energy per correct answer 呈快速上升：Level 5 分别约为 QNN Qwen 118.8 J、T-MAC Qwen 154.1 J、QNN Llama 287.7 J。
- QNN Llama 并没有因模型更大而在当前 MATH-500 部署上形成稳定质量优势，且困难题上的任务能耗最高。
- 该图进一步支持 Finding 2：task-sufficient point 会随 difficulty 移动，不能只用全数据集均值决定部署配置。

## 6. Fig06：decode 提速能否转化为任务收益

![Fig06：Speedup conversion](assets/paper_v2_20260902/fig06_speedup_conversion.png)

该图在同一设备、同一 runtime 内进行归一化：QNN 以 Balanced-512 为 1，T-MAC 以 T6-512 为 1。横轴是相对 decode speed，三个纵轴依次是 20 秒及时正确率、tasks/hour 和 timely tasks/J 的相对值。黑色星号 `(1,1)` 是基准，虚线 `y=x` 表示 decode 变化被等比例转化为任务收益。

读图规则：

- 点落在虚线上：decode 改善大致等比例转化为任务收益；
- 点落在虚线下：任务收益小于 decode 改善；
- 点落在虚线上方：正确率、输出长度或功耗等因素带来了超出 decode 变化的收益。

主要结论：

- tasks/hour 与 decode speed 最接近线性关系，但并非完全重合，因为 prefill 和输出长度同样参与 E2E latency。
- OnePlus T-MAC T6-256 的 decode、及时正确率、tasks/hour 和 timely tasks/J 相对 T6-512 分别约为 1.06、1.07、1.10 和 1.18，说明 20 秒 deadline 下缩短 budget 同时改善吞吐和任务能效。
- Redmi QNN Burst 的相对 decode 仅为 0.74，但及时正确率仍为 0.90；它保留了较多短题的及时成功，却在 tasks/J 上仅为 0.45，表明低延迟局部优势伴随很高能耗成本。
- Redmi T-MAC T8 的相对 decode 为 0.87，而 timely tasks/J 达到 1.70。该点的优势来自本批次较低功耗，而不是速度；考虑到 OnePlus T8 呈相反趋势，应复测后再作强结论。
- 该图支持 Finding 3：throughput improvement 不会自动等比例变成 timely success 或 tasks/J，最终转化取决于任务长度、准确率、deadline 和功耗。

## 7. Fig07：质量—能耗 Pareto 与 deadline 能耗

![Fig07：Energy-quality trade-offs](assets/paper_v2_20260902/fig07_energy_quality_pareto.png)

上半部分比较不同完整 deployment 的 GSM8K 正确率与 J/token；下半部分比较 QNN Balanced 和 T-MAC T6 在不同 deadline 下的 energy per timely correct task。

主要结论：

- OnePlus 上 QNN BitNet 同时比 QNN Qwen 正确率高、J/token 低，因此在这两个指标上支配 QNN Qwen。
- QNN Llama 和 T-MAC Qwen继续提高正确率，但需要更高 J/token；其中 T-MAC Qwen达到 77.86% 的最高正确率，也具有最高或接近最高的单位 token 能耗。
- Redmi 上同样由 QNN BitNet 和 T-MAC Qwen形成低能耗/高质量两端的 Pareto 选择。
- 下半部分显示，虽然较宽 deadline 下 T-MAC 能完成更多正确题，QNN 的 energy per timely correct task 始终更低。
- 因此“正确题最多”和“每个及时正确题能耗最低”是两个不同目标：容量/质量优先可以选择 T-MAC，交互延迟和任务能效优先更倾向 QNN。
- 跨模型、跨 runtime 的 Pareto 图属于 deployment-level 比较。由于模型、量化和 prompt 不完全一致，它适合回答“实际部署选哪个”，不适合把差异归因于某一个底层算子或 runtime。

## 8. Fig08：温度对齐后的持续运行轨迹

Fig08 使用每组运行的最长连续片段，并选择接近 39°C、之后仍有完整 90 分钟数据的位置作为起点。实际起始温度范围为 37.02–40.59°C。数据按 3 分钟聚合，每条曲线包含 30 个统计点；温度为左轴淡虚线，性能为右轴实线。

### 8.1 Temperature × decode

![Fig08a：Temperature and decode](assets/paper_v2_20260902/fig08a_temperature_decode.png)

- QNN Balanced 的 decode 相对稳定：OnePlus 前后三个 bin 均约 15 token/s；Redmi 从约 17.4 降至 16.9 token/s。
- QNN LowPower 始终处于较低温度和较低 decode 区间，体现的是 mode 设定差异，而不能直接解释为温度导致低速度。
- T-MAC 三种 thread 的 decode 波动更明显。以前后三个 bin 均值比较，OnePlus T4/T6/T8 分别约从 14.8/16.7/12.2 降至 12.2/14.5/9.8 token/s；Redmi 分别约从 18.7/18.9/15.5 降至 14.1/15.2/13.6 token/s。
- 这些变化与升温同时出现，但样例长度、题目难度、CPU 调度和后台状态也随时间变化，因此只能称为温度相关的 sustained degradation，不能仅凭此图断言发生硬件 thermal throttling。

### 8.2 Temperature × prefill

![Fig08b：Temperature and prefill](assets/paper_v2_20260902/fig08b_temperature_prefill.png)

- QNN prefill 整体显著高于 T-MAC，并且在 OnePlus 上相对稳定。
- Redmi QNN Balanced 的前三/后三个 bin 从约 664.5 降至 558.2 token/s；Redmi Burst 从约 750.0 降至 665.1 token/s，提示长时间运行后 prefill 可能受系统状态影响。
- T-MAC 的 prefill 在两部手机和三种 thread 下普遍出现前高后低。例如 Redmi T8 从约 37.8 降至 25.8 token/s。
- T-MAC prefill 下降比 decode 更明显，这与 CPU prefill 路径可能对持续频率和调度更敏感的假设一致，但仍需要 CPU frequency、utilization 和 thermal throttling 状态才能确认机制。

### 8.3 Temperature × timely tasks/hour

![Fig08c：Temperature and timely tasks](assets/paper_v2_20260902/fig08c_temperature_timely_tasks.png)

- timely correct tasks/hour 同时受速度、题目输出长度和每个时间 bin 内题目正确性影响，因此比 decode/prefill 更嘈杂。
- QNN Balanced 通常维持较高的短 deadline 任务产出；LowPower 较低。
- T-MAC 的部分 bin 虽然 decode 尚可，但受 TTFT/prefill 和 20 秒 deadline 约束，timely task rate 仍明显波动。
- 该图适合作为 Finding 6 的探索性证据：短时平均性能不能完整代表持续任务产出。但它不应单独用于证明热降频，正式因果实验仍需控制初温、运行顺序和后台负载，并采集 CPU/GPU/DSP 频率及 Android thermal 状态。

## 9. 图表与研究 Finding 的对应关系

| 研究方向 | 主要图表 | 当前能够支持的结论 |
| --- | --- | --- |
| Finding 1：代理指标与任务能效 | Fig01、Fig03、Fig07 | tokens/s、J/token、EDP 和 deadline 下每个正确任务能耗会给出不同排名；需要同时报告系统代理和任务指标。 |
| Finding 2：task-sufficient point | Fig04、Fig05、Fig07 | 最佳配置随 deadline、budget 和 difficulty 改变；短 deadline 偏向 QNN/256，宽 deadline 和质量目标可能偏向 T-MAC/512。 |
| Finding 3：速度到任务收益的转化 | Fig06 | decode 提速与 tasks/hour 较接近，但 timely success 和 tasks/J 还受到准确率、输出长度和功耗影响，不是固定比例。 |
| Finding 5：比较对齐程度 | Fig02、Fig07 | 同模型名不代表同一部署产物；跨 runtime 只能做 deployment-level Pareto，不应把差异全部归因于 runtime。 |
| Finding 6：短测与持续运行 | Fig08 | 多组配置在 90 分钟窗口内出现升温和性能变化，尤其 T-MAC prefill/decode；目前是相关性证据，仍需严格控温复测。 |
