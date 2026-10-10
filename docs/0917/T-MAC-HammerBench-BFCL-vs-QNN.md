# Qwen2.5-3B：T-MAC HammerBench / BFCL 结果及 QNN 对比

更新时间：2026-09-07

## 1. 结论摘要

1. **HammerBench 上 QNN 的系统优势非常明显。**在相同设备、相同数据集和相同官方 prompt/scorer 口径下，QNN 的 prefill 吞吐比 T-MAC 高约 38–51 倍，inference-only latency 低约 8–13 倍，冷启动完成时间低约 7.7–10.2 倍。
2. **HammerBench single-turn 的总体质量也是 QNN 更高。**QNN 为 269/654（41.13%），T-MAC 为 240/654（36.70%），相差 29 题、4.43 个百分点。T-MAC 在相关任务上的 E2E 略高（27.52% 对 25.38%），但无关任务拒答明显更差（45.87% 对 56.88%），最终拖低总体分数。
3. **HammerBench multi-turn 同样由 QNN 领先。**QNN 完整结果为 snapshot accuracy 31.81%、PR 26.91%、SR 15.69%；Redmi T-MAC 当前为 24.58%、18.54%、12.09%。但 T-MAC 多轮结果尚不完整，因此跨 runtime 的多轮结论仍应标记为 provisional。
4. **BFCL simple_python 的质量方向相反：T-MAC 更高。**T-MAC 为 368/400（92.0%），QNN Burst 为 338/400（84.5%），T-MAC 高 7.5 个百分点；代价是平均 BFCL call latency 为 23.64 s，对 QNN Burst 的 5.48 s，慢 4.31 倍。
5. **T-MAC 的主要系统瓶颈是长 prompt prefill，而不是 decode。**HammerBench single-turn 中 T-MAC TTFT 约 20–21 s，而 decode 仅约 1.7–1.8 s；BFCL 中 T-MAC prefill 约 10.1 s、decode 约 2.6 s。QNN 把同类 prefill 压缩到约 0.26–0.54 s。
6. **能效上 QNN 全面领先。**相对同 prompt 的 QNN Burst，T-MAC 的 HammerBench J/token 高 4.5–10.7 倍、EDP 高 27–105 倍；BFCL J/token 高 2.53 倍、EDP 高 7.06 倍。若与 QNN Balanced 比，BFCL 的能效差距更大。

## 2. 范围和可比性

### 2.1 设备与部署

| 项目 | T-MAC | QNN |
|---|---|---|
| 模型族 | Qwen2.5-3B-Instruct | Qwen2.5-3B-Instruct |
| 量化/产物 | GPTQ INT4 W4A16 G128 `INT_N` GGUF | QNN 2.37 hybrid W4A16 block16 |
| 执行单元 | CPU，6 threads | Qualcomm HTP/NPU |
| 最大序列长度 | 1024 | 1024 |
| HammerBench prompt | `hammerbench_official_appendix_d_qwen_v2` | `hammerbench_official_appendix_d_qwen_v2` |
| HammerBench scorer | `hammerbench_official_metrics_v3` | `hammerbench_official_metrics_v3` |
| 设备 | OnePlus PLQ110、Redmi 2510DRK44C，均为 SM8750 / Android 16 | 同左 |

两条路径虽然都标记为 Qwen2.5-3B W4A16，但量化粒度、权重产物、runtime 和 prompt 执行实现并不相同。因此准确率差异应解释为**完整 deployment path 的差异**，不能只归因于“CPU 对 NPU”或 T-MAC/QNN runtime 本身。

HammerBench 的主对比使用 QNN `burst_official_prompt_v2`，因为它与 T-MAC 使用相同官方 prompt 和 v3 scorer。旧 QNN `balanced` HammerBench 结果使用过时 prompt/评分口径，single/multi 的表观 E2E 仅 5.66%/7.23%，不应拿来判断 Balanced mode 的模型质量，也不纳入主对比。

### 2.2 完成状态

| Benchmark | 设备 | T-MAC | QNN 主对比 |
|---|---|---:|---:|
| HammerBench single-turn | OnePlus | 654/654 | 654/654 |
| HammerBench single-turn | Redmi | 654/654 | 654/654 |
| HammerBench multi-turn | OnePlus | 744 successful / 830；仅覆盖 278/306 conversations | 830/830 |
| HammerBench multi-turn | Redmi | 828 successful / 830；827 条有能耗 | 830/830 |
| BFCL v4 simple_python | OnePlus | 400/400 | Burst 400/400；Balanced 性能表 395/400 |

T-MAC OnePlus multi-turn 缺少 85 个数据集快照且另有 1 个 runtime failure；Redmi 缺 2 个成功推理。因此二者的 deadline 能耗文件均标记 `valid_for_full_dataset_reporting=False`。

## 3. HammerBench single-turn

### 3.1 质量

两台设备在同一 deployment 内生成相同答案，所以准确率与设备无关。

| 指标 | T-MAC T6 | QNN Burst | 差值（T-MAC − QNN） |
|---|---:|---:|---:|
| Overall benchmark-aware E2E | 240/654（36.70%） | **269/654（41.13%）** | -4.43 pp |
| 相关任务函数名准确率 | **83.79%** | 79.82% | +3.98 pp |
| 相关任务参数准确率 | **29.97%** | 27.52% | +2.45 pp |
| 相关任务 E2E | **90/327（27.52%）** | 83/327（25.38%） | +2.14 pp |
| 无关任务正确拒答 | 150/327（45.87%） | **186/327（56.88%）** | -11.01 pp |

T-MAC 并不是全面降低了工具调用能力：它选函数和完成相关调用略好于 QNN。总体分数落后主要来自无关任务拒答不足，少正确拒答 36 题；相关任务多答对 7 题，合计净少 29 题。

### 3.2 性能与能耗

下表的 TTFT/inference 不含模型加载；cold completion 为 `model_load + inference`。J/token 和 EDP 使用 inference-only 能耗窗口。

| 设备 | Backend | TTFT (s) | Inference (s) | Cold completion (s) | Prefill tok/s | Decode tok/s | J/token | EDP (J·s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OnePlus | T-MAC T6 | 20.969 | 22.757 | 37.856 | 24.40 | 16.22 | 0.42478 | 272.06 |
| OnePlus | QNN Burst | **0.430** | **1.766** | **3.724** | **1233.60** | **20.05** | **0.06950** | **4.96** |
| Redmi | T-MAC T6 | 20.465 | 22.140 | 33.196 | 25.23 | 17.26 | 0.50777 | 318.87 |
| Redmi | QNN Burst | **0.540** | **1.749** | **3.268** | **949.51** | **20.87** | **0.04743** | **3.04** |

关键倍率：

- OnePlus：T-MAC TTFT / inference / cold completion 分别是 QNN 的 48.7 / 12.9 / 10.2 倍；J/token 和 EDP 是 6.11 / 54.8 倍。
- Redmi：对应倍率为 37.9 / 12.7 / 10.2 倍；J/token 和 EDP 是 10.71 / 105.0 倍。
- Decode 差距相对小：T-MAC 比 QNN 慢约 17–19%。真正拉开交互延迟的是 prefill 和每题重新加载模型。
- `peak_rss_mb` 不作跨 runtime 主比较：T-MAC 进程约 2.8 GiB，而 QNN host 进程仅约 34–71 MiB，但后者没有完整计入共享内存、RPC/HTP context 和 DSP 侧占用。

### 3.3 Deadline

| 设备 | Backend | 5 s | 10 s | 20 s | 30 s | 最终正确 |
|---|---|---:|---:|---:|---:|---:|
| OnePlus | T-MAC T6 | 0 | 3 | 108 | 225 | 240 |
| OnePlus | QNN Burst | **267** | **269** | **269** | **269** | **269** |
| Redmi | T-MAC T6 | 0 | 5 | 121 | 229 | 240 |
| Redmi | QNN Burst | **269** | **269** | **269** | **269** | **269** |

QNN 基本在 5 秒内达到最终质量；T-MAC 因平均 TTFT 已超过 20 秒，在 5 秒内没有任何正确任务，30 秒仍有 11–15 个最终正确答案尚未完成。

按整批总能量除以 deadline 内正确任务数：

| 设备 | Backend | 20 s J/timely-correct | 30 s J/timely-correct |
|---|---|---:|---:|
| OnePlus | T-MAC T6 | 68.09 | 32.68 |
| OnePlus | QNN Burst | **4.12** | **4.12** |
| Redmi | T-MAC T6 | 72.64 | 38.38 |
| Redmi | QNN Burst | **2.81** | **2.81** |

## 4. HammerBench multi-turn

### 4.1 质量

| 设备 / Backend | 统计覆盖 | Snapshot E2E | Function acc | Args acc | PR | SR |
|---|---:|---:|---:|---:|---:|---:|
| OnePlus T-MAC T6 | 745 rows / 278 conv，未完成 | 25.64% | 84.43% | 26.71% | 19.03% | 12.59% |
| OnePlus QNN Burst | 830 / 306 | **31.81%** | **92.53%** | **32.29%** | **26.91%** | **15.69%** |
| Redmi T-MAC T6 | 830 rows，828 success / 306 conv | 24.58% | 84.22% | 25.54% | 18.54% | 12.09% |
| Redmi QNN Burst | 830 / 306 | **31.81%** | **92.53%** | **32.29%** | **26.91%** | **15.69%** |

OnePlus T-MAC 只覆盖 278 个 conversations，不能将 25.64% 与完整 QNN 结果当作严格同分母比较。Redmi 接近完整，当前 QNN 的 snapshot E2E 高 7.23 pp、PR 高 8.37 pp、SR 高 3.60 pp；缺失的 2 条不足以逆转方向，但正式归档前仍应补齐。

### 4.2 性能与能耗

| 设备 | Backend | TTFT (s) | Inference (s) | Cold completion (s) | Prefill tok/s | Decode tok/s | J/token | EDP (J·s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OnePlus | T-MAC T6 | 16.669 | 19.008 | 32.695 | 25.75 | 16.37 | 0.27873 | 207.37 |
| OnePlus | QNN Burst | **0.371** | **2.285** | **4.102** | **1211.32** | **21.10** | **0.06243** | **7.58** |
| Redmi | T-MAC T6 | 17.882 | 20.304 | 29.774 | 24.60 | 16.59 | 0.30717 | 253.67 |
| Redmi | QNN Burst | **0.463** | **2.288** | **3.880** | **926.65** | **21.15** | **0.03904** | **4.34** |

T-MAC 相对 QNN 的 inference latency 高 8.3–8.9 倍，cold completion 高 7.7–8.0 倍，J/token 高 4.46–7.87 倍，EDP 高 27.3–58.4 倍。由于 T-MAC 数据不完整，这些系统指标只描述当前成功样本，不作为完整数据集排名。

## 5. BFCL v4 simple_python

### 5.1 先明确评分口径

当前实验只跑了 BFCL v4 `simple_python` 的 400 条样本，并没有跑完整 BFCL 套件。官方汇总表仍按完整套件权重计算，因此显示的 `Overall Acc` 只有 T-MAC 0.77%、QNN 0.70%；这个值不能代表已跑子集的质量。

本报告使用 `Python Simple AST`：

- T-MAC：368/400，92.0%。
- QNN Burst：338/400，84.5%。
- QNN Balanced：336/400，84.0%。

### 5.2 OnePlus 结果

| Backend | Python Simple AST | BFCL call mean / P95 (s) | TTFT (s) | Inference (s) | Prefill tok/s | Decode tok/s | J/token | EDP (J·s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T-MAC T6 | **368/400（92.0%）** | 23.64 / 33.25 | 10.072 | 12.693 | 26.09 | 15.65 | 0.17609 | 91.55 |
| QNN Burst | 338/400（84.5%） | 5.48 / 8.07 | **0.257** | **2.941** | **1149.57** | **16.78** | 0.06969 | 12.97 |
| QNN Balanced | 336/400（84.0%） | **5.30 / 7.05** | 0.319 | 2.992 | 817.91 | 14.95 | **0.03086** | **6.10** |

QNN Balanced 的准确率来自 400 条官方 score；性能/能耗 `summary_metrics.csv` 只有 395 条，因此其性能均值只基于这 395 条记录。T-MAC 和 QNN Burst 的性能表均为 400/400。

相对 QNN Burst：

- T-MAC 准确率高 7.5 pp（多答对 30 题）。
- BFCL call latency 慢 4.31 倍，TTFT 慢 39.2 倍，inference 慢 4.32 倍。
- Decode 只慢 6.7%，再次说明差距主要来自 prefill/model load。
- J/token 高 2.53 倍，EDP 高 7.06 倍。

相对 QNN Balanced，T-MAC 的 decode 略快 4.7%，但 call latency 慢 4.46 倍、J/token 高 5.71 倍、EDP 高 15.0 倍。QNN Balanced 是现有 BFCL 结果中的能效点；Burst 只缩短了 TTFT，平均 call latency 反而略增，质量仅提高 0.5 pp。

### 5.3 错误结构

T-MAC 的 32 个错误主要为：字符串值不匹配 11、遗漏应推断的 optional/default 参数 6、其他数值错误 5、nested type 3、list/tuple 值错误 3、wrong function count 2、simple type 2。

QNN Burst 的 62 个错误中，`ast_decoder:decoder_failed` 有 16 个、字符串值错误 16 个、wrong function count 9 个，另有 optional/default 参数遗漏 5 个。T-MAC 的 BFCL 优势不仅来自参数值，亦来自更稳定的可解析工具调用格式：QNN Burst 有 16 个 AST 解码失败，而 T-MAC 本批次没有该类错误。

## 6. 综合判断

### 6.1 如果目标是端侧交互体验

优先 QNN。HammerBench/BFCL 都是长工具 schema 输入、短函数调用输出，任务天然偏重 prefill；QNN 的 HTP 路径在这个结构上形成数量级优势。对 HammerBench，QNN 还同时具有更高总体质量、更低 deadline latency 和更低任务能耗。

### 6.2 如果目标是 BFCL simple_python 最终准确率

当前 T-MAC deployment 最好，为 92.0%，比 QNN 高 7.5–8.0 pp。但它需要约 23.6 s/题，且能耗明显更高。这个结果更适合描述为“高质量、低时效”的 operating point，而不是全面胜出。

### 6.3 下一步优先级

1. 补齐 OnePlus T-MAC multi-turn 的 85 个缺失快照和 1 个失败样本；补齐 Redmi 的 2 个失败样本及 1 条缺失能耗。
2. 对 QNN BFCL 的 16 个 AST decoder failures 做输出清洗/终止符修复后重评；这是最可能缩小 7.5 pp 质量差距的软件点。
3. 给 T-MAC 加持久进程或模型常驻模式。当前每题约 9.5–15.1 s 的模型加载显著抬高 cold latency，虽然不会改变 inference-only J/token，仍严重影响真实调用体验。
4. 在相同 prompt 下补跑 QNN Balanced HammerBench；现有 balanced 数据是旧 prompt 结果，不能用于公平质量或 mode 对比。
5. BFCL 扩展到 multiple、parallel、irrelevance 和 multi-turn 后再报告官方 `Overall Acc`，当前只能报告 simple_python 子集。

## 7. 数据来源

HammerBench：

```text
results/T-MAC/{oneplus,redmi}/hammerbench-{single,multi}/qwen2.5-3B-tmac-w4a16/threads_6/
results/QNN/{oneplus,redmi}/hammerbench-{single,multi}/qwen2.5-3B-w4a16/burst_official_prompt_v2/
```

BFCL：

```text
results/T-MAC/oneplus/bfcl/qwen2.5-3B-tmac-w4a16/threads_6/output_budget_no_budget/
results/QNN/oneplus/bfcl/qwen2.5-3B-w4a16/{balanced,burst}/
```

准确率取自 `evaluation_summary.csv`、`conversation_summary.csv` 和 BFCL `score/data_non_live.csv`；性能、能耗取自成功样本的 `summary_metrics.csv`；deadline 取自 `deadline_summary.csv`。J/token 使用 `sum(energy_j) / sum(generated_tokens)`，不是逐题 J/token 的简单平均。
