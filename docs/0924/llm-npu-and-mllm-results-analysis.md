# llm.npu 与原生 mllm v2 现有结果分析

更新日期：2026-09-14

## 1. 结论摘要

截至本报告统计时点，`llm.npu` 已形成较完整的实验矩阵：两台 SM8750 手机上均完成 GSM8K 的三种 HTP mode（512 output budget），并完成 balanced 下的 256、512、no-budget 对照；此外还有 OnePlus MATH-500 balanced/512，以及 Redmi HammerBench single-turn 和 multi-turn burst 全集结果。

原生 `mllm v2` 当前有 4 组完整 GSM8K/512 结果：OnePlus balanced、OnePlus burst、Redmi burst、Redmi low-power-saver。Redmi balanced 只有 897/1319 条，OnePlus low-power-saver 只有 213/1319 条，均属于中断的 partial run，不能作为完整数据集结果发布。mllm v2 尚无完整 MATH-500、HammerBench 或 BFCL 结果。

现有数据支持以下主要判断：

1. `llm.npu` 在 GSM8K/512 上稳定得到 **915/1319，69.37%**；原生 mllm v2 稳定得到 **897/1319，68.01%**。两套 Context 的 SHA-256 相同，tokenizer 相同，1319 份格式化 prompt 也逐字节相同，因此 1.36 个百分点的差异不是数据集、prompt 或模型 Context 不一致造成的，仍需定位 runner/backend 执行栈差异。
2. 对这类较短 GSM8K 输出，**balanced 是当前最稳妥的默认模式**。OnePlus 的 burst 可将平均 inference latency 从 7.33 s 降到 6.58 s，但平均能耗从 4.28 J 增至 6.96 J；Redmi 的 llm.npu burst 反而比 balanced 更慢且更耗电。
3. **low-power-saver 并不节能**：llm.npu 上平均 inference latency 约为 balanced 的 4.2 倍，OnePlus 单题平均能耗增加 74%，Redmi 增加 140%。它只是降低瞬时性能/功率，长得多的执行时间使总能耗上升。
4. GSM8K 从 256 提升到 512 budget，使正确题数从 890 增至 915（+25，+1.90 个百分点）；512 与 no-budget 的正确率完全相同，且仅分别有 3 条和 1 条长度终止。**512 已足够作为 GSM8K 主实验预算**。
5. llm.npu 的 OnePlus MATH-500/512 正确率为 **149/500，29.80%**，但有 217 条显式长度终止（43.4%）。MATH-500 明显比 GSM8K 更受 1024-token Context 和输出预算约束。
6. llm.npu Redmi burst 的 HammerBench 已完整跑完：single-turn end-to-end accuracy 为 **31.65%**，multi-turn turn-level end-to-end accuracy 为 **26.87%**，multi-turn conversation success rate 仅 **9.48%**。函数名选择明显好于参数生成，参数是主要瓶颈。

## 2. 统计口径与可比性

本报告直接读取各 run 的 `manifest.json`、`summary.csv`、`deadline_summary.csv`，以及 HammerBench 的 `evaluation_summary.csv` 和 `conversation_summary.csv`。完整性按数据集实际规模判断：GSM8K 为 1319 条，MATH-500 为 500 条；不能因为目录中已有 summary 文件就认定 run 完整。

表中的延迟、吞吐和能耗均为成功样本的算术平均值：

- TTFT 采用 runner 报告的 prefill time；
- inference latency 为 prefill + decode，不含每条样本重新加载模型的时间；
- model load 是每次独立启动 runner 时，从进程启动到推理窗口开始的反推值；
- energy 是推理窗口内由电池遥测积分得到的净电池能量；外部供电样本不计入有效能耗平均；
- J/token 是逐样本 `energy_j / generated_tokens` 后再求平均，并非总能量除以总 token；
- 正确率分母始终是完整数据集规模，而不是仅成功或可解析样本数。

两套实验的共同配置为 Qwen2.5-3B、LPBQ W4A16 G32、W8A8 KV cache、SM8750 HTP V79、固定 `MAX_SEQ_LEN=1024`、32-token prefill 图和 1-token decode 图。所有列入主表的成功样本均记录了请求的 HTP mode、加载 `model.0.s32`/`model.0.s1`，并通过 HTP execution verification。

这里的命名需要特别澄清：`llm.npu` 是项目中早期 Qwen mllm-QNN AOT bundle 及其 PACE harness；`mllm` 是后来从官方 mllm v2 commit `033d5cd4805383ea9a3d68fe3c162ee0e689887a` 构建的 native bundle 及独立 harness。二者底层都走 mllm/QNN AOT/HTP，不应把 `llm.npu` 理解成完全不同的 NPU API。

## 3. 结果完整性

| Runtime | 设备 | 任务/配置 | 实际进度 | 状态 |
|---|---|---|---:|---|
| llm.npu | OnePlus | GSM8K balanced 256/512/no-budget | 各 1319/1319 | 完整 |
| llm.npu | OnePlus | GSM8K burst 512、low-power-saver 512 | 各 1319/1319 | 完整 |
| llm.npu | Redmi | GSM8K balanced 256/512/no-budget | 各 1319/1319 | 完整 |
| llm.npu | Redmi | GSM8K burst 512、low-power-saver 512 | 各 1319/1319 | 完整 |
| llm.npu | OnePlus | MATH-500 balanced 512 | 500/500 | 完整 |
| llm.npu | Redmi | HammerBench single-turn burst | 654/654 | 完整 |
| llm.npu | Redmi | HammerBench multi-turn burst | 830/830 turns，306 conversations | 完整 |
| mllm v2 | OnePlus | GSM8K balanced 512、burst 512 | 各 1319/1319 | 完整 |
| mllm v2 | Redmi | GSM8K burst 512、low-power-saver 512 | 各 1319/1319 | 完整 |
| mllm v2 | Redmi | GSM8K balanced 512 | 897/1319，其中 894 成功 | **部分结果** |
| mllm v2 | OnePlus | GSM8K low-power-saver 512 | 213/1319，其中 212 成功 | **部分结果** |
| mllm v2 | 两设备 | MATH-500、HammerBench、BFCL | 0 个完整 run | 缺失 |
| llm.npu | 两设备 | BFCL | 0 个结果 run | 缺失 |

Redmi mllm balanced 的中断记录包括一次 adb pull 超时、一次 `dumpsys battery` 超时和一次 device offline；OnePlus mllm low-power-saver 的已记录失败为 device offline。这些是连接/采集链路问题，不是模型精度失败，但 run 仍必须续跑到 1319 条后才能进入主表。

## 4. llm.npu：GSM8K 完整结果

### 4.1 Output budget = 512

| 设备 | HTP mode | 成功 | 正确率 | Model load (ms) | TTFT (ms) | Inference (ms) | Prefill (tok/s) | Decode (tok/s) | Energy (J) | J/token | 有效能耗样本 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OnePlus | balanced | 1319/1319 | 69.37% | 4541 | 337 | 7334 | 462.4 | 22.82 | 4.281 | 0.02682 | 1319/1319 |
| OnePlus | burst | 1319/1319 | 69.37% | 7508 | 251 | 6579 | 640.6 | 25.47 | 6.963 | 0.04335 | 1319/1319 |
| OnePlus | low-power-saver | 1319/1319 | 69.37% | 4520 | 1434 | 30509 | 109.7 | 5.53 | 7.459 | 0.04702 | 1318/1319 |
| Redmi | balanced | 1319/1319 | 69.37% | 2422 | 344 | 7151 | 453.7 | 23.49 | 3.893 | 0.02446 | 1319/1319 |
| Redmi | burst | 1319/1319 | 69.37% | 5212 | 315 | 8319 | 514.9 | 20.59 | 5.481 | 0.03428 | 1318/1319 |
| Redmi | low-power-saver | 1319/1319 | 69.37% | 3147 | 1302 | 30555 | 120.7 | 5.48 | 9.343 | 0.05894 | 1319/1319 |

HTP mode 不改变输出：三种 mode、两台设备均为同一组 915 条正确答案。这说明在当前 runner 与 Context 下，HTP mode 主要改变调度、延迟和能耗，而未观察到数值结果漂移。

OnePlus 上 burst 相对 balanced：TTFT 降低 25.5%，inference latency 降低 10.3%，但能耗增加 62.7%，J/token 增加 61.6%。如果目标是最低延迟，可以选 burst；如果同时考虑能耗，balanced 更合理。

Redmi 上 burst 相对 balanced：TTFT 降低 8.4%，但 decode 从 23.49 tok/s 降至 20.59 tok/s，平均 inference latency 增加 16.3%，能耗增加 40.8%。这不是“burst 必然更快”的正常单调关系，可能受运行时温控、长时间全集运行顺序或 ROM 电源策略影响。现有单次全集 run 足以说明当前实测现象，但不足以把差异归因到 mode 本身；正式论文结论前应做随机化顺序和重复实验。

low-power-saver 在两机上都把 decode 降到约 5.5 tok/s。相对 balanced，OnePlus inference latency 增加 316%，能耗增加 74%；Redmi inference latency 增加 327%，能耗增加 140%。因此它不适合作为“节能模式”主配置。

### 4.2 balanced 下的 output budget

| 设备 | Budget | 正确 | 正确率 | EOS / 长度终止 | 平均生成 token | Inference (ms) | Energy (J) | J/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OnePlus | 256 | 890 | 67.48% | 1222 / 97 | 155.23 | 7436 | 4.227 | 0.02731 |
| OnePlus | 512 | 915 | 69.37% | 1316 / 3 | 159.62 | 7334 | 4.281 | 0.02682 |
| OnePlus | no-budget | 915 | 69.37% | 1318 / 1 | 159.95 | 7429 | 4.280 | 0.02681 |
| Redmi | 256 | 890 | 67.48% | 1222 / 97 | 155.23 | 7324 | 3.808 | 0.02457 |
| Redmi | 512 | 915 | 69.37% | 1316 / 3 | 159.62 | 7151 | 3.893 | 0.02446 |
| Redmi | no-budget | 915 | 69.37% | 1318 / 1 | 159.95 | 7460 | 3.930 | 0.02461 |

256 budget 截断 97 条，损失 25 个正确答案。512 与 no-budget 的准确率一致，no-budget 只比 512 平均多生成 0.33 token，说明 GSM8K 在 512 之后已经饱和。表中 512 比 256 的平均 latency 略低属于不同全集 run 间的系统波动，不能解释成增加 budget 会加速。

### 4.3 Deadline 内正确产出

| 设备 / mode | ≤5 s | ≤10 s | ≤20 s | 最终正确数 |
|---|---:|---:|---:|---:|
| OnePlus balanced | 216 | 831 | 915 | 915 |
| OnePlus burst | 335 | 860 | 915 | 915 |
| OnePlus low-power-saver | 0 | 1 | 190 | 915 |
| Redmi balanced | 247 | 835 | 915 | 915 |
| Redmi burst | 177 | 734 | 915 | 915 |
| Redmi low-power-saver | 0 | 0 | 191 | 915 |

这张表比单独看 tok/s 更接近交互式服务目标。OnePlus burst 在 5 s 和 10 s deadline 下有收益；Redmi burst 则分别比 balanced 少 70 和 101 个及时正确答案。low-power-saver 即使最终正确率不变，在有限 deadline 下几乎失去可用性。

## 5. llm.npu：MATH-500 与 HammerBench

### 5.1 OnePlus MATH-500，balanced，budget 512

| 成功 | 正确率 | Model load | TTFT | Inference | Prefill | Decode | Energy | J/token | 显式长度终止 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500/500 | 149/500（29.80%） | 4543 ms | 309 ms | 17.96 s | 433.8 tok/s | 22.87 tok/s | 11.35 J | 0.02806 | 217/500（43.4%） |

Deadline 内正确题数为：5 s 下 2 条、10 s 下 43 条、20 s 下 143 条、30 s 下 149 条。MATH-500 平均生成 403.6 token，明显长于 GSM8K 的 159.6 token；217 条长度终止说明固定 1024-token Context/512 output budget 对该任务形成实质约束。下一轮应优先比较 512 与 no-budget，并按 prompt 长度和长度终止拆分正确率，但 no-budget 仍受 1024 Context 上限约束。

### 5.2 Redmi HammerBench，burst，无显式 output budget

| Split | 完成 | Parsed | Func acc | Args acc | End-to-end acc | Conversation progress | Conversation success | TTFT | Inference | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-turn | 654/654 | 500/654 | 41.28% | 17.43% | 31.65% | — | — | 739 ms | 1.84 s | 1.273 J |
| multi-turn | 830/830 | 762/830 | 84.46% | 28.92% | 26.87% | 20.62% | 9.48% | 618 ms | 2.11 s | 1.465 J |

single-turn 中有 327 条 irrelevant 样本，irrelevant correct 为 124/327（37.92%）；相关样本上的 function accuracy 为 82.57%，relevant end-to-end accuracy 为 25.38%。因此 overall function accuracy 41.28% 受到 irrelevant 样本拒答能力影响，不能简单理解为函数名识别只有四成。

multi-turn 没有 irrelevant 样本，函数选择准确率为 84.46%，但参数准确率只有 28.92%；306 个 conversation 中仅 29 个全程成功。错误积累使 conversation success 远低于逐 turn end-to-end accuracy。single/multi 的所有最终正确样本都在 5 s inference deadline 内；按“全集总能耗 / 5 s 内正确任务数”计算，分别为 4.02 J 和 5.45 J/及时正确任务。

## 6. 原生 mllm v2：现有完整 GSM8K 结果

| 设备 | HTP mode | 成功 | 正确率 | Model load (ms) | TTFT (ms) | Inference (ms) | Prefill (tok/s) | Decode (tok/s) | Energy (J) | J/token | 有效能耗样本 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OnePlus | balanced | 1319/1319 | 68.01% | 4142 | 328 | 7131 | 475.4 | 23.32 | 4.038 | 0.02543 | 1318/1319 |
| OnePlus | burst | 1319/1319 | 68.01% | 6952 | 248 | 6479 | 653.7 | 25.73 | 6.977 | 0.04377 | 1315/1319 |
| Redmi | burst | 1319/1319 | 68.01% | 5628 | 290 | 7565 | 551.1 | 22.35 | 5.166 | 0.03241 | 1319/1319 |
| Redmi | low-power-saver | 1319/1319 | 68.01% | 3037 | 1450 | 30375 | 107.6 | 5.49 | 9.788 | 0.06209 | 1319/1319 |

四个完整 run 都得到完全相同的 897 条正确答案，说明 mllm v2 自身也在设备和 HTP mode 间保持输出确定性。

两个 partial run 只可用于恢复进度和故障诊断：Redmi balanced 为 897 条记录、894 条成功；OnePlus low-power-saver 为 213 条记录、212 条成功。由于数据可能按固定顺序执行，partial 子集的题目难度分布不保证与全集一致，不应将其正确率或均值与完整 run 直接排名。

## 7. llm.npu 与 mllm v2 直接对比

以下仅比较设备、HTP mode、数据集和 512 budget 都一致的完整 run。百分比是 mllm v2 相对 llm.npu 的变化；延迟/能耗为负代表更好，吞吐为正代表更好。

| 设备 / mode | 正确率变化 | TTFT | Inference | Prefill tok/s | Decode tok/s | Energy | J/token | Model load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OnePlus balanced | -1.36 pp | -2.76% | -2.77% | +2.81% | +2.21% | -5.69% | -5.21% | -8.79% |
| OnePlus burst | -1.36 pp | -1.09% | -1.52% | +2.05% | +1.04% | +0.20% | +0.96% | -7.40% |
| Redmi burst | -1.36 pp | -8.07% | -9.07% | +7.04% | +8.58% | -5.74% | -5.43% | +8.00% |
| Redmi low-power-saver | -1.36 pp | +11.31% | -0.59% | -10.84% | +0.29% | +4.76% | +5.34% | -3.50% |

原生 mllm v2 在 OnePlus balanced 和 Redmi burst 上表现出约 3%—9% 的 inference latency 改善及约 6% 的平均能耗改善，但 OnePlus burst 的能耗基本持平，Redmi low-power-saver 还出现 TTFT 和能耗回退。因此不能概括为“mllm v2 全面更快、更省电”；收益与设备和 HTP mode 有明显交互。

### 7.1 正确率差异诊断

两套 bundle 的 Context 文件 SHA-256 均为：

`38a0544eb898a012011f4c01e07a36b1340cb2473247929d904ffbe920b261a7`

两套 tokenizer SHA-256 均为：

`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`

对 OnePlus balanced 的 1319 条样本逐项检查得到：

- 1319/1319 个 `formatted_prompt.txt` 逐字节相同；
- 1194/1319 个 normalized predicted answer 相同；
- 870 条两套都正确，377 条两套都错误；
- 45 条仅 llm.npu 正确，27 条仅 mllm v2 正确，净差 18 条；
- llm.npu 为 915 条正确，mllm v2 为 897 条正确；
- 两套 runner 报告的 prompt token count 在 40/1319 条上不同，尽管输入字节和 tokenizer 文件相同。

相同的 45/27 翻转集合在 OnePlus burst、Redmi burst 和 Redmi low-power-saver 对照中重复出现，说明差异稳定存在于两套执行 artifact，而不是某台设备、HTP mode 或随机波动。当前本地文件还显示：llm.npu runner SHA-256 为 `2f5a3d...74a5e31`，mllm v2 runner 为 `4f9a33...5d19b4`；三套 `libMllmQNNBackend.so`（llm.npu、Redmi native、OnePlus native）的哈希也各不相同。

因此目前最严谨的说法是：**相同 Context、tokenizer 和 host-rendered prompt，在不同 runner/backend library artifact 下产生了稳定的 token 计数及输出差异**。还不能仅凭现有 CSV 判断是 tokenizer 调用路径、EOS/采样配置、数值实现还是输出解码造成的。下一步应对首个分叉 token 做逐 token 日志对照，并核对 runner 的 sampler、EOS IDs、tokenizer normalization、QNN graph input packing 和生成停止逻辑。

## 8. 当前问题与建议补实验

1. **补齐 mllm v2 GSM8K 矩阵。** 先用 `--resume` 完成 Redmi balanced 与 OnePlus low-power-saver，再补两机 256/no-budget，避免只在 llm.npu 上得出 budget 结论。
2. **补齐 mllm v2 的任务覆盖。** 至少完成两机 MATH-500/512，以及 Redmi burst HammerBench single/multi；目前无法判断 mllm v2 的性能收益是否能推广到长输出和工具调用。
3. **补 BFCL。** 两套 harness 虽已有 BFCL 启动脚本，但结果目录中没有实际 BFCL run，不能报告 BFCL 精度或性能。
4. **定位 18 题净正确率差。** 这是现有结果中最重要的正确性问题。优先抽取 45 条 llm.npu-only-correct、27 条 mllm-only-correct 和 40 条 prompt token count 不一致样本，记录逐 token ID 与首分叉位置。
5. **重复 HTP mode 实验并随机化顺序。** 尤其是 Redmi burst 比 balanced 慢的异常，需要至少 3 次重复、控制起始电量和温度、交错运行顺序，并同时报告均值、标准差和热状态。
6. **保持能耗有效性检查。** 当前主表能耗覆盖率很高，但 OnePlus mllm burst 有 4 条、OnePlus mllm balanced 有 1 条、llm.npu OnePlus low-power 和 Redmi burst 各有 1 条未进入有效能耗平均。发布时应同时给出覆盖数，不能把无效外供电样本当作 0 J。
7. **长任务考虑更长 Context。** MATH-500 的 43.4% 显式长度终止表明 1024 context 是主要限制之一；只提高 output budget 而不重编更长 AOT Context，收益上限有限。

## 9. 数据来源

- llm.npu 数学任务：`results/llm.npu/{oneplus,redmi}/{gsm8k,math500}/...`
- llm.npu HammerBench：`results/llm.npu/redmi/hammerbench-{single,multi}/...`
- 原生 mllm v2：`results/mllm/{oneplus,redmi}/gsm8k/...`
- llm.npu harness 说明：`code/llm.npu/README.md`
- 原生 mllm v2 harness 与构建说明：`code/mllm/README.md`

报告中的准确率、延迟和能耗均由上述现有结果重新汇总，没有把 smoke test、缺失目录或 partial run 补推成完整结果。
