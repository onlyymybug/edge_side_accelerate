# Redmi T-MAC Qwen2.5-3B GSM8K 当前结果

## 1. 范围与状态

本文汇总以下实验截至2026-08-24的当前结果：

```text
设备：Redmi 2510DRK44C（SM8750，Android 16）
Backend：T-MAC CPU
模型：Qwen2.5-3B-Instruct GPTQ-INT4 W4A16G128 INT_N
线程数：4
数据集：GSM8K official test split，1319题
Output budget：512
MAX_SEQ_LEN：1024
Prompt：Qwen ChatML raw v1
Scorer：math_benchmarks_v2_relaxed_prm800k
```

结果目录为：

```text
/home/lyyyy/pace/results/T-MAC/redmi/gsm8k/qwen2.5-3B-tmac-w4a16/threads_4/output_budget_512/
```

该实验目前未完成，不能作为完整 GSM8K test split 的正式结果：

| 项目 | 当前状态 |
| --- | ---: |
| 数据集总题数 | 1319 |
| 成功推理 | 644/1319，48.82% |
| 失败/尚待重试 | 675/1319，51.18% |
| 成功范围 | `gsm8k_0001`–`gsm8k_0644` |
| 待重试范围 | `gsm8k_0645`–`gsm8k_1319` |
| 有效能耗测量 | 644/644个成功样本 |
| 可用于完整数据集报告 | 否 |

之前断线产生的失败行仍保存在 `summary.csv` 中。续跑已经把原先从 `gsm8k_0542` 开始的部分失败修复到 `gsm8k_0644`；在 `gsm8k_0645` 再次发生 `device offline` 后，新逻辑停止。`gsm8k_0646`–`1319` 主要是上一次运行留下的离线失败记录，并不表示修复后的脚本又逐题执行了674次。

## 2. 统计口径

- 正确率使用当前 GSM8K 宽松数值等价 scorer，不要求固定 `The answer is ...` 标记。
- “已完成子集准确率”只以644条成功推理为分母；由于它们是数据集顺序前缀而非随机样本，不能直接外推到完整1319题。
- “全数据集保守下界”将675条未完成题暂记为不正确，只表示当前已经确认的最低正确比例，不是最终模型准确率。
- TTFT 使用 llama.cpp `prompt eval time`，与当前 PACE QNN 的 `first_token == prompt_eval_end` 口径对齐，但不包含每题重新启动进程后的模型加载时间。
- inference/e2e latency 使用 llama.cpp total inference time，不包含模型加载、ADB和主机调度开销。
- J/token 使用644条有效样本的实测总能量除以总生成 token 数，而不是逐题 J/token 的简单平均。
- EDP 为逐题 `energy_j × inference_ms / 1000` 的算术平均。
- 能耗来自整机电池侧采样，包含CPU、内存和手机其他组件，不是隔离的CPU kernel能量。
- 所有 deadline 指标都只覆盖当前644题，统一标记为 provisional。

## 3. 任务质量

| 指标 | 当前结果 |
| --- | ---: |
| 已完成子集正确 | 509/644，79.04% |
| 已完成子集 Wilson 95% 区间 | 75.73%–82.00% |
| 全1319题保守下界 | 509/1319，38.59% |
| 答案可提取 | 644/644，100% |
| 推荐格式合规 | 635/644，98.60% |
| EOS 正常结束 | 641/644，99.53% |
| Length stop | 3/644，0.47% |

答案解析方式为：

| 解析方式 | 数量 | 其中正确 |
| --- | ---: | ---: |
| `strict_answer_phrase` | 635 | 505 |
| `last_line_number` | 7 | 4 |
| `relaxed_answer_phrase` | 2 | 0 |

格式不是当前主要问题：所有644条输出都能提取答案，只有9条没有遵循推荐的严格答案句式。当前已完成前缀中错误主要来自数学答案本身，而不是 parser 丢弃。

79.04% 只能描述前644题。由于 GSM8K 文件按固定顺序排列，Wilson 区间也只是这个前缀内部的二项描述，不能把它当作完整 test split 的统计区间。最终准确率必须等剩余675题补齐后重新计算。

## 4. 性能与资源

| 指标 | 平均 | 中位数 | P95 | 范围 |
| --- | ---: | ---: | ---: | ---: |
| 模型加载时间 | 6808.90 ms | 6546.33 ms | 13172.63 ms | 927.89–18418.57 ms |
| TTFT / prefill latency | 7185.35 ms | 6983.50 ms | 10445.29 ms | 2582.07–13485.58 ms |
| Decode latency | 11254.88 ms | 9949.43 ms | 23084.56 ms | 291.21–39634.59 ms |
| Inference/e2e latency | 18440.24 ms | 17125.30 ms | 31807.99 ms | 6203.76–48181.54 ms |
| Prefill tokens/s | 20.913 | 20.105 | 28.165 | 11.790–43.040 |
| 对齐后的 decode tokens/s | 14.594 | 14.023 | 18.752 | 7.574–24.413 |
| Runner 原生 decode tokens/s | 14.557 | 14.020 | 18.679 | 7.520–24.260 |
| Prompt tokens | 144.06 | 140 | 184 | 107–225 |
| Generated tokens | 158.61 | 146 | 303 | 6–512 |
| Visible output tokens | 156.61 | 144 | 301 | 4–511 |
| Peak RSS | 2717.24 MiB | 2716.69 MiB | 2724.55 MiB | 2710.67–2731.56 MiB |

对齐后的 decode tokens/s 与 runner 原生值只差约0.25%，说明 token 计数与 llama.cpp perf 日志基本一致。

### 4.1 模型加载边界

当前执行方式每题启动一次 `llama-cli` 并重新加载模型。正式表中的 TTFT 只测量 prompt evaluation，不含模型加载。若从冷启动进程开始计算用户可感知时间，当前平均值约为：

| 冷启动指标 | 近似平均值 |
| --- | ---: |
| 启动到首 token | model load + TTFT = 13.99 s |
| 启动到推理结束 | model load + inference = 25.25 s |

因此横向对齐 PACE QNN 表时应使用7.19秒 TTFT和18.44秒 inference；讨论实际逐题冷启动体验时，应同时报告13.99秒 cold-start first token和25.25秒 cold-start completion。不能把前一组数直接表述为包含模型加载的端到端延迟。

实验代码现已直接输出 `cold_start_ttft_ms` 和
`cold_start_e2e_latency_ms`；旧的644条成功记录会在下次 `--resume` 时由
已有 `model_load_ms`、`ttft_ms` 和 `inference_ms` 自动回填，不需要重新推理。

## 5. 能耗与温度

644条成功样本全部具有有效非负电池侧能耗：

| 指标 | 当前值 |
| --- | ---: |
| 总实测能量 | 7820.80 J |
| 平均每题能量 | 12.144 J |
| 加权 J/token | 0.07657 J/token |
| 逐题 J/token 平均 | 0.08173 J/token |
| 平均 EDP | 257.55 J·s |
| 平均功耗 | 0.655 W |
| 当前总能量/正确题 | 15.365 J/correct |
| 当前正确任务/J | 0.06508 correct/J |
| 平均每题最高皮温 | 44.40°C |
| 观测最高皮温 | 47.12°C |

`15.365 J/correct` 和 `0.06508 correct/J` 只针对已完成的前644题。剩余题补齐后，总能量和正确数都会变化，不能把当前数值与完整1319题的QNN结果直接作正式排名。

## 6. Deadline 结果

Energy per correct task under deadline 定义为当前644题的7820.80 J总有效能量，除以相应 deadline 内答对的题数。

| Deadline | 及时正确 | 已完成子集 timely success | Energy/correct under deadline | 状态 |
| ---: | ---: | ---: | ---: | --- |
| 5 s | 0/644 | 0% | 不可计算 | Provisional，未达到任何正确任务 |
| 10 s | 37/644 | 5.75% | 211.37 J | Provisional |
| 20 s | 354/644 | 54.97% | 22.09 J | Provisional |
| 30 s | 480/644 | 74.53% | 16.29 J | Provisional |

最终有509题答对，其中480题在30秒内完成，另29个正确答案超过30秒。5秒内没有正确题的主要原因是平均TTFT本身已经达到7.19秒；这也说明仅看14.59 decode tokens/s会明显高估严格交互 deadline 下的有效进展。

当前 `deadline_summary.csv` 正确标记：

```text
num_completed_tasks = 644
valid_for_full_dataset_reporting = False
```

## 7. 运行过程观察

按题号每100题分段后的均值如下：

| 题号 | TTFT (ms) | Decode tokens/s | Inference (s) | 每题能量 (J) | 平均最高皮温 (°C) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–100 | 6414.20 | 16.235 | 16.21 | 11.65 | 43.69 |
| 101–200 | 7443.91 | 14.514 | 19.12 | 12.71 | 44.69 |
| 201–300 | 7501.26 | 13.932 | 19.54 | 13.15 | 44.51 |
| 301–400 | 7210.74 | 14.215 | 18.78 | 11.88 | 44.96 |
| 401–500 | 7599.25 | 14.077 | 19.18 | 11.79 | 45.14 |
| 501–600 | 7133.93 | 14.377 | 18.44 | 12.13 | 43.26 |
| 601–644 | 6750.86 | 15.072 | 17.03 | 11.11 | 44.70 |

前100题之后吞吐下降、TTFT升高，但后续又部分恢复；这不是单调 thermal-throttling 轨迹。由于运行跨越断线和续跑批次、题目输出长度不同、配置顺序也没有随机化，当前只能说明性能存在明显时间/批次波动，不能仅凭该表把变化归因于温度。

## 8. 当前结论与限制

当前可以支持：

1. T-MAC Qwen2.5-3B W4A16G128 在 Redmi 上可以稳定产生可评分 GSM8K 输出；前644题全部可解析，格式合规率98.60%。
2. 已完成子集平均 decode 为14.59 tokens/s，但TTFT达到7.19秒，严格5秒 deadline 下没有及时正确任务。
3. 当前逐题重载模型的开销很大：平均加载6.81秒，实际冷启动首 token约13.99秒。
4. 644条成功样本的能耗覆盖完整，当前加权能效为0.07657 J/token，平均EDP为257.55 J·s。

当前不能支持：

1. 不能报告完整 GSM8K 准确率；79.04%只是前644题准确率。
2. 不能把当前 deadline energy/correct 与完整QNN结果做正式 winner、regret或Pareto排名。
3. 不能把分段性能变化确定归因于thermal throttling。
4. 不能声称当前 `e2e_latency_ms` 包含模型加载或ADB开销。
5. 当前只有单批次结果，不满足关键 operating point 至少5次独立能耗重复的要求。

## 9. 后续操作

Redmi重新连接后，可从 `gsm8k_0645` 继续：

```bash
cd /home/lyyyy/pace

GSM8K_BUDGET=512 \
./code/T-MAC/run_gsm8k.sh --resume
```

续跑逻辑只跳过 `runner_returncode=0` 的成功题，因此 `gsm8k_0645`–`1319` 会重新执行；断线失败记录会被对应的新结果替换。新的断线保护会在检测到ADB transport错误时保存checkpoint并立即停止，不会再次连续刷出数百条空失败记录。

完整运行后必须重新汇总：

- 1319题最终准确率及Wilson区间；
- 完整 deadline timely success和energy/correct；
- 完整总能量、J/token、EDP和tasks/J；
- 与Redmi QNN Qwen相同模型/配置的对齐比较；
- 至少5个独立关键能耗批次。

## 10. Artifact

```text
manifest:
/home/lyyyy/pace/results/T-MAC/redmi/gsm8k/qwen2.5-3B-tmac-w4a16/threads_4/output_budget_512/manifest.json

sample summary:
/home/lyyyy/pace/results/T-MAC/redmi/gsm8k/qwen2.5-3B-tmac-w4a16/threads_4/output_budget_512/summary.csv

deadline summary:
/home/lyyyy/pace/results/T-MAC/redmi/gsm8k/qwen2.5-3B-tmac-w4a16/threads_4/output_budget_512/deadline_summary.csv

per-sample artifacts:
/home/lyyyy/pace/results/T-MAC/redmi/gsm8k/qwen2.5-3B-tmac-w4a16/threads_4/output_budget_512/gsm8k/<sample_id>/
```

每个成功样本目录包含 formatted prompt、raw output、cleaned completion、runner stderr/log、metrics、telemetry和command，能够追溯本报告中的评分、时延与能耗数据。
