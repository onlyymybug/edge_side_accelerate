# 附录 C：手机物理对照实验

这个目录实现 `paper_draft_zh_v3.pdf` 附录 C 的三项最小补充对照。所有真实测量都必须连接原实验手机后运行；目前仓库中的历史 CSV 只用于确定样本，不会被当成新的物理测量。

## 实验与规模

1. `graph_boundary`：QNN 和 llm.npu 各测试 127/128/129、159/160/161、191/192/193 token，固定输出 8 token，每点 5 次交错重复。共 90 次。
2. `tmac_threads`：从 GSM8K 历史结果中按长度范围确定 12 个样本，测试 `(预填充线程 -tb, 解码线程 -t)` 的 `(6,6)/(8,6)/(6,8)/(8,8)`，每项 3 次交错重复。共 144 次。
3. `budget_termination`：GSM8K 和 MATH500 各选择 EOS 不变、截断后改善、截断后仍错误三类，每类 8 个样本；每个样本跑两个预算、3 次交错重复。共 288 次。

计划文件固定在 `plans/`，随机种子及历史结果路径在 `config.json`。每个请求完成后立即原子更新 `summary.csv`，因此中断后可以用同一个 `--run-id` 续跑。

## 0. 电脑端准备

在仓库根目录运行：

```bash
cd /home/lyyyy/pace
conda activate pace
python code/appendix_c_controls/prepare.py
python code/appendix_c_controls/test_appendix_c.py
python code/appendix_c_controls/run.py graph_boundary --runtime qnn --dry-run
python code/appendix_c_controls/run.py tmac_threads --dry-run
python code/appendix_c_controls/run.py budget_termination --dry-run
```

如果当前 Python 报 `No module named tokenizers`，说明没有进入原实验环境；先执行 `conda activate pace`。不要为此另装一套不同版本的 tokenizer。

手机须已通过 USB/ADB 连接，并保留原来三种 runner 的模型和运行目录。确认：

```bash
adb devices
export ANDROID_SERIAL=你的设备序列号
adb -s "$ANDROID_SERIAL" get-state
```

实验时建议拔掉充电电源、关闭后台应用、固定室温，并在每组开始前让手机回到相近温度。`--cooldown-seconds` 可在请求之间加入固定冷却时间；记录中仍应检查起始和最高温度。
每次命令开始时，脚本会对涉及的 runtime/线程配置做一次不计入结果的 warm-up，并把证据留在 `warmup/`；仅调试时才使用 `--skip-warmup`。

## 1. 图边界与 TTFT

两种 runtime 分开运行，并各自使用固定 run id：

```bash
python code/appendix_c_controls/run.py graph_boundary \
  --runtime qnn --run-id phone1_qnn_20260920 --cooldown-seconds 5

python code/appendix_c_controls/run.py graph_boundary \
  --runtime llm.npu --run-id phone1_llmnpu_20260920 --cooldown-seconds 5
```

脚本严格构造目标 token 长度，并记录按块宽推导的 `expected_prefill_calls`。但当前两个原生 runner 的日志不输出“实际图调用次数”，所以仅靠 Python 无法物理观测它。若以后在原生 runner 中加入例如 `prefill_calls=3` 的日志，可追加：

```bash
--prefill-call-regex 'prefill_calls=(\d+)'
```

没有该原生日志时，TTFT 实验仍可完成，但论文中不能把 `expected_prefill_calls` 写成“实测调用次数”。

## 2. T-MAC 分离线程

```bash
python code/appendix_c_controls/run.py tmac_threads \
  --run-id phone1_threads_20260920 --cooldown-seconds 5
```

本实验依赖本次对 `code/T-MAC/runtime.py` 的兼容扩展：`TMAC_THREADS/-t` 控制解码线程，`TMAC_BATCH_THREADS/-tb` 控制预填充线程。未设置新变量时二者仍相同，原有脚本行为不变。

## 3. 预算、终止与新生成

```bash
python code/appendix_c_controls/run.py budget_termination \
  --run-id phone1_budget_20260920 --cooldown-seconds 5
```

这里会真正重新生成，而不是复用原始文本。每次请求保存格式化输入、原始输出、completion、runner 日志和 telemetry；汇总表包含 token 数、终止原因、正确性、时间与能量。

## 断点续跑与小规模检查

先真机跑一个请求：

```bash
python code/appendix_c_controls/run.py tmac_threads \
  --run-id phone1_threads_smoke --limit 1
```

正式任务中断后，原命令不变、继续使用原 `--run-id`；状态为 `ok` 的 request 会跳过。默认遇到错误即停，便于修复后续跑；诊断时可加 `--continue-on-error`。

正式请求若缺少能量、token 数或关键时延会被标为 `error` 并停止，而不会悄悄混入完整结果。无线 ADB 恢复后，用相同 `--run-id` 重跑即可。

也可以只选择一个 T-MAC 线程组合做诊断，例如：

```bash
python code/appendix_c_controls/run.py tmac_threads \
  --batch-threads 8 --decode-threads 6 --limit 1 \
  --run-id threads_tb8_t6_smoke
```

## 汇总

对每个结果目录运行：

```bash
python code/appendix_c_controls/analyze.py \
  results/appendix_c_controls/tmac_threads/phone1_threads_20260920
```

会生成聚合 CSV 和 `VALIDATION.md`。原始逐请求证据位于该结果目录的 `artifacts/`，提交论文数据时不要只保留聚合 CSV。
