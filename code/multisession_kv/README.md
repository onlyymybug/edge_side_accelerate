# Mobile multi-Session KV experiments

本目录实现
`/home/lyyyy/edge_side_work/docs/mobile_llm_multisession_kv_experiment.md`
中的正式数据准备与回放流程。手机 runner 和部署文件仍位于
`/home/lyyyy/edge_side_work/qnn_multisession_phone_bundle`。

## 当前研究范围

本工程研究单个常驻手机 LLM 在多 Session 交错请求和有限 KV 槽位下的整 Session
保留/驱逐问题。HammerBench/BFCL 只提供多轮工具调用轨迹；当前固定参考历史回放
不执行真实工具，也不评测 Agent 规划、协作或任务成功率。目标指标是 Prefill、
TTFT、KV 重用率、RSS 和策略开销。

使用现有 `pace` 环境；系统 Python 不包含 `tokenizers`：

```bash
PY=/home/lyyyy/miniconda3/envs/pace/bin/python
cd /home/lyyyy/edge_side_work/code/multisession_kv

$PY build_trace.py \
  --config configs/default.yaml \
  --output-dir data/hammerbench_en_50

$PY validate_results.py trace \
  --trace-dir data/hammerbench_en_50

$PY prepare_data.py \
  --config configs/default.yaml \
  --output data/dataset_inventory.json
```

小规模 Redmi 回放示例（serial 不写死在配置中）：

```bash
$PY replay.py \
  --trace data/hammerbench_en_50/fixed_tools.jsonl \
  --output-dir results/hammer_fixed_smoke \
  --bundle /home/lyyyy/edge_side_work/qnn_multisession_phone_bundle \
  --serial 192.168.12.71:5555 \
  --session-limit 2 \
  --schedule round_robin \
  --policy all_keep \
  --max-new-tokens 16 \
  --warmup \
  --telemetry-interval 1
```

`--session-offset` 与 `--session-limit` 用于把固定的 50 Session 切成
1/2/4/8 Session 回放组；配置中不保存无线 adb 地址。

`fixed_tools` 是控制实验：每个 Session 的工具集合固定并稳定排序。
`native_tools` 保留逐 turn 原始工具集合。两者都直接保存完整 ChatML prompt、
token IDs、原始 snapshot ID、参考 function-call 标签以及相邻 turn 的 token LCP。
模型本轮生成不会写入下一 turn 的参考历史，因此这些结果不是 HammerBench 官方
准确率。

## 随机化、可续跑矩阵

先确认手机无线 ADB 在线，并采集空闲温度基线：

```bash
PY=/home/lyyyy/miniconda3/envs/pace/bin/python
cd /home/lyyyy/edge_side_work/code/multisession_kv
SERIAL=192.168.12.71:5555
BUNDLE=/home/lyyyy/edge_side_work/qnn_multisession_phone_bundle

adb -s "$SERIAL" get-state
$PY calibrate_temperature.py \
  --serial "$SERIAL" \
  --duration-seconds 60 \
  --output data/redmi_idle_temperature.json
```

查看输出的 `suggested_max_start_c` 和原始样本稳定性，再把明确选定的阈值写入
`configs/hammer_matrix_formal.yaml` 的 `temperature_gate.max_start_c`。建议值只是本机
当前环境下的起点，不应跨设备直接复用。

先生成并检查 smoke manifest，不启动手机 runner：

```bash
$PY run_matrix.py \
  --config configs/hammer_matrix_smoke.yaml \
  --trace data/hammerbench_en_50/fixed_tools.jsonl \
  --results-root results/hammer_matrix_smoke \
  --bundle "$BUNDLE" \
  --serial "$SERIAL" \
  --dry-run
```

然后执行一个 job 验证完整链路：

```bash
$PY run_matrix.py \
  --config configs/hammer_matrix_smoke.yaml \
  --trace data/hammerbench_en_50/fixed_tools.jsonl \
  --results-root results/hammer_matrix_smoke \
  --bundle "$BUNDLE" \
  --serial "$SERIAL" \
  --max-jobs 1 \
  --stop-on-failure
```

重复同一条命令会跳过 `pass` job，并从随机序列中下一个 `pending` job 继续；去掉
`--max-jobs 1` 可跑完剩余任务。失败任务默认保留现场且不会自动重试，检查后添加
`--retry-failed`；旧 attempt 会移动到矩阵目录的 `attempts/` 下。配置、trace 或
runner hash 变化时必须使用新的 `--results-root`，启动器不会把不同实验混入已有
manifest。每个 job 启动前还会强制检查手机未接外部电源。

汇总当前已通过的 job：

```bash
$PY analyze_matrix.py --matrix-root results/hammer_matrix_smoke
```

输出为 `summary.csv` 和 `summary.json`。加 `--require-complete` 可要求矩阵全部通过。

正式配置为 `configs/hammer_matrix_formal.yaml`。它覆盖全部 50 个 Session 的固定
1/2/4/8 分组、两种调度、各策略和 3 次重复，当前展开为 2,658 个实际 job；小
Session 下映射到相同槽位数的名义 LRU 比例会合并，并同时记录名义比例列表与实际
槽位比例。正式运行建议仍使用 `--max-jobs` 分批推进，并始终复用同一个正式结果
目录。HTP mode 由配置中的 `htp_mode` 指定，当前为 8，需要对照实验时复制一份
配置、修改 mode，并使用另一个结果目录。

### 配对 pilot（建议先运行）

`configs/hammer_matrix_pilot.yaml` 用于正式全量矩阵前的问题验证。它固定选取
2/4/8 Session 各自的第一组，使同一 Session 数下的策略、调度与 3 次重复面对
完全相同的输入。LRU 名义比例映射到相同实际槽位时会自动合并，因此矩阵共 102
个实际 job。它使用独立结果目录，不会修改已经开始的 formal manifest：

```bash
$PY run_matrix.py \
  --config configs/hammer_matrix_pilot.yaml \
  --trace data/hammerbench_en_50/fixed_tools.jsonl \
  --results-root results/hammer_matrix_pilot \
  --bundle "$BUNDLE" \
  --serial "$SERIAL" \
  --max-jobs 10 \
  --stop-on-failure
```

反复执行同一条命令即可续跑。完成后运行：

```bash
$PY analyze_matrix.py \
  --matrix-root results/hammer_matrix_pilot \
  --require-complete
```

Pilot 全部通过后，生成配对统计、图表和问题验证报告：

```bash
$PY analyze_pilot.py \
  --matrix-root results/hammer_matrix_pilot \
  --output-dir results/hammer_matrix_pilot/analysis
```

主要产物是 `analysis/report.md`、`group_summary.csv`、`paired_deltas.csv`
以及 TTFT、Prefill/复用率和 RSS 图。统计单位是一次完整 run；同一 Session 的
turn 不会被当作互相独立的样本。

### 精简 Stage-3 全体 Session 矩阵

Pilot 证明 Grouped 下额外槽位不改变复用，而 RoundRobin 下容量竞争明显。基于该
结果，`configs/hammer_matrix_stage3_reduced.yaml` 将全体 50 Session 的人口扩展
从原始 2,658 个 job 缩减为 388 个：

- 1 Session：只保留 AllKeep/NoReuse；
- 2/4/8 Session Grouped：只保留 AllKeep/NoReuse；
- 2/4/8 Session RoundRobin：保留 AllKeep、SingleSlot 和不同可实现 LRU 槽位；
- 每个固定分组运行一次；Pilot 已用三次重复确认运行级指标 CV 低于 3%。

使用独立结果目录运行，不能写入旧的 `hammer_matrix_formal`：

```bash
$PY run_matrix.py \
  --config configs/hammer_matrix_stage3_reduced.yaml \
  --trace data/hammerbench_en_50/fixed_tools.jsonl \
  --results-root results/hammer_matrix_stage3_reduced \
  --bundle "$BUNDLE" \
  --serial "$SERIAL" \
  --max-jobs 10 \
  --stop-on-failure
```

这一矩阵用于估计不同 Session 分组造成的总体变异；它不能替代 Pilot 的运行级
重复。完成后用 `analyze_matrix.py --require-complete` 汇总。

全体任务完成后，生成以固定 Session 组为统计单位的配对 bootstrap 报告：

```bash
$PY analyze_stage3.py \
  --matrix-root results/hammer_matrix_stage3_reduced \
  --output-dir results/hammer_matrix_stage3_reduced/analysis
```

名义 4/8 Session 分组最后各有一个实际 2 Session 的尾组；分析脚本保留这些原始
run，但不会把它们混入对应名义 Session 数的主要估计。

### 离线 Oracle 方法空间分析

Stage-3 完成后，可以在不使用手机的情况下，用实际 AllKeep 请求序列和复用 token
价值比较实测 LRU、Belady 最远未来访问及 token-optimal 动态规划上界：

```bash
$PY analyze_oracle.py \
  --matrix-root results/hammer_matrix_stage3_reduced \
  --output-dir results/hammer_matrix_stage3_reduced/oracle_analysis
```

脚本首先要求模拟 LRU 与手机 runner 的逐组总复用 token 完全一致，再报告 Oracle
上界。Oracle 使用未来信息，不是可部署策略；它只用于判断在线生命周期预测方法
可能获得的空间，不能作为实测 TTFT 收益。

下一步只实现整 Session 粒度的 `Lifecycle-LRU`：显式 `DONE` 立即释放，`READY`
优先保留，`WAITING/BLOCKED` 按等待时间衰减，状态相同时使用 LRU。由轨迹最终出现
位置推导的 `DONE` 只能用于 what-if；手机在线对照必须使用运行时实际可见的信号。
首批手机点限定为 N4/C2、N4/C3、N8/C2、N8/C4、N8/C6，不再扩大全量 LRU 矩阵。

生命周期日志必须区分 `runtime_explicit`、`controlled_protocol` 和 `future_oracle`。
其中 `future_oracle` 禁止进入在线策略；受控协议注入的 close 可以验证机制，但不能
描述成 HammerBench/BFCL 原生提供的真实生命周期信号。

### 生命周期策略离线消融

在接入手机前，运行严格区分事件可见性的离线消融：

```bash
$PY analyze_lifecycle.py \
  --matrix-root results/hammer_matrix_stage3_reduced \
  --output-dir results/hammer_matrix_stage3_reduced/lifecycle_analysis
```

脚本只分析 N4/C2、N4/C3、N8/C2、N8/C4、N8/C6，并先要求模拟 LRU 与手机实测
LRU 在全部 42 个组/容量案例中完全一致。`serial_arrival` 只暴露当前请求，作为保守
主结果；`queued_next_turn` 只有在下一请求确实已进入调度队列时才成立，单独作为
条件实验。主要产物是 `report.md`、`lifecycle_summary.csv`、
`lifecycle_by_group.csv`、`lifecycle_audit.jsonl` 和 `lifecycle_reuse.png`。

### Lifecycle-LRU 手机验证

28.2°C 配置完成了一个 N8/C4 smoke；其 6,203 个实际复用 token 与离线模拟完全
一致。正式验证使用独立的 30.0°C 配置和结果目录：

```bash
$PY run_matrix.py \
  --config configs/hammer_matrix_lifecycle_phone_t30.yaml \
  --trace data/hammerbench_en_50/fixed_tools.jsonl \
  --results-root results/hammer_matrix_lifecycle_phone_t30 \
  --bundle "$BUNDLE" \
  --serial "$SERIAL" \
  --stop-on-failure
```

矩阵已经完成：84/84 个任务通过，形成 42 组配对；其中 1 个任务在 runner 初始化
超时后重试通过。生成最终逐组统计、bootstrap 区间、图和报告：

```bash
$PY analyze_lifecycle_phone.py \
  --matrix-root results/hammer_matrix_lifecycle_phone_t30 \
  --offline-by-group \
    results/hammer_matrix_stage3_reduced/lifecycle_analysis/lifecycle_by_group.csv \
  --output-dir results/hammer_matrix_lifecycle_phone_t30/analysis
```

五个容量点的 Prefill token 降幅为 4.4%--30.5%。N4/C3、N8/C4、N8/C6 的
TTFT bootstrap 区间不跨 0；N4/C3 和 N8/C6 的 makespan 区间不跨 0。完整结果见
`results/hammer_matrix_lifecycle_phone_t30/analysis/report.md`。DONE 信号来自受控协议，
因此结果验证的是生命周期感知 KV 驱逐机制，不是端到端 Agent 或官方 HammerBench
准确率。

### BFCL V4 跨数据集验证（6144 context）

新 bundle 位于 `/home/lyyyy/edge_side_work/qnn_multisession_phone_bundle_ctx6144`，
手机目录为 `/data/local/tmp/qnn_multisession_ctx6144`。bundle 自带的 1/2/4/8 Session
与 1100/6143-token 边界 smoke 已通过。BFCL V4 Base 和 Long-Context 按完整工具定义、
官方 ground truth 调用以及模拟工具返回构造固定参考历史；由于当前实验使用固定参考历史，
这些结果不代表 BFCL 官方准确率。每个 split 在 6144 窗口下各选 50 个完整 Session。

重新生成并校验 trace：

```bash
$PY build_bfcl_trace.py --config configs/bfcl_v4_ctx6144.yaml \
  --split base --output-dir data/bfcl_v4_base_50_ctx6144
$PY build_bfcl_trace.py --config configs/bfcl_v4_ctx6144.yaml \
  --split long_context --output-dir data/bfcl_v4_long_context_50_ctx6144
$PY validate_results.py trace --trace-dir data/bfcl_v4_base_50_ctx6144
$PY validate_results.py trace --trace-dir data/bfcl_v4_long_context_50_ctx6144
```

先分别运行 4-job Pilot（N4/C3 与 N8/C6，各一组配对）：

```bash
SERIAL=192.168.15.32:5555  # 无线 ADB 地址变化时更新
BUNDLE=/home/lyyyy/edge_side_work/qnn_multisession_phone_bundle_ctx6144
DEVICE_DIR=/data/local/tmp/qnn_multisession_ctx6144

$PY run_matrix.py \
  --config configs/bfcl_v4_lifecycle_pilot_t30.yaml \
  --trace data/bfcl_v4_base_50_ctx6144/reference_history.jsonl \
  --results-root results/bfcl_v4_base_lifecycle_pilot_t30 \
  --bundle "$BUNDLE" --serial "$SERIAL" --device-dir "$DEVICE_DIR" \
  --stop-on-failure

$PY run_matrix.py \
  --config configs/bfcl_v4_lifecycle_pilot_t30.yaml \
  --trace data/bfcl_v4_long_context_50_ctx6144/reference_history.jsonl \
  --results-root results/bfcl_v4_long_context_lifecycle_pilot_t30 \
  --bundle "$BUNDLE" --serial "$SERIAL" --device-dir "$DEVICE_DIR" \
  --stop-on-failure
```

同一命令可续跑；检查失败现场后增加 `--retry-failed`。完成后分别执行：

```bash
$PY analyze_bfcl_phone.py \
  --matrix-root results/bfcl_v4_base_lifecycle_pilot_t30 \
  --output-dir results/bfcl_v4_base_lifecycle_pilot_t30/analysis
$PY analyze_bfcl_phone.py \
  --matrix-root results/bfcl_v4_long_context_lifecycle_pilot_t30 \
  --output-dir results/bfcl_v4_long_context_lifecycle_pilot_t30/analysis
```

Pilot 已全部通过（Base 4/4、Long-Context 4/4），但每点只有一个 Session 组，
不能据此判断泛化收益。正式矩阵每个 split 各 84 个任务；同一命令可续跑：

```bash
SERIAL=192.168.13.185:5555  # 以 adb devices 中的当前地址为准
BUNDLE=/home/lyyyy/edge_side_work/qnn_multisession_phone_bundle_ctx6144
DEVICE_DIR=/data/local/tmp/qnn_multisession_ctx6144

$PY run_matrix.py \
  --config configs/bfcl_v4_lifecycle_full_t30.yaml \
  --trace data/bfcl_v4_base_50_ctx6144/reference_history.jsonl \
  --results-root results/bfcl_v4_base_lifecycle_full_t30 \
  --bundle "$BUNDLE" --serial "$SERIAL" --device-dir "$DEVICE_DIR" \
  --min-battery-level 35 --retry-failed --stop-on-failure

$PY run_matrix.py \
  --config configs/bfcl_v4_lifecycle_full_t30.yaml \
  --trace data/bfcl_v4_long_context_50_ctx6144/reference_history.jsonl \
  --results-root results/bfcl_v4_long_context_lifecycle_full_t30 \
  --bundle "$BUNDLE" --serial "$SERIAL" --device-dir "$DEVICE_DIR" \
  --min-battery-level 35 --retry-failed --stop-on-failure
```

手机不能边充电边执行本配置。电量低于 35% 时，launcher 在下一个任务开始前正常停止；
先中断/退出 launcher、充电并拔掉电源，再执行相同命令续跑。温度门槛仍为 skin
不高于 30°C。若温度等待超时，任务会被标记 failed；冷却后用上述带
`--retry-failed` 的同一命令续跑。两套矩阵完成后分别对其结果目录执行
`analyze_bfcl_phone.py`。
