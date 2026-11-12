# Agent–推理服务闭环 smoke

`agent_closed_loop.py` 是最小链路验证，不是 AppWorld/BFCL 结果，也不证明新调度方法有效。它让两个独立任务共享一个常驻模型：模型产生 `lookup_item` 调用，主机执行工具并等待，随后把真实工具结果送回同一会话的模型；任务完成后发出显式 `evict_session`。没有读取参考轨迹的未来 turn。

现有 runner 仍是串行请求、HTP mode 仅在启动时设定。本脚本只采用 FIFO 和 AllKeep；后续策略实验必须在此基础上增加任务级调度与固定预算对照。

## 无手机测试

```bash
cd /home/lyyyy/edge_side_work/code/multisession_kv
PY=/home/lyyyy/miniconda3/envs/pace/bin/python
$PY -m unittest test_agent_closed_loop.py
$PY agent_closed_loop.py --backend mock --output-dir results/agent_closed_loop_mock_01
```

Mock 的模型回答是脚本预设的，仅验证协议、工具等待、排队时间和日志。`results/agent_closed_loop_mock_01` 必须是一个尚不存在的目录。

## 手机 smoke

确认模型和 runner 已上传到 `/data/local/tmp/qnn_multisession_ctx6144`，并查看当前无线 ADB 地址：

```bash
adb devices
SERIAL=<这里填 adb devices 显示的 IP:端口>
adb -s "$SERIAL" get-state
```

然后运行（手机实测结果可能失败；失败应检查 `runner.log` 和任务日志，不要把 mock 的成功数当作手机成功率）：

```bash
cd /home/lyyyy/edge_side_work/code/multisession_kv
PY=/home/lyyyy/miniconda3/envs/pace/bin/python
$PY agent_closed_loop.py \
  --backend phone \
  --serial "$SERIAL" \
  --bundle /home/lyyyy/edge_side_work/qnn_multisession_phone_bundle_ctx6144 \
  --device-dir /data/local/tmp/qnn_multisession_ctx6144 \
  --output-dir results/agent_closed_loop_phone_01 \
  --tool-delay-ms 1000
```

`events.jsonl` 含任务、请求和工具的时序；`requests.jsonl` 含 runner 原始计时、真实排队时间 `queue_ms` 及主机 prompt 准备时间；`tasks.jsonl` 含任务结果；`telemetry.jsonl` 是已有 Android 采样器的低频温度/RSS 记录。排队时间从请求 ready 到被 FIFO 取出，不包含 prompt tokenization 或 ADB/runner 往返。当前的两任务合成数据只适合验证这些字段是否正确，不能用于论文效果报告。

通过手机 smoke 后的下一项工作是接入可执行的公开任务环境（优先 AppWorld 的小规模可解子集），保留同一事件协议，再测并发任务的真实 ready 队列、工具等待分布、成功率及任务完成时间。只有确认这些现象存在，才值得实现任务关键路径感知的调度器。

2026-10-22 的首轮真机 smoke（`results/agent_closed_loop_phone_20261022_01`）完成了 2 个任务、4 次推理；模型实际生成两个 `lookup_item` 调用，工具返回后的请求分别复用 95 和 97 token。该结果只说明链路可以运行，不代表公开数据集成功率或方法收益。运行前 skin 温度约 38°C，且这次关闭了持续温度采样，不能用其时延作温度受控比较。
