# AndroidWorld 真机实验

2026-11-12：代码已整理到 `/home/lyyyy/edge_side_work/code/android_world_phone`。旧目录为兼容链接。模型包仍在 `/home/lyyyy/edge_side_work/phone_bundle*`，项目内虚拟环境、NDK、实验结果和原始日志随项目保留。

最近正式实验使用三档1K/4K/16K包，四个官方M3A扩展episode均完整结束，40步、77次推理、任务成功0/4；模型调用正常。见 [最新结果](OFFICIAL_M3A_MULTICL16K_STATUS.md) 和 [完整运行报告](results/official_m3a_20261111_140115/RUN_REPORT.md)。

当前正式入口：

```bash
cd /home/lyyyy/edge_side_work/code/android_world_phone
.venv/bin/python -u run_official_m3a.py \
  --serial 189f0fad --tasks extended --repeats 1 \
  --max-new-tokens 512 --skip-context-tight-tasks \
  --request-timeout 600 \
  --remote-bundle /data/local/tmp/qwen3_vl_geniex_multicl16k_3tier

# 将下列目录替换为本次运行输出
.venv/bin/python analyze_official_timing.py results/official_m3a_20261111_140115
```

`run_official_m3a.py` 使用官方M3A动作与动作后反思；`run_tasks.py` 的CLI是早期简化代理，同时该文件提供正式实验使用的 `ResidentVLM` 服务封装。

以下按日期保留此前实验说明，旧4K、固定16K、多档8K结果均为历史对照。

2026-11-10历史扩展试跑：当时3项任务因8K预算不足暂缓，已于11月11日使用三档16K包补测。见 [扩展任务结果](EXTENDED_MARKOR_STATUS.md)。

2026-11-10历史配置：当时 `../../phone_bundle` 更换为 1K/2K/4K/8K 多档包，手机独立部署在 `/data/local/tmp/qwen3_vl_geniex_multicl8k`。短输入和约4K输入已恢复旧包速度，见 [Multi-CL 8K 验收](MULTICL8K_VALIDATION.md)。现已完成两个官方 GUI 任务试跑，20步/38次推理，任务0/2成功，设备服务空闲率41.42%，见 [8K官方运行结果](OFFICIAL_M3A_8K_STATUS.md)。下述16K数据为历史对照。

官方 M3A 状态见 [OFFICIAL_M3A_STATUS.md](OFFICIAL_M3A_STATUS.md)。2026-11-08 新到 16K bundle 已通过短请求与约 14.5K tokens 长输入测试，并已接通官方动作及动作后总结；两个官方episode均已结束但任务未成功，设备服务空闲率17.28%；详情及计时口径见状态报告。此前时间报告属于简化代理。

官方流程使用已独立部署的16K包：

```sh
.venv/bin/python -u run_official_m3a.py --serial 189f0fad --tasks both --repeats 1 --max-new-tokens 512 --remote-bundle /data/local/tmp/qwen3_vl_geniex_ctx16384 --request-timeout 300
```

结束后用 `.venv/bin/python analyze_official_timing.py results/official_m3a_<时间>` 统计。运行期间临时USB保持亮屏，退出恢复原设置。
新旧包decode对照使用 `.venv/bin/python compare_decode_bundles.py`，在GUI测试完全退出后执行；同一手机顺序测试，不并行加载两个模型。

设备：REDMI K90，SM8750，Android 16，ADB serial `189f0fad`。
旧4K模型部署于 `/data/local/tmp/qwen3_vl_geniex_sm8750`，本机备份为 `../../phone_bundle1`；11月10日该目录为8K多档包，现为三档16K包。两者均为 GenieX/QAIRT Qwen3-VL-4B。
上游代码：`../android_world_upstream`，commit `55ddaf5edc6f6c25673aed343907d5b9dca66f84`。

实际结果及限制见 [REPORT.md](REPORT.md)：UI 基线两项通过；VLM 已分别跑通文件夹、笔记任务，笔记使用完成校验反馈重试。不是完整基准成功率。

## 运行

```bash
cd /home/lyyyy/edge_side_work/code/android_world_phone
bash build_native.sh
.venv/bin/python run_tasks.py --serial 189f0fad --tasks both --max-steps 12
# 单测指定笔记，文件名必须尚不存在
.venv/bin/python run_tasks.py --tasks note --note-name fresh_note.md --note-text 'Hello AndroidWorld.'
# 验证输入、保存和原始校验器，不计入模型成绩
.venv/bin/python ui_baseline.py
```

依赖在 `.venv` 中，上游 protobuf 通过其 `setup.py generate_protos` 生成。
`build_native.sh` 使用 Android NDK r27b，或 `ANDROID_NDK_HOME` 指向的 NDK。
首次运行需安装 `assets/net.gsantner.markor_146.apk`，完成欢迎页并授予该应用存储权限。
普通 shell 在此手机上没有输入注入权限，输入和校验命令通过已有 `su` 权限运行。

## 结构

- `phone_env.py`：实现 AndroidWorld `AsyncEnv`，采集截图、UIAutomator XML，执行 `JSONAction`；为上游文件校验器提供真实 ADB protobuf 响应。
- `native/vlm_server.cpp`：使用 GenieX v0.10.0 C API 常驻模型，加载一次，每个请求 reset KV 后执行一次带截图的推理。记录加载、首 token、视觉编码、SDK prompt_time 和 decode 耗时；prompt_time为SDK首token减视觉编码的口径，包含相关准备开销。
- `run_tasks.py`：每步保存截图、UI、提示词、模型原始输出和动作。每次执行动作后调用原始 AndroidWorld `is_successful`，模型自行宣布完成不能通过校验。
- `results/<时间>/command.json`：实际 native 启动命令；`phone_raw.log`：设备原始输出；各任务每步 `response.json`：动作与 SDK 指标；`manifest.json`：环境和最终结论。

## 适配范围

这是 Android 16 真机适配试跑，不能作为原始 API 33 模拟器完整 benchmark 分数。
复用原始 `MarkorCreateFolder`、`MarkorCreateNote` 目标与成功判定；初始化替换为唯一目标名称检查。
不执行上游清共享存储、恢复应用快照、修改系统时间、创建干扰文件等操作。应用在各任务前打开。
Markor 单独设为英文；输入传输目前支持 ASCII，动作仅允许操作 Markor、导航、滚动和等待。
图像等比例缩放并填白至 512×512，UI 元素保留原始屏幕像素坐标。
本节所述简化代理为 `run_tasks.py` 的历史流程。当前正式入口 `run_official_m3a.py` 已接入官方双图和动作后反思；任务初始化采用真机新文件适配。

## 证据说明

`elapsed_seconds` 为单次 native reset/template/generate 总耗时，`host_request_seconds` 另包含请求文件推送；它们不包含 UI 采集和动作执行。
`load_seconds` 单独记录常驻进程创建模型耗时；`ttft_us`、`media_time_us`、`prompt_time_us`、`decode_time_us` 来自 SDK，单位微秒。
UIAutomator 采集和 ADB 输入也消耗时间，整个任务耗时见任务 `result.json`。
GenieX API 未提供旧 runner 的 prefill/decode method call 数和 vision execution 计数，不将 token 数误记为调用数。

## 服务空闲率实验

首轮实测结果见 [时间研究报告](TIMING_STUDY.md)，原始详细分析位于 [20261106_123655/timing_analysis](results/20261106_123655/timing_analysis/REPORT.md)。

```bash
bash build_native.sh
.venv/bin/python run_tasks.py --tasks both --repeats 3 --max-steps 12
.venv/bin/python analyze_timing.py results/<本次时间>
.venv/bin/python -m unittest test_timing_accounting -v
```

`--repeats 3` 连续执行三组文件夹/笔记任务，全程只创建一次模型；任务名称自动区分。
`host_timeline.json` 保存主机单调时钟的阶段及嵌套细节，`service_stats.json` 保存设备单调时钟的 ready、请求处理区间、模型计算区间、shutdown。
分析器输出 `timing_analysis/REPORT.md`、CSV、图及 Chrome/Perfetto trace，并检查时间闭合、请求区间不重叠、SDK 与请求窗口匹配。
任务循环、ready 至 shutdown、冷启动分别给出统计口径；空闲指没有请求处理或没有模型计算，不是 NPU 硬件利用率。
