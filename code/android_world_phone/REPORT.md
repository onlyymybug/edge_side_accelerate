# AndroidWorld 真机首轮适配报告

2026-11-06。在连接的 REDMI K90（SM8750、Android 16 / API 36）上，已跑通截图 → 本机 Qwen3-VL/GenieX 推理 → JSONAction → root ADB 操作 → AndroidWorld 原始成功校验。

当前结论：**真机环境可以执行任务，VLM 已分别完成新建文件夹、新建笔记并保存正文。** 笔记任务使用了完成校验反馈与重试。两个任务的程序化 UI 基线也均通过，不能将该基线结果算作模型成绩。

## 实际结果

| 执行方式 | 任务 | 初始分数 → 最终分数 | 步数 / 整体耗时 | 记录 |
| --- | --- | --- | --- | --- |
| VLM 自主动作 | MarkorCreateFolder | 0 → 1 | 8 步 / 56.16 秒 | [result.json](results/20261106_110507/folder/result.json) |
| VLM 动作 + 校验反馈重试，最后一次尝试 | MarkorCreateNote | 0 → 1 | 10 步 / 77.18 秒 | [result.json](results/20261106_111347/note/result.json) |
| 程序化 UI 基线 | 新建文件夹、笔记及保存 | 两项均为 0 → 1 | 按 UI 标签执行，无 VLM | [result.json](results/ui_baseline_20261106_110921/result.json) |

模型文件夹成功发生于调整过程中的一次运行，笔记为后续调整后的独立运行；这不是固定配置下的一次正式成功率评测，也不能称作一次正式评测 2/2。代理在每步后接收校验反馈，不是原样运行上游 M3A。全部尝试（含失败、中断与早期部署失败）保留在 [summary.json](results/summary.json)，各步的实际提示词也单独保存。

前一次笔记尝试创建 `sprint_plan_1110.md`、输入正文后重复输入并宣布 complete，未点击 Save；校验为 0，独立读取磁盘确认内容为空。随后修正任务循环：模型宣布 complete 而实际校验失败时，拒绝完成声明并反馈给模型，在预算内继续操作。

最后一次笔记创建 `release_notes_1114.md`，正确输入正文，4 次提前完成声明均被拒绝；第 10 个动作由模型选择点击 Save（index 3），原始校验由 0 变为 1。没有由脚本代替模型点击 Save。独立读取磁盘确认正文与目标 **逐字符一致**：`Hello AndroidWorld from Qwen3 VL on this phone.`。程序化基线另行验证输入和保存通道。见 [设备复查](results/device_verification.json)。

成功的文件夹另经 `stat` 确认确为 directory，补足上游该校验器只检查路径存在、未单独区分文件与目录的限制。

## 已解决的环境问题

- 普通 shell 缺少 `INJECT_EVENTS`，`adb install -g` 也缺少授予运行权限的权限。通过已有 root 通道安装独立 Markor、授权并执行输入；模型进程本身仍以 shell 身份运行。
- 提供 AndroidWorld `AsyncEnv` 实现，以截图和 UIAutomator XML 替代模拟器的 gRPC 观测；为原始文件校验器提供真实 ADB protobuf 响应。
- 增加常驻 GenieX C API 进程：一次创建 VLM handle，每步 reset KV、加载当前截图推理。无需每步重启模型。
- UI 元素列表去除部分重复标签；`input_text` 必须定位到可编辑控件，越界或无效目标不会执行。
- 捕获到一次 UIAutomator 超时，已增加远端 `timeout 12`、最多三次重试及新 XML 检查，避免把旧 XML 当作当前界面。
- 每步输入使用独立设备文件路径，保存模型图像 SHA256、原始截图、XML、提示词、动作与 SDK 指标。
- 校验失败时的提前完成声明会被拒绝，反馈给模型继续规划；仍受最大步数限制。校验器不会因模型自称完成而返回成功。

## 耗时与复现

文件夹成功运行的加载耗时为 5.37 秒，逐步推理中位数 3.03 秒；笔记最后一次运行加载为 6.09 秒，逐步推理中位数 3.48 秒。两次运行单步推理范围约 1.56–4.98 秒。任务整体耗时还包括 UIAutomator、截图、ADB 输入与校验，且不包含先前的模型加载时间。

模型原始日志及实际 native 启动命令：

- 文件夹：[phone_raw.log](results/20261106_110507/phone_raw.log)、[command.json](results/20261106_110507/command.json)。
- 最后一次笔记：[phone_raw.log](results/20261106_111347/phone_raw.log)、[command.json](results/20261106_111347/command.json)，每步 `note/00` 至 `note/09` 下有 `response.json`；步骤 `05` 至 `08` 记录 `completion_rejected: true`。

```bash
cd /home/lyyyy/edge_side_work/android_world_phone
# 常规任务入口
.venv/bin/python run_tasks.py --serial 189f0fad --tasks both --max-steps 12
# 最后一次笔记尝试的实际参数；再次复现时换一个尚不存在的文件名
.venv/bin/python run_tasks.py --tasks note --max-steps 12 --note-name release_notes_1114.md
# 独立验证环境通道，结果不计入 VLM 成绩
.venv/bin/python ui_baseline.py
```

## 范围及剩余工作

当前复用上游 commit `55ddaf5edc6f6c25673aed343907d5b9dca66f84` 的原始任务目标和 `is_successful`。为保护个人手机，替换了初始化：不清共享存储、不恢复快照、不改时间、不生成干扰文件；任务前打开 Markor。不是 API 33 模拟器完整 AndroidWorld 基准结果。

目前只接入两个 Markor 任务，ASCII 文本输入，512×512 等比例填白截图。使用自定义逐步 GUI agent；尚未完整接入上游 M3A 的动作后反思摘要及双截图流程。

完成校验反馈已验证有效，但仍发生多次提前完成声明。下一阶段重点是更完整的动作后状态反馈与反思，减少重复操作；再固定提示词配置，验证多次运行和其他应用任务，才能讨论稳定成功率。

已运行 Android 原生编译、Python 语法检查、实际真机多轮动作执行及两项上游校验。没有运行需要模拟器环境的完整上游测试套件。试跑模型进程已退出。
