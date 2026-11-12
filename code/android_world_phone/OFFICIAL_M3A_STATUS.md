# 官方 M3A 真机试跑完成：任务未成功

2026-11-08。新16K包已加载并完成两个官方episode。26步、50次推理（26动作、24总结），任务得分均为0；不能将调用链跑通描述为任务成功。

| 任务 | 步数 | 动作/总结调用 | 主机循环秒 | 得分 |
|---|---:|---:|---:|---:|
| rep01_folder | 10 | 10 / 10 | 1234.311 | 0.0 |
| rep01_note | 16 | 16 / 14 | 1664.493 | 0.0 |

文件夹任务在桌面重复滚动，到10步上限结束；笔记任务打开Markor、新建文件并尝试输入，最后声明完成，但正确名称的.md文件在磁盘上为0字节，未保存要求的正文。见[最终文件校验](results/official_m3a_20261108_010622/rep01_note/final_file_verification.json)。
笔记第13步`input text from`发生10秒ADB超时，该动作无总结；最终status步骤按官方逻辑也无总结。50个模型请求本身全部正常返回。异常事件保存在summary.json的phase_errors，未剔除失败步骤。

## 时间口径

设备steady clock：ready→shutdown 2733.647秒，请求处理忙2261.283秒，空闲472.364秒，**服务handler空闲率17.28%**。模型仍驻留；这不是NPU利用率。
主机perf_counter：任务循环2898.804秒，请求在途2400.876秒，**无请求在途占比17.18%**。
本轮主机/设备计时间隔拟合速率为1.061316，不能直接相减两者的秒数。早期约22%/23%的混合时钟结果已撤回并修正；原始两种时间字段保留、分开标识。[运行后独立钟速探测](results/bundle16k_20261108_0105/post_run_clock_probe.json)也显示两者间隔不同。旧11月6日数据的间隔拟合约1.000000，无本轮这项差异。

主机阶段分解和服务空闲窗口中位数/P95见[完整报告](results/official_m3a_20261108_010622/timing_analysis/REPORT.md)、[summary.json](results/official_m3a_20261108_010622/timing_analysis/summary.json)。仅两个失败episode，不能代表成功GUI任务的一般占比。

## 官方流程与真机适配

直接执行上游M3A.step与episode_runner.run_episode，提交55ddaf5edc6f6c25673aed343907d5b9dca66f84；四个核心源文件git diff为空，SHA256保存在manifest。
保留原始提示词/历史/输出解析、独立双图、正常动作后总结、1秒transition与2秒动作后等待、官方步数分配；没有中途得分反馈或拒绝完成后重试。
真机采用root ADB/UIAutomator代替模拟器gRPC；XML按leaf/content-description/scrollable规则选元素，分别填白缩放双图至512×512。任务初始化用新名称，不清共享存储、不恢复快照、不改时间、不加干扰文件，应用范围Markor。因此不是原始API33模拟器benchmark。
已修正转接层对swipe可选空时长参数的转发；测试临时USB亮屏，结束恢复原设置0。服务均已退出。

证据：[完整命令](results/official_m3a_20261108_010622/command.json)、[manifest](results/official_m3a_20261108_010622/manifest.json)、[phone_raw.log](results/official_m3a_20261108_010622/phone_raw.log)、[host_timeline](results/official_m3a_20261108_010622/host_timeline.json)、[设备区间](results/official_m3a_20261108_010622/service_stats.json)。每步action/summary目录含原始提示词、两张原图/模型图、response.json。

## 模型与历史

16K容量验证及加载日志见[BUNDLE16K_VALIDATION.md](BUNDLE16K_VALIDATION.md)。同输入A/B/A实测旧包16.621token/s、新包4.148token/s，下降75.04%，详见[decode对照](results/decode_ab_20261108_015225/REPORT.md)。
11月6日旧4K失败发生在首步4328纯文字tokens加双图，SDK-200103，见[当时日志](results/official_m3a_20261106_140057/phone_raw.log)。11月8日首轮诊断因swipe空参数与熄屏中止，见results/official_m3a_20261108_010333；它未计入本报告。
