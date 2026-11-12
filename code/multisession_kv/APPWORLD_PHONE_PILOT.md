# AppWorld 手机闭环可行性实验（2026-10-22）

目标：检验现有 Redmi + Qwen2.5-3B-Instruct QNN 常驻 runner 能否接入公开、可执行的 Agent 环境。此实验只验证接线与单任务完成，不评估 KV 策略或提出的方法。

## 环境与协议

- AppWorld 0.1.3.post1，官方 `dev` 数据；数据在 `/home/lyyyy/edge_side_work/appworld_data`，Python 环境在 `/home/lyyyy/edge_side_work/.venv-appworld`。
- 手机通过 ADB 调用已有的 ctx6144 QNN bundle。手机侧执行 LLM 推理；AppWorld 环境、工具调用验证、计时和评估在主机侧执行。
- 单任务 `6bdbc26_1`：查询 Spotify 当前歌曲的歌手粉丝数。每次模型输出一个 JSON 工具调用，适配器仅允许预设 API，并将参数按 API schema 校验后交给 AppWorld。
- 成功的一次使用 `--bootstrap-auth`：Agent Service 先用环境中的账户信息完成登录，并在需要的 API 调用中附加令牌。模型并未生成登录操作；因此这不是 AppWorld 官方端到端协议。

## 当前结果

先前几次调试暴露了 JSON 格式不合规、选错 API、把歌曲字段误当成歌手 API 参数等问题。调整提示及适配器后，`results/appworld_phone_pilot_20261022_06/summary.json` 记录了 3 次手机推理，AppWorld 对该任务的 2/2 项检查通过，总耗时约 12.47 秒。三次调用依次为 `show_current_song → show_artist → complete_task`。

这次成功依赖 1 次可见的 JSON 键名修复，以及 1 次按工具 schema 丢弃多余参数。修复/过滤均记录在 `events.jsonl`。之前的调试尝试和这次成功共享同一任务，不能据此计算成功率或声称模型稳定完成 AppWorld。

固定当时的查询配置后，另两个相同任务模板、不同数据实例 `6bdbc26_2` 和 `6bdbc26_3` 各通过一次（分别约 10.66、12.21 秒）。两者都需要一次 JSON 修复，后者还需要一次多余参数过滤。这是同模板实例验证，不是跨任务泛化评估。

另试了动作型任务 `b119b1f_3`（逐首切歌至目标歌手）。原始 `previous_song` 只返回歌曲 ID，模型会反复切歌而不主动查询歌手。一个服务侧的“观察增强”变体在每次切歌后自动调用 `show_current_song`，把歌手信息合并进工具结果；模型随后在 4 次切歌后到达目标。模型结束时为动作任务附带了不该有的答案，因此另需按 AppWorld 的 API 约束将答案规范为 `None` 才通过。固定该规范后的单次对照：无观察增强在第 3 次推理因 JSON 格式错误失败（约 12.58 秒），有观察增强在第 5 次推理完成任务（约 14.43 秒）。不能把失败运行的完成时间与成功运行直接比较，也不能据此声称观察增强有统计收益；这是接口/模型行为的诊断样例。

## 复现

在项目目录执行（`SERIAL` 换成当前手机 ADB 地址，实验名和输出目录每次必须新建）：

```bash
cd /home/lyyyy/edge_side_work/code/multisession_kv
PY=/home/lyyyy/edge_side_work/.venv-appworld/bin/python
SERIAL=192.168.12.218:5555
adb -s "$SERIAL" get-state
"$PY" appworld_phone_pilot.py \
  --serial "$SERIAL" \
  --task-id 6bdbc26_1 \
  --experiment-name agent_phone_pilot_repeat_01 \
  --output-dir results/appworld_phone_pilot_repeat_01 \
  --bootstrap-auth --max-steps 5
"$PY" -m unittest -v test_appworld_phone_pilot test_agent_closed_loop
```

不要把这一任务的通过率当成论文结果。下一步先冻结接口与提示，选择未用于调试的多个任务，分别报告真实任务成功率、格式修复率、工具参数过滤率和手机推理时延；再引入可检验的 Agent–Inference 协同策略及对照组。

动作型任务的诊断运行可选 `--profile song_navigation --task-id b119b1f_3`；加 `--enrich-navigation` 启用服务侧观察增强。每次运行都要使用新的 `--experiment-name` 和 `--output-dir`。正式比较需在未用过的任务上成对运行，且冻结代码版本、温度与电量条件，并把额外 API 调用及其耗时计入成本。

## 模型能力与提示协议的进一步诊断

当前 bundle 的 `tokenizer_config.json` 没有 `chat_template`，但另附 `device/chat_template.jinja`，`tokenizer.json` 也包含 `<tool_call>` token。根据 [Qwen2.5 官方函数调用说明](https://qwen.readthedocs.io/en/v2.5/framework/function_call.html)及[官方模型配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/tokenizer_config.json)，先新增了 `--prompt-style qwen_hermes`（基于文档手写的近似格式），随后新增 `--prompt-style qwen_bundle_template`（逐字渲染 bundle 自带 Jinja 模板）。

在查询型任务 `6bdbc26_1` 上，该风格成功，3 次推理无需 JSON 修复/参数过滤；但第一轮模型一次输出了 **3 个**工具调用，适配器只执行首个。runner 实际生成 69 token；第一个 `</tool_call>` 约在第 18 token，后续约 51 token 属于不应在看到工具结果前生成的内容。这只是离线 token 估计，未实测提前停止后的时延/能耗。另一个动作型任务 `b119b1f_3` 的 Hermes 风格首轮仍生成不可解析的参数 JSON，任务失败。可见换提示有帮助，但不能把全部问题归因于提示，也不能断言换模型必然解决。

后续 runner 值得支持“第一个完整工具调用结束即停”的受控生成，并把被忽略的后续工具调用数、浪费 token 数和实际提前停止收益作为独立指标。当前代码在新运行中会记录 `ignored_extra_calls`，不会把额外调用悄悄当作成功。

用 bundle 的原样模板再测同一查询任务：首轮模型遗漏 JSON 外层花括号；允许记录式修复后，前两次调用可执行，但第三次在自然语言后输出了未闭合的 `<tool_call>`，任务仍失败。由此不能断言之前的错误全是手写模板造成的；也不能仅凭单任务认定模型整体能力不足。格式/工具选择/任务规划需分别测量。
