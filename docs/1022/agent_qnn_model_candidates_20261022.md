# 手机 Agent 实验的 QNN 模型候选（2026-10-22）

当前手机通过 ADB 读到 `SM8750`、约 16 GB RAM；现有链路是 ExecuTorch + QNN/HTP 的常驻多会话 `.pte` runner，模型为 Qwen2.5-3B-Instruct。AppWorld 单任务实验中模型能在补充观察后到达目标，但多次输出无效 JSON 或多余参数。因此模型替换首先要解决**工具协议稳定性**，同时保持系统实验所需的常驻、多会话、KV 生命周期接口。

| 优先级 | 模型 | 选择理由 | 转换路径与注意点 |
| --- | --- | --- | --- |
| 1 | [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 官方模型卡明确提供 JSON/XML 工具调用模板；参数规模与当前模型接近，适合隔离模型能力因素 | [ExecuTorch Qualcomm 示例](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/README.md)列为支持模型，提供 hybrid 导出示例。必须按其原生 chat/tool template 重新实现 host 提示与解析，不能继续套用 Qwen 模板。 |
| 2 | [Llama 3.2 3B Instruct](https://github.com/pytorch/executorch/blob/main/examples/qualcomm/oss_scripts/llama/README.md) | 与当前规模接近，可作架构/模型对照 | ExecuTorch Qualcomm 提供 hybrid 示例；实际工具调用质量需在同一任务协议下实测，不能预设它更好。 |
| 3 | [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 另一种小模型对照；ExecuTorch Qualcomm 提供 hybrid 示例 | 官方模型卡也提醒函数调用可能臆造函数名或 URL，不能把它当作已解决问题的模型。 |
| 独立路线 | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 官方模型卡强调 Agent/tool calling，非 thinking 的 instruct 版本更适合短工具轮次 | [Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models/qwen3_4b_instruct_2507)列出 QAIRT/GenieX 部署路线，但这**不是**现有 ExecuTorch `.pte` 的直接替换件；页面也未确认这台 Redmi 的精确型号。需另行验证同等多会话/KV 控制能力后才能用于系统对照。 |

建议先在服务器转换 **SmolLM3-3B**，保留当前 Qwen2.5-3B 作为基线。验收不只是能生成文字：必须支持当前 `get_capabilities/create_session/run_request/evict_session` 协议、模型只加载一次、独立 session 的 KV 不串扰、至少覆盖所选 AppWorld 任务的上下文长度，并交付匹配的 tokenizer/chat template。手机端先做 2–3 条小型功能检查，比较原始工具调用合法率、任务成功率与 TTFT/解码时延；通过后才安排大批次测试。

注意：Qualcomm AI Hub 上“列出某模型”不等于这台手机、当前 QNN SDK 版本与自定义常驻 runner 已完成兼容性验证。新模型的量化、上下文长度、KV 占用、热稳态吞吐都需要实测。
