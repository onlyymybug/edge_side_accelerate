# pace/code/QNN/scripts —— 单设备·单模型·单任务运行脚本

这里只保留当前使用的 GSM8K、MATH-500、BFCL 和 HammerBench 入口。已废弃的
W1/W2/W3 已废弃，不会被这些入口调用。目录总览见 `../../README.md`。

每个脚本 = **一个设备 × 一个模型 × 一个任务**,命名 `<device>_<model>_<task>.sh`。
共 2 设备 × 2 模型 × 5 任务 = 20 个,外加共享配置 `_env.sh`。

## 矩阵

| | math500 | gsm8k | bfcl | hammerbench-singleturn | hammerbench-multiturn |
|---|---|---|---|---|---|
| **oneplus · qwen**   | ✅ | ✅ | ✅ | ✅ | ✅ |
| **oneplus · bitnet** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **redmi · qwen**     | ✅ | ✅ | ✅ | ✅ | ✅ |
| **redmi · bitnet**   | ✅ | ✅ | ✅ | ✅ | ✅ |

## 用法

```bash
# 直接跑(前台)
./oneplus_qwen_math500.sh

# 冒烟:只跑前 3 条
./oneplus_qwen_math500.sh --limit 3

# 中断后续跑(reasoning 任务支持 --resume)
./redmi_bitnet_gsm8k.sh --resume

# BFCL 默认先 generate 再 evaluate;也可分开:
./oneplus_bitnet_bfcl.sh generate
./oneplus_bitnet_bfcl.sh evaluate
```

脚本末尾的 `"$@"` 会把额外参数透传给底层 runner,所以 `--limit / --resume / --run-id`
等都能直接追加。建议长任务放 `tmux` 里跑,避免 SSH 断连中断。

## 设备 / 模型 / 部署对照

| 设备 | serial | Qwen pte (balanced) | BitNet pte (burst) |
|---|---|---|---|
| oneplus (PLQ110)   | 192.168.110.67:5555 | `.../qwen25_3b_qnn237_balanced` | `.../single_llama/hybrid_llama_qnn_1024.pte` |
| redmi (annibale)   | 192.168.110.76:5555 | `.../qwen25_3b_qnn237_balanced` | `.../single_llama/hybrid_llama_qnn_1024.pte` |

- **Qwen 两台都用 balanced 档**:redmi 上没有部署 block16 Qwen,统一用 balanced 保证跨设备可比。
- **HammerBench 统一用 `en_mod20_eq0` 数据集**(与 redmi 已有结果口径一致)。

## token 预算(可在命令前用环境变量覆盖)

默认见 `_env.sh`:`GSM8K_BUDGET=128`、`MATH500_BUDGET=256`。BFCL 不设独立
output budget，Qwen 和 BitNet 都生成至 EOS 或 PTE 的 `MAX_SEQ_LEN=1024` 上限。

BFCL 的新结果中，Qwen 写入
`results/QNN/<device>/bfcl/<model>/<mode>/output_budget_no_budget/`，BitNet raw-v1
写入对应的 `output_budget_no_budget_raw_v1/`，避免与旧结果混淆。
HammerBench 官方 Appendix-D prompt 的默认 run-id
对 Qwen 为 `<mode>_official_prompt_v2`，对 BitNet 为
`<mode>_official_prompt_v5`。BitNet v5 使用支持 `--raw_prompt` 的 QNN 2.28
runner，将模型 tokenizer 自带的原生 `System:/User:/Assistant:` chat template
直接送入 tokenizer，不再添加外层 Llama header。需要自定义 run-id 时可设置
`HAMMERBENCH_RUN_ID`。

所有 BitNet 入口现在统一使用 QNN 2.28 的
`qnn_llama_runner_raw_prompt --raw_prompt` 和 `bitnet_native_raw_v1` 模板；
GSM8K/MATH-500/BFCL 的新结果目录追加 `_raw_v1`，避免与旧 Llama-wrapper
结果混合。旧 Llama-wrapper 的 prompt-token calibration 不能用于 raw-v1；
需要精确物理 output budget 时，应先用 raw-v1 结果中的 runner 实测计数生成新
calibration。

HammerBench 的 `end_to_end_acc` 使用 benchmark-aware 口径：普通样例要求函数名
和参数同时正确，`ir-*` 样例要求正确拒绝。single-turn 另报
`function_irrelevant_accuracy`，multi-turn 另报 conversation 级 `progress_rate` 和
`success_rate`。已有推理无需重跑，可离线重评（原汇总自动备份为
`*.pre_official_metrics_v3`）：

```bash
./code/QNN/scripts/rescore_hammerbench.py /path/to/hammerbench/result_dir
```

```bash
# 例:用更大的 math500 预算跑,拿真实准确率
MATH500_BUDGET=512 ./oneplus_qwen_math500.sh
```

> ⚠️ 当前默认预算下,GSM8K/MATH500 大量样本会顶满预算被截断,
> 绝对准确率偏低(见 docs/week2/0806.md)。要真实推理准确率需调大预算重跑。

## 结果落盘

所有入口统一写入以下层级：

```text
pace/results/QNN/<device>/<task>/<model>/<mode>/
pace/results/QNN/<device>/{gsm8k,math500}/<model>/<mode>/output_budget_<budget>/
```

- `device`: `oneplus`、`redmi`
- `task`: `bfcl`、`gsm8k`、`hammerbench-multi`、`hammerbench-single`、`math500`
- `model`: `bitnet-2B-w4a16`、`qwen2.5-3B-w4a16`
- `mode`: `burst`、`balanced`、`low_power`
- 无显式输出预算时使用 `output_budget_no_budget`

GSM8K/MATH500 的 latency 同时保存两套边界：现有 `ttft_ms`、
`inference_ms`/`e2e_latency_ms` 不包含模型加载，用于保持历史可比性；
`model_load_ms` 单独保存加载时间；新增
`cold_start_ttft_ms = model_load_ms + ttft_ms` 和
`cold_start_e2e_latency_ms = model_load_ms + inference_ms`，用于报告逐题启动
runner 时的真实冷启动体验。现有 energy、J/token 和 EDP 仍只覆盖 inference
window，不包含模型加载能量。

GSM8K/MATH500 可将对应预算变量设为 `no_budget`。此时不设置独立的
输出 token 上限，每题的 `seq_len` 使用 PTE 的 `MAX_SEQ_LEN`，生成在 EOS
处提前结束，最晚在模型上下文上限处停止。例如：

```bash
GSM8K_BUDGET=no_budget ./code/QNN/scripts/redmi_bitnet_gsm8k.sh
MATH500_BUDGET=no_budget ./code/QNN/scripts/oneplus_qwen_math500.sh
```

Redmi 上已部署的 Llama 3.1 8B Hybrid-1024 可使用：

```bash
LLAMA_PERFORMANCE_MODE=balanced GSM8K_BUDGET=512 ./code/QNN/scripts/redmi_llama_gsm8k.sh
LLAMA_PERFORMANCE_MODE=burst MATH500_BUDGET=no_budget ./code/QNN/scripts/redmi_llama_math500.sh
```

OnePlus 上部署在 `llama31_8b_qnn237_dynamic` 的 Llama 3.1 8B 可运行
GSM8K/MATH-500。单组运行时通过环境变量选择预算：

```bash
LLAMA_PERFORMANCE_MODE=balanced GSM8K_BUDGET=256 ./code/QNN/scripts/oneplus_llama_gsm8k.sh
LLAMA_PERFORMANCE_MODE=balanced GSM8K_BUDGET=512 ./code/QNN/scripts/oneplus_llama_gsm8k.sh
LLAMA_PERFORMANCE_MODE=balanced GSM8K_BUDGET=no_budget ./code/QNN/scripts/oneplus_llama_gsm8k.sh

LLAMA_PERFORMANCE_MODE=balanced MATH500_BUDGET=256 ./code/QNN/scripts/oneplus_llama_math500.sh
LLAMA_PERFORMANCE_MODE=balanced MATH500_BUDGET=512 ./code/QNN/scripts/oneplus_llama_math500.sh
LLAMA_PERFORMANCE_MODE=balanced MATH500_BUDGET=no_budget ./code/QNN/scripts/oneplus_llama_math500.sh
```

也可以依次运行上述六组（首次运行，不带 `--resume`）：

```bash
LLAMA_PERFORMANCE_MODE=balanced ./code/QNN/scripts/oneplus_llama_math_sweep.sh
```

批量脚本中断后，请对已经创建结果目录的具体任务使用对应单任务脚本加
`--resume`，不要直接给整个 sweep 追加 `--resume`。

脚本只会在实际启动实验时创建对应目录，不会预先创建空组合。
