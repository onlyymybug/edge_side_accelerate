# pace/code/scripts —— 单设备·单模型·单任务 运行脚本

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

BFCL 的新结果写入：
`results/<device>/bfcl/<model>/<mode>/output_budget_no_budget/`，避免与旧的
BitNet budget-256 结果混淆。HammerBench 官方 Appendix-D prompt 的默认 run-id
为 `<mode>_official_prompt_v2`；需要自定义时可设置 `HAMMERBENCH_RUN_ID`。

```bash
# 例:用更大的 math500 预算跑,拿真实准确率
MATH500_BUDGET=512 ./oneplus_qwen_math500.sh
```

> ⚠️ 当前默认预算下,GSM8K/MATH500 大量样本会顶满预算被截断,
> 绝对准确率偏低(见 docs/week2/0806.md)。要真实推理准确率需调大预算重跑。

## 结果落盘

所有入口统一写入以下层级：

```text
pace/results/<device>/<task>/<model>/<mode>/
pace/results/<device>/{gsm8k,math500}/<model>/<mode>/output_budget_<budget>/
```

- `device`: `oneplus`、`redmi`
- `task`: `bfcl`、`gsm8k`、`hammerbench-multi`、`hammerbench-single`、`math500`
- `model`: `bitnet-2B-w4a16`、`qwen2.5-3B-w4a16`
- `mode`: `burst`、`balanced`、`low_power`
- 无显式输出预算时使用 `output_budget_no_budget`

GSM8K/MATH500 可将对应预算变量设为 `no_budget`。此时不设置独立的
输出 token 上限，每题的 `seq_len` 使用 PTE 的 `MAX_SEQ_LEN`，生成在 EOS
处提前结束，最晚在模型上下文上限处停止。例如：

```bash
GSM8K_BUDGET=no_budget ./scripts/redmi_bitnet_gsm8k.sh
MATH500_BUDGET=no_budget ./scripts/oneplus_qwen_math500.sh
```

Redmi 上已部署的 Llama 3.1 8B Hybrid-1024 可使用：

```bash
LLAMA_PERFORMANCE_MODE=balanced GSM8K_BUDGET=512 ./scripts/redmi_llama_gsm8k.sh
LLAMA_PERFORMANCE_MODE=burst MATH500_BUDGET=no_budget ./scripts/redmi_llama_math500.sh
```

脚本只会在实际启动实验时创建对应目录，不会预先创建空组合。
