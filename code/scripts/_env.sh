# Shared configuration for pace/code/scripts/*.sh
# Sourced by every per-(device,model,task) runner script.
#
# 设备:
#   oneplus (PLQ110)   192.168.110.67:5555
#   redmi   (annibale) 192.168.110.76:5555
#
# 模型 / pte 部署 (设备端):
#   Qwen2.5-3B  balanced : /data/local/tmp/lyy/executorch/qwen25_3b_qnn237_balanced  (两台都有)
#   BitNet-2B   burst    : /data/local/tmp/lyy/executorch/single_llama/hybrid_llama_qnn_1024.pte (两台都有)
# 注: redmi 上没有 block16 Qwen,故两台的 Qwen 统一用 balanced,保证可比。

export PACE_ROOT="/home/lyyyy/pace"
export QWEN_CODE="${PACE_ROOT}/code/qwen"
export BITNET_SCRIPTS="${PACE_ROOT}/code/bitnet/scripts"
export BITNET_CODE="${PACE_ROOT}/code/bitnet"

export ADB_BIN="adb"
export ADB_SERVER_PORT="5037"
# 某些环境残留的 ADB_SERVER_SOCKET 会覆盖 -P 端口,统一清掉。
unset ADB_SERVER_SOCKET

# ---- 按硬件序列号自动发现设备 (IP 变了也不用改脚本) ----
# 手机 WiFi 重连 / DHCP 续租会让 <ip>:5555 变化,但 ro.serialno 永不变。
# 遍历 adb devices,对每台读 ro.serialno,匹配到目标序列号就返回其当前
# 传输地址 (network 连接时是 <ip>:5555,USB 直连时是硬件序列号本身)。
#
# 硬件序列号 (adb -s <s> shell getprop ro.serialno):
#   oneplus (PLQ110)     3B15B800A0R00000
#   redmi   (annibale)   189f0fad
export ONEPLUS_HW_SERIAL="${ONEPLUS_HW_SERIAL:-3B15B800A0R00000}"
export REDMI_HW_SERIAL="${REDMI_HW_SERIAL:-189f0fad}"

resolve_serial_by_hwserial() {
  # $1 = 目标 ro.serialno。找不到时返回非 0 并打印提示到 stderr。
  # 每台 getprop 最多重试 3 次:网络 adb 首连后紧接着查第二台偶发超时。
  local target="$1" s hw attempt
  local serials
  serials="$("${ADB_BIN}" -P "${ADB_SERVER_PORT}" devices | awk 'NR>1 && $2=="device"{print $1}')"
  # USB transport 的 adb serial 就是硬件序列号。若同一手机同时保留了
  # USB 和 Wi-Fi ADB 连接，优先选择 USB，避免采样命令误走无线链路。
  for s in ${serials}; do
    if [[ "${s}" == "${target}" ]]; then
      echo "${s}"
      return 0
    fi
  done
  for s in ${serials}; do
    for attempt in 1 2 3; do
      hw="$("${ADB_BIN}" -P "${ADB_SERVER_PORT}" -s "${s}" shell getprop ro.serialno 2>/dev/null | tr -d '\r')"
      [[ -n "${hw}" ]] && break
      sleep 0.5
    done
    if [[ "${hw}" == "${target}" ]]; then
      echo "${s}"
      return 0
    fi
  done
  echo "resolve_serial_by_hwserial: 未找到序列号为 ${target} 的在线设备" >&2
  return 1
}

# ONEPLUS_SERIAL / REDMI_SERIAL 若已由环境显式指定则直接沿用 (手动覆盖),
# 否则按硬件序列号自动解析当前传输地址。
export ONEPLUS_SERIAL="${ONEPLUS_SERIAL:-$(resolve_serial_by_hwserial "${ONEPLUS_HW_SERIAL}" || true)}"
export REDMI_SERIAL="${REDMI_SERIAL:-$(resolve_serial_by_hwserial "${REDMI_HW_SERIAL}" || true)}"

# HammerBench 统一数据集口径 (与 redmi 已有结果一致,保证跨设备可比)。
export HAMMERBENCH_DATASET_ROOT_EQ0="/home/lyyyy/HammerBench/en_mod20_eq0"

# ---- 各任务 token 预算 (集中在此,便于统一调整) ----
export GSM8K_BUDGET="${GSM8K_BUDGET:-128}"
export MATH500_BUDGET="${MATH500_BUDGET:-256}"

# Qwen 使用支持运行时覆盖 HTP performance mode 的新 runner。
# 目录名、manifest 标签和传给 runner 的枚举值由同一个变量生成，避免
# 出现实际跑 burst、结果却落在 balanced 目录的情况。
export QWEN_PERFORMANCE_MODE="${QWEN_PERFORMANCE_MODE:-balanced}"
case "${QWEN_PERFORMANCE_MODE}" in
  burst)     export HTP_PERFORMANCE_MODE=2 ;;
  balanced)  export HTP_PERFORMANCE_MODE=8 ;;
  low_power) export HTP_PERFORMANCE_MODE=5 ;;
  *)
    echo "QWEN_PERFORMANCE_MODE must be burst, balanced, or low_power" >&2
    return 2
    ;;
esac
export HTP_PERFORMANCE_MODE_NAME="${QWEN_PERFORMANCE_MODE}_runtime_override"
export QNN_RUNNER_NAME="qnn_llama_runner_htp_mode"
export QNN_RUNNER_LOCAL_PATH="/home/lyyyy/qwen25_runner_htp_mode/qnn_llama_runner"

# Llama 3.1 8B uses the same QNN 2.37 dynamic-HTP runner/backend as Qwen,
# with decoder_model_version=llama3.  It is currently deployed on Redmi only.
export LLAMA_PERFORMANCE_MODE="${LLAMA_PERFORMANCE_MODE:-balanced}"
case "${LLAMA_PERFORMANCE_MODE}" in
  burst)     export LLAMA_HTP_PERFORMANCE_MODE=2 ;;
  balanced)  export LLAMA_HTP_PERFORMANCE_MODE=8 ;;
  low_power) export LLAMA_HTP_PERFORMANCE_MODE=5 ;;
  *)
    echo "LLAMA_PERFORMANCE_MODE must be burst, balanced, or low_power" >&2
    return 2
    ;;
esac
export LLAMA_HTP_PERFORMANCE_MODE_NAME="${LLAMA_PERFORMANCE_MODE}_runtime_override"

# BitNet has its own runtime-selectable QNN 2.28 stack.  Keep these namespaced:
# bitnet/config.env maps them to the generic runner variables after it is
# sourced, so Qwen's default Balanced mode cannot leak into BitNet runs.
export BITNET_PERFORMANCE_MODE="${BITNET_PERFORMANCE_MODE:-burst}"
case "${BITNET_PERFORMANCE_MODE}" in
  burst)     export BITNET_HTP_PERFORMANCE_MODE=2 ;;
  balanced)  export BITNET_HTP_PERFORMANCE_MODE=8 ;;
  low_power) export BITNET_HTP_PERFORMANCE_MODE=5 ;;
  *)
    echo "BITNET_PERFORMANCE_MODE must be burst, balanced, or low_power" >&2
    return 2
    ;;
esac
export BITNET_HTP_PERFORMANCE_MODE_NAME="${BITNET_PERFORMANCE_MODE}_runtime_override"

# Canonical result hierarchy:
# results/<device>/<task>/<model>/<mode>[/output_budget_<budget>]
export BITNET_RESULT_MODEL_DIR="bitnet-2B-w4a16"
export QWEN_RESULT_MODEL_DIR="qwen2.5-3B-w4a16"
export LLAMA_RESULT_MODEL_DIR="llama3.1-8B-qnn237-hybrid-1024"

output_budget_dir() {
  case "$1" in
    none|no_budget) echo "output_budget_no_budget" ;;
    *) echo "output_budget_$1" ;;
  esac
}
