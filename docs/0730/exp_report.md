# 7.30

## 1. 实验

### 1.1 实验环境

- 设备：OnePlus PLQ110
- SoC：Qualcomm **SM8750**
- QNN SDK/runtime：**2.28.0**

### 1.2 实验进度

- 配置了transformer模型转换为适配手机SoC的环境，将BitNet-2B模型转换，得到对应pte格式与runtime文件
- 转换配置：

| Baseline | Prefill 配置 | Decode 配置 | 粒度        |
| -------- | ------------ | ----------- | ----------- |
| QNN      | `W4A16`      | `W4A16`     | per-channel |

- 模型传到手机并进行了单次的实验
- 部分原始日志：

```
[INFO] [Qnn ExecuTorch]: Deserializing processed data using QnnContextCustomProtocol
[INFO] [Qnn ExecuTorch]: QnnDlcManager Get Qnn Context blob bytes 1887300424
[INFO] [Qnn ExecuTorch]: create QNN Logger with log_level 2
[WARNING] [Qnn ExecuTorch]:  <W> Initializing HtpProvider

...

PyTorchObserver {"prompt_tokens":18,"generated_tokens":23,"model_load_start_ms":1785325413512,"model_load_end_ms":1785325414527,"inference_start_ms":1785325414527,"inference_end_ms":1785325415665,"prompt_eval_end_ms":1785325415025,"first_token_ms":1785325415054,"tokens_per_second":35.9375,"aggregate_sampling_time_ms":1687561,"SCALING_FACTOR_UNITS_PER_SECOND":1000}
```

- 单次输出：

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

User: Who are you? Assistant:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I am an AI assistant designed to help you find information and answer your questions. How can I assist you today?<|eot_id|>
```

- 记录了tokens/s：

```
Tokens per second:
20.2109
```

### 1.3 后续实验

- 现在只对齐了T-MAN中的配置，接下来完善全部输出
- 在给定数据集上进行实验
