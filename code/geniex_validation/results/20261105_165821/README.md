# GenieX + QAIRT 真机验证结果

包：`/home/lyyyy/edge_side_work/phone_bundle`。设备：REDMI K90 / SM8750 / Android 16。
模型 v0.66.0、GenieX v0.10.0、QAIRT 2.50.0.260828221209、W4A16、4096 context、UINT8 KV、512×512 图像。
本地及手机 MANIFEST 校验通过。原包内容保持不变；测试使用独立设备目录 `/data/local/tmp/qwen3_vl_geniex_sm8750`。

## 实际结果

原包三条文本 prompt 全部正常 EOS 完成，算术回答 `42`，JSON 回答 `{"city": "北京"}`，中文说明用途正常。
带图三条也正常 EOS 完成，第三条正确描述“图片显示了一个红色正方形”。

下表为默认 error 日志等级的 SDK 指标，info 复测用于诊断，不混入性能统计：

| 请求 | TTFT ms | media ms | 生成 token | SDK decode token/s | TTFT + decode 秒 |
|---|---:|---:|---:|---:|---:|
| text #1 | 63.474 | 0.000 | 3 | 33.409 | 0.153 |
| text #2 | 60.809 | 0.000 | 7 | 27.123 | 0.319 |
| text #3 | 60.883 | 0.000 | 22 | 23.937 | 0.980 |
| image #1 | 323.973 | 105.061 | 3 | 22.734 | 0.456 |
| image #2 | 330.227 | 102.152 | 10 | 17.567 | 0.899 |
| image #3 | 343.793 | 102.977 | 24 | 17.910 | 1.684 |

文本 TTFT 中位数 60.883ms、decode 中位数 27.123 token/s；带图 TTFT 中位数 330.227ms、decode 中位数 17.910 token/s。
图像 media 约 102–105ms，已包含在带图 TTFT 内，不应再重复相加。
这是几个短请求，不是持续吞吐或温控压力测试。

从启动 shell 到第一条完整回答打印：文本 7.914 秒；带图 7.657 秒。
其中启动文件校验约 2.34–2.37 秒；其余包含 runtime/model 初始化、第一条推理及输出开销。
没有独立 API 初始化计时，不能将上述时间直接标为模型 load time。
`[sep ]` 在 stdout 缓冲中随回答一起刷出，也不能用它计算准确的初始化结束时刻。

## QNN 初始化异常的核对

原始 error 日志保留了 context 4 在 PD0 加载失败、5005 的异常。
info 复测确认随后启用 PD2，并成功加载同一个 context 0x4。
共观察到 5 次成功 context 创建（1 个视觉、4 个文本），全部在首次回答打印之前；后续两个请求没有新的 context 创建。
原始异常没有删除，也没有将它误判成整个模型加载失败。

另一个 `failed to get plugin version for llama_cpp` 出现在 QAIRT 测试完成后查询版本时；本次只装了 QAIRT plugin，主结果依然标记 `plugin=qairt`、`device=npu`。

## 原始证据

- `text/phone_raw.log`、`text/smoke.json`：文本原始回答与 SDK 指标。
- `image/phone_raw.log`、`image/smoke.json`：图片原始回答与 SDK 指标。
- `image_info/phone_raw.log`、`image_info/phone_raw_clean.log`：详细 PD 重试与 context 生命周期。
- 每个目录还有 `command.json`、`line_times.jsonl`、`host_summary.json`。
- `VALIDATION_REPORT.json`：结构化汇总。

SDK JSON 的 `model_path` 被解析成 tokenizer.json，`model_size_bytes` 因而显示 tokenizer 大小，不能当作模型大小；实际上传包共 4,595,172,966 bytes。
`prompt_tokens` 和 prefill 吞吐按官方 benchmark 的 128-token chunk 规则计数，不是原始自然语言 token 数。

## 实际启动命令

```bash
adb -s 189f0fad shell 'cd /data/local/tmp/qwen3_vl_geniex_sm8750 && chmod +x bin/geniex-bench && sh run_smoke.sh'
adb -s 189f0fad shell 'cd /data/local/tmp/qwen3_vl_geniex_sm8750 && chmod +x bin/geniex-bench && sh run_smoke.sh --image /data/local/tmp/qwen3_vl_geniex_sm8750/test_red_square.png'
```

启动脚本内 benchmark 参数为：

```sh
./bin/geniex-bench --plugin qairt --device npu --vlm \
  -m "$PWD/model" --accuracy --prompt-file "$PWD/prompts.txt" \
  -n 128 --temperature 0 --seed 42 \
  --output-json "$PWD/results/smoke.json"
```

`--accuracy` 在这个版本中强制 warmup=0、repetitions=1；每个进程共三条独立请求，共用一个 VLM handle，并在请求之间 reset。
它不是三个独立保留 KV 的 Session，也没有验证满 4096 context、6144 context 或 AndroidWorld。

官方 benchmark 计时语义与生命周期来源：
https://github.com/qualcomm/GenieX/blob/v0.10.0/sdk/benchmark/run.c
