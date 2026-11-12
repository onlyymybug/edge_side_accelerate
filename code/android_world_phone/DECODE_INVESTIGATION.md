# 16K decode 下降的实测与排查线索

2026-11-09因果复核见[DECODE_CAUSAL_REVIEW.md](DECODE_CAUSAL_REVIEW.md)：旧包短输入预计走1K图，表内4K形状只是metadata代表值；不能用“最大CL增4倍→延迟增4倍”或“逐token复制整块KV”作定论。固定大图仍是有机制支持的主因假设，具体瓶颈待实际graph trace与算子profiling。

2026-11-08。初步结论：新包逐 token 生成明显变慢；同输入A/B/A已完成，结论见末节。

## 当前观测

旧4K简化代理67次请求，SDK decode速度中位数17.2355token/s，合并generated_tokens/decode_seconds为17.24966token/s；请求输入832–1815tokens。
旧4K双图诊断：输入558tokens，输出91tokens，decode17.2689token/s。

新16K短请求：decode5.155–6.586token/s；正式M3A首个动作输入4859tokens、输出59tokens，decode4.75841token/s。
前8个完整M3A步骤的所有动作/总结请求合并decode约4.52832token/s；较长输出有一次4.22007token/s。
这些对比尚未控制提示词/图片/输出长度/运行次序，不能给出精确的单因素因果归因。

## 配置证据

同一手机、同一CAPI runner的功耗模式为BURST。旧、新手机部署的 libgeniex.so 与tokenizer.json 哈希相同，运行库均GenieX v0.10.0 / QAIRT2.50。
新包是真正重新编译的16K图；prefill AR128、decode AR1。

| AR1声明输入 | 旧4K | 新16K |
|---|---|---|
| 一个past_key的shape | [8,1,128,4095] | [8,1,128,16383] |
| past_key dtype | uint8 | uint8 |
| attention_mask | [1,1,1,4096] | [1,1,1,16384] |
| 36层K/V声明输入元素的字节数之和 | 301,916,160 | 1,207,885,824 |

上述字节数只是声明张量shape的总和，不是进程RSS、实测DSP内存峰值，也不证明每token实际搬运了这些字节。
固定图的KV/attention维度扩大约4倍，是进一步profiling的线索；性能下降还可能涉及编译策略、kernel、PD映射/分片及温度/频率等，尚未分离。
加载中的QNN8003/1002错误随后恢复，不能仅据此认定它导致decode变慢。

本地原始证据：[图布局对比](results/bundle16k_20261108_0105/graph_layout_comparison.json)、[复用资产哈希](results/bundle16k_20261108_0105/reused_assets_sha256.txt)、[旧67请求](results/20261106_123655/timing_analysis/requests.csv)、[新首个动作](results/official_m3a_20261108_010622/rep01_folder/00/action/response.json)。

官方说明：GenieX的QAIRT上下文容量由编译bundle确定，不能通过运行时nctx扩大，见 [GenieX CLI reference](https://github.com/qualcomm/GenieX/blob/main/docs/en/run/cli/reference.mdx)。
AI Hub导出将AR1用于生成、AR128用于prefill，并适配KV输入/输出布局，见 [LLM onboarding](https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/onboarding.md)。这些文档解释配置语义，不提供本次16K下降的实测根因。

## 同输入对照

`compare_decode_bundles.py` 使用同一组两张截图、同一英文提示词、同runner/BURST、temperature0、seed42、max_new_tokens192。
4K→16K→4K顺序测试，每个session2请求，每次重置KV，保存ready、完整响应、服务日志和SDK分段耗时。它仍是单机小样本，不能完全排除温度与测试次序影响。

## 同输入A/B/A最终结果

[完整报告](results/decode_ab_20261108_015225/REPORT.md)、[summary.json](results/decode_ab_20261108_015225/summary.json)。

4K前两次17.1253/16.2251token/s；16K两次4.13078/4.16598token/s；4K后两次16.2932/16.8739token/s。合并旧包16.62063，新包4.14830，吞吐下降75.04%，每token耗时为原来的4.0066倍。六次输入558tokens、输出91tokens，输出文本完全一致。

所有decode指标均来自手机侧SDK，未跨钟相减。本轮发现主机/手机钟速不同，不影响这组同手机计时的decode比值；官方空闲率也已改用各时钟内部的busy/idle区间。

实测说明整个导出包的性能下降可在相同短输入复现；具体kernel/内存机制仍需单独profiling，KV声明输入约4倍只是线索。
