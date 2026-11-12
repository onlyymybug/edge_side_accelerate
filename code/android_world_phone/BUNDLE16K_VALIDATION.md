# 新 16K bundle 真机验证

2026-11-08，REDMI K90 / SM8750 / 189f0fad。

`/home/lyyyy/edge_side_work/phone_bundle` 当前为 Qwen3-VL-4B-Instruct W4A16，GenieX v0.10.0 / QAIRT 2.50.0.260828221209，context16384、prefill AR128、decode AR1。
云端实际 graph 名为 `ar128_cl16384_*` 和 `ar1_cl16384_*`，KV 输入分别含16256和16383的长度维，已重新编译，不是只改JSON。旧4K包的 libgeniex.so 和 tokenizer.json 与新版 SHA256 相同。

完整本地/手机 MANIFEST 校验通过，4.3GB 文件传到独立目录 `/data/local/tmp/qwen3_vl_geniex_ctx16384`；旧4K手机目录保持。

## 短请求及长输入

- 三次短请求返回42、合法城市JSON、中文用途说明，均 EOS。TTFT0.758–0.783秒，decode5.155–6.586token/s。
- 长输入按要求返回OK，EOS；tokenizer chat template估计14503tokens，SDK prompt_tokens14592（采用其运行时统计）。prefill86.525826秒、decode0.286187秒、生成2tokens。
- 长输入只验证容量与生成链路，不用于证明长文本理解、检索或复杂推理质量。
- 两个脚本的模型加载期间都出现 QNN `map result:8003`、context err1002，之后恢复并成功生成。根因尚未确认，不能描述成无错误加载，也不能由这些行直接判定最终推理失败。
- `long_process_status.txt` 为单次进程内存快照；RSS不包含所有DSP/DMA内存，不能当成整机峰值内存。

原始命令和校验记录见 [validation.json](results/bundle16k_20261108_0105/validation.json)。[短请求日志](results/bundle16k_20261108_0105/smoke_phone_raw.log)、[短请求JSON](results/bundle16k_20261108_0105/smoke.json)、[长输入日志](results/bundle16k_20261108_0105/long_phone_raw.log)、[长输入JSON](results/bundle16k_20261108_0105/long.json)。未改交付 bundle 内的 `phone_tested:false` 记录，实测结果单独保存于本项目。

## 官方 M3A 重试

诊断运行 `official_m3a_20261108_010333` 已证明首个官方双图请求可生成：4859输入tokens，93输出tokens，reset/template/generate48.8481秒，其中TTFT29.387430秒、decode19.252564秒（4.83053token/s），加载6.857309秒。
该诊断随后暴露真机转接层的空swipe时长参数和60秒自动熄屏问题，已中止，不计入正式GUI时间占比。

修正转接层后，`official_m3a_20261108_010622` 已使用原始M3A/episode_runner、双图、官方1秒/2秒等待和动作后总结运行。
测试期间临时USB保持亮屏，结束恢复原值；未修改官方核心源码、提示词、历史、输出或完成判定。
已完成的首个动作请求41.869秒（TTFT29.248078秒，decode12.399111秒/59tokens）；随后实际触发总结请求，6888输入tokens，20输出tokens，45.6734秒。

新包decode约4.8token/s，旧4K简化任务67次请求中位数17.2355token/s。输入长度和导出配置都不同；不能把这个非受控对比当成纯上下文长度的因果结论。
同一提示词/双图、同runner/BURST/sampler的顺序对照由 `compare_decode_bundles.py` 执行，已完成，见下文。

## 最终结果

官方两任务26步/50请求结束，任务均失败；笔记正确文件名存在，但磁盘文件为空。主机无请求占比17.18%，设备服务空闲率17.28%，分别在同一时钟内计算；早期跨时钟计算的约22%/23%已撤回。完整结果见[官方状态](OFFICIAL_M3A_STATUS.md)。

同提示词/双图A/B/A：4K前16.663、16K4.148、4K后16.578token/s；六次输入558tokens、输出91tokens，输出文本完全相同。新包吞吐下降75.04%，逐token耗时4.01倍。[原始对照](results/decode_ab_20261108_015225/REPORT.md)。
