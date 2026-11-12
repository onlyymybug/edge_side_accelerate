# 16K decode 减速的因果复核

2026-11-09 更新：手机重连后已完成实际图跟踪及旧包内强制 CL 对照。**旧包这条请求实际使用 1K，新包实际使用 16K；减速主要发生在 QNN 图执行调用内，CPU KV 更新不是主要耗时。** 仍未获得算子级/NPU 周期数据，不能细化为 Attention、DDR 或具体 kernel。详见[设备复测报告](results/decode_diagnosis_20261109_134535/REPORT.md)。以下保留复测前的推理，并注明当时证据范围。“4K变16K，恰好慢4倍”的简单解释应撤回。

## 已实测与未实测

2026-11-08 A/B/A使用相同手机、runner、BURST、提示词及双图。六次输入均558tokens、输出均91tokens且文本相同。旧包合并16.62063token/s，新包4.14830token/s；单token约60.17ms和241.06ms，减速4.0066倍。切回旧包恢复，说明包级差异可复现。
这是两个完整导出包的对照，不是仅改变CL的单变量实验。没有实际执行graph-ID或算子optrace，不能将全部时间差直接归给Attention。

## 旧包短输入预计走1K，而非固定4K

本机4K备份phone_bundle1的part1二进制包含AR128/AR1各四档CL：512、1024、2048、4096；新包仅包含AR128/AR1的CL16384。见[二进制提取的graph名称](results/decode_ab_20261108_015225/source_review/compiled_graph_names.json)。

复核GenieX v0.10.0锁定的plugin提交ab24abb1f66cb3d8fe624cb8649ec23bd36da41c：

- 加载时从实际graph名称推断并排序CL，而不是直接用genie_config的context.size作为执行图尺寸。
- resetKVCache把active_cl_idx_置0，即最小图。
- prefillLoop按128分块；promoteCL在当前图容量不足时向上升档。
- decode循环使用active_cl_idx_，每次处理一个token。

若该二进制按对应发布源码执行且四档图均已成功加载，558-token请求的预期路径为：前3个128-token块用CL512，第四块需要超过CL512的prefill KV容量384，升至CL1024；剩余输入及约91个输出token均可在1K图内完成。新包则全程只有16K可选。
所以本次短输入对照应理解为**预计约1K执行图对固定16K执行图**；实际图仍需设备日志确认。不能拿metadata中代表性的CL4096张量直接当作这次请求实际执行的tensor。

源码：[resetKVCache](https://github.com/qualcomm/geniex-qairt-plugin/blob/ab24abb1f66cb3d8fe624cb8649ec23bd36da41c/core/src/llm/llm_model.cpp#L1490)、[promoteCL](https://github.com/qualcomm/geniex-qairt-plugin/blob/ab24abb1f66cb3d8fe624cb8649ec23bd36da41c/core/src/llm/llm_model.cpp#L862)、[prefillLoop](https://github.com/qualcomm/geniex-qairt-plugin/blob/ab24abb1f66cb3d8fe624cb8649ec23bd36da41c/core/src/llm/llm_model.cpp#L1050)、[decode](https://github.com/qualcomm/geniex-qairt-plugin/blob/ab24abb1f66cb3d8fe624cb8649ec23bd36da41c/core/src/llm/llm_model.cpp#L1210)。这是与发布tag对应的源码分析，不是已实测的graph trace。

## 为什么大图可能慢，以及不能怎样推导

固定图有更大的Attention/KV张量范围。mask保证未使用位置不影响结果，不保证编译后的密集算子跳过这些位置。较大的工作集也可能改变NPU的分块、片上存储使用、DDR访问及CPU缓存/TLB行为。这些都支持图尺寸影响逐token延迟，但尚未证明哪个占主导。

每token时间包含固定成本（权重/投影/MLP等）、随CL变化的Attention等成本、运行时调度和内存成本。因此预计CL从1K到16K而总延迟只增加约4倍不矛盾；不能假设总时间必然与CL成正比。

**源码的KV写回只复制本次新增的token。** updateKV向copyKV传n_tok，decode调用为1；未溢出的普通KV缓存没有每token整块搬移。prefill/decode相位或CL变化时另有restride操作。不能用“每生成一个token都复制1.2GB缓存”解释本次结果。
源码：[updateKV与reshapeKV](https://github.com/qualcomm/geniex-qairt-plugin/blob/ab24abb1f66cb3d8fe624cb8649ec23bd36da41c/core/src/llm/llm_model.cpp#L778)。

## 其他证据和剩余混杂

- 两个本机保留包的36个.so哈希完全一致；功耗/HTP配置、图像处理配置一致。177个metadata I/O张量的dtype及量化参数全部一致；76个shape差异均为KV输入或attention_mask。它们削弱了“换了精度或运行库”的解释，但不证明QNN编译后的算子调度相同。
- A/B/A电池温度约45.0–45.7℃，切回旧包后仍恢复速度，削弱单纯持续热降频解释；未记录NPU实际频率，不能完全排除温控。
- 加载后MemAvailable，4K约4.08/4.38GB（kB口径换算），16K约1.77GB；仅为整机快照，不是精确模型占用或DDR带宽测量。
- 不同包加载均曾报错后恢复：旧包见context5005，新包见映射8003/context1002。DSP会话布局、编译kernel选择、缓存维护和RPC成本仍是替代机制，不能仅凭加载错误认定CPU回退或内存瓶颈。

## 最有判别力的后续实验

1. 记录实际加载CL列表、每次执行的graph名、phase与AR，先确认旧包确实走1K、新包确实走AR1/16K。
2. 在旧包相同二进制内固定选择1K、2K、4K图，同558-token输入比较逐token时间；不要只改genie_config的context.size来假定已强制选图。
3. 同时采集QNN逐graph/算子耗时，以及CPU侧KV更新、cache维护、RPC等待和NPU频率。若增量主要落在Attention相关执行，图尺寸解释得到直接支持；若落在CPU/映射/RPC或kernel退化，则应据实修正。

复测前 ADB 无连接设备，以上为当时的实验建议。现已完成第 1、2 项和第 3 项的 QNN 调用/CPU KV 分层计时，见顶部链接；尚无算子 optrace、DSP 周期或 NPU 频率数据。
