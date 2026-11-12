# AppWorld 多任务共享手机 runner：小型链路验证

`appworld_multisession_smoke.py` 让每个 AppWorld 任务在独立进程中运行工具环境；父进程独占一个常驻 QNN runner，并按任务实际就绪时间串行派发推理请求。这样不会并发调用未加锁的 `JsonlRunner.request()`，也不会把不同任务的 AppWorld 全局状态放在同一进程。任务结束时显式调用 `evict_session` 清空 KV。

`--max-sessions` 控制 runner 可注册的任务会话数，至少等于不同任务 ID 数；`--kv-slots` 控制主机策略允许保留 KV 的逻辑槽位数。槽位满时按 LRU 调用 `evict_session`，但不重新创建已注册的 Session。该逻辑预算不等于物理内存硬上限：runner 可能预分配多个会话的 KV 空间，必须单独记录 RSS。

该脚本目前覆盖少数已筛选的 Spotify 粉丝数、切歌及最热歌曲任务，并使用 Agent Service 预先登录、限定工具和直接 JSON 提示。因此它是**多任务/多 Session 协议 smoke**，不是 AppWorld 官方成绩，也不是 Agent–Inference 协同方法的收益实验。`--tool-delay-ms` 是额外插入的受控等待，不是 AppWorld API 或真实网络时延。

## 已完成的离线检查

`test_appworld_multisession_smoke.py` 检查两个 worker 共享一个串行后端、请求和 Session 生命周期，并检查 3 任务/2 槽位的 LRU 驱逐。真实 AppWorld 进程 + 脚本化模型的 `results/appworld_multi_mock_n3_c2_20261026_01` 完成 3 个任务、9 次请求，记录 7 次容量驱逐。脚本化模型会从前一工具结果中提取 ID/粉丝数；成功数**不能**当作手机模型准确率，驱逐次数也不能直接外推到真机时序。

## 真机小试

先确认无线 ADB 地址；下面的输出目录和实验名必须未使用过。旧的 `--capacity` 仍是 `--kv-slots` 的别名，不再表示 runner 的 Session 名额。

```bash
cd /home/lyyyy/edge_side_work/code/multisession_kv
PY=/home/lyyyy/edge_side_work/.venv-appworld/bin/python
SERIAL=<当前手机IP:端口>
adb -s "$SERIAL" get-state

"$PY" appworld_multisession_smoke.py \
  --backend phone --serial "$SERIAL" \
  --task-ids 6bdbc26_1 6bdbc26_2 6bdbc26_3 \
  --max-sessions 3 --kv-slots 2 --tool-delay-ms 1000 \
  --experiment-name agent_multi_phone_n3_c2_01 \
  --output-dir results/appworld_multi_phone_n3_c2_01

```

验收重点：同一个 `server_pid`、`model_load_count=1`；两任务请求都有独立 `session_id`；出现真实 `queue_ms` 与工具等待；`tasks.jsonl` 有真实 AppWorld 评估结果。若任一任务因模型格式错误失败，保留原始结果，不把失败当作调度算法失败。

真机容量 2 小试已完成：2 个任务均通过 AppWorld 评估，6 次推理请求；后续请求各有 476–590 个 token 复用。两任务第一步的 TTFT 分别是 1403/1417 ms，后续四步为 396–440 ms；但总排队时间达 8.96 s。每个任务的首个工具调用均经过一次格式修复、一次参数过滤，且预先登录和限定工具集合，因此不代表通用 Agent 能力。每次工具阶段还人为增加 1000 ms 等待。

先前把 runner 的 `max_resident_sessions` 直接设为 1，再创建两个不同任务，会收到 `SESSION_CAPACITY`：`evict_session` 不释放注册名额。这不是“1 个 KV 槽位无法比较”的结论。现在 runner 注册名额设为任务总数，而主机另行实施更小的 KV 逻辑预算，与旧 BFCL 回放的做法一致。只有未来需要在同一 runner 中持续接入超过预设注册名额的新任务时，才需要 `destroy_session(id)` 一类真正注销接口。

若只验收驱逐事件与复用/重算 token，可不设置温度启动门槛，但仍建议另行记录温度。当前脚本尚未自动采集热状态。TTFT、Prefill 时间、吞吐和任务完成时间受热状态影响，不能用温度不受控的小试声称策略时延收益。

## 2026-10-26：3 任务真机功能对照

同一 Redmi、Qwen2.5-3B/QNN 常驻 runner、相同 3 个 AppWorld Spotify 粉丝数任务模板，各运行一次。runner 都允许注册 3 个 Session；主机分别允许保留 2 或 3 份 KV。每次工具调用额外等待 1000 ms，未设置温度启动门槛。运行前后电池温度读数约 23.0–24.6°C；该读数不是芯片温度，也不是持续热状态监测。

| 逻辑 KV 槽位 | 容量驱逐 | 9 次请求的复用 token | Prefill token | AppWorld 任务通过 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 6 | 0 | 5,079 | 2/3 |
| 3 | 0 | 3,187 | 1,892 | 3/3 |

容量 2 下，每次驱逐均得到 runner 的 `state=evicted`；它释放了约 226 MiB 的 KV backing storage，但未注销 Session ID。后续 6 次请求都是 `cache_hit=miss`、`reused_tokens=0`。容量 3 下后续 6 次请求均是 partial hit，复用 476–590 token。两组均由单个模型进程完成，`model_load_count=1`。

容量 2 的失败任务在最后一步生成无法解析的 JSON 工具调用；不能把 2/3 与 3/3 的差异归因于 KV 策略。两次执行还可能有提示内容、请求交错和热状态差异，不是冻结输入的成对时延实验。本对照只证明当前 Agent 链路能施加逻辑 KV 容量压力，并观测到与驱逐一致的重复 Prefill；尚未证明 Agent 状态感知方法的收益。

下一步需要在未参与调试的、不同长度和工具等待的任务上固定 Agent 接口与到达协议，并把“结束即释放”与真正的状态感知决策分开消融。当前小试总在任务完成时释放 KV，因此它本身不能用来估计 close 信号相对纯 LRU 的增量。

## 2026-10-26：就绪队列信号的单组机制验证

脚本增加了 `--policy lru|ready_lru`、`--release-on-done|--no-release-on-done`、按任务设置工具等待及首次到达延迟的参数。`ready_lru` 只读取当前已排队的请求：容量满时，优先从未就绪的驻留 Session 中选最久未使用者；若驻留 Session 全已就绪，则退化为普通 LRU。不读取未来工具结果或任务最后一轮。两个策略都设为“任务结束即释放”，因此这里只比较就绪队列信号的增量，不估计 close 信号的价值。

用同三个已调试的 AppWorld 模板实例、3 个 runner Session 名额、2 个逻辑 KV 槽位，设置任务 1 工具额外等待 0 ms、任务 2 为 3000 ms、任务 3 为 1000 ms；任务 2/3 首次到达分别推迟 10/20 ms。先用脚本化模型验证决策差异，再在 Redmi/Qwen2.5-3B/QNN runner 上各跑一组。两次真机运行的 9 个请求**顺序相同，逐请求输入 token ID 完全相同**，任务结果也同为 2/3；任务 1 均在末步生成格式错误的 JSON。

| 真机策略 | 容量驱逐 | KV 复用 token | Prefill token | 任务通过 |
| --- | ---: | ---: | ---: | ---: |
| LRU | 6 | 0 | 5,079 | 2/3 |
| Ready-protect LRU | 3 | 1,538 | 3,541 | 2/3 |

首次容量冲突时，LRU 驱逐已在就绪队列中的任务 1；ready-protect LRU 改为驱逐仍在等工具的任务 2。后者相对于 LRU 减少 **30.3%** 的 Prefill token。这一对照能说明跨服务就绪信号在该具体交错序列里改变了正确的 KV 决策，但**不能**外推为总体收益：只有一组、三个同模板实例、人工等待，且两次设备热状态并非严格匹配。TTFT、总排队与任务完成时间不作策略收益统计。下一步是构造或选取异质公开任务、多组不同到达/等待负载，并对“纯 LRU、LRU+close、LRU+close+ready”作冻结协议的成对消融。

## 固定事件序列重放与异质任务小试

`appworld_kv_replay.py` 从已有真机运行的 `runner.log`、`requests.jsonl` 和 `events.jsonl` 冻结逐请求输入/生成 token 以及事件顺序；按当前已观察到的 `llm_ready`、`inference_start`、`task_end` 重放 KV 决策。它会先用来源策略逐请求核对真机的 `reused_tokens`，任何不匹配即报错。来源为先前的 LRU/ready 小试和 3 槽位运行时均通过了逐请求精确核对。

```bash
cd /home/lyyyy/edge_side_work/code/multisession_kv
PY=/home/lyyyy/edge_side_work/.venv-appworld/bin/python
"$PY" appworld_kv_replay.py \
  --source-results results/appworld_multi_phone_hetero_cap3_20261026_01 \
  --kv-slots 2 \
  --output results/appworld_kv_replay_hetero_cap2_20261026_01.json
"$PY" -m unittest -q test_appworld_kv_replay test_appworld_multisession_smoke
```

重放只隔离**固定事件序列下的 KV token 决策**；策略改变推理耗时后可能反过来改变任务就绪顺序，所以不能用它估计端到端时延或真实任务成功率。当前来源日志的 `task_end` 是父进程观察到的事件顺序，并非独立于调度策略的外生到达流。

在本地**带 `ground_truth/api_calls.json` 的任务**中，严格按现有 Spotify 工具白名单筛选 API 路径，只有此前三个粉丝数实例完全匹配；没有该文件的任务不在此次筛选结论内。因此多任务脚本增加了按任务指定 `song_navigation` profile 和显式 `--enrich-navigation-task`，让一项此前单任务已调试过的切歌任务进入混合负载。该服务侧观察增强在每次切歌后额外调用 `show_current_song`，必须作为 Agent Service 成本记录，不能称为原始 AppWorld Agent 能力。

无人工工具等待、3 个真实任务（两个粉丝数查询各 3 轮、一个切歌任务 5 轮）、3 个 runner 会话名额、2 个逻辑 KV 槽位的单组真机结果：

| 策略 | 容量驱逐 | KV 复用 token | Prefill token | AppWorld 任务通过 |
| --- | ---: | ---: | ---: | ---: |
| LRU + close | 6 | 1,517 | 5,346 | 3/3 |
| Ready-protect LRU + close | 3 | 3,075 | 3,788 | 3/3 |

两组逐任务输入 token ID 一致，但请求交错顺序不同。就绪保护组在这次小试中 Prefill token 少 **29.1%**；按各自事件顺序离线重放，LRU+close 都为 5,346 token，ready+close 分别为 3,784/3,788 token Prefill，说明该 KV 差异在两种实际交错顺序下都存在。纯 LRU（不使用 close）相比 LRU+close 在这两条固定序列里多 1 次容量驱逐，但 Prefill token 不变，不能据此断言 close 总体无价值。

这仍不是论文级结果：仅三个已调试任务、两种 Spotify 模板、没有独立 holdout 或重复配对；切歌任务依赖观察增强；物理内存预算和芯片温度未受控。下一阶段应扩大可执行任务工具集并冻结 Agent 接口，在新任务/多组到达流上做三段消融，另把真实工具延迟、额外 Agent Service API 调用和 RSS 一并报告。

下一阶段才是冻结 Agent 接口、选未参与调试的多任务负载、实现并对照 FIFO/LRU、结束即释放和 Agent 状态感知策略。大批次需在这些小试通过且电量/温度控制明确后进行。

## 第三种任务类型的一次性验证及失败归因

尝试增加 AppWorld `e85d92a` 系列的“搜索歌手→按播放量排序→提交歌曲名”任务。`_1` 用于开发，经过提示和有记录的平铺参数归一化才成功，不计入未见任务成功率。原计划用 `_2` 验证，但运行后核对任务原文发现它问的是**播放最少**，而固定 profile 强制降序；该次三任务 2/3 是配置错误，标为无效。代码现已在启动 runner 前核对 profile 与任务原文，防止同类错误。

此前未运行的 `_3` 原文确实要求**播放最多**。在 3 个 runner Session 名额、3 个逻辑 KV 槽位、无人工工具等待的单组混合真机运行中，11 次请求，粉丝数和切歌任务成功，`_3` 未通过 AppWorld 评估：搜索歌手结果首项是目标 Jasper Skye（ID 3），模型下一步却调用了另一位歌手的 ID 4，因而返回了错误歌曲。工具调用 JSON 均可解析，`_3` 无参数修复或过滤；这是工具观察到下一次行动的**实体绑定错误**，不是 KV 驱逐（此组容量驱逐为 0）。切歌任务仍依赖服务侧观察增强，另两项旧任务有工具调用修复，不能把 2/3 当作一般 Agent 准确率。

以这次 3 槽位运行的 11 次请求和实际就绪事件作**固定事件序列反事实**，将逻辑 KV 预算改为 2 槽位：纯 LRU、LRU+close、ready-protect LRU+close 的 Prefill token 分别是 6,418 / 6,418 / 4,684；容量驱逐分别为 7 / 6 / 3。来源日志的 3 槽位逐请求 `reused_tokens` 与重放精确一致。这个 27.0% 的 token 差异仅表示**在该固定序列下**就绪信号改变了缓存决策；它不表示真机端到端加速，更不能修复第三项的实体绑定失败。策略改变实际推理耗时后，请求顺序可能变化。日志见 `results/appworld_multi_phone_three_profiles_cap3_20261026_02`，重放见 `results/appworld_kv_replay_three_profiles_cap2_20261026_02.json`。

因此下一步先建立更可靠且明确标注服务侧辅助的 Agent 任务闭环，并从不同任务实例冻结新验证集；之后才做三段 KV 策略真机对照和时延/内存评估。已使用的 `_1/_2/_3` 都不能再作为未见验证实例。

随后筛选的 `396c5a2` 是按歌手和播放次数阈值筛歌并加入播放队列的状态修改任务。开发实例 `_1` 的首轮手机输出把两个 JSON 片段拼接在一起；parser 拒绝执行，任务 0/1。`_2/_3` 未运行。当前小模型在新任务上的格式可靠性和实体绑定能力都不足以支持直接扩大策略实验；应先决定并冻结端侧模型、工具调用协议及可审计的服务侧修复边界，再使用保留实例验证。不能把按任务实例反复修 prompt 之后的通过率当作未见 Agent 准确率。
