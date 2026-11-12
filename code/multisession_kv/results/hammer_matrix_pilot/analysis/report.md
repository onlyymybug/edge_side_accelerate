# HammerBench fixed-tools paired Pilot report

## Scope and integrity

- Source manifest: `results/hammer_matrix_pilot/matrix_manifest.json`
- Manifest SHA256: `95d657f41dfe4239c3394bde171f375656590d0280cd0d4d886b85584da43800`
- Randomization seed: 42
- Completed runs: 102/102; all passed without retry.
- One fixed Session group per 2/4/8-Session condition; 3 run-level repetitions per configuration.
- All request logs report `model_load_count=1`; requests are serial, not continuous batching.
- This is deterministic system replay, not an official HammerBench accuracy score.

## Main finding

RoundRobin exposes real cache competition. When resident slots are fewer than the cyclic working set, Sessions are evicted before reuse, causing repeated Prefill and substantially higher TTFT. Grouped execution avoids this pressure because each Session finishes before the next Session is selected.

| Sessions | RoundRobin policy | Reuse | Prefill/trace | TTFT p50 | Peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | AllKeep | 48.4% | 1090 | 305.8 ms | 86.7 MiB |
| 2 | LRU-1 | 0.0% | 2111 | 591.2 ms | 50.4 MiB |
| 4 | AllKeep | 47.7% | 2391 | 317.7 ms | 158.6 MiB |
| 4 | LRU-3 | 0.0% | 4570 | 605.9 ms | 122.5 MiB |
| 8 | AllKeep | 57.8% | 4836 | 158.7 ms | 317.2 MiB |
| 8 | LRU-2 | 0.0% | 11454 | 619.5 ms | 86.7 MiB |

For 4 Sessions, even three LRU slots realize zero reuse under strict cyclic RoundRobin: every returning Session was already displaced. For 8 Sessions, four and six slots both realize only 20.0% reuse on this variable-turn trace, versus 57.8% for AllKeep.

## Compute rates and memory

Across configuration means, Decode remains stable at 16.25--16.66 token/s. Useful Prefill throughput varies with incremental fragment size and 128-token graph padding; the benefit is the avoided Prefill token/time, not a faster Decode kernel.

Peak runner RSS rises from about 50.5 MiB at one resident slot to about 303--317 MiB at eight slots. Intermediate slot increments are approximately 36 MiB each; the RoundRobin eight-slot endpoint includes additional variation. RSS is process-visible memory only and is not claimed as total device or QNN/DSP memory.

## Repeat stability

| Run-level metric | Median CV | Maximum CV |
| --- | ---: | ---: |
| `useful_prefill_tok_s` | 0.60% | 1.47% |
| `decode_tok_s` | 0.27% | 2.01% |
| `ttft_ms_p50` | 0.62% | 2.58% |
| `trace_makespan_ms` | 0.27% | 1.93% |

Telemetry starts range from 27.562 to 28.228 C; maximum observed skin temperature is 32.446 C. The recovery gate is sampled before model startup/warm-up, so first telemetry may slightly exceed the 28.2 C gate.

## Interpretation and limitations

- The Pilot establishes the mechanism for one fixed group at each Session count; it does not estimate population variation across all 50 Sessions.
- Turns from the same Session are dependent and are not treated as independent statistical samples. With only one Session group per count, no bootstrap confidence interval is reported.
- Session-count curves mix different request-length/turn distributions as the fixed group expands, so they should not be interpreted as a pure scaling curve.
- Natural EOS makes Decode dominate trace makespan; therefore large TTFT changes translate into smaller makespan changes.
- The Pilot does not cover the 897--951-token long-input slowdown observed in one earlier formal run; that anomaly needs a dedicated matched repeat.

## Decision

The problem is validated: cache capacity and Session scheduling materially alter realized prefix reuse, repeated Prefill, TTFT and resident memory, while Decode speed and model loading remain stable. A full real-trace stage should retain RoundRobin capacity points and only minimal Grouped controls, then aggregate all fixed Session groups.

## Generated artifacts

- `group_summary.csv` / `group_summary.json`
- `paired_deltas.csv` / `paired_deltas.json`
- `round_robin_ttft.png`
- `round_robin_prefill_reuse.png`
- `rss_by_resident_slots.png`
