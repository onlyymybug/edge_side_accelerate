# HammerBench fixed-tools Stage-3 population report

## Integrity and scope

- Completed jobs: 388/388; failures/retries: 0/0.
- Full Session groups in primary analysis: 376 runs; tail-group runs retained separately: 12.
- Bootstrap resampling unit is the fixed Session group, never an individual turn.
- This is a fixed-tools system replay and is not an official HammerBench accuracy score.

## RoundRobin population summary

| Sessions | Policy | Groups | Reuse | Prefill tokens | Mean group TTFT p50 | RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | AllKeep | 25 | 69.0% | 31242 | 208.3 ms | 88.1 MiB |
| 2 | LRU-1 | 25 | 3.6% | 97194 | 593.6 ms | 50.6 MiB |
| 2 | LRU-2 | 25 | 69.0% | 31242 | 209.5 ms | 86.7 MiB |
| 2 | SingleSlot | 25 | 3.6% | 97194 | 593.8 ms | 50.5 MiB |
| 4 | AllKeep | 12 | 69.5% | 30032 | 195.8 ms | 165.9 MiB |
| 4 | LRU-1 | 12 | 2.3% | 96252 | 607.8 ms | 50.6 MiB |
| 4 | LRU-2 | 12 | 2.3% | 96252 | 621.9 ms | 86.5 MiB |
| 4 | LRU-3 | 12 | 6.1% | 92542 | 609.9 ms | 122.6 MiB |
| 4 | SingleSlot | 12 | 2.3% | 96252 | 609.3 ms | 50.6 MiB |
| 8 | AllKeep | 6 | 69.5% | 30032 | 157.2 ms | 313.1 MiB |
| 8 | LRU-2 | 6 | 2.3% | 96252 | 618.5 ms | 86.7 MiB |
| 8 | LRU-4 | 6 | 16.3% | 82487 | 612.8 ms | 164.5 MiB |
| 8 | LRU-6 | 6 | 16.3% | 82487 | 616.9 ms | 230.4 MiB |
| 8 | SingleSlot | 6 | 2.3% | 96252 | 610.6 ms | 50.7 MiB |

## Paired effect versus AllKeep

| Sessions | Schedule | Policy | Groups | Prefill change | TTFT change (95% CI) | Makespan change | RSS change |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | grouped | NoReuse | 50 | +218.6% | +272.0% [+230.5, +312.3] | +15.3% | +0.0 MiB |
| 2 | grouped | NoReuse | 25 | +220.8% | +276.0% [+221.9, +326.5] | +14.9% | -35.8 MiB |
| 2 | round_robin | LRU-1 | 25 | +210.2% | +227.8% [+185.9, +268.9] | +13.9% | -37.5 MiB |
| 2 | round_robin | LRU-2 | 25 | +0.0% | +1.3% [-0.1, +2.7] | -0.0% | -1.4 MiB |
| 2 | round_robin | SingleSlot | 25 | +210.2% | +228.0% [+184.0, +268.2] | +14.2% | -37.6 MiB |
| 4 | grouped | NoReuse | 12 | +226.4% | +296.9% [+228.8, +359.8] | +14.7% | -108.0 MiB |
| 4 | round_robin | LRU-1 | 12 | +219.2% | +242.3% [+188.4, +291.1] | +14.1% | -115.3 MiB |
| 4 | round_robin | LRU-2 | 12 | +219.2% | +250.1% [+195.9, +296.6] | +14.5% | -79.3 MiB |
| 4 | round_robin | LRU-3 | 12 | +207.5% | +242.7% [+189.2, +290.0] | +14.5% | -43.3 MiB |
| 4 | round_robin | SingleSlot | 12 | +219.2% | +243.6% [+189.4, +289.6] | +14.1% | -115.3 MiB |
| 8 | grouped | NoReuse | 6 | +226.8% | +371.9% [+363.2, +380.0] | +16.4% | -251.5 MiB |
| 8 | round_robin | LRU-2 | 6 | +219.2% | +294.4% [+283.1, +311.4] | +13.5% | -226.4 MiB |
| 8 | round_robin | LRU-4 | 6 | +173.4% | +290.9% [+275.7, +309.5] | +11.3% | -148.6 MiB |
| 8 | round_robin | LRU-6 | 6 | +173.4% | +293.3% [+280.9, +308.4] | +11.5% | -82.8 MiB |
| 8 | round_robin | SingleSlot | 6 | +219.2% | +289.5% [+275.3, +308.8] | +14.2% | -262.5 MiB |

## Runtime invariants and limitations

- Model load count values: [1].
- Decode throughput across population configurations: 16.16--16.54 token/s.
- Maximum observed skin temperature: 33.763 C.
- Nominal 4- and 8-Session partitions each have a final two-Session tail group. Those runs remain in the manifest but are excluded from primary nominal-count estimates.
- Natural EOS means Decode can dominate makespan; TTFT and Prefill changes need not translate proportionally to total trace time.
- Native-tools and BFCL generalization have not yet been tested.
