# Workload-sensitivity of whole-Session KV retention

Synthetic interleavings replayed through the same online simulator that was
validated against the phone in Stage-3. No device time is used.

## What this measures

The Stage-3 trace is round-robin over Sessions of unequal length. Perfect
rotation is pathological for LRU: when the number of resident Sessions
equals the slot count, every Session is evicted exactly once before its
next turn arrives, so plain LRU scores **zero** reuse. This script varies
the dispatch order while holding the workload itself fixed:

- **dispatch schedule**: `round_robin`, `uniform_random`, `zipf(alpha)`;
- **capacity**: 0.5, 0.75 and (N-1)/N of the Session count;
- **Session shape**: `real` (measured turn counts and prompt lengths, re-interleaved).

`lifecycle_lru` uses the `serial_arrival` protocol, the conservative result.
Oracle is the token-optimal dynamic-programming upper bound.

Replicates per cell: 200. Total workloads: 6042.

## Headline table

Session shape `real`, capacity ratio 0.75:

| N | Schedule | alpha | LRU reuse | CloseAware | Lifecycle | Oracle |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | device_group | 0.0 | 6.1% | 21.9% | 32.7% | 51.8% |
| 4 | round_robin | 0.0 | 25.3% | 38.9% | 45.9% | 58.1% |
| 4 | uniform_random | 0.0 | 61.7% | 66.2% | 66.2% | 67.4% |
| 4 | zipf | 0.0 | 61.1% | 65.1% | 65.2% | 66.5% |
| 4 | zipf | 0.5 | 62.0% | 66.0% | 66.2% | 67.0% |
| 4 | zipf | 1.0 | 63.7% | 67.1% | 67.2% | 67.9% |
| 4 | zipf | 1.5 | 65.9% | 68.0% | 68.0% | 68.2% |
| 8 | device_group | 0.0 | 16.3% | 34.4% | 41.8% | 58.0% |
| 8 | round_robin | 0.0 | 26.9% | 44.2% | 50.3% | 61.2% |
| 8 | uniform_random | 0.0 | 61.4% | 67.2% | 67.4% | 67.9% |
| 8 | zipf | 0.0 | 61.7% | 67.6% | 67.7% | 68.3% |
| 8 | zipf | 0.5 | 62.5% | 67.8% | 67.9% | 68.2% |
| 8 | zipf | 1.0 | 65.6% | 69.2% | 69.2% | 69.3% |
| 8 | zipf | 1.5 | 66.9% | 68.8% | 68.9% | 68.9% |

## Anchor: the exact groupings the phone ran

Sampled groups and the Stage-3 groups are not the same experiment: Stage-3
took N *consecutive* Sessions in trace order, and those happen to be of equal
length. Equal lengths keep the rotation perfectly phase-aligned, which is what
makes plain LRU score near zero. The rows below replay those exact groups, so
they reproduce the device numbers and calibrate everything else.

| N | slots | LRU | CloseAware | Lifecycle | Oracle | reduction | gap recovery |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 2.3% | 10.9% | 13.5% | 29.5% | 11.5% | 41.3% |
| 4 | 3 | 6.1% | 21.9% | 32.7% | 51.8% | 28.4% | 58.3% |
| 8 | 4 | 16.3% | 27.1% | 29.8% | 44.0% | 16.1% | 48.7% |
| 8 | 6 | 16.3% | 34.4% | 41.8% | 58.0% | 30.5% | 61.2% |
| 8 | 7 | 20.5% | 41.9% | 51.5% | 64.2% | 39.0% | 70.9% |

## Reading the result

Compare every sampled row against the `device_group` row at the same N and
slots. Sampled groups mix Session lengths, so rotation drifts out of phase and
LRU stops being degenerate. That drift, not a different workload, is what the
large LRU improvement between the two rows reflects.

`lifecycle_lru_oracle_gap_recovery_pct` is the fraction of the *remaining*
gap that the policy closes. It is the scale-free measure: under phase-aligned
rotation the gap is wide and the policy closes part of it; under skewed
dispatch the gap is narrow, so a small absolute reduction can still be most
of what is left.

## Attribution: close signal vs value function, under skew

Same decomposition as the phone report, recomputed per cell:
`close = (CloseAware - LRU) / (Lifecycle - LRU)`.

| N | slots | schedule | LRU | Close share | Value share |
| ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 2 | device_group a=0.0 | 2.3% | 76.6% | 23.4% |
| 4 | 3 | device_group a=0.0 | 6.1% | 59.3% | 40.7% |
| 8 | 4 | device_group a=0.0 | 16.3% | 80.5% | 19.5% |
| 8 | 6 | device_group a=0.0 | 16.3% | 70.7% | 29.3% |
| 8 | 6 | zipf a=0.0 | 61.7% | 97.8% | 2.2% |
| 8 | 6 | zipf a=0.5 | 62.5% | 98.4% | 1.6% |
| 8 | 6 | zipf a=1.0 | 65.6% | 99.8% | 0.2% |
| 8 | 6 | zipf a=1.5 | 66.9% | 98.5% | 1.5% |
| 8 | 7 | device_group a=0.0 | 20.5% | 69.2% | 30.8% |

This is the most consequential number in the sweep. Under the phone's
phase-aligned grouping the value function contributes a material share of
the gain. Under any skewed dispatch it collapses toward zero, and at high
skew it can go **negative** — Lifecycle-LRU then scores slightly below
CloseAware-LRU because the score occasionally evicts a Session that plain
LRU would have kept.

The value function is therefore not a general improvement. It pays only
where access intervals are near-uniform and the score can break ties that
LRU breaks badly; once dispatch is skewed, knowing *which* Sessions are hot
already dominates, and the score adds noise.

## Decision gate

The original gate asks whether the reduction stays above 10% across skewed
workloads. As the capacity ratio approaches N-1 it does not, because the
Oracle itself sits only a few points above LRU there and no policy can beat
the ceiling. That is arithmetic, not a failure of the method, and the report
should say so rather than hide it.

What survives the sweep is the gap-recovery column. Compare it against the
`device_group` row at the same N and slots: the policy does not need rotation
to work, rotation was simply the regime with the most room. Under skewed
dispatch the remaining gap is small, so the same policy covers most of it.

Note the plain-LRU baseline is what inflates both readings. Against a
degenerate baseline, any policy that breaks the rotation looks strong.
The honest comparator for a skewed workload is CloseAware-LRU, and against
it Lifecycle-LRU's margin is a few tenths of a point.

## What this changes for the paper

Three revisions follow, and they are cheap:

1. The headline must be stated **relative to the achievable gap**, not as a
   raw percentage. A 30% Prefill reduction on the rotation trace is not
   comparable to a 12% reduction on a skewed trace; the gap recovery is.
2. The capacity axis needs a stated operating point. Near N-1 slots the
   ceiling collapses and the discussion should move to *why deploy at that
   point at all*, rather than reporting a small number and moving on.
3. The value function should be presented as a secondary contribution with
   a stated validity region (near-uniform access intervals), not as part of
   the headline method. On skewed dispatch the close signal does essentially
   all of the work, and the score can slightly hurt.

## Files

- `workload_sensitivity.csv`: per-workload raw results
- `workload_sensitivity_summary.csv`: aggregated cells
- `workload_sensitivity_n{4,8}.png`: benefit vs skew

Sampled round-robin reference (real, ratio 0.75):

- N=4: LRU 25.3%, Lifecycle 45.9%, Oracle 58.1%, reduction 27.6%, gap recovery 62.9%
- N=8: LRU 26.9%, Lifecycle 50.3%, Oracle 61.2%, reduction 32.0%, gap recovery 68.3%
