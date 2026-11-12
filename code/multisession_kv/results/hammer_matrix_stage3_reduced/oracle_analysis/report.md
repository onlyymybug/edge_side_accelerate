# Offline Session-cache Oracle analysis

## Method and validation

The analysis uses each full Stage-3 RoundRobin group's measured AllKeep request sequence. AllKeep `reused_tokens` is the realizable value of retaining a Session until its next request. Three whole-Session cache policies are replayed at the exact measured slot budgets:

- LRU: evict the least recently used Session;
- Belady: evict the resident Session whose next request is farthest in the future;
- token-optimal Oracle: dynamic programming that maximizes total reused tokens over the complete known trace.

Simulated LRU exactly matches measured runner LRU for all 104 group/budget cases. The Oracle is an offline upper bound, not an online deployable result, and no TTFT improvement is claimed without a phone run.

## Population result

| Sessions | Slots | LRU reuse | CloseAware-LRU | Belady reuse | Token-optimal reuse | AllKeep reuse | Additional Prefill reduction vs LRU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 3.6% | 3.6% | 3.6% | 3.6% | 69.0% | 0.0% |
| 2 | 2 | 69.0% | 69.0% | 69.0% | 69.0% | 69.0% | 0.0% |
| 4 | 1 | 2.3% | 2.3% | 2.3% | 2.3% | 69.5% | 0.0% |
| 4 | 2 | 2.3% | 10.9% | 28.3% | 29.5% | 69.5% | 27.8% |
| 4 | 3 | 6.1% | 21.9% | 50.4% | 51.8% | 69.5% | 48.7% |
| 8 | 2 | 2.3% | 6.7% | 18.2% | 18.8% | 69.5% | 16.9% |
| 8 | 4 | 16.3% | 27.1% | 42.9% | 44.0% | 69.5% | 33.1% |
| 8 | 6 | 16.3% | 34.4% | 57.1% | 58.0% | 69.5% | 49.9% |

## Interpretation

- Two Sessions with one slot has no policy opportunity: strict alternation exceeds capacity, so LRU and both Oracles are identical.
- Assuming the Agent runtime emits an exact terminal/close signal, immediately releasing completed Sessions recovers 26.1%--43.3% of the available Oracle gap at useful intermediate budgets. This is a causal what-if only when that signal is genuinely available online.
- Four and eight Session traces have heterogeneous completion/future-access patterns. Future-aware eviction recovers substantial reuse that LRU loses.
- Belady is close to the token-optimal result, so most opportunity comes from knowing Session next-use/lifecycle rather than finely weighting token values.
- The largest gaps occur at high but still insufficient capacity: retaining the wrong nearly-complete or distant-future Session causes LRU thrashing.
- This supports investigating an online lifecycle/next-use predictor, but the achievable online fraction of the Oracle gap must be measured separately.

## Decision

There is meaningful eviction-policy headroom for 4/8 Session workloads, while the 2-Session/1-slot case is a fundamental capacity limit. The next method experiment should implement a causal online heuristic using observable lifecycle features (recent interval, turn index, inactive/completed state, KV value/size) and compare it against LRU under the same traces and budgets.
