# Lifecycle-aware Session KV analysis

## Scope and leakage controls

The analysis covers 42 full Stage-3 group/capacity cases at the five preselected N4/N8 points. Simulated LRU is required to match the measured phone LRU exactly before any method result is emitted.

HammerBench does not natively expose runtime lifecycle events. Every method event in this analysis is therefore labeled `controlled_protocol`. `future_oracle` is rejected by the online simulator and is used only by Belady/token-optimal upper bounds.

Two visibility assumptions are reported separately:

- `serial_arrival`: only the dispatched request is READY; other resident Sessions are WAITING.
- `queued_next_turn`: after a Session request completes, its next fixed reference request is released. Queue rank and exact LCP are visible because that prompt is already queued.

A controlled close is emitted after a Session's final request. This validates the mechanism available when an upper layer explicitly closes a Session; it is not evidence that HammerBench provides natural close signals.

## Population result

| Protocol | Point | Policy | Groups | Reuse | Prefill reduction vs LRU | Oracle gap recovery | Evictions | DONE releases |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| queued_next_turn | N4/C2 | close_aware_lru | 12 | 10.9% | 8.8% | 31.6% | 105 | 48 |
| queued_next_turn | N4/C2 | lifecycle_lru | 12 | 28.5% | 26.8% | 96.6% | 73 | 48 |
| queued_next_turn | N4/C2 | lru | 12 | 2.3% | 0.0% | 0.0% | 141 | 0 |
| queued_next_turn | N4/C2 | ready_aware_lru | 12 | 28.3% | 26.6% | 95.9% | 72 | 48 |
| serial_arrival | N4/C2 | close_aware_lru | 12 | 10.9% | 8.8% | 31.6% | 105 | 48 |
| serial_arrival | N4/C2 | lifecycle_lru | 12 | 13.5% | 11.5% | 41.3% | 102 | 48 |
| serial_arrival | N4/C2 | lru | 12 | 2.3% | 0.0% | 0.0% | 141 | 0 |
| serial_arrival | N4/C2 | ready_aware_lru | 12 | 10.9% | 8.8% | 31.6% | 105 | 48 |
| queued_next_turn | N4/C3 | close_aware_lru | 12 | 21.9% | 16.8% | 34.6% | 87 | 48 |
| queued_next_turn | N4/C3 | lifecycle_lru | 12 | 50.8% | 47.6% | 97.8% | 33 | 48 |
| queued_next_turn | N4/C3 | lru | 12 | 6.1% | 0.0% | 0.0% | 123 | 0 |
| queued_next_turn | N4/C3 | ready_aware_lru | 12 | 50.4% | 47.2% | 97.0% | 33 | 48 |
| serial_arrival | N4/C3 | close_aware_lru | 12 | 21.9% | 16.8% | 34.6% | 87 | 48 |
| serial_arrival | N4/C3 | lifecycle_lru | 12 | 32.7% | 28.4% | 58.3% | 71 | 48 |
| serial_arrival | N4/C3 | lru | 12 | 6.1% | 0.0% | 0.0% | 123 | 0 |
| serial_arrival | N4/C3 | ready_aware_lru | 12 | 21.9% | 16.8% | 34.6% | 87 | 48 |
| queued_next_turn | N8/C2 | close_aware_lru | 6 | 6.7% | 4.4% | 26.1% | 111 | 48 |
| queued_next_turn | N8/C2 | lifecycle_lru | 6 | 17.9% | 15.9% | 94.3% | 93 | 48 |
| queued_next_turn | N8/C2 | lru | 6 | 2.3% | 0.0% | 0.0% | 153 | 0 |
| queued_next_turn | N8/C2 | ready_aware_lru | 6 | 18.2% | 16.2% | 95.9% | 90 | 48 |
| serial_arrival | N8/C2 | close_aware_lru | 6 | 6.7% | 4.4% | 26.1% | 111 | 48 |
| serial_arrival | N8/C2 | lifecycle_lru | 6 | 6.7% | 4.4% | 26.1% | 111 | 48 |
| serial_arrival | N8/C2 | lru | 6 | 2.3% | 0.0% | 0.0% | 153 | 0 |
| serial_arrival | N8/C2 | ready_aware_lru | 6 | 6.7% | 4.4% | 26.1% | 111 | 48 |
| queued_next_turn | N8/C4 | close_aware_lru | 6 | 27.1% | 13.0% | 39.2% | 76 | 48 |
| queued_next_turn | N8/C4 | lifecycle_lru | 6 | 41.9% | 30.6% | 92.5% | 53 | 48 |
| queued_next_turn | N8/C4 | lru | 6 | 16.3% | 0.0% | 0.0% | 118 | 0 |
| queued_next_turn | N8/C4 | ready_aware_lru | 6 | 42.9% | 31.8% | 96.0% | 47 | 48 |
| serial_arrival | N8/C4 | close_aware_lru | 6 | 27.1% | 13.0% | 39.2% | 76 | 48 |
| serial_arrival | N8/C4 | lifecycle_lru | 6 | 29.8% | 16.1% | 48.7% | 74 | 48 |
| serial_arrival | N8/C4 | lru | 6 | 16.3% | 0.0% | 0.0% | 118 | 0 |
| serial_arrival | N8/C4 | ready_aware_lru | 6 | 27.1% | 13.0% | 39.2% | 76 | 48 |
| queued_next_turn | N8/C6 | close_aware_lru | 6 | 34.4% | 21.6% | 43.3% | 64 | 48 |
| queued_next_turn | N8/C6 | lifecycle_lru | 6 | 56.7% | 48.3% | 96.9% | 25 | 48 |
| queued_next_turn | N8/C6 | lru | 6 | 16.3% | 0.0% | 0.0% | 106 | 0 |
| queued_next_turn | N8/C6 | ready_aware_lru | 6 | 57.1% | 48.7% | 97.7% | 23 | 48 |
| serial_arrival | N8/C6 | close_aware_lru | 6 | 34.4% | 21.6% | 43.3% | 64 | 48 |
| serial_arrival | N8/C6 | lifecycle_lru | 6 | 41.8% | 30.5% | 61.2% | 54 | 48 |
| serial_arrival | N8/C6 | lru | 6 | 16.3% | 0.0% | 0.0% | 106 | 0 |
| serial_arrival | N8/C6 | ready_aware_lru | 6 | 34.4% | 21.6% | 43.3% | 64 | 48 |

## Interpretation rule

`serial_arrival` is the conservative result for the current fixed replay. Any additional gain under `queued_next_turn` is conditional on the upper scheduler genuinely having those prompts in its ready queue. Neither controlled result is an end-to-end Agent result, and TTFT improvement remains unclaimed until the selected policy is run on the phone.

## Decision

Under `serial_arrival`, Lifecycle-LRU reaches the predeclared 10% Prefill-reduction threshold at N4/C2, N4/C3, N8/C4, N8/C6. It does not reach that threshold at N8/C2.

The offline gate is therefore satisfied at both N=4 and N=8. The next step is a phone comparison at all five points, retaining the failed point as a capacity-pressure negative control. `queued_next_turn` should be a separate conditional experiment, because its near-Oracle result depends on exact queued request visibility.
