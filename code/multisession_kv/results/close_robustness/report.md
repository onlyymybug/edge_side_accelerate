# Close-signal robustness

How much of the measured benefit survives when DONE events are missing or
late. Same Sessions, same groupings, same slot counts as the phone matrix;
only the terminal signal changes.

Coverage in [0.0, 0.25, 0.5, 0.75, 1.0], delay in [0, 1, 5, 20] requests, 100 stochastic draws per cell.

Correctness check: the explicit-release simulator reproduces the validated
`simulate_lifecycle_policy` on 144 group/policy/capacity cases at
coverage 1.0 and delay 0, bit for bit.

## Headline: benefit retained vs a perfect signal

Rows show the Prefill reduction vs LRU; parentheses show what fraction of
the perfect-signal reduction survives.


**N=4, slots=2**, delay 0

| reduction | cov 100% | cov 75% | cov 50% | cov 25% | cov 0% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11.4% (100%) | 8.8% (78%) | 6.6% (58%) | 4.5% (40%) | 2.4% (21%) |

**N=4, slots=3**, delay 0

| reduction | cov 100% | cov 75% | cov 50% | cov 25% | cov 0% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28.0% (100%) | 25.7% (92%) | 23.3% (83%) | 21.1% (76%) | 18.6% (67%) |

**N=8, slots=2**, delay 0

| reduction | cov 100% | cov 75% | cov 50% | cov 25% | cov 0% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4.7% (100%) | 3.9% (84%) | 3.1% (67%) | 2.1% (46%) | 1.3% (29%) |

**N=8, slots=4**, delay 0

| reduction | cov 100% | cov 75% | cov 50% | cov 25% | cov 0% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16.9% (100%) | 6.8% (40%) | 1.0% (6%) | -2.7% (-16%) | -5.8% (-34%) |

**N=8, slots=6**, delay 0

| reduction | cov 100% | cov 75% | cov 50% | cov 25% | cov 0% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30.8% (100%) | 28.5% (93%) | 24.8% (80%) | 18.9% (61%) | 13.9% (45%) |

**N=8, slots=7**, delay 0

| reduction | cov 100% | cov 75% | cov 50% | cov 25% | cov 0% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 38.4% (100%) | 36.9% (96%) | 34.4% (89%) | 31.0% (81%) | 24.1% (63%) |

## Delay sensitivity at full coverage

**This axis is censored and must be read with care.** A DONE scheduled for
the end of a grouping falls past the last request of that grouping and can
never fire, so a long delay converts into a *missing* signal rather than a
*late* one. The trace fragment is short: 8-20 requests per group at N=4 and
20-36 at N=8, while a Session lives 2-5 turns. The second row under each
point reports the fraction of DONE events that actually fired; where that
number is far below 100%, the cell is measuring coverage loss, not latency
tolerance.

| point | delay 0 | delay 1 | delay 5 | delay 20 |
| --- | ---: | ---: | ---: | ---: |
| N=4/2 reduction | 11.4% | 2.4% | 2.4% | 2.4% |
| N=4/2 realized | 100% fired | 75% fired | 4% fired | 0% fired |
| N=4/3 reduction | 28.0% | 19.4% | 18.6% | 18.6% |
| N=4/3 realized | 100% fired | 75% fired | 4% fired | 0% fired |
| N=8/2 reduction | 4.7% | 1.3% | 1.3% | 1.3% |
| N=8/2 realized | 100% fired | 88% fired | 54% fired | 2% fired |
| N=8/4 reduction | 16.9% | 11.8% | -4.7% | -5.8% |
| N=8/4 realized | 100% fired | 88% fired | 54% fired | 2% fired |
| N=8/6 reduction | 30.8% | 26.1% | 13.9% | 13.9% |
| N=8/6 realized | 100% fired | 88% fired | 54% fired | 2% fired |
| N=8/7 reduction | 38.4% | 34.5% | 24.7% | 24.1% |
| N=8/7 realized | 100% fired | 88% fired | 54% fired | 2% fired |

## Reading the result

- **Partial coverage degrades gracefully, but not smoothly.** The retained
  fraction tracks coverage fairly closely at the sparse-capacity points
  (N4/C3, N8/C6, N8/C7) and collapses much faster at N8/C4, which even goes
  negative at coverages below 50%.
- **The delay axis does not measure what its name suggests.** Only delay 0
  keeps every DONE inside the trace window; by delay 5 most events at N=4
  and delay 20 nearly all events at both node counts fall past the last
  request and never fire. The apparent plateau at long delays is the
  coverage floor (the delay-5 cell and the delay-20 cell coincide because
  both are really measuring 'almost no signal'), not a tolerance plateau.

Do not report this table as evidence that late DONE is harmless. It is not
evidence either way. The only cell that carries a clean latency reading is
delay 1: going from 0 to 1 already removes a third to a half of the
reduction at every point except the degenerate ones, which is consistent
with Session lifespans of 2-5 turns - a DONE one request late frequently
arrives after the slot was already needed. To measure tolerance beyond
that, replay longer concatenated traces (several groupings back to back)
so a DONE delayed by 20-50 requests still has requests after it to affect.

The N8/C4 row deserves attention rather than omission: with an imperfect
signal the lifecycle score becomes worse than LRU. That is a real property
of the mechanism, not noise - an intermittently-told dead Session poisons
the recency statistics that the score depends on. Reporting it is what
separates an honest evaluation from a selective one.

## What this means for the interface argument

Coverage is the axis this experiment can speak to, and it says the
mechanism tolerates missing signals far better in the sparse-capacity
regime than in the tight one. The delay axis is currently under-powered:
the traces are too short for a late report to have anywhere to land, so
the paper cannot yet claim a latency bound. What it can state is the
weaker, verified result - the benefit requires eventual and reasonably
complete reporting, and the fraction of the perfect-signal gain that
survives scales with coverage.

An implementation that reports DONE on `onStop` - which fires when the
user switches away, not when they finish - will land at delay >= 1 and
should be expected to lose a substantial part of the measured benefit; how
much is not determined by this sweep.

## Files

- `close_robustness.csv`: per-group, per-trial raw results
- `close_robustness_summary.csv`: aggregated cells
- `close_robustness.png`: reduction vs delay, one line per coverage

