# What part of the queue visibility actually pays

The validated `queued_next_turn` protocol hands the policy three
observations at once: which Sessions are queued, where they sit in the
queue, and the exact LCP of the queued prompt. This sweep holds the
scoring formula fixed and switches those observations on one at a time,
so the marginal value of each is separable.

Conformance: the empty-visibility level reproduces the validated
`serial_arrival` Lifecycle-LRU exactly on 48 device groups.

## The ladder

| level | ready set | queue rank | exact LCP |
| --- | :---: | :---: | :---: |
| `none` | — | — | — |
| `set` | yes | — | — |
| `rank` | yes | yes | — |
| `lcp` | yes | — | yes |
| `rank+lcp` | yes | yes | yes |

## Phone groupings: Prefill reduction vs LRU, by level

Parentheses show the marginal share of the total LRU-to-Oracle gap
that each added observation recovers, relative to the level below.

| point | `none` | `set` | `rank` | `lcp` | `rank+lcp` |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=4/C=2 | 11.4% | 11.4% | 24.8% | 15.3% | 20.9% |
| N=4/C=3 | 28.0% | 28.0% | 46.2% | 28.0% | 43.5% |
| N=8/C=2 | 4.7% | 4.7% | 16.0% | 7.8% | 15.2% |
| N=8/C=4 | 16.9% | 16.9% | 26.6% | 18.9% | 27.2% |
| N=8/C=6 | 30.8% | 30.8% | 45.1% | 34.7% | 45.1% |
| N=8/C=7 | 38.4% | 38.4% | 50.7% | 40.9% | 50.8% |

### Marginal recovery of each observation

Each cell is (level - previous level) / (Oracle - LRU), in percent.
This is the quantity to read when deciding what the platform must
expose; a level that recovers little is not worth building.

| point | +ready set | +queue rank | +exact LCP | full ladder |
| --- | ---: | ---: | ---: | ---: |
| N=4/C=2 | 0.0 | 13.5 | -3.9 | 20.9 |
| N=4/C=3 | 0.0 | 18.3 | -2.7 | 43.5 |
| N=8/C=2 | 0.0 | 11.4 | -0.8 | 15.2 |
| N=8/C=4 | 0.0 | 9.7 | 0.6 | 27.2 |
| N=8/C=6 | 0.0 | 14.4 | 0.0 | 45.1 |
| N=8/C=7 | 0.0 | 12.3 | 0.0 | 50.8 |

### Interaction: is LCP worth more once rank is known?

Two orders of addition, so an interaction effect is visible:
`lcp - set` is LCP without rank, `rank+lcp - rank` is LCP with it.

| point | `set` | `lcp` | LCP w/o rank | `rank` | `rank+lcp` | LCP with rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N=4/C=2 | 11.4 | 15.3 | +3.9 | 24.8 | 20.9 | -3.9 |
| N=4/C=3 | 28.0 | 28.0 | +0.0 | 46.2 | 43.5 | -2.7 |
| N=8/C=2 | 4.7 | 7.8 | +3.1 | 16.0 | 15.2 | -0.8 |
| N=8/C=4 | 16.9 | 18.9 | +2.0 | 26.6 | 27.2 | +0.6 |
| N=8/C=6 | 30.8 | 34.7 | +3.9 | 45.1 | 45.1 | +0.0 |
| N=8/C=7 | 38.4 | 40.9 | +2.5 | 50.7 | 50.8 | +0.0 |

## Robustness under a non-degenerate dispatch order

Mean Prefill reduction vs LRU, Zipf dispatch over measured Session
shapes. The absolute numbers are much larger than on the phone
groupings because plain LRU is not degenerate here; what matters is
whether the *ordering* of the levels survives.

| N | ratio | alpha | `none` | `set` | `rank` | `rank+lcp` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.50 | 0.0 | 10.5% | 10.5% | 15.7% | 15.9% |
| 4 | 0.50 | 1.0 | 5.7% | 5.7% | 13.2% | 12.5% |
| 4 | 0.50 | 1.5 | 9.6% | 9.6% | 12.7% | 12.7% |
| 4 | 0.75 | 0.0 | 10.9% | 10.9% | 13.9% | 13.8% |
| 4 | 0.75 | 1.0 | 8.4% | 8.4% | 8.7% | 8.5% |
| 4 | 0.75 | 1.5 | 6.7% | 6.7% | 6.9% | 7.0% |
| 4 | 1.00 | 0.0 | 0.0% | 0.0% | 0.0% | 0.0% |
| 4 | 1.00 | 1.0 | 0.0% | 0.0% | 0.0% | 0.0% |
| 4 | 1.00 | 1.5 | 0.0% | 0.0% | 0.0% | 0.0% |
| 8 | 0.50 | 0.0 | 17.3% | 17.3% | 23.6% | 24.0% |
| 8 | 0.50 | 1.0 | 16.6% | 16.6% | 21.5% | 22.3% |
| 8 | 0.50 | 1.5 | 14.2% | 14.2% | 16.9% | 16.8% |
| 8 | 0.75 | 0.0 | 16.3% | 16.3% | 17.4% | 17.5% |
| 8 | 0.75 | 1.0 | 8.2% | 8.2% | 8.1% | 8.0% |
| 8 | 0.75 | 1.5 | 7.1% | 7.1% | 7.1% | 7.1% |
| 8 | 0.88 | 0.0 | 7.3% | 7.3% | 7.2% | 7.3% |
| 8 | 0.88 | 1.0 | 2.9% | 2.9% | 2.9% | 2.9% |
| 8 | 0.88 | 1.5 | 1.0% | 1.0% | 1.0% | 1.0% |

## Reading the result

The finding is not that queue visibility is worthless - it is that two
of its three advertised components are structural zeros on this trace,
and only one carries the whole effect.

### The ready set is a structural zero here, not a small effect

At all 114 eviction decisions across the phone
groupings, residents split 100.0% all-ready,
0.0% mixed, 0.0% none-ready.
There is never a decision where one resident is READY and another is
not, so the ready bit cannot order candidates. Confirming this is not
a scoring artefact: raising the ready bonus from 0 to 100 changes
reused Prefill by 0.00% of total
input - exactly nothing.

This is not a property of the round-robin trace, and it is not an
artefact of the weights - it follows from the release mechanism
already in place. Whole-Session DONE release evicts a Session at its
final request, so when the next eviction decision is taken, every
resident by construction still has a future request. 'No future
request' and 'already released' are the same state; the ready bit
can only ever report that state, which the policy has already acted
on. The synthetic sweep below confirms this independently: `set`
matches `none` to the digit under Zipf dispatch too, where the
schedule is not lockstep at all.

The consequence for the interface argument is direct: **once the
platform reports DONE, the ready set is redundant.** It is not a
second signal to be obtained, it is a weaker restatement of the
first.

### Exact LCP is the cached-length term under another name

On this fixed-reference trace, a turn's LCP equals the previous
turn's input length in 720/720
cases - exactly. The serial score already reads that same quantity as
`cached_tokens` alongside its recency term.

So `+exact LCP` is not new information. It is a reweighting of a term
the policy already has, which is why its marginal value is near zero
and sometimes negative: the extra copies of the same signal distort
the recency balance. Any argument that the KV manager needs prompt
token identity from the upper layer has to be made on a trace where
LCP and prior input length actually diverge - replayed
conversational text with partial prefix reuse, for example - not on
these fixed-reference traces.

### Queue rank is the real signal

That leaves one observation doing all the work: how many requests away
the queued prompt is. It is content-free - it needs no prompt bytes and
no token overlap, only the scheduler's own ordering - and it recovers
the bulk of what the full `queued_next_turn` protocol was credited with.

The interface claim that survives is therefore narrower and stronger
than the original one: **the KV manager needs the scheduler's dispatch
order, not its prompts.**

## What this means for the earlier near-Oracle number

The validated `queued_next_turn` result should be re-read as a rank
result. Reporting it as evidence that the manager needs prompt-level
visibility would attribute the gain to the two components this sweep
shows carry none.

Everything here is a controlled-protocol observation. It bounds what a
cooperative scheduler *could* give, not what Android currently does;
no device measurement backs any level above `none`.

## Files

- `visibility_device.csv`: per-group ladder results on phone groupings
- `visibility_synthetic.csv`: per-workload ladder under Zipf dispatch
- `visibility.png`: reduction by level, phone groupings

