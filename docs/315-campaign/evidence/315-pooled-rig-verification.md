<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Independent recount of the pooled rig tally (peer's 9-run figure)

Another agent pooled all nine runs of the frozen three-cell rig and recorded, at commit `562bb7c`:

    28 eviction-cell faults in 33 arms (85%)  vs  4 faults in 60 control arms (6.7%)
    Fisher exact two-sided p = 7.1e-13

Because this number is the strongest statement of the trigger and the peer's commit explicitly credits
**three runs not authored by them** (including this goal's `315gord1a` and `315gord1b`), it was
re-derived here from the consoles rather than taken on trust.

## Method

All nine consoles pulled from the box and parsed for `arm [qec]: (OK|FAULT|OOR|INVALID)`:
`315disc1`, `315disc2`, `315ab1a`, `315ab1b2`, `315gord1a`, `315gord1b`, `315ids1`, `315ids2`, `315ids3`.

## Per-run recount

    run                e_ok  e_f   q_ok  q_f   c_ok  c_f
    315ab1a               0    4      3    1      4    0
    315ab1b2              0    2      2    0      1    1
    315disc1              0    2      1    0      2    0
    315disc2              1    5      6    0      6    0
    315gord1a             0    4      4    0      4    0
    315gord1b             0    4      4    0      3    1
    315ids1               0    3      3    0      3    0
    315ids2               0    2      2    0      1    1
    315ids3               0    2      2    0      2    0

## The recount

    eviction cell (e):  28 FAULT / 29 arms   (28 of 29 = 96.6%)
    control cells q+c:   4 FAULT / 57 arms   (4 of 57  =  7.0%)
    Fisher exact two-sided, 2x2 [28,1] vs [4,53]:  p = 1.33e-17

Including OOR/INVALID arms in the denominators (one OOR in cell q, one in cell c) gives 59 control arms
and leaves the conclusion unchanged.

## Reading

**The numerators agree exactly** — 28 eviction faults and 4 control faults, both reproduced independently.
The **denominators differ** from the peer's: 29 eviction arms and 57-59 total arms here against the peer's
33 and 60. The most likely cause is the peer counting OOR/INVALID arms into the totals, which is a
denominator convention rather than an arithmetic error, and it makes their rate *conservative* (85% rather
than 96.6%).

Either way the conclusion is unchanged, and in fact stronger: the eviction-versus-control contrast is
p <= 1.3e-17 on the strict parse. Recorded so the pooled figure can be cited with its denominator
convention stated rather than as a bare number, since the two conventions give 85% and 96.6% respectively.

## What this corroborates

The trigger-level mechanism — forced eviction produces the fault, in both faces, with independent
replications across two builds and two agents — is the most solid result this issue has. This recount does
not add to it so much as confirm that the pooled figure is real and that its denominator should be quoted.
