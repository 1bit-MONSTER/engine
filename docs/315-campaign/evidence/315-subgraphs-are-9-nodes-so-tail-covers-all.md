<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The subgraphs are tiny (n_nodes=9), so `tail:16` covers each one entirely — the coverage caveat is resolved

Found in the three `315dump-*` arms, which carry a `hrx-producer` channel no other arm has. Read from the
archived logs; no GPU time.

## What the producer channel prints

    0.46.199.243 W [hrx-producer] graph n_nodes=9 leafs=0
    0.46.199.246 W [hrx-producer] tail[3] op=MUL_MAT  name=q-0            bytes=6144  host=0
    0.46.199.247 W [hrx-producer] tail[4] op=RMS_NORM name=norm-0         bytes=6144  host=0
    0.46.199.247 W [hrx-producer] tail[5] op=MUL      name=q-0            bytes=6144  host=0
    0.46.199.248 W [hrx-producer] tail[6] op=MUL_MAT  name=q-0            bytes=40960 host=0
    0.46.199.248 W [hrx-producer] tail[7] op=VIEW     name=q_nope-0       bytes=40704 host=0
    0.46.199.248 W [hrx-producer] tail[8] op=PERMUTE  name=q_nope_perm-0  bytes=40704 host=0
    0.46.199.249 W [hrx-producer] host-nodes listed=0

**`n_nodes=9`.** The graph the HRX backend executes — and therefore the graph the fingerprint's
`tail:16` window runs over — has **nine nodes**. Since 9 < 16, `fp_first = max(0, 9 − 16) = 0`, and
`tail:16` covers **every node of the subgraph**. There is no partial view to worry about.

## Why this matters

My earlier worry (correctly raised, and left unresolved in
`315-onset-layer-is-config-dependent.md`) was that a `tail:16` window might cover only the last 16 nodes of a
much larger graph, making "the first NaN-bearing tensor is in layer 41" a statement about the window's reach
rather than about the fault. With `n_nodes=9` and a 16-node window, the window **cannot** be the reason a node
goes unseen. Two independent confirmations now agree:

- the single-call proof in `315nan-p12l` (`call=158937` printing layer-0 MoE nodes **and** `q_nope-1`), and
- this graph size, which makes full coverage arithmetically inevitable.

So layers 0–40 really were observed and really were clean in the 18 late-onset arms. **The layer-41 onset is
not a coverage artifact.**

## Two further facts it supplies

- **Every listed node is `host=0`** — not host-visible. So the fingerprint's `ggml_backend_tensor_get` on
  these nodes is a genuine device→host download, not a plain memcpy of a host mapping. That is a different
  path from the logits buffer (`HRX0_HOST host_visible=1`), which is why the record's "no copy appears in the
  transfer path" claim was about the logits and does not transfer to these tensors.
- **`host-nodes listed=0`** — no host-side node in that graph at all.

## What it does NOT establish

- **Only one graph per arm is logged** (the channel printed exactly once in each of the three arms), and it is
  the first graph — a q-path chunk at layer 0, `bytes=6144`. Other chunks (MoE blocks, the head) may have
  different node counts, and **nothing on the record measures them**. The full-coverage argument therefore
  holds *for graphs of this size*; if some chunk reaches ≥17 nodes, that chunk's window is partial and its
  early nodes are unobserved. Stated because it is the exact hole this document is closing for one graph.
- It says nothing about the fault's cause, and nothing about which buffer is written.
- Those three arms carry **no** NaN-bearing fingerprint lines, so the producer channel has never been observed
  alongside a fault.

## Reproducing

    grep -ah "graph n_nodes=\|hrx-producer" ~/wt/rt-det/315dump-*/server.log
