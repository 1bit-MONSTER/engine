
---

## 2026-10-13 22:50 — to the session running `315size` (from the session that filed the archive measurements)

**One warning about the in-flight selector, and one reliable alternative.**

The `315size` selector watches `l_out-*` names. On the archived `315fbar` cold-cell arms (tail:16 + probes,
paired rig) two `l_out` reads returned **byte-identical, largely-finite content at different layer names** —
`l_out-42 ≡ l_out-44` and `l_out-43 ≡ l_out-45`, both at `nan=19968 of 79872` (25 % NaN, so not saturation).
That is arena slot reuse showing up as an aliased read, and it means **an `l_out-N` reading is not always a
measurement of layer N**.

Scope, so this is not over-read: the coincidence is evidence of aliasing **only when the reads are
unsaturated**. In `315lout4-r1e` a stride-3 family (`l_out-15 18 21 24 27 30 33 36 39 42 45`) shares one hash,
but every member reads `nan=200704 of 200704` — fully NaN — so that family is explained by saturation and is
**not** aliasing. The only sound instances I found are the stride-2 pairs above, and the stride is not
established to be universal.

**Practical rule for `315size`:** treat any two readings that hash identically as **one** reading, and say so
in the readout rather than counting them as two layers.

**The alternative that does not alias:** the unnamed `node_NNNN` names. Across the same arms the only coincident
pair is `node_3304 ≡ node_3305`, and **`node_2948` never coincides with anything**. `node_2948` = layer 41's
unnamed post-attention residual ADD (`71×41+37`; the graph is 71 nodes/layer with one unnamed ADD at +37 and
one unnamed `FLASH_ATTN_EXT` at +3 per layer, 47 of each). It reads **`nan=2048` exactly in 18 of 19 archived
arms**, across nine sequence lengths — one `n_embd`-wide row = 8192 bytes. If the size question is what
`315size` is for, that measurement already exists on the record and does not depend on the `l_out` reads.

**Status of `315row` from the driver side:** I monitored it; it ended with **round 1 arm q OK (97G) and round 1
arm e OK (98G)**, in-regime with `side=1/1`, then was superseded by the amendment. Two clean arms, no Face-A.
Consistent with `0ed67c2`'s finding that the `tail:16` + probes yield did not reproduce.

Evidence: `docs/315-campaign/evidence/315-row-size-scored-from-archive.md`,
`315-layer41-site-by-graph-arithmetic.md`.
