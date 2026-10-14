
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

---

## 2026-10-14 03:55 — the onset layer is set by the READ PATTERN; and `315lout4` did have probes

**1. A correction to `0ed67c2`'s table, because it drove the config switch.** That commit describes
`315lout4` as *"disc rig, 16 `l_out` names, **no probes**"* and infers the configs that yield Face A are the
name list *without* the probes. The console says otherwise:

    extra_env='GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000
               GGML_HRX_TENSOR_FP=l_out-0,l_out-3,...,l_out-47 GGML_HRX_TENSOR_FP_NANONLY=1'

All three probes were present. The amended selector (15 `l_out` names, probes retained) is therefore very
close to `315lout4` — the config that already worked — so the switch is probably right in outcome, but its
stated reason is not what the logs show, and the inference that the probes suppress the yield is unsupported.

**2. The onset layer is a property of the read pattern, not of the probes.** Three conditions already exist in
the archive, and they separate the two variables:

| condition | selector | probes | onset |
|---|---|---|---|
| A — `315size` | 15-name `l_out` list | none | **layer 0** (`l_out-0 nan=1984`, `1790`) |
| B — `315lout4`, `315inp2` | `l_out` name list | **yes** | **layer 0** (`l_out-0 nan=1984`, `nan=46`) |
| C — `315nan`, `315fbar` | **`tail:16`** (all ~9 nodes/chunk) | yes | **layer 41** (`node_2948 nan=2048`, 18/19) |

A vs B holds the selector fixed and changes the probes: onset does not move. B vs C holds the probes fixed and
changes the selector: onset moves 0 → 41. **So the read pattern is the cause**, which also explains why
`tail:16`'s yield would not reproduce: the instrument that reads every node is a different regime from one that
reads three.

`hrx-producer` in the three `315dump-*` arms shows `graph n_nodes=9`, so `tail:16` covers **every** node of a
subgraph — ~9 reads per chunk, against ~1–3 for a name list.

**3. What this implies for the goal.** Neither layer 0 nor layer 41 is *the* site; both are where one
instrument first sees NaN. Any "the corruption is at layer N" claim is instrument-relative unless the read
pattern is fixed. The instrument-independent quantity appears to be the **size** — one `n_embd` row, 2048
floats, `finite + nan = extent` exactly — and that should be re-measured under condition A before being called
universal.

**4. `315cfg` (mine, running).** It tests A′ vs B, which the table above already answers; it is left to finish
only as a formal pre-registered check. Its interesting half is `A vs C`, which the archive has supplied. If it
is still thermally blocked when you read this, stopping it loses little — and note it holds `box.lock`, so it
is the one thing blocking the box right now.

Full evidence: `docs/315-campaign/evidence/315-onset-is-set-by-read-pattern-not-probes.md`.

---

## 2026-10-14 05:39 — `315prod` is QUEUED and holds `box.lock`; the box is load-blocked by foreign work

`315prod` (the producer-side sentinel/payload test, pre-registered in
`docs/315-campaign/evidence/315-producer-probe-preregistration.md` with its two amendments) took `box.lock`
at 05:37:34 and is in its **gate wait**: `tctl = 96 °C` against a required < 80, and `load1 = 30` against a
required < 4.

The load is **not** 315 work and is **not** junk — it is `60 decompile` processes plus java workers (Ghidra /
JADX), i.e. someone's reverse-engineering lane. **I am not touching them**, per the objective's "do not
disturb other workloads". Consequences:

- The gate requires `load1 < 4`, which that CPU work controls, so `315prod` cannot start until it finishes.
  Its own cap is 2 h (aborts with exit 6 at ~07:38), and it will report either way.
- While waiting it holds the lock **and** three composed GLM neighbours at `gtt = 79G`. That is ~79 GB of
  GTT held by a run that is doing nothing. If you need the box or the GTT for the 315 lane, the run is
  yours to stop — kill `~/wt/315prod.sh` and the `315-disc.sh 8 315prod` child; the lock fd is inherited by
  the script's `sleep`, so check `/proc/*/fd` for a leftover holder rather than assuming `flock` cleared.

**Why this run matters, in one line:** it is the only instrument on this record that asks *did the device
write the bytes the consumer reads* — it fills the host-visible region with its own payload `0x7FC0DEAD`
**before** each hooked graph and classifies **after a drained stream**, and the archive contains **zero**
`[hrx-producer] armed` and **zero** `[hrx-producer] anomaly` lines, i.e. it has never once been enabled.
