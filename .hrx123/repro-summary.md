# engine#123 repair — reproduction baseline (this box)

Pin: llama.cpp 895d63f0 (engine hrx-123-moe-router-expert-id, engine pin == engine main).
Decode-split: ON (the default; GGML_HRX_DISABLE_DISPATCH unset). Device: HRX0.
Box: Strix Halo, 122 GiB RAM. Page-mover state: ksm=1, compaction_proactiveness=20,
THP always, compact_unevictable_allowed=1 ("box defaults").

## Probe A — 0.6B, 300 identical greedy requests (the #140 probe shape)
Qwen3-0.6B-Q4_K_M, n_predict 1, n_probs 5, temperature 0, seed 1, cache_prompt false;
divergence = first-token top-5 logprob vector differs from the modal vector.

| condition | n | divergent | rate |
|---|---|---|---|
| 16-token prompt, box defaults | 300 | 0 | 0.0% |
| 16-token prompt + forced compaction every 0.2 s | 300 | 0 | 0.0% |
| 16-token prompt, server mlockall()'ed + forced compaction | 300 | 0 | 0.0% |
| 716-token prompt (12 KV blocks) + forced compaction | 300 | 0 | 0.0% |
| 716-token prompt, GGML_HRX_FA_PARTIAL_ALIGN=256 (the rig oracle) + forced compaction | 100 | 0 | 0.0% |

The decode-split kernel is genuinely exercised: the dumped command program for the
16-token request contains 84x flash_attention_decode_split_f32_f16_wmma_next_q8 per
program (GGML_HRX_DUMP_COMMAND_PROGRAM_DIR dump).

## Probe B — 30B MoE, output oracle (GPU vs the known-correct greedy output)
Qwen3-Coder-30B-A3B-Instruct-Q4_K_M, the existing /tmp/longprompt.json (~2671 tokens;
eval'd 2113 ctx), greedy 16 tokens, fresh llama-server per sample.
Correct reference string (matches depth_rate.sh REF and the CPU backend):
` Paris. \n\nThe quick brown fox jumps over the lazy dog near the riverbank`

| condition | n | match | divergent | rate |
|---|---|---|---|---|
| fresh server, one at a time | 8 | 8 | 0 | 0.0% |
| 8 fresh servers concurrently (self-induced memory pressure) | 13 valid | 7 | 6 | 46.2% |

Every divergent sample degenerates into the unknown token after the first output(s):
run 3 ` Paris. \n\nThe quick???????????`, run 9 ` Paris. \n\n?????????????`,
run 10 ` Paris???????????????`, run 11 ` Paris.??????????????`,
run 12 ` Paris???????????????`, run 14 ` Paris.??????????????`.
(3 further concurrent samples failed to load - concurrent 30B server load.)

## Reading
The wrongness does not appear on a quiet box under the documented short-context probes,
but appears at ~46% when the process is under memory pressure - the same driver
(compaction moving pages behind in-flight HRX work) that engine#140 measured. This is
the silent wrong-output symptom: the token stream is corrupted with no error raised.

## Post-repair measurements (pin llama.cpp c16424f5 = the loud all-NaN repair)

The repair turns the all-NaN router/attention case into a loud abort
("HRX returned NaN logits ... (engine#123)"), so corruption now shows up as an
abort instead of a wrong token.

| condition | n | match | silent divergent | loud abort |
|---|---|---|---|---|
| 30B MoE, 2113 ctx, fresh server, one at a time, movers ON | 8 | 8 | 0 | 0 |
| 30B MoE, 2113 ctx, 6 fresh servers concurrently, KSM=1 proact=20 | 6 | 0 | 0 | 6 |
| 30B MoE, 2113 ctx, 6 fresh servers concurrently, KSM=0 proact=0 | 6 | 0 | 1 | 5 |

Reading. The driver is memory pressure, not the two named knobs: with 6 concurrent 30B
instances (~108 GiB of a 122 GiB box) the corruption persists even with KSM and proactive
compaction off, because direct compaction under that pressure still migrates pages. This is
the caveat the engine#123/#140 write-up already records ("direct compaction under memory
pressure can still move pages"). Under no induced pressure the fault never appeared
(0/300 and 0/8 above).

Known residual, out of scope for this repair: 1 of the 6 movers-off samples was silently
divergent *without* NaN logits (a truncated wrong token stream), so page migration can also
corrupt a finite value. The loud guard covers the NaN class only; a finite-value corruption
remains silent and would need a different (checksum/parity) mechanism.

## CORRECTION (supersedes the "Known residual" paragraph above)

That paragraph was a **measurement bug**, not a finding. The reference string had been passed
through bash single quotes, so its `\n` was a literal backslash-n and **no** sample could ever
compare equal; a correct match was therefore counted as "silent divergent". With the reference
fixed (real newlines) and every sample archived under `logs/verify-<tag>/`
(`verify_repair.sh` classifies MATCH / LOUD-ABORT / SILENT-WRONG / LAUNCH-FAILED):

| condition | n | match | loud abort | silent wrong | launch OOM |
|---|---|---|---|---|---|
| pre-repair (`895d63f0`), 8 fresh servers concurrently | 16 | 7 | n/a (no guard) | **6** | 3 |
| post-repair (`c16424f5`), 6 concurrent, movers ON | 6 | 1 | 4 | **0** | 1 |
| post-repair (`c16424f5`), 6 concurrent, movers OFF | 6 | 1 | 4 | **0** | 1 |

After the repair there are **zero** silent-wrong samples: every corrupted sample is either the
correct output or a loud abort. The end-to-end repro is deterministic after the repair.

Honest caveats about these runs, stated rather than hidden:
- The `LAUNCH-FAILED` samples are 30B servers that failed to load (6 concurrent instances is
  ~108 GiB of a 122 GiB box). They are launch failures, not corrupted decodes.
- In the movers-OFF run the `fuser` quiet line was taken while one earlier server was still
  shutting down, so that run is "6 concurrent + one held fd" rather than perfectly quiet. The
  per-sample classification is unaffected (each sample is judged by its own outcome).
