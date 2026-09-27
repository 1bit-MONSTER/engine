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
