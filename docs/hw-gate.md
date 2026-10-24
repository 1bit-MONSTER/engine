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
# HRX hardware gate

`ci.yml` runs on a GitHub runner with no GPU, so it cannot see an HRX kernel or pin that computes
the wrong thing. Eight HRX correctness bugs were found by hand within ten days: #308, #309, #310, #313, #318, #319, #369
and #370. The hardware gate checks every HRX pin bump, and every PR that asks for it, on the Strix Halo box before merge.

## When it runs

`.github/workflows/hw-gate.yml`, job `hw_gate`, on the self-hosted runner `strixhalo-engine`:

- **A labelled PR.** Add the label `hw-gate` to a PR from this repository. Adding the label starts
  the job, and every later push runs it again.
- **Every PR that bump-hrx opens** (branch `bump-hrx/*`).
- **By hand.** Actions -> hw-gate -> Run workflow, with a ref (a branch, tag, SHA or `pull/N/head`).
  The run posts the commit status `hw-gate` on the commit it checked, so a PR shows the result.

PRs from forks never run there: the runner is the owner's box.

## What it checks

`scripts/hw-gate.sh` builds this checkout's HRX pair (`third_party/llama.cpp` and
`third_party/hrx-system`) with the configuration of `cmake/hrx.cmake`, plus `test-backend-ops`
and `llama-perplexity`, with the private HIP kernels compiled in as the release builds them: the
private add-on at the commit the release config pins, fetched by
the private-fetch helper with the runner box's `gh` login and passed as
`GGML_HRX_HIP_ADDON_DIR`. A llama.cpp pin and an add-on pin that do not compile together fail
the build step (2026-10-20: an add-on pin used a kernel-side type the pinned
llama.cpp did not have yet), and so does a build whose `libggml-hrx.so` lacks the add-on. A box
that cannot read the add-on stops with exit 2; `HW_GATE_PRIVATE=0` gates the public plumbing
only. On a failed build the job log shows the compiler's error lines without their source
excerpts, since the log is public and the add-on's sources are not. It then runs six checks:

| check | what | catches |
|---|---|---|
| ops | one `test-backend-ops -b HRX0 -j 4` run over MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET, SSM_CONV, GET_ROWS, ROPE, RMS_NORM, SOFT_MAX | a kernel that disagrees with the CPU (#310, #313, #370, the IQ ports) |
| fa-decode-q8 | the add-on's `ggml-hrx-fa-decode-bench` on its q8_0 KV decode kernel against a double-precision reference: K q8_0 with V f16 (`-ctk q8_0` alone) and with V q8_0, Hq/Hkv 16/8 and 32/8, n_kv 1031, 2048 and 4103, six runs in one hold | a mixed K/V type bug test-backend-ops cannot see: it has no FLASH_ATTN_EXT case with K and V of different types (gpu-kernels #11 read f16 V dims 0..31 only) |
| kat | `tools/kat.py check` on five models (below), against `tests/golden/hw-gate/kat-<model>.json` | wrong answers, empty replies, NaN logprobs, nondeterminism, cached-prefix and multi-ubatch bugs, drift from the release pin (#308, #309, #315, #318, #319, #369) |
| kld | `llama-perplexity --kl-divergence` on wiki.test.raw, 512 tokens x 8 chunks, against base logits stored at the release pin | small numeric drift that greedy answers do not show |
| needle | `tests/hrx_long_prompt.py`: the 320-record vault-code needle (7.1K tokens on Qwen3, 7.4K on ZAYA1) on Qwen3-0.6B and ZAYA1-8B | #319, #369 |
| placement | every graph the server scheduled for each model (load, warmup, the kat prompts, the needle) is all on HRX, read from ggml-hrx's placement report | a pin that leaves a node on the CPU, which `1bit serve` now refuses to load by default ([serve.md](serve.md#off-hrx-placement-strict-by-default)) |

The models (`MODEL_TABLE` in the script, files under `~/models`):

| name | file | covers | kat repeats | needle |
|---|---|---|---|---|
| qwen3-0.6b | Qwen3-0.6B-Q4_K_M.gguf | small dense K-quant | 3 | yes |
| zaya1-8b | zaya1-8b-Q4_K_M.gguf | ZAYA1 (16 experts, 1 active per token), K-quant | 2 | yes |
| qwen3.8-27b | qwen38/UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf | gated delta net + attention hybrid, Unsloth dynamic mix | 2 | no |
| qwen3-coder-30b-a3b | Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf | qwen3moe: 128 experts, 8 active per token (MUL_MAT_ID with top-8 routing) | 2 | no |
| qwen3-4b-iq4xs | Qwen3-4B-IQ4_XS.gguf | an IQ quant (IQ4_XS matmul and GET_ROWS) | 2 | no |
| qwen3-0.6b-kq8 | Qwen3-0.6B-Q4_K_M.gguf, served with `-fa on -ctk q8_0` | a q8_0 K cache with an f16 V cache | 2 | yes |
| qwen3-0.6b-kvq8 | Qwen3-0.6B-Q4_K_M.gguf, served with `-fa on -ctk q8_0 -ctv q8_0` | q8_0 K and V caches | 2 | yes |

The last two rows give `llama-server` extra arguments (the fifth column of `MODEL_TABLE`). Their kat
needle (about 4.7K tokens) and the 7.1K-token needle decode past 1,024 KV cells, where the add-on's
split decode kernel (`fa_decode_q8_d128`) takes over from its prompt kernel; the short kat cases
decode through the prompt kernel with one query. Both bugs of gpu-kernels #11 showed only on that
path: a launch parameter registered on the wrong kernel (every long decode failed with "compiled
ABI does not match manifest", HTTP 500) and an f16 V read that covered dims 0..31 only. The rows
run kat, the needle and placement but not kld, since a KLD base is stored per model file and
recorded with the default f16 cache.

kat repeats every case and fails on any of:
- a missing expected answer;
- an empty reply or a NaN logprob;
- two different texts for the same request;
- text that leaves the golden before token 8;
- a logprob more than 0.5 from the golden.

KLD thresholds live per model in `tests/golden/hw-gate/manifest.json` (`limits`). The defaults are:
- mean KLD at most 0.01;
- KLD p99 at most 0.2;
- the same top token at least 95% of the time.

A rerun at the recording pin gives 0 on every one.

The placement check reuses engine #390's report. The kat server runs with `GGML_HRX_REPORT=2`
(one line per graph) and `-lv 4` (llama-server drops the library's INFO lines below it), and with
`GGML_HRX_STRICT=0`, so a node off HRX is counted rather than aborting the server and losing the
kat. `tools/hw_gate_manifest.py placement` then reads the server log the way serve does:

```
placement_observer: placement: 512 nodes, 0 off HRX (none), 1 graph splits
```

Any graph with a node off HRX fails the check, with the worst graph's counts and op types
(`20 of 354 nodes off HRX (MUL_MAT x 20), 5 graph splits; 1 of 41 graphs had nodes off HRX`). A
log with no report at all fails too: the check saw nothing. With `--phases placement` alone the
server answers one short chat request instead of the kat, so a prompt graph and decode graphs are
scheduled. The judge's verdicts are in `tests/hw_gate_test.py` (ctest `hw_gate_judges`).

One `test-backend-ops` case is a known intermittent failure (KNOWN-ISSUES.md, engine#370:
`MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)`, about 1 run in 40). When that is the only
failing case, the case runs alone three more times and the check passes if all three are clean;
the summary says so.

## Box etiquette

The gate shares the box with other work. It takes the GPU nine times: once for all the ops, once
for the fa-decode-q8 runs, and once per `MODEL_TABLE` row (the server for kat and the needle, then
llama-perplexity for the KLD, in one hold).
Before the first of them it takes the fleet's GPU token, `~/GPU-TOKEN`: the file long GPU jobs
(Hyperloom, agents' benchmark holds) write their name into while they own the GPU. The gate waits
while another job holds it, up to `HW_GATE_TOKEN_WAIT` seconds (default 3600), logging the holder;
then it writes `hw-gate <run> pid <pid> <time>` into it and removes it on exit, only if the content
is still its own. A wait that times out fails the run with exit 2 and names the holder. Two gate
runs on 2026-10-20 (#481) died with "server failed (rc 137)" while Hyperloom held the token and
loaded models of its own. Hyperloom's own token_take waits while a `scripts/hw-gate.sh` process
exists, so the two take turns. `HW_GATE_TOKEN=none` skips the token for a caller that already
holds it.

Each of those steps:
- **Lock.** It holds `~/npu.lock` (flock) only while it runs, so other GPU work keeps its turn between steps.
- **Heat.** It waits for Tctl to drop to 75 C, then runs under `~/lb/thermal-run.sh`, which freezes it at 78 C and kills it at 93 C.
- **Memory.** It runs under `scripts/mem-guard.sh` (kills the step below 12 GB MemAvailable). Before a step starts, `tools/mem_budget.py` estimates it. A step that would take GTT to 95 GiB or more (the engine#315 fault zone), or past MemAvailable, is refused. The gate then releases the lock and retries every minute, up to ten times.
- **The PR-Agent model.** coder-llm (Vulkan, :8090) loses its device when long HRX dispatches starve it. The gate runs on `strixhalo-engine`, the runner that also runs the pr-agent jobs, and a runner takes one job at a time, so no review is generating during the gate. Before each step the gate also waits for coder-llm's slots to be idle. Afterwards it reports any new `ring ... timeout` in the kernel log.

The build runs under thermal-run too, and resumes after a thermal kill. Build trees and ccache
persist under `~/.cache/hw-gate`, so a PR that does not move the pins rebuilds in seconds.

## Goldens: where they live

- **kat goldens are in git:** `tests/golden/hw-gate/kat-*.json`. They are the text and per-token logprobs of every case, a few KB per model.
- **KLD base logits are not in git.** They are 0.4-1.6 GB per model (`llama-perplexity` stores
  the whole vocabulary for every scored token), past GitHub's file limit, and only the gate box
  reads them. They live on that box under `~/.cache/hw-gate/kld/<model>.kld`.
  `tests/golden/hw-gate/manifest.json` pins each one: its sha256, the model file fingerprint
  (size + sha256 of the first 64 MiB), the text's sha256, and the pins it was recorded at. The gate
  refuses a base, model or text that does not match. A private repository would add a token and a
  multi-GB download to every run for no gain while the gate has one box.

Record them again only at a release pin, after the gate passes there:

```sh
scripts/hw-gate.sh --phases build,kat,kld --record   # writes kat-*.json, the .kld files and manifest.json
```

and commit the changed `tests/golden/hw-gate/` files in the same PR as the pin they belong to.

## Running it by hand

```sh
scripts/hw-gate.sh                                     # everything, on this checkout
scripts/hw-gate.sh --phases ops --models qwen3-0.6b    # a subset
scripts/hw-gate.sh --bin /path/to/llama.cpp/build/bin  # check binaries built elsewhere (no build)
```

The settings are environment variables at the top of the script: `HW_GATE_STORE`, `HW_GATE_MODELS`,
`HW_GATE_LOCK`, `HW_GATE_OPS`, `HW_GATE_MEM_FLOOR` and the others. Logs and `summary.md` go to
`$HW_GATE_OUT` (default `~/.cache/hw-gate/runs/<time>`). Exit code: 0 pass, 1 a check failed, 2
the box could not run the gate (lock timeout, a model or base file missing, memory refused).

## Measured

Strix Halo, 2026-10-12, at the release pin (main d2a56a4, llama.cpp 0e255d77, hrx-system
98d05d94), on a box shared with about ten other GPU jobs:

| step | holds the GPU lock |
|---|---|
| build, incremental (no pin change) | 6 s, no lock |
| build, cold (llama.cpp + hrx-system, 4 cores under thermal-run) | about 20 min, no lock |
| ops: 611/611 cases pass | 269 s |
| Qwen3-0.6B: kat (3 repeats) + needle | 67 s |
| ZAYA1-8B: kat (2 repeats) | 172 s |
| Qwen3.8-27B: kat (2 repeats) | 213 s |
| KLD: 0.6B / 8B / 27B, 4,096 tokens each | 6 / 24 / 29 s |

That is about 13 minutes of GPU time in all.

Strix Halo, 2026-10-20, performance power mode, main 55929b5 (llama.cpp 00f2a68d, hrx-system
82804197) plus this coverage change, a quiet box (3 s waiting in all):

| step | holds the GPU lock |
|---|---|
| ops: 611/611 cases pass | 20 s |
| Qwen3-0.6B: kat + needle + KLD | 20 s |
| ZAYA1-8B: kat + needle + KLD (the needle: 7,421-token prompt at 1,207 tok/s, about 8 s) | 70 s |
| Qwen3.8-27B: kat + KLD | 184 s |
| Qwen3-Coder-30B-A3B (MoE): kat + KLD | 38 s |
| Qwen3-4B IQ4_XS: kat + KLD | 31 s |

363 s of GPU time in all; the MoE model, the IQ model and the ZAYA1 needle add about 77 s of it.
Waiting for `~/npu.lock` costs more than the gate
itself: flock is not first-come-first-served, and with about ten jobs queueing, each acquisition
waited 10 to 60 minutes. The summary reports both numbers (waiting, holding).

What the goldens record at the release pin:
- Qwen3-0.6B misses one short needle (`varied-720w`: "25"). The check holds that case to the golden's text instead of the answer.
- On `varied-1620w`, the cached path answers differently from a fresh request at a near-tie: top-2 margin 0.19, a WARN.
- ZAYA1-8B and Qwen3.8-27B answer every case.

The MoE and IQ rows (`qwen3-coder-30b-a3b`, `qwen3-4b-iq4xs`) were recorded on 2026-10-20 at
llama.cpp 00f2a68d / hrx-system 82804197 (`manifest.json` `recorded_at`). Both answer every kat case.
Their HRX base logits against a CPU run of the same file (`--device none`) give:

| model | mean KLD | KLD p99 | same top token |
|---|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | 0.0054 | 0.043 | 96.7% |
| Qwen3-4B IQ4_XS | 0.0054 | 0.057 | 96.4% |
| Qwen3-0.6B Q4_K_M (the release-pin base, for comparison) | 0.0080 | 0.039 | 95.8% |

That is the usual GPU-vs-CPU rounding gap of a 4-bit file, not a wrong kernel.

## Not covered yet

- **Lock fairness.** `~/npu.lock` is a plain flock, so a job that re-queues quickly can starve the
  gate for an hour. A first-come-first-served queue, or a reserved window for the gate, would bound
  the wall time. The gate's own GPU time is 6 to 13 minutes.
- **Not exercised:** `1bit serve` itself (the gate talks to llama-server with serve's HRX flags),
  `--parallel`, `--mtp` / `--dflash`, vision, and contexts past 8K.
- **One copy of the KLD bases** on the box. A spare copy of the release-pin set is on ryzen
  `~/backups/hw-gate-kld-0e255d77`; the MoE and IQ bases are in `~/backups/hw-gate-kld-00f2a68d`.
- **An intermittent bug** that fires about once in 40 runs (#370's class) can pass a single gate
  run. kat repeats every case and the ops step re-runs a known flake, but only repeated runs
  catch such a bug reliably.
