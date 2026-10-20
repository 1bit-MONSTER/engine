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
the wrong thing. Eight HRX correctness bugs were found by hand within ten days: #308, #309, #310,
#313, #318, #319, #369 and #370. The hardware gate checks every HRX pin bump, and every PR that asks
for it, on the Strix Halo box before merge.

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
and `llama-perplexity`. It then runs four checks:

| check | what | catches |
|---|---|---|
| ops | one `test-backend-ops -b HRX0 -j 4` run over MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET, SSM_CONV, GET_ROWS, ROPE, RMS_NORM, SOFT_MAX | a kernel that disagrees with the CPU (#310, #313, #370, the IQ ports) |
| kat | `tools/kat.py check` on Qwen3-0.6B, ZAYA1-8B and Qwen3.8-27B (Q4_K_M / UD-Q4_K_XL), against `tests/golden/hw-gate/kat-<model>.json` | wrong answers, empty replies, NaN logprobs, nondeterminism, cached-prefix and multi-ubatch bugs, drift from the release pin (#308, #309, #315, #318, #319, #369) |
| kld | `llama-perplexity --kl-divergence` on wiki.test.raw, 512 tokens x 8 chunks, against base logits stored at the release pin | small numeric drift that greedy answers do not show |
| needle | `tests/hrx_long_prompt.py`: the 4.7K-token vault-code needle on Qwen3-0.6B | #319, #369 |

kat repeats every case (three times on Qwen3-0.6B, twice on the others) and fails on any of:
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

One `test-backend-ops` case is a known intermittent failure (KNOWN-ISSUES.md, engine#370:
`MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)`, about 1 run in 40). When that is the only
failing case, the case runs alone three more times and the check passes if all three are clean;
the summary says so.

## Box etiquette

The gate shares the box with other work. It takes the GPU four times: once for all the ops, and
once per model (the server for kat and the needle, then llama-perplexity for the KLD, in one hold).
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

That is about 13 minutes of GPU time in all. Waiting for `~/npu.lock` costs more than the gate
itself: flock is not first-come-first-served, and with about ten jobs queueing, each acquisition
waited 10 to 60 minutes. The summary reports both numbers (waiting, holding).

What the goldens record at the release pin:
- Qwen3-0.6B misses one short needle (`varied-720w`: "25"). The check holds that case to the golden's text instead of the answer.
- On `varied-1620w`, the cached path answers differently from a fresh request at a near-tie: top-2 margin 0.19, a WARN.
- ZAYA1-8B and Qwen3.8-27B answer every case.

## Not covered yet

- **Lock fairness.** `~/npu.lock` is a plain flock, so a job that re-queues quickly can starve the
  gate for an hour. A first-come-first-served queue, or a reserved window for the gate, would bound
  the wall time. The gate's own GPU time is about 13 minutes.
- **Private kernels.** The gate builds the public HRX plumbing only. The private HIP add-on
  (gpu-kernels, `ONEBIT_GPU_PRIVATE`) is not built or checked, because the runner serves a public
  repository.
- **Not exercised:** `1bit serve` itself (the gate talks to llama-server with serve's HRX flags),
  `--parallel`, `--mtp` / `--dflash`, vision, and contexts past 8K.
- **One copy of the KLD bases** on the box. A spare copy of the release-pin set is on ryzen
  `~/backups/hw-gate-kld-0e255d77`.
- **An intermittent bug** that fires about once in 40 runs (#370's class) can pass a single gate
  run. kat repeats every case and the ops step re-runs a known flake, but only repeated runs
  catch such a bug reliably.
