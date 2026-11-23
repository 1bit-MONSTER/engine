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
# Inkling-2-Mini on the Loom route as a Hyperloom 1.2.0 custom workload

[Hyperloom](https://github.com/AMD-AGI/Hyperloom) 1.2.0 (MIT) runs a `--framework custom` workload by running one
script per round and reading one JSON report. This directory is that script for **Inkling-2-Mini served by
loom-serve (hrx-loom 0.3.0) from the loom-fun Loom card** on the Strix Halo box, with Hyperloom itself running on a
different machine (the orchestration host). The box is an appliance with no compiler and is shared by several lanes,
so every round is exactly one ticket on its device lock.

| File | Runs on | What it is |
| --- | --- | --- |
| `custom_radeon8060s.sh` | orchestration host | The entrypoint Hyperloom resolves for `--gpu-type radeon8060s`. Validates the round's levers against a fixed list (names and values), ships the box side, runs ONE `box-hold gpu` ticket over ssh, fetches the output, calls `hl_score.py`. |
| `box_inkling.sh` | box, inside `box-hold gpu sudo -n timeout` | One measurement: RAM check (declared need + 16 GB floor, watchdog), a card overlay with the levers in `model.json`, loom-serve, the gauntlet, cleanup, `provenance.json`. |
| `gauntlet.py` | box | The Inkling gauntlet rows (code, reasoning, sustained 2048, 16.8k needle, 17*23), vendored from the cl-loomfun harness; keeps full replies. |
| `hl_score.py` | orchestration host | Score + fail-closed gate -> `inferencex_result.json`. `--self-test` covers the gate. |
| `levers.json` | Hyperloom | The seed grid EXPLORE sweeps. |

The device-free test is `tests/hyperloom_inkling_loom_test.py` (stub ssh/box-hold/sudo and a fake loom-serve).

## What changed from the 1.1.x adapter (`../custom/`)

* No `HYPERLOOM_BENCHMARK_BACKEND=bypass`, no Magpie, no Ray port: 1.2.0 removed them and measures every round
  through InferenceSuite; `--framework custom` needs only `--framework-path` and `--benchmark-scripts-dir`.
* The script is found by board name (`custom_radeon8060s.sh`); there is no `custom_strixhalo.sh` indirection.
* Hyperloom no longer runs on the box: the box has no toolchain and must not carry agent processes, and its memory
  guard kills large jobs. The measurement crosses ssh as one bounded ticket instead.
* Threshold-named pins (`*_MAX`, `*_MIN`) are refused by 1.2.0, so the gate's bars are constants in `hl_score.py`, not
  `--extra-env` pins.

## One round

1. Levers allowed (anything else in the round's environment stays on the orchestration host):

   | Variable | Values | Means |
   | --- | --- | --- |
   | `HL_DRAFT_DEPTH` | 0-8 | preset `draft_depth` (DSpark; card default 3) |
   | `HL_ROWS`, `HL_CONTEXT` | 1-16, tokens | preset `rows` x `context_capacity` (the slots/context preset; card 4 x 262144; refused above 2x the card's pool) |
   | `HL_PREFILL_CAP` | tokens | preset `prefill_capacity` (card 4096; at most **8192**: the card's `llm.loom` declares `@llm.tokens` in `range(1, 8192)` and refuses to register above it) |
   | `HL_MAX_TOKENS` | tokens | preset `max_tokens` (card 4096) |
   | `HL_SPIN_NS` | ns or `default` | amdgpu `completion_spin_ns` (the card's flags use 4294967295) |
   | `HL_WAIT_MODE` | `active`/`blocking` | `--wait_mode` |
   | `HL_POOL_BACKING` | `fixed`/`elastic` | `--pool_backing` (card flags: fixed) |
   | `HL_WORKERS` | 1-32 | `--workers` (default 8) |
   | `HL_RUNS`, `HL_GAUNTLET_ROWS`, `HL_RAM_NEED_GB` | | repetitions (3), rows, the job's declared RAM need (40 GB) -- pins, not levers |
   | `HL_CARD_NAME` | a name | which card under the box's `cl-hyperloom/cards/` (default the Inkling card; `../loom-qwen38/` pins the 27B) -- a pin, a name only, never a path |

   An unset preset lever keeps the **card's own** value, and spin defaults to none: the baseline is exactly the
   shipped route (preset `chat`, `amdgpu://`, `--pool_backing=fixed`).

   **KV format is not a lever here**: this card's `llm.loom` stores KV as f16 in its kernels; another format is a Loom
   program change, not a config value.
2. The ticket: `box-hold gpu sudo -n timeout -k 30 <HL_TICKET_MIN*60> env <levers> bash box_inkling.sh <out>`.
   `HL_TICKET_MIN` defaults to 40 (the fleet's ticket cap is 45); a measurement takes ~6 minutes.
3. Score: geometric mean of the medians of code decode, reasoning decode, sustained-2048 decode, 16.8k-needle prefill
   and 16.8k-needle decode, in tok/s.
4. Gate (all three halves, fail-closed): every needle run recalls the code, every 17*23 run answers 391, and at least
   80 % of the greedy replies are identical to the reference's over their first 160 characters. The reference is the
   first measurement of the state directory (`HL_STATE`, keyed by card revision + GGUF hashes + loom-serve hash), i.e.
   Hyperloom's baseline. **No KLD half**: loom-serve 0.3.0 exports no logits for this card; every report says so in
   `quality_gate.kld_note`. When a logit reference exists the KLD half (bar 0.01) is added here.
5. Every report carries `bench_summary.provenance`: card revision, loom-serve and GGUF hashes, the exact loom-serve
   command, the power-mode log line, die temperature at start/end, MemAvailable at start and its low-water, and the
   count of amdgpu fault/reset kernel lines in the last hour.

## Launch (orchestration host)

```sh
python -m hyperloom optimize -v --framework custom \
    --framework-path <disposable public engine checkout> \
    --benchmark-scripts-dir <tracked checkout>/tools/hyperloom/inkling-loom \
    --model /data/dev/cl-hyperloom/cards/inkling2-mini-a8354e77 \
    --gpu-type radeon8060s --tp 1 --max-hours 4 --no-kernel --no-enable-roofline \
    --extra-env HL_STATE=<state dir> --extra-env HL_RUNS=3
```

### Orchestration-host notes (1.2.0, measured on ryzen 22-23 Nov 2026)

* Run Hyperloom under its own account (home 0700) with a public-only workspace; only specialists are bwrap-sandboxed.
* Preflight exits 2 unless a credential is in the environment (a bare `claude` login is not accepted). A
  subscription token (`claude setup-token`) read into the launch environment only leaves `~/.claude/config.json`
  alone. `install.sh` also refuses without one and persists whatever it is given into `.env` and the kernel-agent env
  file, so run it with a placeholder and delete that line afterwards.
* `../host_venv_fixup.sh VENV` applies the fix below. With `--framework custom`, the benchmark interpreter is `/opt/venv/bin/python` whenever that file exists; only a
  `vllm` on `PATH` takes precedence (its sibling `python` is used). On a host whose `/opt/venv` belongs to someone else,
  put a refusing `vllm` shim in the run's venv `bin/` (first on `PATH`) and pre-install `ray[default]>=2.44.1` there
  (Python 3.14 has no 2.44.1 wheel; 2.55.0 passes the probe).
* The box is reached with a key restricted by `command=` to `../box_gate.sh`, so the agents' account can run tickets
  of allowlisted box sides and nothing else.

Rounds the box cannot serve are **refused, not measured** (`hl_refuse_unservable` in `../box_ticket.sh`): a model
proposal that carries server arguments (`EXTRA_CUSTOM_ARGS`; there is no server) and a round whose framework checkout
has uncommitted changes (the box serves the Loom card / the installed payload, not that checkout). Before this, five
such rounds of the first run measured the baseline and read as ties. Only the `HL_*` levers reach the box.

Keep the framework agent ON: in 1.2.0 `levers.json` is seeded only by EXPLORE, which is part of the optimize
(framework-agent) phase; `--no-framework-agent --no-kernel` together collapse the run to baseline -> SWEEP and sweep
nothing (the CLI prints that warning). `--no-kernel` alone keeps EXPLORE and drops kernel rewrites.

Run the entrypoint by hand once first (`RESULT_DIR=/some/dir bash custom_radeon8060s.sh`) and read
`inferencex_result.json`.
