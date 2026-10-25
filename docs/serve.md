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
# `1bit serve`: the engine behind one OpenAI-compatible API

The engine runs inside Lemonade ([lemonade.md](lemonade.md)) as a backend, the way Lemonade runs
`llama-server`. Lemonade runs `1bit serve`, and for a GGUF it starts the `llama-server` that serve
names, one HTTP hop ([below](#one-hop-from-lemonade---print-backend)).
On HRX the whole graph runs on the GPU or the model does not load: 43 of 43 configurations in the
2026-10-20 census had 0 nodes off HRX. `1bit serve` exposes nothing but an OpenAI-compatible API,
so it also works on its own with any OpenAI client.

```sh
1bit serve -m <model> [--port 8000] [--host 127.0.0.1]
           [--device auto|npu|hrx|hybrid|cpu|ds4] [--ctx-size N] [--alias NAME]
           [--llama-server PATH] [--ds4 PATH] [--ssd-streaming] [--hrx-libhsa PATH]
           [--prefill-provider PATH]      --device hybrid: NPU prefill, HRX decode (docs/hybrid.md)
           [--hrx-cwsr-placement=auto|on|off]
           [--mtp HEAD.gguf | --dflash DRAFT.gguf | --no-mtp] [--mtp-max N] [--mtp-p-min P] [--mmproj MMPROJ.gguf]
           [--laya | --laya-model DIR] [--route-policy FILE]
           [--pm [--pm-experts FILE] [--lemonade-url URL]]
           [--moe-slots N|auto] [--moe-subst R] [--moe-prefetch N]   # refused in this build, below
           [--parallel N]
           [--embed MODEL.gguf] [--rerank MODEL.gguf]
           [--npu-opt KEY=VALUE ...]
           [--no-mem-check | --mem-check]
           [--watchdog SECS] [--watchdog-prompt-tps N]
           [--strict-device | --no-strict-device] [--placement-report]
           [--print-backend | --print-backend-env]
```

One model per process:

| Endpoint | |
|---|---|
| `GET /health`, `GET /v1/health` | 503 while the model loads, then 200 while every backend is up and answers its own health route; 503 again if one has exited, does not answer within 1.5 s, or the watchdog found it hung ([below](#when-a-backend-hangs)) |
| `GET /v1/models` | the one model (`--alias`, else the file or directory name) |
| `POST /v1/chat/completions` | streamed (SSE) or not; a thinking model's `<think>` block comes back as `reasoning_content` (`"reasoning_format": "none"` keeps it in `content`) |
| `POST /v1/completions` | |
| `POST /v1/embeddings` | with `--embed` (RAG) |
| `POST /v1/rerank` | with `--rerank` (RAG) |
| `POST /v1/embeddings` or `/v1/rerank` | with `--embedding` or `--reranking`: the model itself is an embedding or reranking model |
| `POST /v1/responses` | OpenAI Responses API, streamed or not (llama-server devices) |
| `POST /tokenize`, `/detokenize`, `/apply-template` | llama-server devices |
| `GET /slots`, `POST /slots/<id>?action=…`, `GET /props` | llama-server devices; slot save and restore answer 501 (llama-server runs without `--slot-save-path`) |
| `GET /metrics` | Prometheus text: the engine's own series (GTT, Tctl, CPU fallbacks, program-cache flushes, starts, watchdog kills) on every child-process device, then the backend's own ([below](#metrics)) |

"llama-server devices" means `hrx` and `cpu`. These routes go to the llama-server
behind the model, which is how Lemonade's llamacpp backend reaches them (the `onebit` recipe
inherits it). On `npu` and `ds4` they answer 501.

`1bit serve` has no authentication (SECURITY.md), so it refuses two kinds of request a web
page could make with 403: on a loopback `--host`, any whose `Host` header is not a loopback
name (`localhost`, `127.x.x.x`, `[::1]`), and on any host, a POST whose `Origin` is neither a
loopback origin nor the server's own. API clients send no `Origin`. A reverse proxy in front
of a loopback serve must pass the upstream host (nginx's default `proxy_set_header Host
$proxy_host`), not the public one.

## Will it fit?

Before any llama-server starts (`hrx`, `cpu`, and the `--embed` / `--rerank` servers), serve
estimates what the model will hold and refuses a load the box has no room for, in one line and
in milliseconds, so Lemonade shows the reason instead of an allocation failure or an OOM minutes
into the load. ggml-hrx reports free memory equal to total, so llama.cpp's own `--fit` cannot do
this on HRX0.

```
$ 1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --device hrx --mem-check     # strixhalo, 2026-10-12
Qwen3.8-27B-UD-Q4_K_XL: fits: need 33.00 GiB (Qwen3.8-27B-UD-Q4_K_XL.gguf: weights 16.35 + KV/state 16.15 + compute 0.50) at the model's own ctx 262144 (no --ctx-size), free 57.71 GiB (GTT 73.3 of 120.0 GiB free, MemAvailable 57.7 GiB)
```

A load that does not fit prints the same line as `1bit serve: <model> does not fit: need …;
free memory, lower --ctx-size, or pass --no-mem-check` and exits 3 before any backend starts.
Lemonade's loader logs that line with the exit code and fails the load ("1bit serve failed to
start").

The estimate (`app/mem_admit.h`) reads only the GGUF header, the way `tools/mem-budget`'s
`mem_budget.py model` does:

- **weights:** the GGUF's size, every shard;
- **KV:** per attention layer, KV heads x (key + value length) x context x the cache type's
  bytes (`-ctk` / `-ctv`, f16 by default). MLA models (`kv_lora_rank`) cache the latent and the
  rope dimensions once. Qwen3.5 / 3.8 / Next attend in one layer of `full_attention_interval`.
  Per-layer `head_count_kv` arrays count only the layers with heads. Sliding-window layers count
  the window per sequence plus a ubatch, with llama.cpp's per-architecture pattern for which
  layers are windowed (gpt-oss and Gemma 2: every other layer; Gemma 3: 5 of 6).
- **state:** the gated delta-net recurrent state, per sequence;
- **compute:** 512 MiB (HRX measured 20-505 MiB);
- **extras:** a draft or MTP head (`-md`, with its KV), `--mmproj`.

The context is `--ctx-size` (or a recipe's `-c`). Without one, llama-server takes the model's own
on HRX0 (HRX reports its whole pool free, so `--fit` shrinks nothing), and serve judges that. On the
CPU `--fit` shrinks an unset context, so serve judges llama.cpp's 4096 floor.

Free is `min(GTT total - GTT used, MemAvailable)` when a backend runs on HRX0, since the GTT pool
is the same DRAM, and MemAvailable on the CPU. A load is refused only when the need plus 1 GiB is
more than that. Two things grow after load, so they only warn:
- the HRX graph-program cache (2-9 GiB as prompt shapes vary);
- landing at or above 95 GiB of GTT, where engine #315's faults were seen.

- `--no-mem-check` (or `ONEBIT_MEM_CHECK=0`) skips the check.
- `--mem-check` prints the estimate and exits: 0 when it fits, 3 when it does not.
- A refused load exits with status 3 too.

ctest `mem_admit` checks the KV and state sizes against HRX0 server logs:

| model | ctx | KV/state |
|---|---|---|
| Qwen3-0.6B | 4096 | 448 MiB |
| GLM-4.7-Flash (MLA) | 2048 | 105.75 MiB |
| ZAYA1-8B | 2048 | 80 MiB |
| Qwen3.6-35B-A3B | 2048 | 40 + 62.8 MiB |

ctest `mem_admit_route` checks that a sparse 4 TiB GGUF is refused in milliseconds, before any
backend starts. On strixhalo (2026-10-12), estimates for 15 model and context pairs agree with
`mem_budget.py model` (KV from llama-fit-params) within 1.1%. The pairs cover Qwen3 0.6B/8B,
GLM-4.7-Flash, ZAYA1-8B, Qwen3.6-35B-A3B, Qwen3.8-27B, Qwen3-Coder-30B-A3B, gpt-oss-20b and
MiniCPM4-8B, at 2K-262K context.
## When a backend hangs

A hung GPU dispatch (a wedged HRX queue, a #315 fault) stops llama-server's main loop, while its
HTTP thread keeps answering `/health` with 200. So `1bit serve` checks two things itself, for the
backends it runs as child processes (`hrx`, `cpu`, `ds4`; the in-process NPU route
is not covered):

- **`/health` follows the backend.** Once the model is loaded, serve asks each backend's own
  health route (`/health`, or `/v1/models` for DwarfStar) and answers 503 with the reason when
  one has exited, does not answer within 1.5 s (Lemonade's probe gives up at 2 s), or the watchdog below has fired. Lemonade's
  BackendWatchdog probes this route.
- **A progress watchdog.** Every chat, completion and Responses request serve forwards is
  tracked. Progress is a byte from the backend on that request, or the backend's main loop
  answering `GET /slots`, which serve asks every few seconds while requests are in flight
  (llama-server answers it only between decode steps). A request with no progress for
  `--watchdog` seconds (default 120; `$ONEBIT_WATCHDOG_SECS` sets the default; 0 turns it off)
  means the backend is hung. Before a request's first byte it gets prompt tokens /
  `--watchdog-prompt-tps` seconds more (default 50 tok/s; tokens are estimated as request bytes
  / 4), so a long prompt is not taken for a hang.

When the watchdog fires, serve logs the request, how long it waited, the CPU's Tctl and the
backend's last stderr lines (serve passes the backend's stderr through and keeps its tail, which
is where an HSA status shows up), SIGKILLs the backend and exits with status 1. Lemonade then
sees the backend process gone and loads the model again. A process that does not die after
SIGKILL (stuck in the kernel) is reported as such; serve exits anyway.

**A lost GPU.** After a GPU reset (an MES hang, a MODE2 reset), a backend keeps running but
cannot use the device again: ggml-hrx logs `AMDGPU hardware exception` and llama-server answers
every request with HTTP 500 "Compute error". Serve watches the backend's stderr for that line
(and Vulkan's device-lost error) and treats it like the watchdog firing, whatever `--watchdog`
is set to: it logs the line, kills the backend and exits with status 1, so systemd
(`Restart=always`) or Lemonade starts it again on the reset device.

```
1bit serve: watchdog: POST /v1/chat/completions (streamed) on the hrx backend (port 34901, pid 3672241) made no progress for 121 s (limit 120 s; 211 bytes received in 125 s); killing it and exiting so Lemonade reloads it
1bit serve: watchdog: Tctl 71.5 C
1bit serve: watchdog: the backend's last lines:
1bit serve: watchdog: | ...
```

How this meets Lemonade's BackendWatchdog (`wrapped_server.cpp` in our Lemonade fork, which owns
the restart policy; serve builds none):
- Every 5 s Lemonade checks that the backend process is running. Exiting is serve's signal: Lemonade
  marks the backend reset and reloads the model on the next request. That request is answered
  with `backend_watchdog_reset` ("reloaded automatically on retry").
- During an active request, after 90 s with no activity, Lemonade also probes `/health` (2 s,
  status 200 required). After 3 failed probes it stops the backend itself. A wedged llama-server
  still answers its own `/health`, so this probe alone does not catch a hung GPU. serve's
  watchdog exit is what does, within `--watchdog` seconds plus Lemonade's 5 s poll.
- `LEMONADE_BACKEND_WATCHDOG_*` tune Lemonade's side. `onebit_args` in a recipe passes
  `--watchdog` to serve.

A backend with no `/slots` (DwarfStar, or a llama-server started with `--no-slots`)
shows progress only by sending bytes, so on those only streamed requests are judged: a long
non-streamed answer and a hang look the same from outside. `tests/watchdog_serve.sh` (ctest
`watchdog_serve`, CI included) checks all of this against `tests/fake_backend.py`, whose
`"fake_hang"` request stops its stream and its `/slots` while `/health` stays 200.

## Metrics

Lemonade's own `/metrics` already reports, per model:
- requests, tokens, time to first token and tok/s (`lemonade_model_*`);
- CPU and memory use;
- GTT plus VRAM, as `lemonade_vram_used_gb`.

So `GET /metrics` on serve adds only what the engine sees and Lemonade cannot, in Prometheus
text (format 0.0.4), on every device serve runs as a child process (`hrx`, `cpu`, `ds4`).
The in-process NPU route has none yet. serve's own series come first. The backend's own
`/metrics` follows when it answers one: llama-server is started with `--metrics`, so its
`llamacpp:*` series follow.

| Series | What |
|---|---|
| `onebit_up{device}` | 1; `device` names the route |
| `onebit_serve_starts_total` | loads of this model by serve, kept across restarts |
| `onebit_watchdog_kills_total` | backends the [progress watchdog](#when-a-backend-hangs) killed, kept across restarts |
| `onebit_backend_exits_total` | backends that exited on their own, kept across restarts |
| `onebit_program_cache_flushes_total` | ggml-hrx's "HRX graph program cache over its limit" WARN lines |
| `onebit_cpu_fallback_ops{op}` | engine #390's load report of ops placed off HRX, per op and `op="all"` |
| `onebit_gtt_bytes{kind="total"\|"used"}` | the amdgpu GTT pool HRX allocates from (`mem_info_gtt_*`) |
| `onebit_gtt_fault_zone` | 1 when GTT use is at or above 95 GiB, where engine #315's faults were seen |
| `onebit_tctl_celsius` | CPU Tctl (k10temp) |

The log-derived series come from the backend's stderr, which serve passes through unchanged and
reads line by line (Linux). A series whose line never came is left out, not reported as 0.
The #390 report is read from a line that names "CPU fallback" (or "off HRX"), followed by `OP=count`,
`OP xcount` or `OP: count` pairs, or by a total alone. llama-server drops the library's INFO lines
at its default verbosity, because they map to its trace level. So the report reaches serve only
when it is logged as a WARN or above, as the program-cache flush line is.

Starts, watchdog kills and exits outlive one serve process, because Lemonade restarts a backend
by starting `1bit serve` again. They are kept per model id in
`~/.local/state/1bit/serve/<hash of the id>.json` (the file names the model inside; the home
directory comes from the user database, not `$HOME`).

**How Lemonade picks them up.** Lemonade appends a backend's `/metrics` to its own when the
backend's recipe descriptor sets `exposes_prometheus_metrics`. Each line is renamed
`lemonade_llamacpp_<name>` and labelled with the model. Our fork's `onebit` descriptor has it
`false` today, so turning it on is a one-line change in the fork, and it is Lemonade's call.
Until then, scrape serve's port directly.

`tests/metrics_serve.sh` (ctest `metrics_serve`, CI included) checks every series against
`tests/fake_backend.py`.

## One hop from Lemonade (`--print-backend`)

For a `.gguf` on HRX, Lemonade does not need serve in the request path. Lemonade's `onebit`
recipe (our fork, branch `1bit/one-hop`) runs serve's command with `--print-backend` added:

```
$ 1bit serve -m Qwen3-4B-Q4_K_M.gguf --port 8001 --device hrx --alias Qwen3-4B-1bit --print-backend
{"argv":[".../llama-server","-m","Qwen3-4B-Q4_K_M.gguf","--host","127.0.0.1","--port","8001","--metrics","--device","HRX0","-ngl","99","--jinja","--alias","Qwen3-4B-1bit","--cache-reuse","256"],"device":"hrx","env":{"GGML_HRX_REPORT":"1","GGML_HRX_STRICT":"1",...},"port":8001}
```

serve makes every check it makes before a load (the architecture, [memory](#will-it-fit), the
placement flags, [recipes](#recipes---recipes---no-recipes)) and prints the llama-server it would
start, on the port Lemonade chose, as one JSON line on stdout, then exits. Lemonade starts that
llama-server itself and forwards to it as it does for its own `llamacpp` recipe: one HTTP hop and
two processes (lemond, llama-server) instead of two hops and three.

| Exit | Meaning | Lemonade |
|---|---|---|
| 0 | the plan is on stdout | starts it |
| 3 | the model does not fit ([above](#will-it-fit)) | fails the load with serve's line in its log |
| 5 | the route needs serve in the request path: an NPU model directory, `--device npu`, `hybrid` or `ds4`, `--pm`, `--laya`, `--embed` or `--rerank` | runs `1bit serve` as before |

Any other exit (an older engine without the flag) also falls back to `1bit serve`, and
`LEMONADE_ONEBIT_ONE_HOP=0` in lemond's environment turns one hop off. `env` lists only variables
the caller's environment does not set already, so a value the user set still wins, as under serve.

`--print-backend-env` prints only `env`, one `NAME=VALUE` line each, for a caller that builds the
argv itself: `lemonade-hrx`, the stock Lemonade PPA's HRX recipe on Ubuntu 26.04
([lemonade.md, "Ubuntu 26.04: the PPA's lemond"](lemonade.md#ubuntu-2604-the-ppas-lemond)).

What serve did on the request path is the backend's own on this route:

- **Health.** llama-server's `/health` answers 503 while its main loop has work queued and has
  made no progress for `LLAMA_SERVER_HEALTH_STALL_SECS` (30 s; llama.cpp fork #111), so Lemonade's
  BackendWatchdog probe (above, [When a backend hangs](#when-a-backend-hangs)) now catches a hung
  GPU: three failed probes, and Lemonade stops the backend and reloads the model on the next request.
- **A lost GPU.** llama-server exits with status 1 when ggml-hrx logs `AMDGPU hardware exception`
  (llama.cpp fork, `tools/server/server-device-lost.cpp`; `LLAMA_SERVER_EXIT_ON_DEVICE_LOST=0` keeps
  it up), the same signal serve's #466 exit gave Lemonade.
- **Placement.** Strict placement, the default on `hrx` ([below](#off-hrx-placement-strict-by-default)),
  is `GGML_HRX_STRICT=1` in the plan's `env`, so the llama-server Lemonade starts aborts on the
  first node off HRX, naming it, and Lemonade fails the load with that line in its log.
  `--no-strict-device` (or `ONEBIT_STRICT_DEVICE=0`, or `GGML_HRX_STRICT=0` in lemond's
  environment) leaves it out. serve's own summary line and `onebit_*` [metrics](#metrics) are not
  printed on this route; llama-server's `--metrics` still are.
- **Names.** `--alias` gives replies Lemonade's model name; `reasoning_content` is llama-server's
  own (`--jinja`).
- **Prefix reuse.** `--cache-reuse 256` lets llama-server shift a cached prefix into place when the
  middle of a conversation changes. A recipe's own `--cache-reuse` wins.
- **Warm-up.** serve sends a backend two synthetic requests once it is ready (HRX only): a chat
  request (a 360-token prompt, 8 tokens, temperature 0, `cache_prompt` off), so the first client
  request does not pay HRX's one-time per-shape program build (#492), and a 16-token `/completion`
  prompt (`" a"` x 16, `n_predict` 0), below. On this route the plan carries them as
  `"warmups": [{"path", "body"}, ...]` (and `"warmup"`, the first one, for an older lemond), on `hrx`
  chat plans only; Lemonade's onebit recipe sends them in order after `/health` answers and before it
  marks the model loaded. `ONEBIT_NO_WARMUP=1` in lemond's environment leaves them out of the plan, as
  it turns serve's own warm-up off.

  **Why the 16-token request.** The first batch of 13 to 20 tokens after a load takes ggml-hrx's
  small-batch matmul kernels, which read the weights in a layout ggml-hrx builds on the host the first
  time it is needed: it re-quantizes the affected tensors (Qwen3.8-27B UD-Q4_K_XL: 6.1 s of CPU time,
  +1.0 GiB of GTT, held for the life of the server). A chat request is split into batches at the last
  user message and before its last 4 tokens, so short chats land in that range: Lemonade's chat-short
  prompt (67 tokens) runs as 50 + 13 + 4. First-request prompt time per length on a fresh server
  (Qwen3.8-27B, JIT cache warm, 4 cores): 1-12 tokens 0.1-0.5 s, **13-20 tokens 2.03-2.07 s**, 21-512
  tokens 0.45-1.23 s, rising with the length. The extra request moves
  that build to load time: +2.1 s per load on the 27B, +0.2 s on Qwen3.5-4B. Outputs are unchanged
  (greedy chats with and without it are identical; it changes when the layout is built, not which
  kernels run).

  Lemonade's standard benchmark (`lemonade bench --scenarios chat-short code-short --runs 5
  --warmup 1`, a reload before every run, one hop, lemond on cores 0-3), 2026-10-21, TTFT ms:

  | model | chat-short before -> after | code-short before -> after |
  |---|---|---|
  | Qwen3.8-27B UD-Q4_K_XL (MTP on) | 2,643 -> **796** | 824 -> 818 |
  | Qwen3.5-4B UD-Q4_K_XL | 263 -> **138** | 268 -> 142 |

  Decode is unchanged (27B 25.6 / 29.3 -> 26.2 / 29.6 tok/s, 4B 61.3 / 61.0 -> 61.5 / 60.9).
  The host build runs one thread per hardware thread, limited by the cores the backend may run on;
  these numbers are with 4 (not measured unpinned).
- **Residency.** Keeping the model loaded, and so its HRX graph-program cache warm, is Lemonade's:
  `lemonade load --pinned` (or `"pinned": true`), `max_loaded_models`, and the `evict_idle_timeout`
  recipe option when `auto_evict` is on. A reload starts a new llama-server, whose program cache
  starts cold.

Measured on Strix Halo, 2026-10-20 (performance mode), with `tools/onehop_bench.py`: Lemonade fork
`23972095` (lemond), engine `1e9de48` (`1bit`), the llama-server of package `1bit-hrx-3092596`
(llama.cpp `00f2a68de`, sha256 `5c3669cdf3c4104e...`) for both arms,
Qwen3-4B-Instruct-2507 Q4_K_M (`unsloth/Qwen3-4B-Instruct-2507-GGUF`, sha256 `3605803b982cb64a...`)
on `hrx`, ctx 8192, with `HSA_USERPTR_FOR_PAGED_MEM=0` as the services run. Six rounds per arm
(three for `/tokenize`, 200 requests each), alternating arms; each cell is the range of the rounds'
medians. Every request opens a new connection, as Lemonade's client does.

```sh
tools/onehop_bench.py --arm two --out out --lemond lemond --onebit build/1bit --llama-server <llama-server>
tools/onehop_bench.py --arm one --out out --lemond lemond --onebit build/1bit --llama-server <llama-server>
```

| | two hops (`LEMONADE_ONEBIT_ONE_HOP=0`) | one hop | llama-server direct |
|---|---|---|---|
| processes for the model | 1bit serve + llama-server | llama-server | |
| load, `/api/v1/load` to ready | 1.62 s (2.33 s on the first, cold page cache) | 1.35-1.56 s | |
| `/tokenize` (no GPU work: the hops alone) | 0.53-0.64 ms | 0.36-0.39 ms | 0.11-0.15 ms |
| 1-token chat | 19.9-21.5 ms | 19.3-21.0 ms | 17.5-19.4 ms |
| time to first token, repeated prompt (prompt cache hit) | 19.9-21.1 ms | 19.3-21.4 ms | |
| time to first token, fresh 333-word prompt, seen length | 116.7-125.3 ms | 115.1-119.8 ms | |

The serve hop cost about 0.2 ms a request, under the run-to-run noise of a GPU request: one hop
removes a process and a quarter second of load, not latency anyone sees. The time to first token
is the GPU's.

Reload, one hop (three rounds): a prompt length the server had seen answered its first token in
105.6-118.5 ms after Lemonade was asked to load the model again (pinned) and loaded a second model
(Qwen3-0.6B Q4_0) beside it: the same llama-server, its program cache warm. After an unload and a
load, a new llama-server answered the same length in 227.5-235.7 ms, then about 150 ms and 120 ms
as its program cache filled.

`tests/print_backend.sh` (ctest `print_backend`, CI included) checks the plan and the exit codes
without a GPU.

## Where the model runs

| Model | `--device` | Runs on |
|---|---|---|
| NPU model directory (`model.q4nx` + `npu/`, docs/npu.md) | `auto`, `npu` | the NPU engine, in process (in progress) |
| Qwen3.6-35B-A3B Q4NX directory (`model_type` `qwen3_5_moe`) | `auto`, `npu` | a closed NPU add-on route, in process, in builds with `-DONEBIT_NPU_PRIVATE` ([npu.md](npu.md#add-on-routes)) |
| `.gguf` | `auto`, `hrx` | the HRX build's llama-server on `HRX0` with our HIP kernels (docs/hrx.md), `--mtp` included; every node on HRX in all 43 configurations of the 2026-10-20 census (#476, #480, #481). `auto` exceptions: below |
| `.gguf` | `cpu` | the same llama-server with no GPU layers; `auto` in a build without HRX |
| `.gguf` | `hybrid` | the HRX llama-server decodes, one slot, from a KV cache a prefill provider computed for the prompt (the NPU engine, named by `--prefill-provider` / `ONEBIT_HYBRID_PREFILL`) and serve restored into the slot; without a provider, HRX prefills too ([hybrid.md](hybrid.md)) |
| DwarfStar `.gguf` (DeepSeek V4 Flash, GLM 5.x, Qwen3.8-Flash-Next in its own layouts) | `ds4` | this build's DwarfStar `ds4-server` (HIP; `--ssd-streaming` streams routed experts; docs/dwarfstar.md) |

What `--device auto` does with a `.gguf` that HRX has no route for. Until RFC #213 stage 3 these
fell back to Vulkan (engine #271); with no Vulkan or ROCm build left:

| Case | `auto` |
|---|---|
| an architecture our llama.cpp does not build: anything outside `gguf_architectures.hrx` in registry/architectures.json, which `tools/registry_build.py` reads from the pinned fork and the build compiles in (among them Zyphra Zamba, Zamba2, BlackMamba; Spark2.5, BailingMoeV3, HRM text, MuseGlimmer, Kimi K3, Maple, GraniteSwitch, Granite SWA, HY v4, MiniMax, Dots3 Note, PocketTTS, Qwen3-TTS) | refused on hrx and cpu, pointing at Lemonade's llamacpp backend; Zyphra is to be ported to HRX (Flash-Next's `qwen4exp` was, at pin `e57beb97`) |
| `--moe-slots` | refused: not in this build ([below](#moe-models-larger-than-memory-not-in-this-build)) |
| `--parallel N` > 1 on a gated delta-net model (Qwen3.5, Qwen3.8, Qwen3-Next) | HRX0 with one slot, and a warning on stderr: HRX runs one such sequence at a time (no multi-sequence delta-net yet). `--device cpu` keeps `--parallel N` |
| `--mmproj` | HRX0. **Unverified:** vision on HRX is an open RFC #213 gate, to be checked on ZAYA1-VL; if it fails there, `auto` goes back to the CPU for `--mmproj` |
| `--mtp` | HRX0: Qwen3.8-27B decodes at 36.0 / 26.8 / 29.9 tok/s on code / prose / short with the default draft length (#487; [below](#multi-token-prediction---mtp)) |
| an H32 file (a Hadamard-rotated Q4_0 stamped `onebit.hadamard_q4_0`) | refused on every device: the format is dropped; use the model's UD-Q4_K_XL or another standard GGUF. PrismML's own rotation (`prism.hadamard`, Ternary Bonsai) is a different thing and still runs on HRX ([hrx.md](hrx.md#ternary-bonsai-prismmls-hadamard-folded-ggufs)) |

### Off-HRX placement: strict by default

Engine #390. HRX could leave part of a graph on the CPU - an op its kernels did not serve, a
`GET_ROWS` outside the supported range - and the model still answered, slower, with nothing said
so. The HRX census of 2026-10-20 closed with every runnable cell (43 of 43) all on HRX in every
graph, so a node off HRX is now a bug to report, not a fallback to take: on `hrx` serve refuses it.

**The default.** Every `hrx` backend serve starts (the model's own, an `--embedding` or
`--reranking` model, and the `--embed` / `--rerank` servers beside it) gets `GGML_HRX_STRICT=1`.
ggml-hrx checks each graph the scheduler places and aborts on the first node off HRX, naming it,
and serve exits non-zero with its own line:

```
.../ggml-hrx.cpp:447: placement_observer: GGML_HRX_STRICT=1: "kq-0" is MUL_MAT f32 ne=[2,2,12,1] and is not placed on HRX; 12 of 152 nodes off HRX (MUL_MAT x 8, SOFT_MAX x 4), 9 splits
1bit serve: HRX strict placement refused the model: "kq-0" is MUL_MAT f32 ne=[2,2,12,1] and is not placed on HRX; 12 of 152 nodes off HRX (MUL_MAT x 8, SOFT_MAX x 4), 9 splits. Pass --no-strict-device (or set ONEBIT_STRICT_DEVICE=0) to run those nodes on the CPU instead
```

(jina-reranker-v1-tiny-en Q8_0 with `--reranking`, on a llama-server built before the jina-bert
attention kernels, census gap G10; Strix Halo, 2026-10-20. With `--no-strict-device` the same
load serves `/v1/rerank`.)

The load, the warmup graph and every later graph are checked. A graph that only a request
schedules (a `--parallel N` multi-sequence batch, a long prompt) therefore aborts at that request,
and serve exits with the same line, so Lemonade sees the backend go. The same holds on the
[one-hop route](#one-hop-from-lemonade---print-backend): the plan's `env` carries
`GGML_HRX_STRICT=1`. The default needs no extra log volume: the abort is an error line, and the
backend keeps its default verbosity. `--device cpu` and the other devices are not affected.

**The opt-out.** `--no-strict-device`, or `ONEBIT_STRICT_DEVICE=0` in serve's environment, leaves
`GGML_HRX_STRICT` unset, and HRX runs any node it declines on the CPU, as before. A
`GGML_HRX_STRICT` you set yourself wins over either (`0` off, `1` on), under serve and in the
one-hop plan.

**`--placement-report`** passes `GGML_HRX_REPORT=2`, so the backend reports **every** graph, and
serve prints its own line whenever a graph has more nodes off HRX than every graph before it:

```
placement_observer: placement: 512 nodes, 34 off HRX (MUL_MAT x 30, GET_ROWS x 4), 7 graph splits
1bit serve: HRX placement: 34 of 512 nodes off HRX (GET_ROWS x 4, MUL_MAT x 30), 7 graph splits
```

A graph that is all on HRX prints `1bit serve: HRX placement: all 512 nodes on HRX, 1 graph
splits`; with `, worst of N graphs` when more than one reported. The same numbers are
`onebit_cpu_fallback_ops{op}` in [Metrics](#metrics) (`op="all"` is the total). The warmup graph
alone is not the model: the census found LFM2 `--parallel 4` (20 nodes, 5 splits) and ZAYA1-8B
(1-3 nodes) clean in the warmup report and split in prompt and multi-sequence graphs. To see the
counts for a model that strict mode refuses, pass `--no-strict-device --placement-report`.
Without `--placement-report` serve keeps `GGML_HRX_REPORT=1` (the first graph only), which costs
no log volume.

llama-server drops the library's INFO lines below `-lv 4` (an INFO line is a trace line there,
`common/log.cpp`), so at its default verbosity the report never reaches serve. `--placement-report`
passes `-lv 4` to the backend for that load, which prints the report along with the rest of
llama.cpp's INFO lines.

**`--strict-device`** is the default plus the report: `GGML_HRX_REPORT=2` and `-lv 4`, and if a
report arrives and names off-HRX nodes, serve refuses with the same numbers (a backend that
reached ready despite `GGML_HRX_STRICT`). It needs `--device hrx` or `auto`; `ONEBIT_STRICT_DEVICE=1`
is the same. It and `--no-strict-device` do not combine.

Every flag only sets what the backend already reads (llama.cpp fork #102), and a `GGML_HRX_REPORT`
or `GGML_HRX_STRICT` in the environment wins over what serve passes. `tests/hrx_placement_test.cpp`
(ctest `hrx_placement`, CI included) covers the report line, the summary, the strict abort line and
the default and opt-out environments with no GPU, and `tests/print_backend.sh` (ctest
`print_backend`) the plan's `env`. The [hardware gate](hw-gate.md) fails any gate model with a node
off HRX.

### Removed devices and flags

On the GPU the engine is pure HRX with HIP kernels; no Vulkan, ROCm backend, Metal, ZINC or CUDA (RFC #213 stage 3, 2026-10-01: HRX + NPU, zero Vulkan, zero ROCm; the core strip,
2026-10-04: no ZINC, so no CUDA; 2026-10-12: no MLX/Metal route). Lemonade ships its own backends for those
([lemonade.md](lemonade.md#vulkan-rocm-metal-and-cuda-are-lemonades)). `serve` exits at once, with the reason,
when asked for any of these:

| Asked for | What it was |
|---|---|
| `--device vulkan` | upstream llama.cpp's release on `Vulkan0` |
| `--device rocm` | the ROCmFPX tree's ROCm build on `ROCm0` |
| `--lean` | ROCmFPX's ROCmFP4 / ROCmI4 formats |
| `--prefill-device hrx`, `--prefill-min-tokens` | Vulkan decode with the prompt prefilled on HRX0 over one shared KV cache |
| `--adaptive`, `--adaptive-at` | Vulkan for the first requests, ROCm batching the overflow |
| `--long-model`, `--long-from` | long conversations to a Hadamard-rotated Q4_0 on the ROCm W4A4 route |
| `--device zinc`, `--zinc PATH` | ZINC (zolotukhin/zinc), a Zig GGUF engine on Vulkan, ROCm or CUDA |
| `--device mlx`, `--mlx-server PATH` | lemon-mlx-engine's server (MLX on Metal, Apple Silicon); Lemonade has its own MLX backend |

The DFlash2 one-server route (a Hadamard-rotated file with W4A4 prompts and DFlash2 decode on ROCm)
went with them, and so did its recipes (`rotated-moe-ub1024`, `rocm-dflash-p-min-0.4`), and so did
the H32 format it served (`tools/hadamard_q4_0.py` is removed; serve refuses those files). `--dflash`
itself stays, on HRX and the CPU.

A build without the add-on answers a Qwen3.6-35B-A3B directory saying that its NPU route is not
part of this build. `--npu-opt KEY=VALUE` passes an option through to an add-on route. On a route that keeps its cache between requests,
`usage.prompt_tokens_details.cached_tokens` counts the reused prompt tokens and
`timings.cache` says where they came from. On the NPU a streamed request with
`stream_options.include_usage` gets a last chunk with `usage` and `timings`, as OpenAI's
API sends it.

`chat_template_kwargs.enable_thinking: false` works on every device. The GPU
backends apply the model's own chat template. The NPU route emits what Qwen3's
template does: an empty think block after the assistant prefix. For Qwen3.6-35B-A3B it
also opens `<think>\n` when thinking is on, as that model's template does.

For a `.gguf` the engine starts that server as a private child on a loopback
port and forwards the OpenAI routes to it, streaming included. Replies carry
the served model name. `auto` means HRX for GGUF: it met RFC #213's decode gates (docs/hrx.md), and
it is the engine's only GPU route. With `--laya` (or
`--laya-model DIR`), Laya classifies each conversation and the route policy picks the device:
the built-in policy sends every class to HRX, long documents included, since HRX prefills
Qwen3.8-27B at 335 tok/s pp512 (docs/laya.md). The first turn of
a conversation pays about 0.5 s for the decision.

HRX needs TheRock's HSA runtime: the distro `libhsa` rejects gfx1151's
PM4-emulation probe, and then HRX registers no device. `serve` sets
`IREE_HAL_AMDGPU_LIBHSA_PATH` itself unless you did. It uses `--hrx-libhsa`,
else the build's copy, else the first one under `/opt/rocm-therock`.

The child binaries default to this build's (`-DONEBIT_HRX`, `-DONEBIT_DS4`),
then `$ONEBIT_LLAMA_SERVER` / `$ONEBIT_DS4`, then `llama-server` / `ds4-server` on PATH.

Stopping `serve` (SIGTERM, SIGINT) sends each backend SIGTERM and waits
`ONEBIT_BACKEND_STOP_GRACE_S` seconds (default 30) before SIGKILL, with a warning line
first; a backend that already exited is reaped where `serve` noticed, so its stop
is free. A backend that exits on its own ends `serve` with exit code 1 and the way it
ended on stderr (`a cpu backend exited: signal 6`). The grace matters for a backend that
holds an NPU hardware context: it must get to close it itself.

## The Project Manager (`--pm`)

`--pm` makes the served model the Project Manager ([pm.md](pm.md)): on `/v1/chat/completions`
it runs a tool loop instead of a plain forward. The model answers general conversation itself and
delegates a task that fits an expert's domain with `delegate(expert, task, context)`; the engine
turns that into one chat request to Lemonade (`--lemonade-url`; the `onebit` recipe passes Lemonade's own
address, and by hand the default is `http://127.0.0.1:8000`, not Lemonade's default port 13305, so pass it)
for the expert's catalog id, Lemonade loads the expert beside the PM, and the expert's answer
comes back as the tool result. The experts are listed in `config/pm-experts.json` (built in;
`--pm-experts FILE` replaces it). At most `pm.max_rounds` (4) rounds; a Lemonade that is down
becomes a tool result that says so, and the PM answers alone. Every other route forwards
unchanged. Lemonade's catalog entry `ZAYA1-8B-PM-1bit` runs this (recipe option `onebit_pm`).

## Images (`--mmproj`)

`--mmproj <mmproj.gguf>` hands llama-server a vision projector on the llama.cpp routes
(hrx, cpu; `auto` picks the CPU until HRX vision is checked). Chat messages may then carry `image_url` parts (a `data:` URL or a file
URL), which llama.cpp's mtmd encodes and places in the prompt. The mmproj comes from
`convert_hf_to_gguf.py --mmproj` on the same checkpoint as the model. With it, llama-server runs
with `-b 4096 -ub 4096`: an image is decoded as one ubatch, which models that attend to an image
bidirectionally (ZAYA1-VL, Gemma 3) need. Zyphra's Zamba2-VL and ZAYA1-VL-8B were checked on the
removed Vulkan build ([vulkan.md at 0baf286](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/vulkan.md)). The NPU and DwarfStar routes take no `--mmproj`.

## Multi-token prediction (`--mtp`)

`--mtp <head.gguf>` turns on llama-server's `draft-mtp` speculative decoding: the model's
own MTP head (Unsloth ships it as `MTP/mtp-<model>-*.gguf`) drafts tokens on the same
device, and the model checks them in one batch. `--mtp-max N` caps the draft length;
`--mtp-p-min P` drafts a token only when the head is at least that sure of it.
It works on the llama.cpp devices (`hrx`, `cpu`), and `--device auto` keeps it on HRX0.
A drafted token is kept only when the model agrees, so the output is the model's own.

**A GGUF's own MTP head is on by default on `hrx`.** When the `-m` file carries its MTP layer
(`<architecture>.nextn_predict_layers` > 0, e.g. Unsloth's Qwen3.8-27B UD-Q4_K_XL, `blk.64.nextn.*`)
and no other drafter is asked for, serve adds `--spec-type draft-mtp` with no `-md`: llama-server
loads the head from the model file. That is what Lemonade's own llamacpp recipe does for a model
with the `mtp` label, which Lemonade sets from the same key; both `1bit serve` and the
[one-hop plan](#one-hop-from-lemonade---print-backend) do it. `--mtp-max` and `--mtp-p-min` apply.
Not on `cpu`, not with `--parallel` > 1, not for `--embedding` / `--reranking`, and `--no-mtp` (or
`ONEBIT_NO_MTP=1`; Lemonade: `onebit.args=--no-mtp`) turns it off. A file without the key is
unchanged.

Lemonade's standard benchmark (`lemonade bench --scenarios chat-short code-short --runs 5
--warmup 1`, the pinned CI's protocol, a reload before every run), onebit recipe on the one-hop
route, Qwen3.8-27B UD-Q4_K_XL, Strix Halo, 2026-10-21 (decode tok/s, chat-short / code-short):

| | tok/s | TTFT ms |
|---|---|---|
| no drafter (before) | 11.1 / 11.9 | 2298 / 724 |
| own MTP head (now) | **24.9 / 29.1** | 2424 / 807 |
| own MTP head + one-hop warm-up | **26.2 / 29.5** | 2446 / 824 |
| Lemonade's llamacpp Vulkan, same file, `--spec-type draft-mtp` (README of the 2026-10-21 run) | 24.2 / 28.0 | 1882 / 1882 |

Greedy outputs with and without the head (three 128-token chats): two identical, one diverges at a
near-tie (" Earth" -0.684 against " the" -0.705 without MTP), where the verify batch's numerics
pick the other token; a drafted token is still only kept when the model's verify pass agrees.

On HRX, Qwen3.8-27B UD-Q4_K_XL with its MTP head (default draft length 3, greedy 256-token chats,
medians of 3 fresh servers) decodes at **36.0 / 26.8 / 29.9 tok/s** on code / prose / short, up
from 25.9 / 20.5 / 21.9, since our HIP kernels check 2 to 8 drafted tokens in one pass: the 4-token
verify step went from 121.5 to 88.0 ms ([#487](https://github.com/1bit-MONSTER/engine/pull/487)).
Without MTP it decodes at 12.15 tok/s (tg128, [hrx.md](hrx.md#not-yet)).

The sweep below was taken on the removed Vulkan build (2026-09-24) and is kept for the settings it
found; it has not been repeated on HRX. The sweep (draft length x
`--mtp-p-min`, code / prose / short; 12.2 / 12.3 / 12.3 without MTP):

| Setting | tok/s | Drafts accepted |
|---|---|---|
| `--mtp` (draft length 3, the default) | 35.9 / 28.2 / 28.8 | 94% / 68% / 75% |
| `--mtp-max 4 --mtp-p-min 0.5` | 38.1 / 28.0 / 26.9 | 93% / 71% / 76% |
| **`--mtp-max 6 --mtp-p-min 0.5`** | **42.0** / 27.3 / 25.2 | 84% / 60% / 72% |
| `--mtp-max 8` | 21-25 / 13-20 / 10-20 | 76-88% / 39-65% / 33-78% |

Use the default for chat and `--mtp-max 6 --mtp-p-min 0.5` for code (3.4x): code is
predictable enough to keep long drafts, prose is not. Draft length 8 collapses on every
prompt. Adding n-gram drafting to MTP gains nothing.

**Cap the draft on HRX**: `config/recipes.json` pins `--mtp-max 3` for `qwen35` on `hrx`, which is
also llama-server's default — 32.6 / 26.9 / 28.0 tok/s on the 27B on 2026-10-24. Do not raise it
past 7: from 8 drafts on, acceptance falls to 0.17-0.60 and the longer verify batch costs more than
the drafts return (`--mtp-max 7 --mtp-p-min 0`: 37.9 / 22.4 / 22.4; `--mtp-max 15`: 33.6 / 14.9 /
13.5, even with 16-token verify kernels). Code is the one case where a longer draft still pays, and
only up to 7. Small-active MoE models are the
opposite case: on Qwen3-Coder-30B-A3B (3B active, 88 tok/s then) every draft model
tried (Qwen3 0.6B / 1.7B / 4B) was slower than no drafting, even at 82-87% acceptance.

### Ternary-Bonsai-2-27B: the Qwen3.8 head, and two drafts by default

PrismML's Ternary-Bonsai-2-27B is a `qwen35` model with ternary weights, and its three published files
(PTQ1_0, PQ2_0, the packed Q4_0) hold **the same weights**: the 4096 blocks we compared decode
identically, so picking a file is picking a size, not a model. **PQ2_0 is the pick** — 9.8 GiB against
17.2 for the packed Q4_0, and the fastest of the three at both ends (pp512 553, tg128 21.4 tok/s,
against 237/19.0 and 176/16.1).

Because the geometry is Qwen3.8's, **Qwen3.8-27B's own MTP head attaches** to it: `--mtp
MTP/mtp-Qwen3.8-27B-Q4_0.gguf` drafts and the ternary model verifies. That head is the recommendation —
PQ2_0 with MTP decodes at **43 / 37 / 27 tok/s** on code / prose / short (39/40 on GSM8K, against 20-21
tok/s without drafting), and a DFlash2 drafter works on it too since llama.cpp fork #143 but trails the
MTP head (0.37 against 0.67 acceptance), so it is not the pick here.

The ternary files accept 0.76 of **two**-token drafts against 0.67 at three, so the draft length is 2 for
prose, short answers and translation and 4 when the traffic is code (40.4 / 38.3 / 30.4 / 34.7 tok/s at
two drafts over code / prose / short / translation, 43.7 / 36.0 / 24.6 / 30.1 at four).
`config/recipes.json` pins the 2-draft default for these files only: the recipe matches on the
PrismML metadata key the Hadamard-folded files carry (`prism.hadamard.version`), so it cannot catch a
plain Qwen3.8-27B, which has the same architecture. Raising the draft length is `--mtp-max 4`.

Both the PQ2_0 dense prefill kernels and the ternary verify kernels come with the next private-add-on
pin; on the current one, PQ2_0 prefill runs at 188 tok/s instead of 553.

### Qwen3.8-Flash-Next: drafting from part of the vocabulary

Flash-Next's MTP head computes logits over all 248,320 tokens for every drafted token. That
is the model's 644 MiB Q8_0 output matrix, read three times per decode step at draft length 3.
A draft only proposes; the model checks every token, so the head can be cut down to the
tokens it is likely to propose. `tools/mtp_draft_vocab.py` writes such a head from Unsloth's
*shared* head file and the model's own output rows. Our llama.cpp fork (`qwen4exp`,
`src/models/qwen4exp-draft-vocab.cpp`) runs it on HRX since pin `e57beb97`:

```sh
python3 tools/mtp_draft_vocab.py \
  --mtp MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --target UD-Q4_K_XL/ \
  --ids config/draft-vocab/qwen38-en-code-65536.txt --out mtp-Qwen3.8-Flash-Next-dv65k.gguf
```

`config/draft-vocab/qwen38-en-code-65536.txt` is dime-online's 65,536-id list, picked from
English prose and source code
([qwen3.8-Flash-DGX-UltraFast](https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast),
Apache-2.0, see NOTICE). It makes the head 170 MiB.

Measured with the fork's Vulkan `llama-server` on Strix Halo, 2026-10-01. Settings:
- UD-Q4_K_XL, greedy, draft length 3, `-lm mmap -lzm on`
- decode tok/s, median of rounds 2-3 of 3

| Prompt | Full head | 65,536-id head |
|---|---|---|
| code, 256 tokens | 47.8 | **52.1** |
| prose, 256 tokens | 38.6 | **42.2** |
| code, 768 tokens | 45.8 | **48.3** |
| French, 26 tokens | **46.4** | 39.6 |

- **English and code:** 8-9% faster, the same acceptance, and about 6 ms less per decode step (76 -> 70 ms).
- **Other languages:** text outside the list loses, as the French row shows. Use the full head for such text.
- **Output text:** decoding with a draft is not token-identical to decoding without one, even with the full head. Batched verification flips near-ties (one flip each in prose and long code here). The cut head gives the full head's text in rounds 2-3, with two other flips in round 1.
- **Q4_K_M:** Unsloth's shared Q4_K_M head was no faster and flipped more often.
- **`-lm mmap`:** Flash-Next needs it on this llama.cpp. Without it, the 27.5 GiB per-layer embedding table is copied into RAM next to the ~82 GB the GPU holds, which does not fit in 128 GB.

## DFlash draft models (`--dflash`)

> **Not on this pin.** On llama.cpp pins before fork `e91b608cd` (this one, and the release candidate built on it) a DFlash2 drafter fails to load with `done_getting_tensors: wrong number of tensors; expected 81, got 58`; fixed by fork #139 + #140, lands with the next pin bump.

`--dflash <draft.gguf>` uses a DFlash block-diffusion draft model instead of the MTP head: the
draft emits a whole block of tokens in one pass, reading the model's hidden states, and the
model checks the block in one batch. It turns on llama-server's `draft-dflash` (DFlash2's
candidate selector included). A drafter is trained for one target model; z-lab publishes them
on Hugging Face (for example
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), 2B, Apache-2.0).
`--dflash` and `--mtp` are exclusive. `--mtp-max` and `--mtp-p-min` set the draft length and
threshold for either; without `--mtp-max`, `--dflash` drafts full blocks (llama-server's own
default of 3 throws most of the gain away).

Convert the drafter with upstream llama.cpp's converter. It needs only the target's
`config.json` and tokenizer files, not its weights:

```sh
hf download Qwen/Qwen3.8-27B --include "*.json" "*.jinja" "merges.txt" "vocab.json" --local-dir Qwen3.8-27B-hf
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir DFlash2
python convert_hf_to_gguf.py DFlash2 --target-model-dir Qwen3.8-27B-hf --outtype bf16 --outfile dflash2-bf16.gguf
llama-quantize dflash2-bf16.gguf dflash2-q8_0.gguf Q8_0
1bit serve -m Qwen3.8-27B-Q4_0.gguf --dflash dflash2-q8_0.gguf
```

Measured through `1bit serve` on Strix Halo on 2026-10-24: Qwen3.8-27B UD-Q4_K_XL, the same three
chat prompts as above (code / prose / short, decode tok/s), greedy 256-token answers, medians of
three fresh servers, performance mode, **strict placement on** — the drafter's candidate selector
(its top-k, fill and row-gather nodes) runs on HRX, so the whole drafter graph is on the GPU in one
graph split:

| Drafter | code | prose | short |
|---|---|---|---|
| none | 12.1 | 12.1 | 12.5 |
| `--mtp` (the file's own head, draft length 3) | 32.6 | 26.9 | 28.0 |
| `--dflash` (DFlash2 Q8_0, whole blocks) | **56.6** | **30.8** | 20.6 |
| `--dflash` (DFlash2 Q4_K, whole blocks) | **58.8** | — | — |

DFlash2 is the pick for code (+73% over MTP, 4.7x without drafting) and about level on prose, but a
short answer stops before a block is worth filling, so it loses a third there. **The default stays
MTP for chat; `--dflash` is for code-heavy serving.** Acceptance on the three prompts is 0.88 / 0.42
/ 0.23, and a drafted token is kept only when the model's verify pass agrees, so the output is the
model's own. The Q8_0 drafter is what the numbers above use; a Q8_0 file is also what the drafter's
own README recommends over BF16.

Without `--mtp-max`, `--dflash` drafts the drafter's block minus one (its `dflash.block_size`; 16
when the file does not say). **DFlash2's block is 8 tokens**, not 16: the seven drafts per verify
step are what `1bit serve --dflash` asks for, and they fit the 2-8-token verify kernels.

A non-zero draft threshold throws whole blocks away: on the 27B, `--spec-draft-p-min` 0.2 / 0.4 /
0.6 cut prose from 30.8 to 22.5-23.3 tok/s, so `config/recipes.json` pins `--spec-draft-p-min 0`
for `qwen35` on `hrx` (docs/recipes.md). `--mtp-p-min` sets the same threshold. (The DFlash2
one-server route on the lean ROCm build is gone, above.)

## Recipes (`--recipes`, `--no-recipes`)

Tuned llama-server settings that depend on the model and route go in `config/recipes.json` as
recipes, each with the measurement behind it. The built-in set has two entries today, both for
Qwen3.8-27B on `hrx`: `--spec-draft-p-min 0` with a DFlash2 drafter and `--mtp-max 3` with the file's
own MTP head ([recipes.md](recipes.md#the-built-in-recipes)). Serve prints each recipe it applies. Anything serve sets
itself wins over a recipe. `--recipes FILE` replaces the built-in set and `--no-recipes` turns
recipes off ([recipes.md](recipes.md)). To measure a setting before making it a recipe, use
[`tools/bench.py`](bench.md).

## MoE models larger than memory: not in this build

`--moe-slots` is kept (the option, `--moe-subst`, `--moe-prefetch` and
serve's code for them), but the expert streamer lives in the removed Vulkan llama.cpp pin. Until it
moves to HRX together with Qwen3.8-Flash-Next, `serve` refuses `--moe-slots` with "not available in
this build". What follows describes it as it ran on Vulkan0 (docs/moe-streaming.md).

`--moe-slots N` keeps a MoE model's routed experts in the file and
holds N of them, across all layers, in GPU memory. Decode reads the missing ones from the drive
as the router picks them. The rest of the model loads on the GPU as usual.

```sh
1bit serve -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --moe-slots 4608 --ctx-size 8192
```

Qwen3-Coder-30B has 6,144 experts; at 4,608 slots it decodes 39 tok/s warm, at 1,536 about
6-7 (docs/moe-streaming.md, "Streaming in the inference path"). A model that fits in memory is
faster without it (79-91 tok/s resident).

`--moe-subst R` trades a little accuracy for fewer reads: when the router picks an expert that
is not in memory, a resident one among its next choices takes its place if it scores at least
R times as high. `0.5` cut Coder-30B's drive reads by 20-27% and sped decode up 10-35% at 1,536
and 3,072 slots, for a KL divergence of 0.008-0.012 against the exact model (docs/moe-streaming.md,
"Routing that prefers resident experts"). Without it, routing is exact.

Qwen3.8-Flash-Next UD-Q4_K_XL (111 GB) runs this way from 18-32 GiB of GPU memory: 3.5-6 tok/s
exact at 4,608-9,216 slots. On that model use `--moe-subst 0.9` or none: 0.5 costs a KL
divergence of 0.07 there.

`--moe-slots auto` sizes the cache from free memory: `MemAvailable`, less the other tensors the
GPU holds and a reserve (8 GiB or 15%), divided by the model's mean expert size. On a box with
40 GiB free it gave Flash-Next 9,331 of 24,576 experts (27.2 GiB) beside 5.1 GiB of other
tensors, and `serve` held 145 MiB of anonymous memory. Other processes can take that memory
back later; on a shared box, give a number instead. The gate-ahead prefetch is off unless
`--moe-prefetch N` asks for it (it doubled Flash-Next's decode time).

Prompts are slow when streaming: a batch larger than 8 tokens runs the experts on the CPU
from the mapped file (Flash-Next: 0.4-0.8 tok/s of prompt). Keep prompts short, or see
docs/moe-streaming.md, Next.

## Many requests at once (`--parallel`)

`--parallel N` gives llama-server N slots (`-np N`): requests that arrive together decode
together, one read of the weights per step for all of them (continuous batching, what
vLLM is built on). `--ctx-size` is split across the slots, so size it for all of them.

On HRX, `--parallel N` shares one KV cache across the slots (`-kvu`) so attention stays on the
GPU, and turns off AMD's Qwen attention path, which ignores the other sequences. HRX does not yet
run a gated delta-net model (Qwen3.5, Qwen3.8, Qwen3-Next) with more than one sequence: `serve`
refuses `--parallel` > 1 there, and without `--parallel` gives those models one slot.

The earlier measurements here (Vulkan and ROCm batching, and `--adaptive`, which ran both) went
with those builds (RFC #213 stage 3); see this page at
[0baf286](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/serve.md#many-requests-at-once---parallel).
`--adaptive` and `--long-model` are refused now ([above](#removed-devices-and-flags)).

## RAG (`--embed`, `--rerank`)

`--embed MODEL.gguf` serves `/v1/embeddings` and `--rerank MODEL.gguf` serves `/v1/rerank`,
each from its own llama-server beside the chat model (`--embedding`, `--reranking`, the whole input
in one batch). In a build with HRX that server runs on `HRX0` with one slot and inputs of up to 2048
tokens (HRX runs one sequence per batch); a build without HRX runs it on the CPU, 4 slots and 8192 tokens. `/v1/models` lists them with their
role. A RAG client embeds its documents, retrieves by cosine similarity, reranks the best
few and asks the chat model with them as context, all against the one server:

`--embedding` and `--reranking` serve the model itself in that role, as Lemonade loads
embedding and reranking models: `1bit serve -m nomic-embed-text-v2-moe.Q8_0.gguf --embedding`.
They run on `hrx` or `cpu` (`auto`: HRX where the build has it). On Strix Halo (2026-10-02, two
repeats each): Qwen3-Embedding-0.6B Q8_0 embeddings repeat bit for bit on HRX; bge-reranker-v2-m3
Q8_0 scores repeat exactly (8.609 / -6.756 / -0.401 / -11.020). jina-reranker-v1-tiny still
fails on HRX: its JIT can't link the fp32 matmul-with-bias kernel it needs.

```sh
1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --mtp mtp-Qwen3.8-27B-Q4_0.gguf \
  --embed Qwen3-Embedding-0.6B-Q8_0.gguf --rerank qwen3-reranker-0.6b-q8_0.gguf
```

Measured end to end on Strix Halo (2026-09-24): the engine's own docs and README as the
corpus (93 passages of about 120 words), four questions with known answers, Qwen3.8-27B
UD-Q4_K_XL with `--mtp` answering, Qwen3-Embedding-0.6B Q8_0 embedding (queries prefixed
with its `Instruct: ... Query: ` line):

| Stage | Time |
|---|---|
| Embed the 93 passages | 3.4-5.6 s |
| Embed a question | 17-32 ms |
| Cosine search | 4-11 ms |
| Rerank 8 passages | 0.4-0.6 s |
| Answer (700-1,560-token prompt) | 4-5.6 s at 27-33 tok/s |

| Retrieval | Right |
|---|---|
| **Top 8 by embedding, all 8 as context** | **3/4** (the fourth answered 11.0 ms/token, the same fact as 91 tok/s) |
| Top 3 by embedding | 1/4 |
| Top 8, reranked to 3 by Qwen3-Reranker-0.6B | 1/4: it scores every passage 0.96-1.0 through llama.cpp (two conversions tried) |
| Top 8, reranked to 3 by bge-reranker-v2-m3 | 0/4: it discriminates, but keeps the wrong passages |

So use embeddings with a wide context and no reranker: retrieval is about 25 ms, and a
few thousand extra prompt tokens cost little at 270-320 tok/s prefill. `--rerank` stays
for larger corpora, where cutting 50 candidates to 8 matters more.

## Verified (Strix Halo, 2026-09-23)

`tests/serve_e2e.sh` with Qwen3-0.6B: the Q4_K_M GGUF for the GPU devices and the Q4NX model directory for the NPU. The test checks `/health` 200,
`/v1/models`, a chat that answers "Paris." under the served name, and
streaming:

| Device | Result |
|---|---|
| `npu` | PASS (33 SSE chunks): Qwen3-0.6B NPU model directory |
| `hrx` | PASS (32 SSE chunks), with no environment set up |

Qwen3.6-35B-A3B on the NPU is tested by its add-on ([npu.md](npu.md#add-on-routes)).
Without it, `serve` on that directory exits with the message above.

(The `vulkan` row, PASS with 32 SSE chunks, went with the Vulkan build, and the `zinc` row, PASS
with 6, with ZINC.) With `-DONEBIT_HRX=ON`,
ctest runs `serve_e2e_hrx` and `serve_e2e_cpu`, and in general `serve_e2e_<device>`, when configured with
`-DONEBIT_SERVE_TEST_GGUF=<gguf>`. `smoke_serve` runs everywhere, CI included:
`tests/fake_backend.py` stands in for llama-server. It checks the proxy (`/health`,
`/v1/models`, the reply rename, SSE relay), that no backend outlives a
SIGKILLed `serve`, and that a backend that exits on its own ends `serve` at once.
`watchdog_serve` runs everywhere too (see "When a backend hangs").

## `--precise`: a follow-up turn is bitwise identical to a cold run

A follow-up turn in the same conversation is not bit-identical to a cold run of that conversation. The
follow-up continues from the KV cache and prefills only its new user message (28 tokens in the test below),
where a cold run prefills the conversation in 512-token ubatches - and HRX serves a different token count with a
different plan, so the last bits of the logits differ and a near-tie can flip. Measured on
2026-10-22, 10 greedy two-turn chats per model, follow-up against a cold run of the same conversation.

`--precise` (or `ONEBIT_PRECISE=1`) makes every request re-prefill its whole conversation, which is exactly the
path a cold run takes: with it, 10/10 chats came out identical on Qwen3.8-27B and 9/10 on Qwen3-4B (the
remaining 4B difference is at a 0.0385-nat tie with both arms prefilling the same tokens, i.e. a question about
that path's determinism rather than about batch size). The cost is the full time-to-first-token of the
conversation on every turn instead of only the new tokens: on the 27B a follow-up goes from 672 ms to 4785 ms
after a ~1.6k-token conversation and from 676 ms to 12348 ms after ~5.4k tokens. It is off by default, and the
token stream of a single turn is unchanged either way.

