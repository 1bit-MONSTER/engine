---
base_model: unsloth/Inkling-2-Mini-GGUF
license: apache-2.0
tags:
- loom
- loom-serve
- gguf
- strix-halo
- speculative-decoding
- dflash
---

# Inkling-2-Mini (UD-Q4_K_XL) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`Inkling-2-Mini-UD-Q4_K_XL.gguf`), generated on an AMD Ryzen AI Max+ 395 "Strix Halo" with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow. Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/).

## Quick start

```bash
pipx install "hrx-loom[hf]==0.2.0"
loom-fetch huggingface:loom-fun/unsloth-Inkling-2-Mini-loom:chat --materialize=copy --output=/home/sawsriva/Projects/loom/work/inkling2/pub/info.json
loom-serve --model=/home/sawsriva/Projects/loom/work/inkling2/pub/info.json '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed --parameter_mode=preload
```

- `--materialize=copy` puts copies of the fetched package files in the output directory.
- `completion_spin_ns=4294967295` enables active waiting for GPU completions (uses CPU while GPU work is in flight); the card's performance measurements use this setting.
- `--pool_backing=fixed` commits the preset's memory pool up front instead of growing elastic backing.
- `--parameter_mode=preload` only changes how the weights are loaded on the host, so the performance rows were measured without it unless their job served with it.

## Send a request

In another terminal (optional thinking is turned off):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Inkling-2-Mini", "messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": {"enable_thinking": false}}'
```

## Performance vs llama.cpp

| metric | llama.cpp | Loom | margin vs best llama.cpp | roofline | Loom % of roofline |
|---|---|---|---|---|---|
| decode tok/s, short prompt | 15.3 | 46.0 | +201 % | 98.5 | 47 % |
| decode tok/s, 2k context | 14.4 | 43.1 | +199 % | 93.8 | 46 % |
| decode tok/s, 32k context | 10.8 | 14.8 | +37 % | 78.0 | 19 % |
| prefill tok/s, 1k document | 288.9 | 1456.3 | +404 % | 9583.4 | 15 % |
| prefill tok/s, 2k document | 273.1 | 1639.3 | +500 % | 9283.9 | 18 % |
| prefill tok/s, 8k document | 241.3 | 1580.3 | +555 % | 8991.6 | 18 % |
| prefill tok/s, 16k document | 206.1 | 1529.2 | +642 % | 8642.8 | 18 % |
| prefill tok/s, 32k document | 159.7 | 1252.9 | +685 % | 8020.5 | 16 % |
| 4 streams, aggregate decode tok/s | 39.7 | 89.8 | +126 % | 379.1 | 24 % |
| speculative decoding, drafted tokens accepted | — | — | — | n/a | n/a |

**Methodology**

- Compared: Loom against llama.cpp ROCm `llamacpp-rocm/b1383`; the same GGUF, the same DevLab runner and job, the same prompts; Loom served with the active wait. Values are the median of 1 sessions (4 streams: median of 1); Loom is timed by the client, llama.cpp by its server.
- Out of the box on both engines: Loom serves the preset shipped in this package; llama.cpp runs its build defaults plus only the server arguments listed below (offload, context and slots). No tuning sweep on either engine.
- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): Loom DFlash depth 3; llama.cpp ROCm no drafting (NOT the same choice on both engines). Each engine is loaded once for all these rows.
- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.
- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within +-3 % (the measured same-runner noise) is a tie.
- llama.cpp ROCm server arguments: `-ngl 99 -cram 0 -c 38000 -kvu -np 1 (c4: -c 35024 -np 4)`.
- Measured on AMD DevLab Dispatch Strix Halo runners (`devlab-perf` jobs); runners differ between sessions.
- Margin vs best llama.cpp: (Loom / max(ROCm, Vulkan) - 1) × 100 %, using the available measured tokens/s for each row on this GPU. With only one valid backend, that backend is used; with neither, the margin is not measured. Raw backend values and their measurement sources are retained below and in the table.
- Measured 2026-11-22; device AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 24.04; hrx-loom unknown; source: DevLab Dispatch run.
- Drafted tokens accepted: the share of the drafter's proposed tokens the model accepted during the `spec` gate's requests.
- Roofline: the hardware ceiling for this GGUF on AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151) at 228 GB/s and 44 TFLOP/s sustained at 85 W (spec 59), from loom-roofline `roofline.py` (loom-skills unknown commit); single-pass ceiling (no acceptance measured on this package) for decode, decode at 2k, decode at 32k; 4 streams run without drafting, so its ceiling is single-pass; the 4-stream ceiling of this mixture-of-experts model assumes the four streams route to the same experts (the bound a run may not exceed).
- — not measured; n/a not applicable.

## Features

| feature | status | detail |
|---|---|---|
| API | ✅ | OpenAI-style `/v1/chat/completions` |
| tool calling | ✅ | OpenAI-style `tools` and `tool_calls` |
| reasoning / thinking | ✅ | on by default; `chat_template_kwargs: {"enable_thinking": false}` turns it off |
| reasoning effort | ✅ | `reasoning_effort`: none, minimal, low, medium, high, xhigh, max (default high) |
| speculative decoding | ✅ | DFlash, depth 3, on by default; drafter [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) |
| image input | ❌ | not supported by Loom (the upstream model has it) |
| audio input / output | — | not applicable |
| sampling | ✅ | `temperature`, `top_p`, `top_k`, `seed` |
| context size | ✅ | 262,144 tokens per request |
| concurrency | ✅ | 4 concurrent requests |
| max output tokens | ✅ | 4,096 (preset `max_tokens`) |

## Correctness

| gate | description | result |
|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | ❌ no evidence |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | ✅ pass ¹ |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | ✅ pass ¹ |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | ✅ pass ¹ |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | ✅ pass ¹ |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | ✅ pass ¹ |
| `agent_think_off` | The tool-calling loop with thinking off | ✅ pass ¹ |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | ✅ pass ¹ |
| `logits` | Host logits readback (llm.logits on): at T=0 every returned row's argmax is the device's token, the top-k matches llama.cpp, greedy text is unchanged | ❌ no evidence |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | — not applicable |

- ¹ local run, LEVEL=full, 2026-11-22; device AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 24.04; hrx-loom unknown.
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the files above.
- `REPORT.md`: how the values above were measured: setup, commands and the value of every run.

Versions: card schema `loom-card/3`; loom-skills unknown; hrx-loom unknown; card generator unknown.

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) @ `ea0c79b`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`
- drafter: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) @ `ea0c79b`, `dspark-Inkling-2-Mini-Q8_0.gguf`
