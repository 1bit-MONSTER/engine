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
# Threat model

## What this project does and where untrusted input enters

The 1bit engine serves language models on AMD Strix Halo machines (the HRX GPU runtime and the
XDNA2 NPU) behind one OpenAI-compatible HTTP API, `1bit serve`, which Lemonade launches as a
backend (docs/lemonade.md, docs/serve.md). On the machine the scanner has, the GPU and NPU routes
cannot run; everything that parses input on the host can, and that is where untrusted input enters:

- **HTTP requests** to `1bit serve` (app/serve.cpp and app/*.cpp): JSON bodies on the chat,
  completion, embedding and reranking routes, headers (`Host`, `Origin`), query strings, and
  streamed responses coming back from the child `llama-server` that serve proxies. serve listens on
  127.0.0.1 and has no authentication of its own; its only request guards are the loopback `Host`
  and `Origin` checks (tests/http_guard_test). Lemonade is the layer that adds auth and binding.
- **Model files** opened by the engine itself: GGUF headers and tensor tables (gguf_meta,
  moe_gguf_index, the registry and routing code that reads architecture metadata), Q4NX model
  directories and their chunk tables (npu/q4nx*, npu_q4nx_test, npu_model_bounds_test), the NPU
  full-ELF containers and the ELF/PDI parsing around them (npu/full_elf*, npu_full_elf_test,
  npu_forward_loader_test), tokenizer JSON (tokenizer_*), and Laya's encoder checkpoint
  (laya_encoder_test, laya_bounds_test). A model file is downloaded from a public hub, so it is
  untrusted.
- **Configuration**: config/recipes.json, config/pm-experts.json, registry/architectures.json and
  the Lemonade-supplied command line (`--npu-opt`, `--pm-experts FILE`, `--lemonade-url`).
- **Tool outputs**: with `--pm`, serve runs a tool loop and calls other models through Lemonade's
  API; the text those models return is untrusted.

## Components that matter most / least

- **Most**: everything under `app/` (serve, the proxy, routing, the Project Manager loop), `npu/`
  host-side loaders and bounds code, the tokenizer loaders, `laya/` scorer input handling,
  `moe/` GGUF indexing, and the Python tools under `tools/` that rewrite model files.
- **Less**: build scripts and `scripts/` (they run with the developer's rights, not on a request
  path), docs, the registry generator.
- **Out of scope here**: `third_party/` (llama.cpp, HRX, Lemonade, the XDNA driver, tokenizers and
  the rest are pinned upstreams; report those to their own projects), GPU and NPU kernel dispatch
  that needs the hardware, and the weekly packaging workflows.

## How to exercise it

- `ctest --test-dir /src/build` runs the host-side suite (the Dockerfile already ran it once).
- `/src/build/1bit --help`, `1bit serve --help`, `1bit serve --recipes`, `1bit route ...` run with
  no hardware. Starting serve needs a model and a child `llama-server`, which this image does not
  have; the request-path code can still be driven through the unit tests (tests/http_guard_test,
  tests/smoke_serve, tests/think_split_test) and by reading app/serve.cpp.
- Loader fuzzing targets: the GGUF, Q4NX, full-ELF and tokenizer readers all take a file path and
  have small fixtures under tests/.

## How you rate severity

- Memory corruption (overflow, use-after-free, out-of-bounds read of attacker-controlled size)
  reachable from an HTTP request is **critical**; reachable from a model, tokenizer or config file
  is **high**.
- A request from a non-loopback origin that bypasses the `Host`/`Origin` guards, or any way for a
  web page to make serve do work, is **high**.
- Path traversal or arbitrary file read/write through a request field or a model directory is
  **high**.
- Denial of service of serve (crash, hang, unbounded allocation) from one request is **medium**;
  from a model file is **low** unless it is also memory corruption.
- Information leaks of local file contents through the API are **high**; leaks of file names or
  versions are **low**.

## Anything to leave alone

- `third_party/` and anything that only runs on the GPU or NPU.
- serve having no authentication and binding to loopback by default is a design decision
  (SECURITY.md); report bypasses of the loopback guards, not the absence of auth.
- The weekly release and bump workflows under `.github/workflows/` run with repository tokens on
  our own runners; they are not a request path.
