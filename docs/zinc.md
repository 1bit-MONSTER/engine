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
# ZINC

[ZINC](https://github.com/zolotukhin/zinc) (MIT) is a single-binary GGUF engine
written in Zig. It has its own kernels for four GPU backends, and serves
`/health`, `/v1/models` and `/v1/chat/completions`:

| Backend | Hardware | Build needs |
|---|---|---|
| `vulkan` | AMD RDNA (Strix Halo gfx1151), any Vulkan GPU | `glslc`, the Vulkan loader |
| `rocm` | AMD, HIP | ROCm in `ROCM_PATH` (default `/opt/rocm`) |
| `cuda` | NVIDIA Ada `sm_89`, Blackwell `sm_120` | CUDA toolkit in `CUDA_HOME` (default `/usr/local/cuda`) |
| `metal` | Apple Silicon | Xcode command-line tools |

The CUDA backend is how the engine reaches NVIDIA GPUs. It is a C shim over the
driver API, NVRTC and cuBLAS (`src/cuda/cuda_shim.c`). Its int8 GEMMs use
`__dp4a`, the CUDA counterpart of AMD's `v_dot4_i32_i8`. Upstream's design notes
are in `third_party/zinc/docs/cuda-backend.md`.

## The pin

`third_party/zinc` is upstream `zolotukhin/zinc` `main` at a fixed commit. ZINC
changes daily, so the engine builds only the commit it pins, and
`.github/workflows/bump-zinc.yml` moves that pin forward (see "How it stays
current").

## Build

```sh
git submodule update --init third_party/zinc
scripts/build-zinc.sh ~/.cache/zinc-pin vulkan   # or rocm, cuda
~/.cache/zinc-pin/vulkan/bin/zinc -m model.gguf --prompt "The capital of France is"
```

The script needs nothing from the system except the backend's own toolkit:

- **Zig.** It downloads the Zig release that ZINC's `build.zig.zon` names as
  `minimum_zig_version` (currently 0.15.2) into `<prefix>/zig-<version>`. The
  download is checked against the sha256 that ziglang.org publishes.
- **Build output.** Zig's caches go under `<prefix>`, so the submodule stays
  clean. The binary is `<prefix>/<backend>/bin/zinc`.

## Verified (pin `c50b4add`, 2026-09-23)

| Backend | Result |
|---|---|
| `vulkan` | builds. Qwen3-0.6B Q4_K_M, prompt `785,6722,315,9625,374`: first token **12095** (" Paris"), the same as the NPU lane and the CPU reference. Prefill 817 tok/s; decode **295 tok/s** (3.4 ms/tok) with `RADV_PERFTEST=coop_matrix`, which ZINC expects on RADV. |
| `rocm` | builds and links the system ROCm 7 (`libamdhip64.so.7`, `libhsa-runtime64`). Not run on a model: ZINC's ROCm forward pass implements only the Qwen3.5/3.6 family, and it rejects `qwen3` GGUFs with `UnsupportedArchitecture`. |
| `cuda` | **RTX 5090** (`sm_120`, rented on Clore.ai; driver 615.71, CUDA 13.2 from NVIDIA's apt repo, `CUDA_HOME=/usr/local/cuda-13.2`). `scripts/build-zinc.sh <prefix> cuda` builds. With `qwen35-9b-q4k-m` (Qwen3.5-9B Q4_K_M, ZINC's catalog) the continuation of "The capital of France is" is **" Paris."**. Decode 167–173 tok/s (5.8–6.0 ms/tok); prefill 309 tok/s on the 5-token prompt. |

ZINC's CUDA and ROCm forward passes share one implementation. It covers only
the Qwen3.5/3.6 family: `qwen3` GGUFs fail with `UnsupportedArchitecture`, and
Qwen3.5-0.8B (`UD-Q4_K_XL`) fails with `MissingTensor`. Vulkan runs `qwen3`.

ZINC has no BF16 weight path, so use a quantized GGUF. For example, the
BF16 golden `Qwen3-0.6B-BF16.gguf` fails with `UnsupportedQuantType` (type 30).

## Served by Lemonade

With `-DONEBIT_ZINC=ON` (and `-DONEBIT_ZINC_BACKEND=vulkan|rocm|cuda`, default
`vulkan`), the build runs `scripts/build-zinc.sh` into `build/zinc/`, and
`1bit lemonade` sets the `zinc` recipe's `zinc.zinc_bin` to that binary. A value
you set yourself wins. The recipe is the local Lemonade delta 7
(`third_party/lemonade/UPSTREAM.md`):

- **Models.** Checkpoints are GGUF files that Lemonade downloads and resolves
  with llamacpp's own rules. `Qwen3-0.6B-ZINC` is
  `unsloth/Qwen3-0.6B-GGUF:Q4_K_M`.
- **Load.** `zinc -m <gguf> -p <port> -c <ctx_size>`, with
  `RADV_PERFTEST=coop_matrix` unless you already set it, then wait on `/health`.
- **Requests.** Sent without `model`, because zinc rejects any id except its
  own. Replies carry the Lemonade name back.

`tests/zinc_lemonade_e2e.sh` (ctest `zinc_lemonade_e2e`) passes on Strix Halo:
pull, "Paris." under `Qwen3-0.6B-ZINC`, served by this build's zinc, and
streaming. Lemonade's `/stats` reports 0 tok/s for zinc, because zinc's replies
have no `timings` block.

## How it stays current

`.github/workflows/bump-zinc.yml` runs daily. When upstream `main` has moved, it
opens a PR that moves `third_party/zinc` and lists the new upstream commits. It
uses the same `HRX_BUMP_TOKEN` secret as `bump-hrx.yml` (see [hrx.md](hrx.md)).
GitHub-hosted CI has no GPU, so before merging a bump, rerun the build and the
smoke test above on Strix Halo.
