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
# DwarfStar

[DwarfStar](https://github.com/antirez/ds4) (`antirez/ds4`, MIT) is a native
inference engine written for a few large MoE models: DeepSeek V4 Flash (and V4.1
Flash and PRO on bigger machines), GLM 5.2/5.3 Flash and Qwen3.8-Flash-Next. It has
its own kernels for Metal, CUDA and ROCm, runs Strix Halo (`gfx1151`) as a first-class
target, and can stream routed experts from the SSD. It is not a general GGUF runner:
it loads its own GGUF layouts (`antirez/deepseek-v4-gguf`,
`antirez/deepseek-v4.1-flash-gguf`, and the others its `download_model.sh` lists), and
llama.cpp cannot load those.

`1bit serve --device ds4 -m <DwarfStar .gguf>` runs its `ds4-server` as the model's
backend, behind the same OpenAI API as every other device.

## Build

`third_party/ds4` pins upstream `antirez/ds4` main; `.github/workflows/bump-ds4.yml`
opens a PR when it moves.

```
git submodule update --init --depth 1 third_party/ds4
cmake -B build -G Ninja -DONEBIT_DS4=ON        # ONEBIT_DS4_BACKEND=rocm|cuda|metal|cpu
cmake --build build
```

`scripts/build-ds4.sh <prefix> [backend]` does the work: it builds a copy of the
source under `<prefix>/src/<backend>` (DwarfStar builds in its tree; the submodule
stays clean) and puts `ds4-server`, `ds4` and `ds4-bench` in `<prefix>/<backend>`.
`DS4_TEST=1` then runs DwarfStar's model-free routed-MoE test on the GPU.

**ROCm (Strix Halo)** needs HIP, hipBLAS, hipBLASLt, rocBLAS, rocWMMA and hipCUB. The
script takes `ROCM_PATH`, else TheRock's SDK
(`/opt/rocm-therock/lib/python3*/site-packages/_rocm_sdk_devel`), else `/opt/rocm`.
It passes that SDK's `include/` with `-isystem`: clang otherwise searches it after
`/usr/include`, and a distro HIP there (another version) shadows TheRock's headers
and the build fails (`use of undeclared identifier '__ocml_exp10_f32'`).

## Serving

```
1bit serve --device ds4 -m ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --ctx-size 8192
```

| `1bit serve` | `ds4-server` |
|---|---|
| `-m FILE` | `-m FILE` |
| `--ctx-size N` | `--ctx N` |
| `--ssd-streaming` | `--ssd-streaming`: routed experts from the SSD instead of full residency (GLM 5.x needs it on 128 GB) |
| `--ds4 PATH` | a different `ds4-server` (else `$ONEBIT_DS4`, else this build's, else `ds4-server` on PATH) |

`ds4-server` opens its port only after the model has loaded, so `1bit serve` waits
for its `/v1/models` (it has no `/health`). `--parallel` and `--mtp` are llama.cpp
options and are refused on `ds4`; DwarfStar has its own batching
(`--batched-session`) and MTP/DSpark drafting, which `serve` does not expose yet.

Memory: the resident DeepSeek V4 Flash Q2 needs about 81 GiB plus runtime buffers,
and the GPU must see that much (`amdgpu.gttsize` / `ttm.pages_limit`, DwarfStar's
[STRIX_HALO.md](https://github.com/antirez/ds4/blob/main/docs/STRIX_HALO.md)).

## Verified (Strix Halo, ds4 `0aaea5a238fb`, TheRock ROCm)

| Check | Result |
|---|---|
| `DS4_TEST=1 scripts/build-ds4.sh build/ds4 rocm` | builds; routed-MoE MXFP4 test PASS (0 failures at 128 and 512 tokens, variants bitwise OK) |
| `-DONEBIT_DS4=ON` engine build | builds; ctest 7/7 |
| `1bit serve --device ds4 -m <missing file>` | DwarfStar's "cannot open model", then serve exits: backend did not become ready |
| `--ssd-streaming` with another device | refused |
| DeepSeek V4 Flash Q2 (`antirez/deepseek-v4-gguf` rev `f71f23d5`, `...-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, 81 GiB) through `1bit serve --device ds4 --ctx-size 8192` | "The capital of France is Paris.", a working ISO-8601 parser, a correct B-tree walkthrough; decode **14.8 tok/s** (DwarfStar's own log, steady over 256 tokens); first prompt 6.7 s cold, then ~0.5 s |
