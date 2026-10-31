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
# onnx/ — ONNX Runtime in-process (plugin EP + `1bit ort-info`)

Optional, off by default. `-DONEBIT_ORT=ON` links the pinned ONNX Runtime into
the engine and adds `1bit ort-info`; OFF builds contain no ORT code path.

## The pin

| item | value |
|---|---|
| ONNX Runtime | **1.23.2** (OrtApi v23) |
| package | `libonnxruntime1.23` / `libonnxruntime-dev` `1.23.2+dfsg-6ubuntu1` (Ubuntu 26.04 *resolute*) |
| library | `/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23.2` |
| sha256 | `9c8f6112e9dcf09ed4e621a1d0a7b6ba49575c7d09f27a652d24a22c8564b13c` |
| headers | `/usr/include/onnxruntime/` (same package, so header and binary are one version) |
| discovery | `pkg-config libonnxruntime` — a vendored build that ships `libonnxruntime.pc` is found the same way |
| licence | MIT, `THIRD_PARTY_NOTICES.md` |

Verify the pin on a new box:

```sh
sha256sum /usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23.2
pkg-config --modversion libonnxruntime
```

## Owning the runtime

ORT is MIT, so we build it ourselves and link that: `onnx/build_ort.sh` clones `v1.23.2`, applies
the one patch GCC 15 needs (`core/common/semver.h` uses `uint32_t` without `<cstdint>`) and installs
to `~/vendor/onnxruntime-1.23.2`. Nothing in `onnx/` changes — discovery is still `pkg-config`, so
the switch is two environment variables:

```sh
export PKG_CONFIG_PATH=$HOME/vendor/onnxruntime-1.23.2/lib/pkgconfig   # find it
export LD_LIBRARY_PATH=$HOME/vendor/onnxruntime-1.23.2/lib            # run it
```

That build carries the same three builtin EPs as the distro package — `[0] DnnlExecutionProvider`,
`[1] XnnpackExecutionProvider`, `[2] CPUExecutionProvider` — because a pin that changes which EPs a
session can partition to is not a pin. `ort_info` prints the same roster for both.

The numbers do not move: with DNNL and XNNPACK present the cached graph decodes a 256-token prompt at
62.3 ms/token against 63.7 for the CPU-only build, and the per-step logits against transformers are
byte-for-byte the same comparison (mean KL 1.238e-07, max|dlogit| 2.122e-05) — i.e. neither EP claims
this graph's fp32 MatMuls and MLAS still runs them. They are there for parity, not speed.

Two build quirks. ORT pins `DNNL_TAG v3.0.1`, which still declares `cmake_minimum_required` below
3.5, so the script exports `CMAKE_POLICY_VERSION_MINIMUM=3.5`; and oneDNN installs to its own prefix, so
`libdnnl.so.3` has to be copied next to `libonnxruntime` or the DNNL provider `.so` loads and the EP is
simply never registered.

## The cached graph (`--cached`)

The default export re-runs the whole sequence every step, so its cost grows with the answer. With
`--cached` the graph takes the cache as an input and returns it grown by one position:

| | uncached | `--cached` |
|---|---|---|
| inputs | `input_ids`, `attention_mask` | `input_ids` `[1,seq]`, `past` |
| outputs | `logits` | `logits`, `present` |

`past`/`present` are **one stacked tensor**, not 56 named ones:
`[2*28, 1, 8, past, 128]` — layer-major, `key` before `value` in each layer pair. `onnx/ort_run.cpp`
carries that single tensor through the loop; 56 separate names would need a name map and dynShapes
cannot place a cache axis against a `*kv` argument tree anyway.

There is no `attention_mask` and no `position_ids` input: for one contiguous sequence both follow
from the cache length, and the traced mask path expands a zero-length `past` axis that ONNX Runtime
rejects (`Expand … invalid expand shape`). Prefill therefore starts from a cache with `past = 0`.

```sh
# export, then decode 64 tokens against the cache
python3 onnx/export_qwen3.py --model-dir ~/models/hf-qwen3-0.6b \
  --out ~/wt/onnx-artifacts/qwen3-0.6b-cached.onnx --cached
./onnx/build/ort_run --model ~/wt/onnx-artifacts/qwen3-0.6b-cached.onnx \
  --ids "15 23 40" -n 64 --threads 8 --cached
```

```
  prefill : 3 tokens in 48.3 ms (logits [151936])
  kv      : 28 layers x 8 heads x 128 (224.0 KB/token)
  decode  : 4 tokens in 197.5 ms (20.26 tok/s, 49.36 ms/token)
  ids     : 13 16 13 16
```

The point of it, measured on a 256-token prompt, `-n 16 --threads 8`, same box:

| | decode ms/token | tok/s |
|---|---|---|
| uncached (re-runs 256 positions per step) | 291.22 | 3.43 |
| `--cached` | 63.66 | 15.71 |

4.6x at 256 tokens, and the gap widens with context because only the cache is re-read instead of the
whole prefix being recomputed. At a 3-token prompt it is only 1.2x (50.72 vs 60.07 ms/token) — there
the 2.4 GB weight stream and `lm_head` dominate, not attention. Prefill costs the same either way
(210 ms for 256 tokens).

Correctness is checked against transformers running the same greedy loop with a real cache:
`compare_logits.py` over both `--dump-logits` directories agrees to a mean KL of 1.2e-07 nats with
top-1 matching at every step, so the cached graph is the same function, not just a plausible one.

## Build and run

```sh
# standalone probe: no HIP, no submodules, seconds
cmake -S onnx -B onnx/build && cmake --build onnx/build -j
ctest --test-dir onnx/build --output-on-failure
./onnx/build/ort_info

# in the engine (adds the `1bit ort-info` subcommand)
cmake -S . -B build -DONEBIT_ORT=ON   # plus the usual engine flags
cmake --build build --target onebit
./build/1bit ort-info
```

Expected shape of the output:

```
onebit ort-info
  ortv    : 1.23.2
  api     : v23 (compile-time ORT_API_VERSION, negotiated above)
  lib     : /usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23
  builtin : 3
    [0] DnnlExecutionProvider
    [1] XnnpackExecutionProvider
    [2] CPUExecutionProvider
  eproster: 1 (partition order; first EP gets the node, no tie-break)
    [0] CPUExecutionProvider   vendor=Microsoft  device=CPU
```

`builtin` is what the shipped `.so` was compiled with; `eproster` is what a
session can actually partition to, **in priority order**. Registration order is
strict priority with no tie-break, so our plugin EP must be registered first and
CPU last.

## Why a probe and not a session

`GetApi(ORT_API_VERSION)` returning non-null is the whole ABI check: it fails
loudly when the header we compiled against is newer than the loaded library,
which is the one mistake that otherwise corrupts memory instead of erroring.
The roster comes from `GetEpDevices`, the same call the session's partitioner
uses.

## Files

- `ort_info.cpp` — `onebit_ort_info_main()`, the probe body (no `main`, so the
  single binary can dispatch to it).
- `ort_info_main.cpp` — standalone `main()` for the `ort_info` target/ctest only.
- `CMakeLists.txt` — works both as an in-tree subdirectory and standalone.
