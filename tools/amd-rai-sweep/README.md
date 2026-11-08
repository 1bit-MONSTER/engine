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
<!--
-->
# AMD Ryzen AI ONNX packs — image + Whisper sweep

Tools for the image-generation and Whisper half of `~/briefs/TAMDCOL.md`: AMD's public Hugging Face
collections ship four kinds of artifact, and only two of them can run on Linux at all. This directory is
how the rows were measured, with the traps that produced false results written next to the code that
avoids them.

| tool | what it does |
|---|---|
| `run_onnx_image.py` | runs one pack's `*.onnx` on the CPU EP with the repo's own preprocessing; prints load/infer, shapes and the box's load average |
| `run_whisper_onnx.py` | runs a Whisper pack's ONNX encoder+decoder pair, greedy decode, transcript, `RTF(compute)` vs `RTF(incl load)` |
| `sweep_family.py` | per-repo loop: free-space floor → minimal download (with `.onnx.data` sidecars) → run → write the `result:<repo>` row → delete |

## What AMD's packs contain (measured, 2026-11-08)

| family | contents | runs on Linux? |
|---|---|---|
| NPU SD 1.8.0 (`*-amdnpu`, 15) | diffusers layout + `replaced.onnx` **stubs**: the UNet is one `com.ryzenai::DynamicDispatch` node (13 nodes total), the VAE 5 nodes; `dd/*.fconst|.ctrlpkt|.super` | **no** — a plain ORT session refuses with `com.ryzenai:CastAvx(-1) is not a registered function/op`, and AMD's RyzenAI EP is Windows-only |
| NPU creativity 1.7 (`ryzenai-*`, 9) | **plain `ai.onnx`** (realesrgan 482 nodes/156 Conv/0 MatMul, sesr, psfrgan, hrnet-bg-seg); tiles variants add an ORT-VitisAI cache (`context.json` + `*.xmodel`) that is inert without that EP | **yes** (CPU EP) |
| Whisper NPU (7) | `ggml-<size>-encoder-vitisai.rai` + `*_encoder.onnx`/`encoder_model.onnx` + decoder ONNX | **yes** for the ONNX pair; the `.rai` needs the Windows-only runtime |
| NPU M1.0 SD (`*-medusa`, 9) | `.xar` + one `.meta` per static shape | untested — **HF-gated (manual)**, the box token has access to none |

Measured with these tools: `ryzenai-realesrgan` 5.03 s fp32 per 512²→2048² tile (round-trip PSNR 32.6 dB,
shuffled control 8.1); `realesrgan-512x512-tiles-amdnpu` via **AMD's own** `onnx_inference.py` 4.45 s/tile and
**PSNR 33.22 dB / corr 0.9922 against the repo's own reference pair**; `ryzenai-hrnet-bg-seg` mean IoU@0.5
0.669 on DUT-OMRON (40 images, GT from the repo's own downloader); `whisper-tiny`/`base`/`small`/`small-en`
exact transcripts on whisper.cpp's `jfk.wav` (dec 0.16 / 0.16 / 0.28 / 0.37 s per token).

## The four traps, each of which looked like a broken model

1. **Preprocessing is not uniform.** `realesrgan`/`sesr` take **0..255 fp32 with no `/255`**; `hrnet-bg-seg`
   takes `/255` + ImageNet mean/std; `psfrgan` takes `(x/255-0.5)/0.5`. Take it from the repo's own
   `onnx_runner.py:preprocess`. A `/255` on a 0..255 graph scored 18.6 dB for a model that is correct.
2. **Layouts differ.** Packs named `nhwc` (e.g. `sesr-m7`, `psfrgan` int8) declare `[1,H,W,3]` on both input
   and output. Detect from the graph; do not assume NCHW.
3. **A missing `.onnx.data` sidecar is a silent wrong answer.** `whisper-small.en`'s graphs are ~5 MB and
   ~3 MB stubs whose weights live in `<name>.onnx.data`; ORT opens the stub with zero weights and the decoder
   emits `endoftext` at step 1. `sweep_family.py` fetches the sidecar whenever the repo has one.
4. **The mel-bin count differs by export.** `whisper-large-v3`/`large-turbo` are exported with **128** mel bins,
   tiny/base/small/medium with **80**. A `whisper-tiny` feature extractor against a large-v3 graph is rejected by the
   graph — and in a sweep driver that surfaces as "no output line", not as a shape error. The runner reads the bin
   count from the encoder's declared input and picks the matching reference model.
5. **The tokenizer is not shipped, and `.en` vocabularies shift every style id by one**
   (`<|startoftranscript|>` 50257 vs 50258, `<|transcribe|>` 50358 vs 50359, `<|notimestamps|>` 50362 vs
   50363). Ids are derived from the chosen tokenizer's own vocab, never hard-coded.

And one about the *box*: a timing without the load average beside it is not evidence. The same 128² graph
timed 0.5 s, 1.6 s and 4.3 s at `load_avg` 81 on 32 cores, and one Whisper load took 236 s at `load_avg` ≈60.
Both runners print `load_avg` and mark a contended result.

## Usage

```sh
# one repo, writing into the board's cache (tools/amd-rai-board.py renders it)
tools/amd-rai-sweep/sweep_family.py whisper amd/whisper-small-en-onnx-npu \
    --cache ~/briefs/.amd-rai-cache.json --rows ~/briefs/pi-demo-amdcol-rows.json

# the image family needs a test image of the pack's own static size
tools/amd-rai-sweep/sweep_family.py image amd/sesr-m7-512x512-tiles-amdnpu --src=512.png \
    --cache ~/briefs/.amd-rai-cache.json
```

Requires `onnxruntime` (CPU), `numpy`, `pillow` and, for Whisper, `transformers` (its feature extractor and
tokenizer only — no torch). One download at a time; the driver refuses to start below 150 GB free.
