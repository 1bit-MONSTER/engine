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

# ComfyUI.cpp

ComfyUI.cpp runs ComfyUI workflows in C++. It is a port of ComfyUI's workflow engine and of
the Stable Diffusion 1.5 nodes on LibTorch, pinned as the submodule `third_party/comfyui.cpp`
([1bit-MONSTER/comfyui.cpp](https://github.com/1bit-MONSTER/comfyui.cpp)).

```
git submodule update --init third_party/comfyui.cpp
cmake -S . -B build -DONEBIT_COMFYUI=ON
cmake --build build
build/1bit comfy workflow.json
```

`1bit comfy` takes a workflow in ComfyUI's API format (the JSON that ComfyUI's "Save (API)"
writes) and runs it. SaveImage writes the PNG to the node's `path` input.

## License

ComfyUI is GPL-3.0, and so is ComfyUI.cpp. The engine is Apache-2.0, so it never links
ComfyUI.cpp. `scripts/build-comfyui.sh` builds it as its own program,
`build/comfyui/comfyui_cpp`, and `1bit comfy` runs that program in place of itself.
`1bit comfy` looks for the binary in `$ONEBIT_COMFYUI` first, then this build's, then
`comfyui_cpp` on `PATH`.

## What runs

Everything a Stable Diffusion 1.5 txt2img or img2img workflow needs:
- CheckpointLoaderSimple, CLIPTextEncode, EmptyLatentImage
- KSampler with `euler` / `normal`, including denoise below 1
- VAEEncode, VAEDecode, LoadImage and SaveImage
- the conditioning, latent and image nodes

It runs on the CPU for now, through LibTorch; the build takes LibTorch from the Python `torch`
package.

## Parity with ComfyUI

ComfyUI.cpp pins ComfyUI itself (`third_party/ComfyUI`, Comfy-Org/ComfyUI `1568e6c`).
`tools/parity_sd15.py` in that repository runs the same workflows through both on CPU fp32 and
compares the images. Measured on Strix Halo with `v1-5-pruned-emaonly.safetensors`, 512x512,
20 steps, cfg 7:

| workflow | PSNR vs ComfyUI | max / mean abs diff (0-255) | pixels off by > 2 |
|---|---|---|---|
| txt2img, "a cat", seed 0 | 50.6 dB | 15 / 0.515 | 0.286% |
| img2img, denoise 0.6, seed 7 | 51.2 dB | 1 / 0.498 | 0.000% |

The remaining difference is fp32 rounding: ComfyUI batches the cond and uncond passes and uses
PyTorch's attention kernel. Through `1bit comfy`, the txt2img workflow takes 48 s and gives the
same 50.6 dB. ComfyUI takes about 35 s on the same CPU.

## Next

- The GPU: LibTorch on ROCm for gfx1151.
- More model families and samplers beyond SD1.5 with Euler.
