#!/usr/bin/env python3
"""Copyright 2026 bong-water-water-bong
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

Run one of AMD's public "Ryzen AI" image packs on plain ONNX Runtime and report what it actually did.

usage: run_onnx_image.py MODEL.onnx INPUT.png OUT.png --pre no255|imagenet [--ref REFERENCE.png]

Why the two flags exist (both were measured the hard way, and each one turned a correct model into a
plausible-looking defect):
  --pre   AMD's packs do NOT normalise the same way. realesrgan/sesr feed uint8->fp32 with NO /255;
          hrnet-bg-seg feeds /255 + ImageNet mean/std; psfrgan feeds (x/255-0.5)/0.5. Always take it
          from the repo's own onnx_runner.py:preprocess - a /255 on a 0..255 graph scored 18.6 dB for a
          model that is fine.
  --ref   For a QUALITY claim, run the repo's own onnx_inference.py/OnnxRunner against its own reference
          output. This runner zero-pads to the static input and crops to origin*scale, which is exact for
          square inputs but is not bit-identical to AMD's tiling: against the same reference pair this
          scored 17.2 dB where AMD's own runner scored 33.2 dB.

Layouts: AMD ships both NCHW ([1,3,H,W]) and NHWC ([1,H,W,3]) graphs - the "nhwc" packs are named for it
(sesr-m7, psfrgan int8). The layout is read from the graph's declared input AND output shapes, never assumed.

A timing is only evidence with the box's load average beside it: this prints load_avg and flags a contended
result, because the same 128x128 graph timed 0.5 s, 1.6 s and 4.3 s across runs at load_avg 81 on 32 cores.
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image


def load_image(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float32)


def preprocess(a, mode):
    if mode == "no255":                      # realesrgan / sesr: 0..255 fp32, RGB, CHW
        return a.transpose(2, 0, 1)[None]
    if mode == "imagenet":                   # hrnet-bg-seg: /255 + ImageNet mean/std
        x = a / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return x.transpose(2, 0, 1)[None]
    raise SystemExit(f"unknown preprocessing {mode}")


def postprocess(y, mode):
    """Layout-aware: the graph's own output shape decides (NHWC packs output [1,H,W,3] as well)."""
    if mode != "no255":
        return y
    a = y[0]
    hwc = a if (a.ndim == 3 and a.shape[-1] == 3) else a.transpose(1, 2, 0)
    return np.clip(hwc, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pre", choices=("no255", "imagenet"), required=True,
                    help="the repo's own preprocessing; see the module docstring")
    ap.add_argument("--ref", help="a reference output shipped by the repo (optional, enables PSNR/corr)")
    a = ap.parse_args()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.time()
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    t_load = time.time() - t0
    iname = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape          # static packs: NCHW [1,3,H,W] or NHWC [1,H,W,3]
    nhwc = len(shape) == 4 and shape[-1] == 3 and shape[1] != 3
    h, w = (shape[1], shape[2]) if nhwc else (shape[2], shape[3])

    img = load_image(a.input)
    oh, ow = img.shape[:2]
    padded = np.zeros((h, w, 3), np.float32)     # AMD's runners pad bottom/right to the static input
    padded[:min(oh, h), :min(ow, w)] = img[:h, :w]
    x = preprocess(padded, a.pre)
    if nhwc:
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    t0 = time.time()
    y = sess.run(None, {iname: x})[0]
    dt = time.time() - t0
    out = postprocess(y, a.pre)
    if out.dtype != np.uint8:
        out = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
    scale = int(round(y.shape[1 if nhwc else 2] / h))
    if scale > 1 and oh * scale <= out.shape[0] and ow * scale <= out.shape[1]:
        out = out[:oh * scale, :ow * scale]
    Image.fromarray(out).save(a.output)

    line = (f"{os.path.basename(a.model)}: load={t_load:.2f}s infer={dt:.3f}s "
            f"{tuple(x.shape)}->{tuple(y.shape)} pre={a.pre}")
    if a.ref:
        ref = np.asarray(Image.open(a.ref).convert("RGB"), np.float32)
        mine = np.asarray(Image.open(a.output).convert("RGB"), np.float32)
        hh, ww = min(mine.shape[0], ref.shape[0]), min(mine.shape[1], ref.shape[1])
        mse = float(np.mean((mine[:hh, :ww] - ref[:hh, :ww]) ** 2))
        line += (f" PSNR={-10 * np.log10(max(mse, 1e-12) / 255 ** 2):.2f}dB"
                 f" corr={float(np.corrcoef(mine[:hh, :ww].ravel(), ref[:hh, :ww].ravel())[0, 1]):.4f}")
    try:
        loadavg = os.getloadavg()[0]            # NOT `load`: that name is the image loader (shadowed it twice)
    except OSError:
        loadavg = float("nan")
    line += f" load_avg={loadavg:.1f}/{os.cpu_count()}c"
    if loadavg > os.cpu_count() * 0.5:
        line += "  [CONTENDED: not quotable as this model's speed - re-run in a quiet window]"
    print(line)


if __name__ == "__main__":
    main()
