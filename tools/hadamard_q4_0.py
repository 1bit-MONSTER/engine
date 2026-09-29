#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""hadamard_q4_0.py — make a Hadamard-rotated Q4_0 GGUF for the lean ROCm route (docs/lean.md).

    tools/hadamard_q4_0.py SOURCE.gguf OUT.gguf [--imatrix IMATRIX.gguf] [--quantize PATH]
                           [--work DIR] [--keep]

SOURCE is a high-precision GGUF (Q8_0, BF16, F16 or F32). Every matmul weight in a transformer
block (attention and FFN projections, the delta-net alpha/beta projections) is rotated along K by
the normalized 32-point Walsh-Hadamard matrix, per 32-element block, in a copy of SOURCE. The
imatrix gets the matching change (each rotated column's importance becomes its block's mean).
llama-quantize then writes Q4_0 for exactly those tensors and a non-Q4_0 type for every other
matmul weight, and stamps `onebit.hadamard_q4_0 = 32`. `1bit serve` sees the stamp, sets
GGML_Q4_0_HADAMARD=1 (the ROCm activation quantizers apply the same rotation) and
GGML_W4A4_TENSORS=all, and refuses the file on devices without the rotation.

The rotation is exact: on the int8 path the rotated file matches an unrotated one quantized the
same way (Qwen3.8-27B: KLD 0.031 vs 0.029 against BF16). What it buys is 4-bit activations
that lose less (KLD 0.055 instead of 0.084) at the W4A4 kernel's prompt speed.

The last step checks the invariant the runtime relies on: every Q4_0 tensor in OUT is a rotated
one (a tied output embedding is kept at Q8_0). A file that breaks it is deleted.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "third_party", "llama.cpp-rocmfpx", "gguf-py"),
          os.path.join(HERE, "..", "third_party", "llama.cpp", "gguf-py")):
    if os.path.isdir(p):
        sys.path.insert(0, p)
        break
import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

STAMP = "onebit.hadamard_q4_0"
ROLES = ("attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_v", "attn_output",
         "ffn_gate", "ffn_up", "ffn_down", "ssm_alpha", "ssm_beta",
         # MoE: the routed experts (one [experts, rows, K] stack each, rotated along K like any
         # matmul weight) and the shared expert; the router (ffn_gate_inp) is not quantized
         "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
         "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
         # MLA (DeepSeek-2 family: GLM-4.7-Flash, ...): the query and KV low-rank projections
         # and the per-head K/V expansions
         "attn_q_a", "attn_q_b", "attn_kv_a_mqa", "attn_kv_b", "attn_k_b", "attn_v_b")
ROT = re.compile(r"^blk\.\d+\.(" + "|".join(ROLES) + r")\.weight$")

H = np.array([[1.0]])
for _ in range(5):
    H = np.block([[H, H], [H, -H]])
H = (H / np.sqrt(32.0)).astype(np.float32)  # Sylvester order, symmetric: H == H.T, H @ H == I


def rotate_rows(w):
    rows, k = w.shape
    return (w.reshape(rows, k // 32, 32) @ H).reshape(rows, k)


def rotate_model(src, dst):
    shutil.copyfile(src, dst)
    r = gguf.GGUFReader(dst, "r+")
    n, t0 = 0, time.time()
    for t in r.tensors:
        if not ROT.match(t.name):
            continue
        typ = t.tensor_type
        if typ.name not in ("F32", "F16", "BF16", "Q8_0"):
            raise SystemExit(f"{t.name} is {typ.name}: start from Q8_0, BF16, F16 or F32")
        w = quants.dequantize(t.data, typ).astype(np.float32)
        w = rotate_rows(w.reshape(-1, w.shape[-1]))
        if typ.name == "F32":
            t.data[...] = w.reshape(t.data.shape)
        else:
            t.data[...] = np.asarray(quants.quantize(w, typ)).reshape(t.data.shape).view(t.data.dtype)
        n += 1
        if n % 50 == 0:
            print(f"  rotated {n} tensors, {time.time() - t0:.0f} s", flush=True)
    r.data.flush()
    return n


def rotate_imatrix(src, dst):
    shutil.copyfile(src, dst)
    r = gguf.GGUFReader(dst, "r+")
    n = 0
    for t in r.tensors:
        if t.name.endswith(".in_sum2") and ROT.match(t.name[: -len(".in_sum2")]):
            a = np.array(t.data, dtype=np.float32)
            k = a.shape[-1]
            m = a.reshape(-1, k // 32, 32).mean(axis=-1, keepdims=True)
            t.data[...] = np.broadcast_to(m, a.reshape(-1, k // 32, 32).shape).reshape(t.data.shape)
            n += 1
    r.data.flush()
    return n


def check(out):
    r = gguf.GGUFReader(out)
    names = {t.name for t in r.tensors}
    tied = "output.weight" not in names
    bad = [t.name for t in r.tensors if t.tensor_type.name == "Q4_0" and not ROT.match(t.name)
           and not (t.name == "token_embd.weight" and not tied)]
    stamp = r.fields.get(STAMP)
    return bad, stamp is not None, sum(t.tensor_type.name == "Q4_0" for t in r.tensors)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--imatrix")
    ap.add_argument("--quantize", default=os.path.join(HERE, "..", "build", "lean", "llama", "bin", "llama-quantize"))
    ap.add_argument("--work", help="directory for the rotated copy (default: next to OUT)")
    ap.add_argument("--keep", action="store_true", help="keep the rotated source copy")
    a = ap.parse_args()
    if not os.path.exists(a.quantize):
        raise SystemExit(f"llama-quantize not found at {a.quantize} (pass --quantize)")
    work = a.work or os.path.dirname(os.path.abspath(a.out))
    rot_src = os.path.join(work, os.path.basename(a.out) + ".rotated-source.gguf")
    print(f"rotating {a.source} -> {rot_src}", flush=True)
    n = rotate_model(a.source, rot_src)
    print(f"rotated {n} tensors", flush=True)
    cmd = [a.quantize, "--allow-requantize", "--override-kv", f"{STAMP}=int:32"]
    if a.imatrix:
        rot_im = rot_src + ".imatrix.gguf"
        print(f"imatrix: {rotate_imatrix(a.imatrix, rot_im)} entries -> block means", flush=True)
        cmd += ["--imatrix", rot_im]
    for role in ROLES:
        cmd += ["--tensor-type", f"{role}=q4_0"]
    # every other matmul weight stays off Q4_0: the runtime rotates the activations of every
    # Q4_0 matmul (ssm_out and the MTP projection as in Unsloth's Q4_0; the head at Q6_K)
    cmd += ["--tensor-type", "ssm_out=q5_k", "--tensor-type", "eh_proj=q8_0", "--output-tensor-type", "q6_k"]
    src_names = {t.name for t in gguf.GGUFReader(a.source).tensors}
    cmd += ["--token-embedding-type", "q4_0" if "output.weight" in src_names else "q8_0"]
    cmd += [rot_src, a.out, "Q4_0"]
    print("quantizing:", " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)), flush=True)
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode
    if not a.keep:
        for f in (rot_src, rot_src + ".imatrix.gguf"):
            if os.path.exists(f):
                os.remove(f)
    if rc != 0:
        raise SystemExit(f"llama-quantize failed ({rc})")
    bad, stamped, nq = check(a.out)
    if bad or not stamped:
        os.remove(a.out)
        raise SystemExit(f"invariant broken, {a.out} deleted: unrotated Q4_0 {bad[:6]}{'...' if len(bad) > 6 else ''}, stamp {stamped}")
    print(f"ok: {a.out}, {nq} Q4_0 tensors, all rotated, stamped {STAMP}=32")


if __name__ == "__main__":
    main()
