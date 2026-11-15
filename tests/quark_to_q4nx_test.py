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
"""tools/quark_to_q4nx.py on synthetic Quark parts: exact int4, refused uint4, values from the parts.

Every fixture is synthesised (numpy only): a tiny HF checkpoint and a "parts" safetensors shaped like Quark
0.13's frozen model (<base>.weight dequantized, <base>._weight_quantizer.scale / .zero_point).  The decisive
checks are the ones a silent fallback would pass: the projection must decode to Quark's dequantized weight
EXACTLY for int4 (an RTN re-fit would not), and a parts file the tool finds nothing in must be refused.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "quark_to_q4nx.py")
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import q4nx_pack as Q  # noqa: E402
sys.path.insert(0, HERE)
from safetensors_to_q4nx_test import read_q4nx, write_safetensors  # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails += 1


H, I, V = 256, 512, 64
rng = np.random.default_rng(7)


def bf16(a):
    return Q.from_bf16(Q.bf16_bits(np.asarray(a, dtype=np.float32)))


def make_hf(d):
    os.makedirs(d, exist_ok=True)
    arrays = {
        "model.embed_tokens.weight": rng.standard_normal((V, H)) * 0.02,
        "model.layers.0.input_layernorm.weight": np.ones(H),
        "model.layers.0.mlp.down_proj.weight": rng.standard_normal((H, I)) * 0.05,
        "lm_head.weight": rng.standard_normal((V, H)) * 0.02,
    }
    write_safetensors(os.path.join(d, "model.safetensors"), arrays, {k: "BF16" for k in arrays})
    json.dump({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": H,
               "intermediate_size": I, "num_hidden_layers": 1, "vocab_size": V},
              open(os.path.join(d, "config.json"), "w"))
    return arrays


def quark_parts(path, hf_arrays, G, signed, norm_scale=2.0, perturb=False, no_quantizer=False):
    """A parts file with down_proj quantized the way Quark stores it, and the norm rescaled (AWQ-like)."""
    w = hf_arrays["model.layers.0.mlp.down_proj.weight"]
    N, K = w.shape
    g = w.reshape(N, K // G, G)
    if signed:
        s = bf16(np.abs(g).max(-1) / 7.0)
        zp = np.zeros_like(s)
        q = np.clip(np.rint(g / s[..., None]), -8, 7)
    else:
        lo, hi = g.min(-1), g.max(-1)
        s = bf16((hi - lo) / 15.0)
        zp = np.clip(np.rint(-lo / s), 0, 15)
        q = np.clip(np.rint(g / s[..., None]) + zp[..., None], 0, 15)
    deq = bf16(((q - zp[..., None]) * s[..., None]).reshape(N, K))
    if perturb:
        deq = deq.copy()
        deq[0, 0] += 0.5 * float(s[0, 0])
    arrays = {"model.layers.0.input_layernorm.weight": hf_arrays["model.layers.0.input_layernorm.weight"] * norm_scale,
              "model.layers.0.mlp.down_proj.weight": deq}
    dt = {k: "BF16" for k in arrays}
    if not no_quantizer:
        arrays["model.layers.0.mlp.down_proj._weight_quantizer.scale"] = s
        arrays["model.layers.0.mlp.down_proj._weight_quantizer.zero_point"] = zp
        dt["model.layers.0.mlp.down_proj._weight_quantizer.scale"] = "BF16"
    write_safetensors(path, arrays, dt)
    return deq, q, s, zp


def run(*argv):
    return subprocess.run([sys.executable, TOOL, *argv], capture_output=True, text=True)


def decode(out, name, shape):
    table, raw = read_q4nx(os.path.join(out, "model.q4nx"))
    t = table[name]
    blob = raw[t["data_offsets"][0]:t["data_offsets"][1]]
    if t["dtype"] == "I8":
        return Q.untile(np.frombuffer(blob, dtype=np.uint8).reshape(t["shape"]), *shape)
    return Q.from_bf16(np.frombuffer(blob, dtype=np.uint16)).reshape(shape)


with tempfile.TemporaryDirectory() as tmp:
    hf = os.path.join(tmp, "hf")
    hfa = make_hf(hf)

    # 1-3. int4 (symmetric), group 32 and group 128: Quark's codes, decoded exactly.
    for G in (32, 128):
        parts = os.path.join(tmp, f"int4_{G}.safetensors")
        out = os.path.join(tmp, f"q_int4_{G}")
        deq, q, s, zp = quark_parts(parts, hfa, G, signed=True)
        r = run(parts, hf, out)
        check(r.returncode == 0, f"int4 G={G}: converts (rc {r.returncode}) {r.stderr.strip()[-200:]}")
        if r.returncode:
            continue
        got = decode(out, "model.layers.0.mlp.down_proj.weight", (H, I))
        exact = ((q - zp[..., None]) * s[..., None]).reshape(H, I).astype(np.float32)
        check(np.array_equal(got, exact), f"int4 G={G}: down_proj decodes to Quark's (q - zp) * s exactly "
                                          f"(max |d| {np.abs(got - exact).max():.3g})")
        check(np.array_equal(bf16(got), deq), f"int4 G={G}: and rounds to Quark's own bf16 weight bit for bit")
        st = json.loads(r.stdout.strip().splitlines()[-1])
        check(st["quark_linears"] == 1 and st["int4"] == 1 and st["max_zero_round_steps"] == 0.0,
              f"int4 G={G}: one Quark linear, zero rounding 0.0 (stats {st})")
    # The control the exact check needs: an RTN re-fit of the same weight is NOT exact.
    codes, sc, zr = Q.quantize_q4_1(quark_parts(os.path.join(tmp, "c.safetensors"), hfa, 32, True)[0])
    rtn = (codes * np.repeat(sc, 32, 1) + np.repeat(zr, 32, 1))
    check(not np.array_equal(rtn, deq), "control: an RTN re-fit of Quark's weight does not reproduce it")

    # 4. values from the parts (AWQ rescaled the norm), lm_head RTN like the baseline packer.
    out = os.path.join(tmp, "q_int4_32")
    norm = decode(out, "model.layers.0.input_layernorm.weight", (H,))
    check(np.array_equal(norm, bf16(np.ones(H) * 2.0)), "the norm is the parts' rescaled value, not the checkpoint's")
    head = decode(out, "lm_head.weight", (V, H))
    c, sc, zr = Q.quantize_q4_1(bf16(hfa["lm_head.weight"]))
    check(np.array_equal(head, c * np.repeat(sc, 32, 1) + np.repeat(zr, 32, 1)),
          "lm_head (no quantizer in the parts) is RTN'd exactly as safetensors_to_q4nx does")

    # 5-6. uint4: refused by default (the bf16 zero rounds), converted with --allow-uint4 and the rounding reported.
    parts = os.path.join(tmp, "uint4.safetensors")
    quark_parts(parts, hfa, 128, signed=False)
    r = run(parts, hf, os.path.join(tmp, "q_uint4"))
    check(r.returncode != 0 and "uint4" in (r.stderr + r.stdout), "uint4 is refused without --allow-uint4")
    r = run(parts, hf, os.path.join(tmp, "q_uint4b"), "--allow-uint4")
    ok = r.returncode == 0
    st = json.loads(r.stdout.strip().splitlines()[-1]) if ok else {}
    check(ok and st.get("uint4") == 1 and st.get("max_zero_round_steps", 0) > 0,
          f"--allow-uint4 converts and reports the zero rounding ({st.get('max_zero_round_steps')})")

    # 7. nothing to convert -> refused, never a silent RTN pack.
    parts = os.path.join(tmp, "none.safetensors")
    quark_parts(parts, hfa, 32, signed=True, no_quantizer=True)
    r = run(parts, hf, os.path.join(tmp, "q_none"))
    check(r.returncode != 0 and "no Quark-quantized linear" in (r.stderr + r.stdout),
          "a parts file with no quantizer tensors is refused")

    # 8. a weight off Quark's grid (wrong convention or a mismatched file) is refused.
    parts = os.path.join(tmp, "off.safetensors")
    quark_parts(parts, hfa, 32, signed=True, perturb=True)
    r = run(parts, hf, os.path.join(tmp, "q_off"))
    check(r.returncode != 0 and "grid" in (r.stderr + r.stdout), "a weight half a step off Quark's grid is refused")

print("quark_to_q4nx_test: " + ("PASS" if fails == 0 else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
