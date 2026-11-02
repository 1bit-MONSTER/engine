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
"""tools/safetensors_to_q4nx.py: the converter, the verifier and the d3 head, without hardware.

usage: tests/safetensors_to_q4nx_test.py        (needs numpy; no gguf, no model, no device)

Every fixture is synthesised in a temporary directory, so the test says what the converter does
rather than depending on a downloaded pack:

1. a single-shard HF model (one 32x256 projection, a norm, an embedding) packs to the Q4NX
   container with I8 tiles for the projection and BF16 for the rest, and `verify` accepts it;
2. the same model sharded into two files with an index.json packs to the same tensor set;
3. `decision` turns a [codes, hidden] readout plus decision_config.json into decision.q4nx;
4. the refusals: a code count that does not match the head, a dtype the reader does not handle,
   and a missing readout.
5. a d3-shaped (`qwen3_5`) pack: the GDN renames the engine loads, and the three transforms --
   `A_log` -> `-exp(A_log)` in F32, the `conv1d` transpose to tap-major, and the alpha/beta rows
   padded to a 32-row tile -- each with a known answer, plus the refusal for a tensor with no
   mapping, and `--dtype bf16` writing the same tensor set with no tiles at all.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "safetensors_to_q4nx.py")
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import q4nx_pack as Q  # noqa: E402

fails = 0


def check(ok, what):
    global fails
    if ok:
        return
    fails += 1
    print(f"FAIL {what}")


def write_safetensors(path, arrays, dtypes=None):
    """(name -> numpy array) -> a safetensors file, dense row-major."""
    dtypes = dtypes or {}
    header, blobs, off = {}, [], 0
    for name, a in arrays.items():
        dt = dtypes.get(name, "F32")
        if dt == "BF16":
            raw = Q.bf16_bits(a.astype(np.float32)).tobytes()
        elif dt == "F16":
            raw = a.astype(np.float16).tobytes()
        else:
            raw = a.astype(np.float32).tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def read_q4nx(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        table = json.loads(f.read(n))
        data = f.read()
    return table, data


def make_model(d, weights):
    os.makedirs(d, exist_ok=True)
    write_safetensors(os.path.join(d, "model.safetensors"), weights)
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 256,
           "num_hidden_layers": 1, "vocab_size": 8}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        f.write("{}")
    return d


def run(*argv):
    return subprocess.run([sys.executable, TOOL, *argv], capture_output=True, text=True)


def bf16r(x):
    """The value the converter actually reads: every fixture tensor is stored bf16."""
    return Q.from_bf16(Q.bf16_bits(np.asarray(x, dtype=np.float32)))


# d3-lite's GDN geometry at test scale: hidden 256, 16 value heads of 16, key head dim 8,
# conv kernel 4 -> KD 128, VD 256, CD 512.  Every shape clears the 32x256 tile grid.
H, NVH, KHD, VHD, CK = 256, 16, 8, 16, 4
KD, VD = NVH * KHD, NVH * VHD
CD = 2 * KD + VD


def qwen35_model(d):
    """A two-layer d3-shaped pack: one GDN layer, one full-attention layer, plus a visual tensor."""
    rng = np.random.default_rng(11)
    n = lambda *s: (rng.standard_normal(s).astype(np.float32) * 0.02)   # noqa: E731
    p = lambda *s: (np.abs(rng.standard_normal(s)).astype(np.float32) + 1.0)  # noqa: E731
    A_log = np.linspace(0.5, 4.0, NVH).astype(np.float32)
    dt_bias = np.linspace(-5.0, 19.0, NVH).astype(np.float32)
    conv1d = rng.standard_normal((CD, 1, CK)).astype(np.float32) * 0.1   # [CD,1,CK], asymmetric
    alpha, beta = n(NVH, H), n(NVH, H)
    weights = {
        "language_model.embed_tokens.weight": n(8, H),
        "language_model.norm.weight": p(H),
        "language_model.layers.0.input_layernorm.weight": p(H),
        "language_model.layers.0.post_attention_layernorm.weight": p(H),
        "language_model.layers.0.mlp.gate_proj.weight": n(H, H),
        "language_model.layers.0.mlp.up_proj.weight": n(H, H),
        "language_model.layers.0.mlp.down_proj.weight": n(H, H),
        "language_model.layers.0.linear_attn.in_proj_qkv.weight": n(CD, H),
        "language_model.layers.0.linear_attn.in_proj_z.weight": n(VD, H),
        "language_model.layers.0.linear_attn.in_proj_a.weight": alpha,
        "language_model.layers.0.linear_attn.in_proj_b.weight": beta,
        "language_model.layers.0.linear_attn.out_proj.weight": n(H, VD),
        "language_model.layers.0.linear_attn.conv1d.weight": conv1d,
        "language_model.layers.0.linear_attn.norm.weight": p(NVH * KHD // NVH),
        "language_model.layers.0.linear_attn.A_log": A_log,
        "language_model.layers.0.linear_attn.dt_bias": dt_bias,
        "language_model.layers.1.input_layernorm.weight": p(H),
        "language_model.layers.1.post_attention_layernorm.weight": p(H),
        "language_model.layers.1.mlp.gate_proj.weight": n(H, H),
        "language_model.layers.1.mlp.up_proj.weight": n(H, H),
        "language_model.layers.1.mlp.down_proj.weight": n(H, H),
        "language_model.layers.1.self_attn.q_proj.weight": n(2 * H, H),
        "language_model.layers.1.self_attn.k_proj.weight": n(H, H),
        "language_model.layers.1.self_attn.v_proj.weight": n(H, H),
        "language_model.layers.1.self_attn.o_proj.weight": n(H, 2 * H),
        "language_model.layers.1.self_attn.q_norm.weight": p(H),
        "language_model.layers.1.self_attn.k_norm.weight": p(H),
        "visual.patch_embed.proj.weight": n(8, 8),      # skipped without reading its bytes
    }
    os.makedirs(d, exist_ok=True)
    write_safetensors(os.path.join(d, "model.safetensors"), weights,
                      {k: "BF16" for k in weights})
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
           "text_config": {"hidden_size": H, "num_hidden_layers": 2, "vocab_size": 8,
                           "linear_num_value_heads": NVH, "linear_key_head_dim": KHD,
                           "linear_value_head_dim": VHD, "linear_conv_kernel_dim": CK}}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    open(os.path.join(d, "tokenizer.json"), "w").write("{}")
    return {"dir": d, "weights": weights, "A_log": A_log, "dt_bias": dt_bias,
            "conv1d": conv1d, "alpha": alpha, "beta": beta}


def main():
    rng = np.random.default_rng(7)
    proj = rng.standard_normal((32, 256)).astype(np.float32) * 0.02   # exactly one tile
    norm = np.abs(rng.standard_normal(256)).astype(np.float32) + 1.0
    emb = rng.standard_normal((8, 256)).astype(np.float32) * 0.02
    weights = {"model.layers.0.self_attn.q_proj.weight": proj,
               "model.layers.0.input_layernorm.weight": norm,
               "model.embed_tokens.weight": emb}

    with tempfile.TemporaryDirectory() as tmp:
        src = make_model(os.path.join(tmp, "hf"), weights)
        out = os.path.join(tmp, "q4nx")
        r = run("pack", src, out)
        check(r.returncode == 0, f"pack exits 0 (stderr: {r.stderr[-300:]})")
        table, data = read_q4nx(os.path.join(out, "model.q4nx"))
        check(table["model.layers.0.self_attn.q_proj.weight"]["dtype"] == "I8",
              "a 32x256 projection becomes I8 tiles")
        check(table["model.layers.0.self_attn.q_proj.weight"]["shape"] == [1, Q.TILE],
              "one tile is 5120 bytes")
        check(table["model.layers.0.input_layernorm.weight"]["dtype"] == "BF16", "a norm stays bf16")
        check(table["model.embed_tokens.weight"]["dtype"] == "BF16", "the embedding stays bf16")
        check(table["__metadata__"]["format"] == "q4nx", "the container is marked q4nx")
        cfg = json.load(open(os.path.join(out, "config.json")))
        check(cfg["hidden_size"] == 256 and cfg["model_type"] == "qwen3", "the HF config is carried over")
        check(cfg["q4nx_source"]["converter"] == "tools/safetensors_to_q4nx.py", "the source is recorded")
        check(os.path.exists(os.path.join(out, "tokenizer.json")), "the tokenizer is copied")

        # The tiles really hold the projection: decode them back.
        ent = table["model.layers.0.self_attn.q_proj.weight"]
        lo, hi = ent["data_offsets"]
        back = Q.untile(np.frombuffer(data[lo:hi], dtype=np.uint8).reshape(-1, Q.TILE), 32, 256)
        rel = float(np.sqrt(((back - proj) ** 2).mean()) / np.sqrt((proj ** 2).mean()))
        check(rel < 0.15, f"the q4_1 tiles decode back to the projection (rel rms {rel:.4f})")

        v = run("verify", src, out)
        check(v.returncode == 0, f"verify accepts its own output (out: {v.stdout[-200:]})")
        check("3/3 tensors ok" in v.stdout, "verify reports all three tensors")

        # ---- sharded: the same tensors split across two files with an index
        sh = os.path.join(tmp, "hf_sharded")
        os.makedirs(sh)
        names = list(weights)
        write_safetensors(os.path.join(sh, "model-00001-of-00002.safetensors"),
                          {n: weights[n] for n in names[:2]})
        write_safetensors(os.path.join(sh, "model-00002-of-00002.safetensors"),
                          {n: weights[n] for n in names[2:]})
        shard_of = {n: ("model-00001-of-00002.safetensors" if i < 2 else "model-00002-of-00002.safetensors")
                    for i, n in enumerate(names)}
        with open(os.path.join(sh, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": shard_of}, f)
        for extra in ("config.json", "tokenizer.json"):
            with open(os.path.join(src, extra)) as f:
                open(os.path.join(sh, extra), "w").write(f.read())
        out2 = os.path.join(tmp, "q4nx_sharded")
        r2 = run("pack", sh, out2)
        check(r2.returncode == 0, f"a sharded model packs too (stderr: {r2.stderr[-200:]})")
        table2, _ = read_q4nx(os.path.join(out2, "model.q4nx"))
        check(set(table2) - {"__metadata__"} == set(table) - {"__metadata__"},
              "the sharded pack has the same tensors")

        # ---- dtype the reader refuses
        odd = os.path.join(tmp, "hf_odd")
        make_model(odd, weights)
        write_safetensors(os.path.join(odd, "model.safetensors"),
                          {"model.layers.0.self_attn.q_proj.weight": proj}, {"model.layers.0.self_attn.q_proj.weight": "F64"})
        r3 = run("pack", odd, os.path.join(tmp, "q4nx_odd"))
        check(r3.returncode != 0 and "F32, F16 and BF16" in (r3.stderr + r3.stdout),
              "an unsupported dtype is refused by name")

        # ---- the d3 readout head
        d3 = os.path.join(tmp, "d3")
        os.makedirs(d3)
        head = rng.standard_normal((3, 8)).astype(np.float32)
        write_safetensors(os.path.join(d3, "readout.safetensors"), {"weight": head})
        dc = {"format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3",
              "codes": ["A", "B", "C"], "token_ids": [32, 33, 34], "temperature": 1.0}
        json.dump(dc, open(os.path.join(d3, "decision_config.json"), "w"))
        dout = os.path.join(tmp, "d3out")
        r4 = run("decision", d3, dout)
        check(r4.returncode == 0, f"decision converts a head (stderr: {r4.stderr[-300:]})")
        dtable, ddata = read_q4nx(os.path.join(dout, "decision.q4nx"))
        check(dtable["weight"]["dtype"] == "F32" and dtable["weight"]["shape"] == [3, 8],
              "the head keeps its f32 [codes, hidden] shape")
        check(dtable["__metadata__"]["kind"] == "d3-readout", "the container says what it is")
        lo, hi = dtable["weight"]["data_offsets"]
        back = np.frombuffer(ddata[lo:hi], dtype=np.float32).reshape(3, 8)
        check(np.array_equal(back, head), "the head is stored exactly (no quantization)")
        check(os.path.exists(os.path.join(dout, "decision_config.json")), "decision_config.json is copied")

        # ---- refusals on the head
        bad = dict(dc)
        bad["codes"] = ["A", "B"]
        json.dump(bad, open(os.path.join(d3, "decision_config.json"), "w"))
        r5 = run("decision", d3, os.path.join(tmp, "d3bad"))
        check(r5.returncode != 0 and "codes" in (r5.stderr + r5.stdout),
              "a head whose shape does not match the codes is refused")
        json.dump(dc, open(os.path.join(d3, "decision_config.json"), "w"))

        os.remove(os.path.join(d3, "readout.safetensors"))
        r6 = run("decision", d3, os.path.join(tmp, "d3missing"))
        check(r6.returncode != 0 and "readout.safetensors" in (r6.stderr + r6.stdout),
              "a missing readout is refused by name")

        # ---- the d3 pack's convention: language_model.* names, GDN renames, three transforms
        q = qwen35_model(os.path.join(tmp, "hf_qwen35"))
        qout = os.path.join(tmp, "q4nx_qwen35")
        r7 = run("pack", q["dir"], qout)
        check(r7.returncode == 0, f"a qwen3_5 pack converts (stderr: {r7.stderr[-300:]})")
        check("arch qwen3_5" in r7.stdout, "the arch is detected from the language_model.* names")
        t, d = read_q4nx(os.path.join(qout, "model.q4nx"))
        check(not any(n.startswith(("language_model.", "visual.")) for n in t),
              "nothing keeps its HF name and no visual tensor is written")
        check("skipped 1 tensor(s)" in r7.stdout, "the skipped visual tensor is counted, not silent")
        # the renames the engine loads (configs/q4nx/qwen3.5_0.8b.json's q4nx_name column)
        for hf, want in (("in_proj_qkv.weight", "linear_attn.qkv_proj.weight"),
                         ("in_proj_z.weight", "self_attn.gate_proj.weight"),
                         ("in_proj_a.weight", "linear_attn.ssm_alpha_proj.weight"),
                         ("in_proj_b.weight", "linear_attn.ssm_beta_proj.weight"),
                         ("out_proj.weight", "linear_attn.ssm_out_proj.weight"),
                         ("conv1d.weight", "linear_attn.ssm_conv1d.weight"),
                         ("norm.weight", "linear_attn.ssm_norm.weight"),
                         ("A_log", "linear_attn.ssm_a"),
                         ("dt_bias", "linear_attn.ssm_dt.bias")):
            qn = f"model.layers.0.{want}"
            check(qn in t, f"linear_attn.{hf} -> {want}")
        check("model.embed_tokens.weight" in t and "model.norm.weight" in t,
              "embed_tokens and the final norm keep their engine names")
        check(t["model.embed_tokens.weight"]["shape"] == [8, H],
              "the embedding keeps its 2-D shape (model_load reads vocab/hidden off it)")
        qcfg = json.load(open(os.path.join(qout, "config.json")))
        check(qcfg.get("num_hidden_layers") == 2 and qcfg.get("hidden_size") == H,
              "text_config's keys are lifted to the top level, where the engine reads them")
        check(qcfg["text_config"]["num_hidden_layers"] == 2, "text_config itself is left in place")
        check("model.layers.1.self_attn.q_proj.weight" in t, "a full-attention layer keeps its names")

        # transform 1: A_log -> ssm_a = -exp(A_log), F32 (the engine's expf(ssm_a*softplus(dt)))
        ent = t["model.layers.0.linear_attn.ssm_a"]
        check(ent["dtype"] == "F32", "ssm_a is F32, which is what the loader reads")
        lo, hi = ent["data_offsets"]
        got = np.frombuffer(d[lo:hi], dtype=np.float32)
        check(np.array_equal(got, (-np.exp(bf16r(q["A_log"]))).astype(np.float32)),
              "ssm_a is exactly -exp(A_log)")
        check(bool((got < 0).all()), "every ssm_a is negative (a positive one decays backwards)")

        # ... and dt_bias is carried raw, in F32
        ent = t["model.layers.0.linear_attn.ssm_dt.bias"]
        check(ent["dtype"] == "F32", "ssm_dt.bias is F32")
        lo, hi = ent["data_offsets"]
        check(np.array_equal(np.frombuffer(d[lo:hi], dtype=np.float32), bf16r(q["dt_bias"])),
              "ssm_dt.bias is the pack's dt_bias, unchanged")

        # transform 2: conv1d [CD,1,CK] -> tap-major [CK,CD], the engine's conv[j*CD + c]
        ent = t["model.layers.0.linear_attn.ssm_conv1d.weight"]
        check(ent["dtype"] == "BF16" and ent["shape"][0] == CD * CK, "conv1d is 1-D bf16 [CK*CD]")
        lo, hi = ent["data_offsets"]
        conv = Q.from_bf16(np.frombuffer(d[lo:hi], dtype=np.uint16))
        src_conv = bf16r(q["conv1d"])             # [CD, 1, CK], as stored
        wrong = sum(1 for j in range(CK) for c in range(0, CD, 7)
                    if conv[j * CD + c] != src_conv[c, 0, j])
        check(wrong == 0, f"conv entry j*CD+c holds conv[c,0,j] ({wrong} mismatches)")
        check(conv[0] != conv[CD], "the transpose is not a flatten (tap 0 and tap 1 differ)")

        # transform 3: alpha/beta rows zero-padded to 32 so they tile as q4_1
        for suffix in ("ssm_alpha_proj", "ssm_beta_proj"):
            ent = t[f"model.layers.0.linear_attn.{suffix}.weight"]
            check(ent["dtype"] == "I8", f"{suffix} packs as tiles")
            check(ent["shape"][0] == NVH // Q.ROWS + 1,
                  f"{suffix} is padded to 32 rows ({ent['shape'][0]} tiles)")
            lo, hi = ent["data_offsets"]
            back = Q.untile(np.frombuffer(d[lo:hi], dtype=np.uint8).reshape(-1, Q.TILE), 32, H)
            src = bf16r(q["alpha"] if suffix == "ssm_alpha_proj" else q["beta"])
            rel = float(np.sqrt(((back[:NVH] - src) ** 2).mean()) / np.sqrt((src ** 2).mean()))
            check(rel < 0.15, f"{suffix} decodes back to the source rows (rel rms {rel:.4f})")
            check(float(np.abs(back[NVH:]).max()) == 0.0 if NVH < 32 else True,
                  f"{suffix} pad rows decode to zero")

        v7 = run("verify", q["dir"], qout)
        check(v7.returncode == 0, f"verify accepts the converted d3 pack (out: {v7.stdout[-300:]})")
        check("MISSING" not in v7.stdout, "verify finds every renamed tensor")

        # ---- --dtype bf16: no tiles at all, so a run can hold the weights fixed and ask
        #      whether the forward agrees with a reference (engine #796's parity row).
        qexact = os.path.join(tmp, "q4nx_qwen35_bf16")
        r7b = run("pack", q["dir"], qexact, "--dtype", "bf16")
        check(r7b.returncode == 0, f"pack --dtype bf16 succeeds (out: {r7b.stderr[-300:]})")
        dt, _ = read_q4nx(os.path.join(qexact, "model.q4nx"))
        kinds = {k: v["dtype"] for k, v in dt.items() if k != "__metadata__"}
        check("I8" not in kinds.values(),
              f"--dtype bf16 writes no q4_1 tiles (dtypes: {sorted(set(kinds.values()))})")
        check(kinds.get("model.layers.0.linear_attn.ssm_a") == "F32",
              "the transform-forced F32 tensor stays F32 under --dtype bf16")
        check(kinds.get("model.layers.0.mlp.gate_proj.weight") == "BF16",
              "a tileable projection is BF16 under --dtype bf16")
        v7b = run("verify", q["dir"], qexact)
        check(v7b.returncode == 0, f"verify accepts the bf16 container (out: {v7b.stdout[-300:]})")

        # ---- refusals: an unmapped qwen3_5 tensor is an error, not a silent drop
        bogus = qwen35_model(os.path.join(tmp, "hf_qwen35_bad"))
        write_safetensors(os.path.join(bogus["dir"], "model.safetensors"),
                          {**bogus["weights"],
                           "language_model.layers.0.linear_attn.in_proj_mystery.weight":
                               rng.standard_normal((32, 256)).astype(np.float32)})
        r8 = run("pack", bogus["dir"], os.path.join(tmp, "q4nx_bogus"))
        check(r8.returncode != 0 and "in_proj_mystery" in (r8.stderr + r8.stdout),
              "an unmapped language_model tensor is refused by name")

    print("FAILED" if fails else "safetensors_to_q4nx: ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
