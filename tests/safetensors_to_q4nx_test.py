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

    print("FAILED" if fails else "safetensors_to_q4nx: ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
