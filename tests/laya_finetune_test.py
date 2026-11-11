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
"""tools/laya_finetune on synthetic data (engine #1313): no checkpoint, no game, no device.

A tiny checkpoint with the real head's tensor names (d 32, the 16-head layout of laya/scorer.cpp) and one encoder
tensor the fine-tune must not touch, plus `--dump-encoder`-shaped dumps whose label is carried by the marker row of
the right option. Checked:
  1. train_head.py lowers dev log loss and does not lower dev accuracy;
  2. every byte of the new model.safetensors outside the head tensors is identical (header, encoder tensor), the
     head tensors changed, and the temperature landed in rl_agent_config.json's bucket;
  3. it refuses to overwrite an existing --out;
  4. metrics.py on a hand-computed --per-case file.
Exit 77 (ctest skip) when torch or numpy is missing: the tool needs them, the engine build does not.
"""
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import torch  # noqa: F401
except ImportError:
    print("SKIP: torch/numpy not installed")
    sys.exit(77)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "laya_finetune")
sys.path.insert(0, TOOL)
import metrics  # noqa: E402

D, F_, DH = 32, 64, 32
fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    fails += 0 if ok else 1


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        raw = a.astype(np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_checkpoint(d, rng):
    t = {"encoder.final_norm.weight": rng.standard_normal(D), "type_emb.weight": 0.02 * rng.standard_normal((3, D))}
    for i in range(2):
        p = f"head.layers.{i}."
        t[p + "norm1.weight"], t[p + "norm1.bias"] = np.ones(D), np.zeros(D)
        t[p + "self_attn.in_proj_weight"] = 0.05 * rng.standard_normal((3 * D, D))
        t[p + "self_attn.in_proj_bias"] = np.zeros(3 * D)
        t[p + "self_attn.out_proj.weight"] = 0.05 * rng.standard_normal((D, D))
        t[p + "self_attn.out_proj.bias"] = np.zeros(D)
        t[p + "norm2.weight"], t[p + "norm2.bias"] = np.ones(D), np.zeros(D)
        t[p + "linear1.weight"], t[p + "linear1.bias"] = 0.05 * rng.standard_normal((F_, D)), np.zeros(F_)
        t[p + "linear2.weight"], t[p + "linear2.bias"] = 0.05 * rng.standard_normal((D, F_)), np.zeros(D)
    t["scorer.0.weight"], t["scorer.0.bias"] = np.ones(D), np.zeros(D)
    t["scorer.1.weight"], t["scorer.1.bias"] = 0.1 * rng.standard_normal((DH, D)), np.zeros(DH)
    t["scorer.3.weight"], t["scorer.3.bias"] = 0.1 * rng.standard_normal((1, DH)), np.zeros(1)
    t["temperature"] = np.ones(3)
    os.makedirs(d)
    write_safetensors(f"{d}/model.safetensors", t)
    json.dump({"temperature_by_options": {}}, open(f"{d}/rl_agent_config.json", "w"))


def make_dump(d, n, rng):
    """n states, each written in both option orders (the --rotate shape); the right option's marker row carries +3 on
    feature 0 and the other one -3, plus noise, so the class is learnable from the head's input alone."""
    os.makedirs(d)
    rows, cases, L = [], [], 6
    for i in range(n):
        y = i % 2  # 0 = reflex, 1 = plan
        for rot in (0, 1):
            options = ["reflex", "plan"] if rot == 0 else ["plan", "reflex"]
            label = options.index(["reflex", "plan"][y])
            h = 0.5 * rng.standard_normal((L, D))
            markers = [2, 4]
            for k, m in enumerate(markers):
                h[m, 0] += 3.0 if k == label else -3.0
            cases.append({"src": f"s{i}", "rotation": rot, "label": label, "options": options, "qtype": 0,
                          "offset": len(rows) * L, "len": L, "markers": markers})
            rows.append(h)
    np.concatenate(rows).astype(np.float32).tofile(f"{d}/h.f32")
    with open(f"{d}/cases.jsonl", "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    json.dump({"d": D, "rows": len(rows) * L, "cases": len(cases), "rotate": True}, open(f"{d}/dump.json", "w"))


def tensor_ranges(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def main():
    work = tempfile.mkdtemp(prefix="laya-finetune-", dir=os.environ.get("ONEBIT_TEST_SCRATCH")
                            or os.path.expanduser("~/.cache/scratch-tests"))
    try:
        rng = np.random.default_rng(1313)
        ck, tr, dv, out = (os.path.join(work, x) for x in ("ckpt", "train", "dev", "out"))
        make_checkpoint(ck, rng)
        make_dump(tr, 24, rng)
        make_dump(dv, 8, rng)
        cmd = [sys.executable, os.path.join(TOOL, "train_head.py"), "--model", ck, "--train", tr, "--dev", dv,
               "--out", out, "--epochs", "6", "--lr", "2e-3", "--threads", "2"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        check(r.returncode == 0, "train_head.py exits 0" + ("" if r.returncode == 0 else ": " + r.stderr[-400:]))
        if r.returncode != 0:
            return
        meta = json.load(open(f"{out}/finetune.json"))
        b, a = meta["dev_before"], meta["dev_after"]
        check(a["nll"] < b["nll"], f"dev log loss falls: {b['nll']:.4f} -> {a['nll']:.4f}")
        check(a["acc"] >= b["acc"], f"dev accuracy does not fall: {b['acc']:.3f} -> {a['acc']:.3f}")
        check(a["acc"] >= 0.9, f"the synthetic boundary is learned: dev accuracy {a['acc']:.3f} >= 0.9")
        check(meta["train_sequences"] == 48 and meta["class_counts"] == {"reflex": 24, "plan": 24},
              "finetune.json records the data it trained on")

        base0, hdr0 = tensor_ranges(f"{ck}/model.safetensors")
        base1, hdr1 = tensor_ranges(f"{out}/model.safetensors")
        raw0, raw1 = open(f"{ck}/model.safetensors", "rb").read(), open(f"{out}/model.safetensors", "rb").read()
        check(hdr0 == hdr1 and base0 == base1 and len(raw0) == len(raw1), "header, offsets and size are unchanged")
        changed, frozen_same = 0, True
        for name, info in hdr0.items():
            lo, hi = info["data_offsets"]
            same = raw0[base0 + lo:base0 + hi] == raw1[base1 + lo:base1 + hi]
            if name.startswith(("head.", "type_emb.", "scorer.")):
                changed += 0 if same else 1
            elif not same:
                frozen_same = False
        check(frozen_same, "every non-head tensor (encoder, temperature) is byte-identical")
        check(changed > 0, f"the head tensors were rewritten ({changed} changed)")
        cfg = json.load(open(f"{out}/rl_agent_config.json"))
        check("choice:2" in cfg.get("temperature_by_options", {}), "the temperature is written to the choice:2 bucket")

        r2 = subprocess.run(cmd, capture_output=True, text=True)
        check(r2.returncode != 0 and "refusing to overwrite" in (r2.stdout + r2.stderr), "an existing --out is refused")

        # metrics.py, by hand: 3 cases, 2 right; p(true) 0.9, 0.6, 0.3
        pc = os.path.join(work, "per.jsonl")
        with open(pc, "w") as f:
            for lab, cho, p, ms in (("reflex", "reflex", 0.9, 100), ("plan", "plan", 0.6, 200), ("plan", "reflex", 0.3, 300)):
                other = "plan" if lab == "reflex" else "reflex"
                f.write(json.dumps({"label": lab, "chosen": cho, "probabilities": {lab: p, other: 1 - p},
                                    "decision_ms": ms}) + "\n")
        m = metrics.m(pc)
        ll = -(math.log(0.9) + math.log(0.6) + math.log(0.3)) / 3
        # top p 0.9 (right), 0.6 (right), 0.7 (wrong) -> bins 9, 6, 7, each |p - correct|: 0.1, 0.4, 0.7
        ece = (0.1 + 0.4 + 0.7) / 3
        check(m["correct"] == 2 and m["acc"] == round(2 / 3, 4), "metrics: accuracy 2/3")
        check(m["log_loss"] == round(ll, 4), f"metrics: log loss {m['log_loss']} == {round(ll, 4)}")
        check(m["ece"] == round(ece, 4), f"metrics: ECE {m['ece']} == {round(ece, 4)}")
        check(m["ms_p50"] == 200 and m["min_p_true"] == 0.3, "metrics: decision p50 and min p(true)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    os.makedirs(os.path.expanduser("~/.cache/scratch-tests"), exist_ok=True)
    main()
    print("all laya_finetune checks passed" if fails == 0 else f"{fails} laya_finetune check(s) FAILED")
    sys.exit(1 if fails else 0)
