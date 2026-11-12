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
"""Head-only fine-tune of Laya's routing head on CPU, over `1bit route --dump-encoder` dumps.

  train_head.py --model DIR --train DUMP [--dev DUMP] --out NEW_DIR [--epochs 4] [--lr 2e-5] [--seed 1313]

The encoder is frozen (its output is in the dump); type_emb, the two head layers and the scorer train.
Loss: Laya's proper reward (log score + 0.5 spherical score), maximised, class-weighted.
Early stop on dev log loss; then the choice:k temperature is refit on dev (1-D search, minimum NLL).
NEW_DIR is a copy of DIR with only the head tensors' bytes replaced in model.safetensors (same names, dtypes,
shapes and offsets, so every other byte is identical), the new temperature in rl_agent_config.json, and
finetune.json (data, seed, epochs, dev numbers before and after, command line). It is an ordinary
`--laya-model` directory: measure it with `1bit route --laya-models T --laya-model NEW_DIR --eval test.jsonl`.
"""
import argparse
import json
import math
import os
import random
import shutil
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head as H  # noqa: E402


def logits_of(w, dump, idx, bs=8):
    out = []
    for s in range(0, len(idx), bs):
        h, pad, mk, qt, y, cs = dump.batch(idx[s:s + bs])
        out.append((H.forward(w, h, pad, mk, qt), y, cs))
    return out


def evaluate(w, cfg, dump, temp=None):
    """dev log loss, accuracy, n over labelled sequences"""
    idx = [i for i, c in enumerate(dump.cases) if c["label"] >= 0]
    nll, ok = 0.0, 0
    with torch.no_grad():
        for lg, y, cs in logits_of(w, dump, idx):
            for b, c in enumerate(cs):
                k = len(c["markers"])
                t = temp if temp is not None else H.temperature(w, cfg, c["qtype"], k)
                lp = torch.log_softmax(lg[b, :k] / t, -1)
                nll -= lp[y[b]].item()
                ok += int(lp.argmax().item() == y[b].item())
    return nll / max(1, len(idx)), ok / max(1, len(idx)), len(idx)


def refit_temperature(w, dump):
    """The temperature with the lowest NLL on dump, over a 0.2..4.0 grid; the head runs once."""
    idx = [i for i, c in enumerate(dump.cases) if c["label"] >= 0]
    rows = []
    with torch.no_grad():
        for lg, y, cs in logits_of(w, dump, idx):
            for b, c in enumerate(cs):
                rows.append((lg[b, :len(c["markers"])].clone(), int(y[b])))
    grid = [round(0.2 + 0.02 * i, 2) for i in range(191)]  # 0.2 .. 4.0

    def nll(t):
        return -sum(torch.log_softmax(lg / t, -1)[y].item() for lg, y in rows) / len(rows)

    return min(grid, key=nll)


def proper_reward(logp, y):
    """Laya's proper scoring reward: log score + 0.5 spherical score (both strictly proper)."""
    p = logp.exp()
    log_score = logp.gather(-1, y[:, None])[:, 0]
    sph = p.gather(-1, y[:, None])[:, 0] / p.norm(dim=-1).clamp_min(1e-9)
    return log_score + 0.5 * sph


def write_checkpoint(src_dir, out_dir, w, new_temp, k_bucket):
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} exists; refusing to overwrite")
    shutil.copytree(src_dir, out_dir)
    path = f"{out_dir}/model.safetensors"
    with open(path, "r+b") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, t in w.items():
            if name == "temperature" or not name.startswith(H.HEAD_PREFIXES):
                continue
            info = header[name]
            assert list(t.shape) == info["shape"], name
            a, b = info["data_offsets"]
            v = t.detach().contiguous()
            if info["dtype"] == "F32":
                raw = v.float().numpy().tobytes()
            elif info["dtype"] == "F16":
                raw = v.half().numpy().tobytes()
            elif info["dtype"] == "BF16":
                raw = v.bfloat16().view(torch.int16).numpy().tobytes()
            else:
                raise SystemExit(f"{name}: dtype {info['dtype']} not handled")
            assert len(raw) == b - a, name
            f.seek(base + a)
            f.write(raw)
    cfgp = f"{out_dir}/rl_agent_config.json"
    cfg = json.load(open(cfgp))
    cfg.setdefault("temperature_by_options", {})[k_bucket] = new_temp
    json.dump(cfg, open(cfgp, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the Laya checkpoint dir the dumps were made with")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1313)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t_start = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)

    w, cfg = H.load_head(a.model)
    train = H.Dump(a.train)
    dev = H.Dump(a.dev) if a.dev else None
    lab = [i for i, c in enumerate(train.cases) if c["label"] >= 0]
    if not lab:
        raise SystemExit("no labelled sequences in --train")
    ks = {len(train.cases[i]["markers"]) for i in lab}
    qts = {train.cases[i]["qtype"] for i in lab}
    if len(ks) != 1 or qts != {0}:
        raise SystemExit("one choice question with one option count per dump is supported")
    k = ks.pop()
    kb = H.bucket(0, k)
    # class weights over the labels by class name (rotations move the index, not the class)
    names = [train.cases[i]["options"][train.cases[i]["label"]] for i in lab]
    freq = {n: names.count(n) for n in set(names)}
    cw = {n: len(names) / (len(freq) * c) for n, c in freq.items()}

    params = [v for n, v in w.items() if n != "temperature"]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    T0 = H.temperature(w, cfg, 0, k)
    before = evaluate(w, cfg, dev) if dev else None
    best = (before[0] if before else math.inf, {n: v.detach().clone() for n, v in w.items()}, 0)
    log = []
    for ep in range(1, a.epochs + 1):
        random.shuffle(lab)
        tot, nb = 0.0, 0
        for s in range(0, len(lab), a.bs):
            idx = lab[s:s + a.bs]
            h, pad, mk, qt, y, cs = train.batch(idx)
            lg = H.forward(w, h, pad, mk, qt)[:, :k]
            logp = torch.log_softmax(lg / T0, -1)
            wt = torch.tensor([cw[c["options"][c["label"]]] for c in cs])
            loss = -(proper_reward(logp, y) * wt).sum() / wt.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        d = evaluate(w, cfg, dev) if dev else None
        log.append({"epoch": ep, "train_loss": tot / nb, "dev_nll": d[0] if d else None, "dev_acc": d[1] if d else None})
        print(json.dumps(log[-1]), flush=True)
        if d is None or d[0] < best[0]:
            best = (d[0] if d else -ep, {n: v.detach().clone() for n, v in w.items()}, ep)
    if dev:
        for n in w:
            w[n].data.copy_(best[1][n])
    # temperature refit on dev (or train when there is no dev)
    ref = dev or train
    temp = refit_temperature(w, ref)
    refit_kept = True
    if temp in (0.2, 4.0):
        # A refit at the grid's edge means the dev set is (nearly) separable and says nothing about how sure to be.
        # ft1 (refit 0.2 on a template-separable dev) made its one out-of-template miss confidently wrong, p(true)
        # 0.446 -> 0.001. Keep the checkpoint's own temperature instead.
        print(f"warning: the refit temperature {temp} is at the grid's edge; keeping {T0:.4f}", flush=True)
        temp, refit_kept = float(T0), False
    after = evaluate(w, cfg, dev, temp) if dev else None
    write_checkpoint(a.model, a.out, w, temp, kb)
    meta = {"from": os.path.abspath(a.model), "train": os.path.abspath(a.train),
            "dev": os.path.abspath(a.dev) if a.dev else None, "train_sequences": len(lab), "class_counts": freq,
            "class_weights": cw, "seed": a.seed, "epochs": a.epochs, "best_epoch": best[2], "lr": a.lr,
            "weight_decay": a.wd, "batch": a.bs, "temperature_bucket": kb, "temperature_before": T0,
            "temperature_after": temp, "temperature_refit_kept": refit_kept, "dev_before": {"nll": before[0], "acc": before[1], "n": before[2]} if before else None,
            "dev_after": {"nll": after[0], "acc": after[1], "n": after[2]} if after else None, "log": log,
            "seconds": round(time.time() - t_start, 1), "argv": sys.argv}
    json.dump(meta, open(f"{a.out}/finetune.json", "w"), indent=1)
    print(json.dumps({k2: meta[k2] for k2 in ("best_epoch", "temperature_before", "temperature_after",
                                               "dev_before", "dev_after", "seconds")}))


if __name__ == "__main__":
    main()
