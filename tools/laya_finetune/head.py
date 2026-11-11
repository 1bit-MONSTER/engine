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
"""Laya's decision head in plain torch, over the encoder states `1bit route --dump-encoder` writes.

It is the head of laya/scorer.cpp (and of DecisionModel upstream): h + type_emb[qtype], two pre-norm
TransformerEncoderLayers (d 1024, 16 heads, ff 4096, relu, key padding), the option markers, and the scorer
LayerNorm -> Linear -> GELU(erf) -> Linear(1); probabilities are softmax(logit / T) with T from
temperature_by_options. The encoder is frozen: only these tensors are trained.
"""
import json
import struct

import numpy as np
import torch
import torch.nn.functional as F

HEAD_PREFIXES = ("head.", "type_emb.", "scorer.")
DT = {"F32": np.float32, "F16": np.float16, "BF16": None}


def read_safetensors(path, want=lambda name: True):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__" or not want(name):
                continue
            a, b = info["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if info["dtype"] == "BF16":
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
            else:
                t = torch.from_numpy(np.frombuffer(raw, dtype=DT[info["dtype"]]).astype(np.float32).copy())
            out[name] = t.reshape(info["shape"])
        return header, out


def load_head(model_dir):
    _, w = read_safetensors(f"{model_dir}/model.safetensors",
                            lambda n: n.startswith(HEAD_PREFIXES) or n == "temperature")
    cfg = json.load(open(f"{model_dir}/rl_agent_config.json"))
    return {k: v.clone().requires_grad_(k != "temperature") for k, v in w.items()}, cfg


def bucket(qtype, k):
    sz = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return ("choice:", "score:", "noul:")[qtype] + sz


def temperature(w, cfg, qtype, k):
    return float(cfg.get("temperature_by_options", {}).get(bucket(qtype, k), w["temperature"][qtype]))


def forward(w, h, pad, markers, qtype, nhead=16):
    """h [B, L, D] (encoder after final norm), pad [B, L] True = padding, markers [B, K] long, qtype [B] long.
    Returns the logits [B, K] before temperature."""
    x = h + w["type_emb.weight"][qtype][:, None, :]
    B, L, D = x.shape
    for i in range(2):
        p = f"head.layers.{i}."
        y = F.layer_norm(x, (D,), w[p + "norm1.weight"], w[p + "norm1.bias"], 1e-5)
        a, _ = F.multi_head_attention_forward(
            y.transpose(0, 1), y.transpose(0, 1), y.transpose(0, 1), D, nhead,
            w[p + "self_attn.in_proj_weight"], w[p + "self_attn.in_proj_bias"], None, None, False, 0.0,
            w[p + "self_attn.out_proj.weight"], w[p + "self_attn.out_proj.bias"], training=False,
            key_padding_mask=pad, need_weights=False)
        x = x + a.transpose(0, 1)
        y = F.layer_norm(x, (D,), w[p + "norm2.weight"], w[p + "norm2.bias"], 1e-5)
        y = F.linear(F.relu(F.linear(y, w[p + "linear1.weight"], w[p + "linear1.bias"])),
                     w[p + "linear2.weight"], w[p + "linear2.bias"])
        x = x + y
    m = torch.gather(x, 1, markers[:, :, None].expand(-1, -1, D))
    m = F.layer_norm(m, (D,), w["scorer.0.weight"], w["scorer.0.bias"], 1e-5)
    m = F.gelu(F.linear(m, w["scorer.1.weight"], w["scorer.1.bias"]))
    return F.linear(m, w["scorer.3.weight"], w["scorer.3.bias"])[..., 0]


class Dump:
    """A `1bit route --dump-encoder --out DIR` directory."""

    def __init__(self, d):
        self.dir = d
        self.info = json.load(open(f"{d}/dump.json"))
        self.cases = [json.loads(x) for x in open(f"{d}/cases.jsonl") if x.strip()]
        D = self.info["d"]
        self.h = np.memmap(f"{d}/h.f32", dtype=np.float32, mode="r", shape=(self.info["rows"], D))

    def batch(self, idx):
        cs = [self.cases[i] for i in idx]
        L = max(c["len"] for c in cs)
        K = max(len(c["markers"]) for c in cs)
        D = self.info["d"]
        h = torch.zeros(len(cs), L, D)
        pad = torch.ones(len(cs), L, dtype=torch.bool)
        mk = torch.zeros(len(cs), K, dtype=torch.long)
        for b, c in enumerate(cs):
            h[b, :c["len"]] = torch.from_numpy(np.array(self.h[c["offset"]:c["offset"] + c["len"]]))
            pad[b, :c["len"]] = False
            mk[b, :len(c["markers"])] = torch.tensor(c["markers"])
        qt = torch.tensor([c["qtype"] for c in cs])
        y = torch.tensor([c["label"] for c in cs])
        return h, pad, mk, qt, y, cs
