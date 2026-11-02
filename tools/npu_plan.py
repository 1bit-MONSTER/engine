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
"""Generate a model's plan.json -- the stream plan the NPU forward reads (engine #799 P1).

    tools/npu_plan.py <model dir> [--kernels DIR] [--out PATH] [--check] [--print]

The forward used to decide three things by looking at which files were in --kernels: whether gate/up is
fused (`full_i8_GU_...` beating split G/U), which artifact family to run (`_b8` / `_pb16i`), and where the
M=16 decode twins are (`kernels-m16`, `m16`, then the kernel dir).  The same model dir served with a
different --kernels therefore changed path silently.  This tool makes those decisions once, from the model
and the kernel set, and writes them down; `npu/plan/plan.h` reads them and refuses by name when the plan
and the directory disagree.

Sources, in order of authority: the container's own tensor table (model.q4nx) for layers, roles and
projection shapes; `config.json` for the layout/language-model dimensions; the kernel directory only for
"which of these artifacts exists" -- the question the plan exists to answer once.

`--check` re-generates and compares against the plan already on disk (rc 1 on any difference), so a stale
plan is caught by a gate rather than by a wrong number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TOOL_VERSION = "1.0"
FORMAT = "1bit-npu-plan/1"


def read_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path, "rb") as f:
            cfg = json.load(f)
    except OSError as exc:
        raise SystemExit(f"{path}: {exc.strerror}")
    except ValueError as exc:
        raise SystemExit(f"{path}: not JSON ({exc})")
    # Some packs nest the language model under text_config.
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        merged = dict(cfg["text_config"])
        for k, v in cfg.items():
            merged.setdefault(k, v)
        return merged
    return cfg


def layer_roles(cfg: dict, n_layers: int) -> list[str]:
    """Per-layer role from the architecture's own layer_types, which is what the forward reads too."""
    types = cfg.get("layer_types")
    if isinstance(types, list) and len(types) == n_layers:
        out = []
        for i, t in enumerate(types):
            if t in ("linear_attention", "linear", "gdn"):
                out.append("gdn")
            elif t in ("full_attention", "sliding_attention", "attention"):
                out.append("attn")
            else:
                raise SystemExit(f"config.json layer_types[{i}] = {t!r} is not a layer type this plan knows")
        return out
    # No layer_types: a homogeneous stack.  Hybrid GDN models always declare them, so this is the
    # dense-attention case (Qwen3 and friends).
    if cfg.get("linear_num_value_heads"):
        raise SystemExit("config.json declares GDN dimensions but no layer_types: the layer mix is unknown")
    return ["attn"] * n_layers


def layers_from_config(cfg: dict, n_layers: int, fused_gu: bool) -> list[dict]:
    """Layer plans: role from layer_types, projection shapes derived the way the engine derives them.

    The shapes are K x N (the engine's projection convention), computed from the model's own
    dimensions -- not the container's stored tensor shapes, which are tiled and are not N x K.
    """
    hidden = cfg["hidden_size"]
    im = cfg["intermediate_size"]
    out = []
    for i, role in enumerate(layer_roles(cfg, n_layers)):
        entry = {"index": i, "role": role}
        if role == "gdn":
            kh = cfg.get("linear_num_key_heads", 0) * cfg.get("linear_key_head_dim", 0)
            vd = cfg.get("linear_num_value_heads", 0) * cfg.get("linear_value_head_dim", 0)
            if not (kh and vd):
                raise SystemExit(f"layer {i} is GDN but config.json lacks the linear_* head dimensions")
            entry["qkv"] = {"K": hidden, "N": 2 * kh + vd, "fused": True}
            entry["attn_gate"] = {"K": hidden, "N": vd}
            entry["ssm_out"] = {"K": vd, "N": hidden}
        else:
            nh = cfg.get("num_attention_heads", 0)
            nkv = cfg.get("num_key_value_heads", nh)
            hd = cfg.get("head_dim") or (hidden // nh if nh else 0)
            if not (nh and hd and nkv):
                raise SystemExit(f"layer {i} is attention but config.json lacks the head dimensions")
            # qwen3_5's attn_output_gate makes q_proj emit q|gate, so the fused QKV is
            # [q | gate | k | v] -- the same doubling the engine's own derivation does
            # (npu/forward/src/q4nx_pack.c:150).  Without this the plan asks for a 3072-wide
            # QKV kernel on a container whose rows are 5120.
            q = nh * hd * (2 if cfg.get("attn_output_gate") else 1)
            entry["qkv"] = {"K": hidden, "N": q + 2 * nkv * hd, "fused": True}
        entry["gate_up"] = {"K": hidden, "N": 2 * im} if fused_gu else {"K": hidden, "N": im}
        entry["down"] = {"K": im, "N": hidden}
        out.append(entry)
    return out


def artifact(model_dir: str, kernels: str, name: str) -> bool:
    for d in (kernels, model_dir):
        if os.path.isfile(os.path.join(d, name)):
            return True
    return False


def decode_dirs(model_dir: str, kernels: str) -> list[dict]:
    """The M=16 twin search order the forward uses, restricted to the ones that exist.

    Each entry names the root (`model` = the -m directory, `kernels` = --kernels) and a relative path
    under it, so the runtime can build the path without guessing which base a bare name meant.
    """
    dirs = []
    for root, rel, absolute in (("model", "kernels-m16", os.path.join(model_dir, "kernels-m16")),
                                ("kernels", "m16", os.path.join(kernels, "m16"))):
        if os.path.isdir(absolute):
            dirs.append({"root": root, "path": rel})
    dirs.append({"root": "kernels", "path": "."})  # the kernel dir itself, always the last candidate
    return dirs


def generate(model_dir: str, kernels: str) -> dict:
    cfg = read_config(model_dir)
    hidden = cfg.get("hidden_size")
    if not isinstance(hidden, int) or hidden <= 0:
        raise SystemExit(f"{model_dir}/config.json: hidden_size is {hidden!r}")
    n_layers = cfg.get("num_hidden_layers")
    if not isinstance(n_layers, int) or n_layers <= 0:
        raise SystemExit(f"{model_dir}/config.json: num_hidden_layers is {n_layers!r}")
    layout = cfg.get("q4nx_weight_layout", "rm")
    if layout not in ("rm", "b8", "pb16i", "t2s"):
        raise SystemExit(f"{model_dir}/config.json: q4nx_weight_layout {layout!r} is not rm, b8, pb16i or t2s")
    # a t2s container names its wire format by version: 5 = v1 (`_t2s`), 6 = v2 (`_t2s2`)
    t2s_v2 = layout == "t2s" and cfg.get("q4nx_weight_layout_version", 5) == 6
    tag = {"pb16i": "_pb16i", "b8": "_b8"}.get(layout, "_t2s2" if t2s_v2 else ("_t2s" if layout == "t2s" else ""))

    im = cfg.get("intermediate_size")
    fused_gu = True
    if isinstance(im, int) and im > 0:
        # The probe the plan replaces, run once and written down: the fused kernel wins, and when
        # neither exists the split layout is assumed (the forward then refuses by name).
        #
        # The gate/up probe uses the *prefill* family's names, which is not the decode suffix: a b8 container
        # blocks the tap at every M (so `_b8`), while a pb16i or t2s container's M=128 prefill stays the plain
        # int8 family and only its decode kernels carry the stream's suffix.  Probing with the decode tag on a
        # t2s container found neither artifact and wrote "fused", which the box's own pb16i container
        # contradicts (split) -- 2026-11-02.
        prefill_tag = "_b8" if layout == "b8" else ""
        gu = f"full_i8_GU_K{hidden}_N{2 * im}{prefill_tag}.elf"
        g = f"full_i8_G_K{hidden}_N{im}{prefill_tag}.elf"
        fused_gu = artifact(model_dir, kernels, gu) or not artifact(model_dir, kernels, g)
    else:
        raise SystemExit(f"{model_dir}/config.json: intermediate_size is {im!r}")

    layers = layers_from_config(cfg, n_layers, fused_gu)
    plan = {
        "format": FORMAT,
        "generated_by": f"tools/npu_plan.py {TOOL_VERSION}",
        "model": {
            "hidden": hidden,
            "layers": n_layers,
            "arch": cfg.get("model_type", ""),
        },
        "container": {
            "layout": layout,
            "chunk_kind": "pb16i" if layout in ("pb16i", "t2s") else "mode0",
            "pb16i": layout in ("pb16i", "t2s"),
        },
        "fusion": {"gate_up": "fused" if fused_gu else "split"},
        "prefill": {"M": 128},
        "decode": {
            "M": 16,
            "enabled": True,
            "dirs": decode_dirs(model_dir, kernels),
            # One full ELF per container: pb16i's key for pb16i, t2s's own key for t2s (a
            # layout never borrows the other's image format).  A t2s decode family shipped as
            # per-role ELFs pays a hw_context switch on every launch -- measured 2026-11-02 at
            # +2.0-2.3 ms, more than the kernels themselves.
            "sealed_elf": cfg.get("q4nx_t2s_sealed_elf" if layout == "t2s" else "q4nx_pb16i_sealed_elf", "")
            if layout in ("pb16i", "t2s") else "",
        },
        "kernels": {"tag": tag},
        "moe": {"batched_expert_gu": moe_batched(cfg, hidden, model_dir, kernels)},
        "layers": layers,
    }
    return plan


def moe_batched(cfg: dict, hidden: int, model_dir: str, kernels: str) -> dict:
    """The batched-expert gate/up kernel, when the model has experts and the kernel exists."""
    experts = cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
    top_k = cfg.get("num_experts_per_tok") or cfg.get("top_k_experts") or 0
    if not (isinstance(experts, int) and experts > 0 and isinstance(top_k, int) and top_k > 1):
        return {"enabled": False, "K": 0, "N": 0}
    for n in (8192, 16384):  # the widths the kernel set carries
        if artifact(model_dir, kernels, f"full_i8_G_K{hidden}_N{n}.elf") and artifact(
            model_dir, kernels, f"full_i8_U_K{hidden}_N{n}.elf"
        ):
            return {"enabled": True, "K": hidden, "N": n}
    return {"enabled": False, "K": 0, "N": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", help="the model directory (config.json + model.q4nx)")
    ap.add_argument("--kernels", default=None,
                    help="kernel directory to describe (default: <model dir>/kernels, else the model dir)")
    ap.add_argument("--out", default=None, help="where to write the plan (default <model dir>/plan.json)")
    ap.add_argument("--check", action="store_true",
                    help="re-generate and compare with the plan on disk; rc 1 if it differs")
    ap.add_argument("--print", dest="show", action="store_true", help="print the plan instead of writing it")
    args = ap.parse_args(argv)

    model_dir = args.model_dir.rstrip("/")
    if not os.path.isdir(model_dir):
        raise SystemExit(f"{model_dir}: not a directory")
    kernels = args.kernels
    if kernels is None:
        candidate = os.path.join(model_dir, "kernels")
        kernels = candidate if os.path.isdir(candidate) else model_dir
    out = args.out or os.path.join(model_dir, "plan.json")

    plan = generate(model_dir, kernels)
    text = json.dumps(plan, indent=1, sort_keys=False) + "\n"

    if args.show:
        sys.stdout.write(text)
        return 0

    if args.check:
        try:
            with open(out, "r", encoding="utf-8") as f:
                on_disk = f.read()
        except OSError as exc:
            print(f"{out}: {exc.strerror} (run without --check to write it)", file=sys.stderr)
            return 1
        if on_disk != text:
            print(f"{out}: differs from what the model dir generates -- regenerate it", file=sys.stderr)
            for line in __import__("difflib").unified_diff(
                on_disk.splitlines(), text.splitlines(), fromfile="on disk", tofile="generated", lineterm=""
            ):
                print(line, file=sys.stderr)
            return 1
        print(f"{out}: matches the model dir")
        return 0

    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, out)
    print(f"wrote {out} ({len(plan['layers'])} layers, gate/up {plan['fusion']['gate_up']}, "
          f"layout {plan['container']['layout']}, tag {plan['kernels']['tag'] or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
