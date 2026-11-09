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

TOOL_VERSION = "1.1"
FORMAT = "1bit-npu-plan/1"

# The two grids a projection's K and N live on, mirroring npu/forward/include/q4nx_pack.h
# (q4nx_pad_k / q4nx_pad_n).  They are the container's chunk width (npu/q4nx.h kCols = 256) and the
# full ELF's column pass (32 rows x 4 x cols(4) = 512).  A model already on both grids -- every one
# validated before this existed -- is unchanged; a pack with hidden 896 is stored as K=1024 and its
# kernels are built and named on the same grid (engine#845).
K_GRID = 256
N_GRID = 512


def mla_config(cfg: dict) -> bool:
    """The MLA family, which the C side flags from the same keys (q4nx_pack.c derive_mla's caller)."""
    return bool(cfg.get("q_lora_rank") or cfg.get("kv_lora_rank"))


def pad_k(k: int) -> int:
    return (k + K_GRID - 1) // K_GRID * K_GRID


def pad_n(n: int) -> int:
    return (n + N_GRID - 1) // N_GRID * N_GRID


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
            entry["qkv"] = {"K": pad_k(hidden), "N": pad_n(2 * kh + vd), "fused": True}
            entry["attn_gate"] = {"K": pad_k(hidden), "N": pad_n(vd)}
            entry["ssm_out"] = {"K": pad_k(vd), "N": pad_n(hidden)}
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
            entry["qkv"] = {"K": pad_k(hidden), "N": pad_n(q + 2 * nkv * hd), "fused": True}
        entry["gate_up"] = {"K": pad_k(hidden), "N": pad_n(2 * im if fused_gu else im)}
        entry["down"] = {"K": pad_k(im), "N": pad_n(hidden)}
        out.append(entry)
    return out


def attention_mode(model_dir: str) -> str:
    """The mask the pack itself declares, from `decision_config.json` beside the model.

    A d3 readout pack is trained noncausal on its full-attention layers and says so; every other model
    on this box has no `decision_config.json` and is causal, which is also the field's default in
    `npu/plan/plan.h`.  An unknown value is refused here rather than defaulted, because the two modes
    produce different numbers and a guessed mask is a silently different model.
    """
    path = os.path.join(model_dir, "decision_config.json")
    if not os.path.isfile(path):
        return "causal"
    try:
        with open(path, "rb") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: cannot read the attention mode ({exc})")
    mode = cfg.get("attention_mode", "causal")
    if mode not in ("causal", "noncausal_full_attention"):
        raise SystemExit(f"{path}: attention_mode {mode!r} is not causal or noncausal_full_attention")
    return mode


def rope_block(cfg: dict) -> dict:
    """The rotary the model declares: `plain` one index, or the vision-conditioned `mrope` planes.

    Qwen3-VL states `rope_scaling: {mrope_interleaved: true, mrope_section: [24, 20, 20]}` -- on the text
    tower, so a VL pack's copy sits under `text_config`.  The split is written into the plan because a
    caller that builds the position block has to know how the rotary width divides over (t, h, w); without
    it the three planes have no declared width and a caller would guess.  Everything else (including a
    `rope_scaling` that carries no `mrope_section`, e.g. a plain linear/YaRN scaling) means one index, which
    is what every plan meant before this field existed.
    """
    def sections_of(block) -> object:
        if not isinstance(block, dict):
            return None
        scaling = block.get("rope_scaling")
        return scaling.get("mrope_section") if isinstance(scaling, dict) else None

    sections = sections_of(cfg)
    if sections is None:
        sections = sections_of(cfg.get("text_config"))
    if sections is None:
        return {"kind": "plain", "planes": 1}
    if not isinstance(sections, list) or not sections or not all(
            isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in sections):
        raise SystemExit(f"rope_scaling.mrope_section {sections!r} is not a list of positive integers "
                         "(one width per plane)")
    return {"kind": "mrope", "planes": len(sections), "sections": list(sections)}


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


def tile_order_claim(model_dir: str) -> tuple[str, str]:
    """(the tile_order this plan may declare, a note to print) from the directory's repack stamp.

    The plan's `container.tile_order` is the only thing standing between a container whose tiles are
    another order and a clean 200 with noise in it (engine#1110): `npu/plan/plan.h` refuses any value
    but "engine-1".  So the value is a measurement, not a constant -- `scripts/repack_gguf.py` holds
    both the source GGUF and the tiles and writes what it measured into `repack-stamp.txt`, including
    on the path where it refuses the directory (measured: a refused dir planned anyway declared
    "engine-1" and the loader accepted it).

    No stamp is not evidence of anything: those directories predate the field, so they keep the
    previous default silently rather than turning every plan run into a warning.
    """
    stamp = os.path.join(model_dir, "repack-stamp.txt")
    if not os.path.isfile(stamp):
        return "engine-1", ""
    fields = {}
    with open(stamp, encoding="utf-8", errors="replace") as f:
        for line in f:
            k, sep, v = line.partition("=")
            if sep:
                fields[k.strip()] = v.strip()
    order = fields.get("tile_order", "")
    checked = fields.get("tile_check", "")
    if not order:
        # A stamp written before this field existed records provenance and says NOTHING about the tiles, so
        # it is no evidence rather than evidence against: both repack-cache directories on this box
        # (~/.cache/1bit/q4nx/Qwen3-0.6B-Q4_K_M and ZR1-1.5B-Q4_K_M) carry exactly that stamp, and treating
        # it as a measurement would be this check breaking two working directories.
        return "engine-1", ""
    if order == "engine-1":
        return "engine-1", f"tile order measured at repack: {checked}" if checked else ""
    if order == "unverified":
        return "engine-1", (f"tile order NOT measured ({checked or 'no reason recorded'}): the plan declares "
                            f"engine-1 because that is the only order this loader accepts, not because anything "
                            f"checked these tiles")
    raise SystemExit(f"{stamp}: tile_order={order!r} -- the repack measured these tiles as NOT the engine-1 "
                     f"order this engine decodes ({checked or 'no verdict line recorded'}), so this plan may not "
                     f"declare engine-1.  Re-convert with a converter that writes the engine-1 order "
                     f"(engine#1110; `scripts/repack_gguf.py --tile-check <dir> <model.gguf>` re-measures).")


def generate(model_dir: str, kernels: str) -> dict:
    # A measurement that contradicts the only order this loader decodes is refused before anything is
    # derived from the directory (engine#1110): no plan for tiles the repack already rejected.
    tile_order, tile_note = tile_order_claim(model_dir)
    if tile_note:
        print(f"{model_dir}: {tile_note}", file=sys.stderr)
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
        #
        # K and N are the padded grids (engine#845), the same ones the forward derives: a pack whose hidden is
        # not a multiple of 256 has no artifact named after its un-padded shape.
        prefill_tag = "_b8" if layout == "b8" else ""
        gu = f"full_i8_GU_K{pad_k(hidden)}_N{pad_n(2 * im)}{prefill_tag}.elf"
        g = f"full_i8_G_K{pad_k(hidden)}_N{pad_n(im)}{prefill_tag}.elf"
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
            # The tile order of the payload this plan describes (engine#1110).  Declared so a container
            # from another converter -- a different order that would decode as noise with no error -- is
            # refused by name at load instead of being served.  Absent means this same value.  Measured
            # from the directory's repack stamp when one exists (see tile_order_claim), never assumed.
            "tile_order": tile_order,
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
        "attention": {"mode": attention_mode(model_dir)},
        # The rotary, from the pack's own declaration (engine#799): `plain` one index, or `mrope` with the
        # planes' split.  Stated rather than left to a key's absence, like kv below -- a plan should read as
        # what its positions are.
        "rope": rope_block(cfg),
        # P7 (#806) part A: the KV contract.  f16 with no rotation is what every container shipped so
        # far means; the generator states it rather than leaving it to a key's absence, so a plan reads
        # as what its cache is.  Turning int8 on is a per-target decision (its rotation and scale
        # sidecar are kernel-adjacent), so nothing here enables it -- a plan that asks for int8 is
        # either written by hand or by a target-aware generator, and the plan loader validates it.
        "kv": {"key": "f16", "value": "f16", "hadamard": {"block": 0, "signs": ""}},
        "moe": {"batched_expert_gu": moe_batched(cfg, hidden, model_dir, kernels)},
        "layers": layers,
    }
    # The per-role kernel set (engine#803 A2a).  Written only when this tool can state the WHOLE set:
    # the plan's section is checked both ways, so a partial one would be refused at load and a missing
    # one is exactly today's behaviour (the engine derives it).
    designs = designs_from_config(cfg, prefill_tag if layout == "b8" else "", fused_gu)
    if designs is not None:
        plan["designs"] = designs
    return plan


def missing_artifacts(plan: dict, dirs: list[str]) -> list[str]:
    """The designs whose artifact is in none of `dirs`, as "designs[i] (ROLE): elf" strings.

    This is rule (c) at import time (engine#799): a plan can be complete as JSON and unservable in fact, and
    the question "is this op's kernel here" belongs at generation, where the model and kernel directories are
    both in hand.  `Plan::check_artifacts()` in npu/plan/plan.h asks the same question at load with the same
    rule -- present and non-empty, first directory that has it wins -- so the two cannot disagree about what
    "here" means.
    """
    missing = []
    for i, d in enumerate(plan.get("designs") or []):
        elf = d.get("elf", "")
        if not elf:
            continue
        found = False
        for root in dirs:
            if not root:
                continue
            path = os.path.join(root, elf)
            try:
                if os.path.getsize(path) >= 1:
                    found = True
                    break
            except OSError:
                continue
        if not found:
            missing.append(f"designs[{i}] ({d.get('role', '?')}): {elf}")
    return missing


def designs_from_config(cfg: dict, prefill_tag: str, fused_gu: bool) -> list[dict] | None:
    """The engine's per-role kernel set for this config, or None when this tool cannot state it fully.

    These are the rules q4nx_derive_designs() applies (npu/forward/src/q4nx_pack.c), ported here so a
    plan can NAME the set instead of the engine re-deriving it (engine#803 A2a).  The port is checked
    against the C derivation by tests/npu_plan_designs_parity_test.py, which runs both on the same
    dims and requires identical sets -- the reason this function exists as a function.

    WHY IT RETURNS None FOR SOME MODELS: the plan's `designs` section is checked BOTH ways, so a
    section that omitted a role the container derives would be refused at load. The families below
    derive roles this tool does not model yet (MLA's q_a/q_b/kv_a, Gemma4's second attention type and
    per-layer embeddings and double-wide MLP, GDN's fused q|k|v / z / ssm_out), and for those the right
    output is NO section -- absent means the engine keeps deriving, which is what every plan written
    before this field means. Emitting a partial set would be worse than emitting nothing.
    """
    if mla_config(cfg) or cfg.get("layer_types") or cfg.get("global_head_dim") or cfg.get("ple_dim") \
            or cfg.get("use_double_wide_mlp") or cfg.get("linear_num_key_heads"):
        return None
    hidden = cfg.get("hidden_size")
    im = cfg.get("intermediate_size")
    nh = cfg.get("num_attention_heads", 0)
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim") or (hidden // nh if nh else 0)
    if not (hidden and im and nh and hd and nkv):
        return None

    def elf(tag: str, k: int, n: int) -> str:
        return f"full_i8_{tag}_K{pad_k(k)}_N{pad_n(n)}{prefill_tag}.elf"

    # q_rows is the attention width WITHOUT the output gate: attn_output_gate doubles what the q
    # projection emits (so it doubles QKV's N), but O still reads one head width per head
    # (q4nx_derive_designs: `q_rows = NH*HD` for O's K, `q_rows * (gate?2:1)` for QKV's N).
    q_rows = nh * hd
    qkv_rows = q_rows * (2 if cfg.get("attn_output_gate") else 1)
    kv_rows = nkv * hd
    out = [
        {"role": "QKV", "M": 128, "K": pad_k(hidden), "N": pad_n(qkv_rows + 2 * kv_rows),
         "elf": elf("QKV", hidden, qkv_rows + 2 * kv_rows)},
        {"role": "O", "M": 128, "K": pad_k(q_rows), "N": pad_n(hidden),
         "elf": elf("O", q_rows, hidden)},
    ]
    # gate/up is a plan decision the caller already made (the artifact probe above), so it is passed in
    # rather than probed a second time here: two copies of that rule is how a plan and its images drift.
    if fused_gu:
        out.append({"role": "GU", "M": 128, "K": pad_k(hidden), "N": pad_n(2 * im),
                    "elf": elf("GU", hidden, 2 * im)})
    else:
        out.append({"role": "G", "M": 128, "K": pad_k(hidden), "N": pad_n(im),
                    "elf": elf("G", hidden, im)})
        out.append({"role": "U", "M": 128, "K": pad_k(hidden), "N": pad_n(im),
                    "elf": elf("U", hidden, im)})
    out.append({"role": "D", "M": 128, "K": pad_k(im), "N": pad_n(hidden),
                "elf": elf("D", im, hidden)})
    return out


def moe_batched(cfg: dict, hidden: int, model_dir: str, kernels: str) -> dict:
    """The batched-expert gate/up kernel, when the model has experts and the kernel exists."""
    experts = cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
    top_k = cfg.get("num_experts_per_tok") or cfg.get("top_k_experts") or 0
    if not (isinstance(experts, int) and experts > 0 and isinstance(top_k, int) and top_k > 1):
        return {"enabled": False, "K": 0, "N": 0}
    for n in (8192, 16384):  # the widths the kernel set carries
        if artifact(model_dir, kernels, f"full_i8_G_K{pad_k(hidden)}_N{n}.elf") and artifact(
            model_dir, kernels, f"full_i8_U_K{pad_k(hidden)}_N{n}.elf"
        ):
            return {"enabled": True, "K": pad_k(hidden), "N": n}
    return {"enabled": False, "K": 0, "N": 0}


def _only_defaulted_tile_order(on_disk: str, generated: str) -> bool:
    """True when the ONLY difference is a container.tile_order the on-disk plan does not carry.

    engine#1110 added `container.tile_order`; absent means "engine-1", which is what the generator writes,
    so an older plan is not stale -- it is silent about a field whose default is the generated value.
    """
    try:
        a = json.loads(on_disk)
        b = json.loads(generated)
    except ValueError:
        return False
    ca, cb = a.get("container"), b.get("container")
    if not isinstance(ca, dict) or not isinstance(cb, dict):
        return False
    if "tile_order" in ca or cb.get("tile_order") != "engine-1":
        return False
    ca = dict(ca)
    ca["tile_order"] = "engine-1"
    a["container"] = ca
    return a == b


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
        # Rule (c), import-time half: a plan that names an op with no kernel in this directory is refused
        # before it is compared or written, naming the design -- the same refusal the loader makes, asked at
        # the point where the model dir and the kernel dir are both in hand.
        missing = missing_artifacts(plan, [kernels])
        if missing:
            for m in missing:
                print(f"{model_dir}: {m} is in no kernel in {kernels}", file=sys.stderr)
            print(f"{model_dir}: the plan would name {len(missing)} op(s) with no kernel here -- refusing",
                  file=sys.stderr)
            return 1
        try:
            with open(out, "r", encoding="utf-8") as f:
                on_disk = f.read()
        except OSError as exc:
            print(f"{out}: {exc.strerror} (run without --check to write it)", file=sys.stderr)
            return 1
        if on_disk != text and _only_defaulted_tile_order(on_disk, text):
            # A plan written before container.tile_order existed means exactly what the generated one says
            # ("engine-1"), so a plan gate must not go red over a field whose absence has the same meaning.
            # The loader makes the same assumption (npu/plan/plan.h), which is why this is not a fudge.
            print(f'{out}: matches the model dir (written before container.tile_order, which defaults to '
                  f'"engine-1" -- the value it generates)')
            return 0
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
          f"layout {plan['container']['layout']}, tag {plan['kernels']['tag'] or 'none'}, "
          f"attention {plan['attention']['mode']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
