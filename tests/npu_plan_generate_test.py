#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""tests/npu_plan_generate_test.py -- tools/npu_plan.py on a synthetic model dir, no model and no kernels.

What it pins down is the decision the plan replaces: gate/up is fused when the fused kernel is there OR no
split kernel is, and `--check` catches a plan that no longer matches its model dir.
"""
import json
import os
import subprocess
import sys
import tempfile

TOOL = sys.argv[1]
failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
        print(f"FAIL {what}")


def model_dir(root, *, layout="rm", layers=2, layer_types=None, sealed="", **extra):
    d = os.path.join(root, "model")
    os.makedirs(os.path.join(d, "kernels"), exist_ok=True)
    cfg = {
        "model_type": "qwen3_5",
        "hidden_size": 1024,
        "num_hidden_layers": layers,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 8,
        "linear_value_head_dim": 128,
        "q4nx_weight_layout": layout,
    }
    cfg.update(extra)
    if layer_types:
        cfg["layer_types"] = layer_types
    if sealed:
        cfg["q4nx_pb16i_sealed_elf"] = sealed
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "model.q4nx"), "wb") as f:
        f.write(b"\x00" * 16)  # the generator reads config.json, not the container
    return d


def run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def touch(path):
    with open(path, "wb") as f:
        f.write(b"\0" * 8)


with tempfile.TemporaryDirectory() as root:
    # 1. Only the fused GU kernel: the plan says fused.
    d = model_dir(os.path.join(root, "a"), layer_types=["full_attention", "linear_attention"])
    k = os.path.join(d, "kernels")
    touch(os.path.join(k, "full_i8_GU_K1024_N6144.elf"))
    r = run(d)
    check(r.returncode == 0, f"generate rc=0 (got {r.returncode}: {r.stderr[:200]})")
    plan = json.load(open(os.path.join(d, "plan.json")))
    check(plan["format"] == "1bit-npu-plan/1", "format is the versioned one")
    check(plan["fusion"]["gate_up"] == "fused", "fused GU kernel present -> gate_up fused")
    check([l["role"] for l in plan["layers"]] == ["attn", "gdn"], "layer_types decide the roles")
    check(plan["layers"][0]["qkv"]["N"] == (16 + 2 * 8) * 128, "attn QKV N is (nh + 2*nkv) * hd")
    check(plan["layers"][1]["attn_gate"]["N"] == 8 * 128, "GDN gate N is the value width")
    check(plan["decode"]["dirs"][-1] == {"root": "kernels", "path": "."}, "the kernel dir is the last decode dir")
    # engine#1110: the generator declares the payload's q4_1 tile order, so a container from another
    # converter (a different order that would decode as noise with no error) is refused at load.
    check(plan["container"]["tile_order"] == "engine-1",
          "the plan declares container.tile_order = engine-1 (engine#1110)")

    # 2. --check agrees with what is on disk, and catches a kernel set that changed under it.
    r = run(d, "--check")
    check(r.returncode == 0, f"--check passes on a fresh plan (rc={r.returncode}: {r.stderr[:200]})")
    os.remove(os.path.join(k, "full_i8_GU_K1024_N6144.elf"))
    touch(os.path.join(k, "full_i8_G_K1024_N3072.elf"))  # only the split kernel now
    r = run(d, "--check")
    check(r.returncode == 1, "--check fails when the kernel set no longer matches the plan")
    check("gate_up" in r.stderr, "--check prints the difference")
    r = run(d)
    plan = json.load(open(os.path.join(d, "plan.json")))
    check(plan["fusion"]["gate_up"] == "split", "split kernel only -> gate_up split")

    # 3. Neither kernel: FUSED, which is the legacy probe'"'"'s own rule ("the fused kernel wins, and
    #    with no split kernel on disk there is nothing to say the model is split").  Behaviour
    #    preservation means the generator answers what the probe answered -- the difference is that
    #    the answer is now written down, and the runtime refuses by name instead of re-deciding.
    os.remove(os.path.join(k, "full_i8_G_K1024_N3072.elf"))
    run(d)
    check(json.load(open(os.path.join(d, "plan.json")))["fusion"]["gate_up"] == "fused",
          "no GU kernel at all -> fused, exactly as the legacy probe decided")

    # 4. pb16i: the tag and the sealed ELF come from the container's config.
    d2 = model_dir(os.path.join(root, "b"), layout="pb16i", sealed="pb16i_sealed.elf",
                   layer_types=["full_attention", "full_attention"])
    touch(os.path.join(d2, "kernels", "full_i8_G_K1024_N3072_pb16i.elf"))
    run(d2)
    p2 = json.load(open(os.path.join(d2, "plan.json")))
    check(p2["kernels"]["tag"] == "_pb16i" and p2["decode"]["sealed_elf"] == "pb16i_sealed.elf",
          "pb16i layout -> _pb16i tag and the sealed ELF named")

    # 4b. t2s names its OWN sealed ELF.  A t2s decode family shipped as eight per-role ELFs pays a
    #     hw_context switch on every launch -- measured 2026-11-02 in-loop at +2.0-2.3 ms against this
    #     engine's documented ~2.7 ms between contexts, more than the kernels' own device time -- so the
    #     one-context sealed set is the t2s layout's lever too, not a pb16i special case.
    d2b = model_dir(os.path.join(root, "b2"), layout="t2s", q4nx_weight_layout_version=6,
                    q4nx_t2s_sealed_elf="SEALED-T2S.elf",
                    layer_types=["full_attention", "full_attention"])
    touch(os.path.join(d2b, "kernels", "full_i8_G_K1024_N3072_t2s2_m16.elf"))
    run(d2b)
    p2b = json.load(open(os.path.join(d2b, "plan.json")))
    check(p2b["kernels"]["tag"] == "_t2s2" and p2b["decode"]["sealed_elf"] == "SEALED-T2S.elf",
          "t2s v2 layout -> _t2s2 tag and its own sealed ELF named")

    # 4c. and a layout never borrows the other's key: a t2s container carrying only the pb16i key must stay
    #     per-role, because a pb16i ELF reads 4,608-byte tiles out of 2,176-byte images.
    d2c = model_dir(os.path.join(root, "b3"), layout="t2s", q4nx_weight_layout_version=6,
                    sealed="pb16i_sealed.elf", layer_types=["full_attention", "full_attention"])
    run(d2c)
    check(json.load(open(os.path.join(d2c, "plan.json")))["decode"]["sealed_elf"] == "",
          "a t2s container does not inherit q4nx_pb16i_sealed_elf")

    # 4d. engine#1990: a wide prefill family is carried into prefill{}, with its row count; a pack without
    #     one keeps the two-key-free block it always had; and a layout reads only its own key.
    check(p2b["prefill"] == {"M": 128}, "no wide key in config -> prefill block unchanged")
    d2d = model_dir(os.path.join(root, "b4"), layout="t2s", q4nx_weight_layout_version=6,
                    q4nx_t2s_sealed_elf="SEALED-T2S.elf", q4nx_t2s_sealed_wide_elf="SEALED-T2S-m32.elf",
                    layer_types=["full_attention", "full_attention"])
    run(d2d)
    p2d = json.load(open(os.path.join(d2d, "plan.json")))
    check(p2d["prefill"] == {"M": 128, "sealed_elf": "SEALED-T2S-m32.elf", "sealed_rows": 32},
          "t2s wide key -> prefill.sealed_elf and the default 32 rows")
    d2e = model_dir(os.path.join(root, "b5"), layout="t2s", q4nx_weight_layout_version=6,
                    q4nx_t2s_sealed_elf="SEALED-T2S.elf", q4nx_t2s_sealed_wide_elf="W64.elf",
                    q4nx_sealed_wide_rows=64, layer_types=["full_attention", "full_attention"])
    run(d2e)
    check(json.load(open(os.path.join(d2e, "plan.json")))["prefill"]["sealed_rows"] == 64,
          "q4nx_sealed_wide_rows is carried")
    d2f = model_dir(os.path.join(root, "b6"), layout="t2s", q4nx_weight_layout_version=6,
                    q4nx_t2s_sealed_elf="SEALED-T2S.elf", q4nx_pb16i_sealed_wide_elf="PB-WIDE.elf",
                    layer_types=["full_attention", "full_attention"])
    run(d2f)
    check(json.load(open(os.path.join(d2f, "plan.json")))["prefill"] == {"M": 128},
          "a t2s container does not inherit q4nx_pb16i_sealed_wide_elf")

    # 4e. engine#2073: the narrow decode family is carried into decode{} the same way.
    check("narrow_elf" not in p2b["decode"] and "narrow_rows" not in p2b["decode"],
          "no narrow key in config -> decode block unchanged")
    d2g = model_dir(os.path.join(root, "b7"), layout="t2s", q4nx_weight_layout_version=6,
                    q4nx_t2s_sealed_elf="SEALED-T2S.elf", q4nx_t2s_sealed_narrow_elf="SEALED-T2S-m4.elf",
                    layer_types=["full_attention", "full_attention"])
    run(d2g)
    dg = json.load(open(os.path.join(d2g, "plan.json")))["decode"]
    check(dg.get("narrow_elf") == "SEALED-T2S-m4.elf" and dg.get("narrow_rows") == 4,
          "t2s narrow key -> decode.narrow_elf and the default 4 rows")
    d2h = model_dir(os.path.join(root, "b8"), layout="t2s", q4nx_weight_layout_version=6,
                    q4nx_t2s_sealed_elf="SEALED-T2S.elf", q4nx_pb16i_sealed_narrow_elf="PB-NARROW.elf",
                    layer_types=["full_attention", "full_attention"])
    run(d2h)
    check("narrow_elf" not in json.load(open(os.path.join(d2h, "plan.json")))["decode"],
          "a t2s container does not inherit q4nx_pb16i_sealed_narrow_elf")

    # 5. The plan printer writes nothing.
    r = run(d2, "--print")
    check(r.returncode == 0 and '"format": "1bit-npu-plan/1"' in r.stdout, "--print emits the plan on stdout")

    # 5b. attention.mode comes from the pack's own decision_config.json, which is the file the d3 runtime
    #     obeys; no such file means causal, which is what every non-readout model on this box is.
    check(json.load(open(os.path.join(d2, "plan.json")))["attention"]["mode"] == "causal",
          "no decision_config.json -> attention causal")
    with open(os.path.join(d2, "decision_config.json"), "w") as f:
        json.dump({"attention_mode": "noncausal_full_attention", "format_id": "d3-code-readout-v1"}, f)
    run(d2)
    check(json.load(open(os.path.join(d2, "plan.json")))["attention"]["mode"] == "noncausal_full_attention",
          "a noncausal pack's decision_config.json sets attention.mode")
    r = run(d2, "--check")
    check(r.returncode == 0, "--check passes with the attention block on disk")
    with open(os.path.join(d2, "decision_config.json"), "w") as f:
        json.dump({"attention_mode": "backwards"}, f)
    r = run(d2)
    check(r.returncode != 0 and "attention_mode" in r.stderr,
          f"an unknown attention_mode is refused, not defaulted (rc={r.returncode})")
    os.remove(os.path.join(d2, "decision_config.json"))

    # 6. attn_output_gate (qwen3_5's q|gate): q_proj emits q AND the output gate, so the fused QKV
    #    carries twice the q rows.  Found by running this generator on the real d3-lite pack, whose
    #    q_proj is [4096,1024] with 8 heads of 256 while the tool asked for a 3072-wide kernel.
    d3 = model_dir(os.path.join(root, "c"), layers=1, layer_types=["full_attention"],
                   attn_output_gate=True)
    run(d3)
    p3 = json.load(open(os.path.join(d3, "plan.json")))
    check(p3["layers"][0]["qkv"]["N"] == (2 * 16 + 2 * 8) * 128,
          "attn_output_gate doubles the q rows in the attn QKV (q4nx_pack.c:150)")

print("npu_plan_generate: ok" if not failures else "npu_plan_generate: FAILED")
# ---- rule (c) at import time: --check refuses a plan that names an artifact no kernel here provides ----
#
# The plan can be complete as JSON and unservable in fact.  The loader refuses it (Plan::check_artifacts,
# engine#799), and generation has to refuse it too -- otherwise the plan is written happily and the failure
# surfaces at load, on the box, instead of where the model dir and the kernel dir are both in hand.
with tempfile.TemporaryDirectory() as root:
    d = os.path.join(root, "dense")
    kdir = os.path.join(d, "kernels")
    os.makedirs(kdir, exist_ok=True)
    # A DENSE config on purpose (model_dir()'s default carries GDN dimensions, which this tool refuses
    # without layer_types): hidden 1024 / IM 3072 / 16 heads / 8 KV / head_dim 128.
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 2,
                   "intermediate_size": 3072, "num_attention_heads": 16,
                   "num_key_value_heads": 8, "head_dim": 128, "q4nx_weight_layout": "rm"}, f)
    with open(os.path.join(d, "model.q4nx"), "wb") as f:
        f.write(b"\0" * 16)

    ok = run(d, "--kernels", kdir, "--print")
    check(ok.returncode == 0, "a dense model generates a plan: " + ok.stderr.strip()[:160])
    plan = json.loads(ok.stdout) if ok.returncode == 0 else {}
    designs = plan.get("designs") or []
    check(len(designs) > 0, "the generated plan carries a designs section")

    plan_path = os.path.join(d, "plan.json")
    # Written by the tool itself, so the comparison later is about the artifacts and not about formatting.
    wrote = run(d, "--kernels", kdir)
    check(wrote.returncode == 0, "the tool writes the plan: " + wrote.stderr.strip()[:160])

    # Nothing on disk: --check refuses, names the design, and leaves the plan alone.
    refused = run(d, "--kernels", kdir, "--check")
    check(refused.returncode == 1, "--check refuses a plan whose artifacts are absent")
    check("designs[0]" in refused.stderr, "the refusal names the design index: " + refused.stderr.strip()[:120])
    with open(plan_path) as f:
        check(json.load(f) == plan, "--check does not rewrite the plan it refused")

    # Every artifact present: --check is green again (the fixtures are the names the plan asked for).
    for des in designs:
        touch(os.path.join(kdir, des["elf"]))
    accepted = run(d, "--kernels", kdir, "--check")
    check(accepted.returncode == 0, "--check accepts the plan once every named artifact is present: "
          + accepted.stderr.strip()[:160])

sys.exit(1 if failures else 0)
