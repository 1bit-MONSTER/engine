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
"""tests/npu_plan_designs_parity_test.py -- the plan's `designs` section against the engine's own derivation.

engine#803 A2a: a plan may name the per-role kernel set, and the engine refuses a plan that disagrees with
what the container derives. That check is only useful if the GENERATOR states the same set -- so this test
runs both sides on the same dims and requires identical sets:

  * tools/npu_plan.py's designs_from_config() (Python, what a plan will carry)
  * npu_derive_designs_dump (the engine's q4nx_derive_designs(), the same call the forward makes)

A disagreement here is worse than a missing section: the section is checked both ways, so a generator that
named a different ELF would refuse every plan-carrying model at load. The last part of the test pins the
other half -- the families this tool does not model yet must emit NO section at all, so they keep loading
with the engine's derivation.

Usage: npu_plan_designs_parity_test.py <tools/npu_plan.py> <npu_derive_designs_dump>
"""
import importlib.util
import sys

TOOL, DUMP = sys.argv[1], sys.argv[2]
failures = []


def check(ok, what):
    if not ok:
        print(f"FAIL {what}")
        failures.append(what)


spec = importlib.util.spec_from_file_location("npu_plan", TOOL)
npu_plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npu_plan)


def engine_designs(cfg) -> list[str]:
    """The engine's set for this config, as `ROLE M K N elf` lines."""
    import subprocess

    argv = [DUMP, str(cfg["hidden_size"]), str(cfg["num_attention_heads"]),
            str(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
            str(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]),
            str(cfg["intermediate_size"]), "1" if cfg.get("_fused_gu", True) else "0",
            "1" if cfg.get("attn_output_gate") else "0", cfg.get("_tag", "")]
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{DUMP} failed: {out.stderr.strip()}")
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def plan_designs(cfg) -> list[str] | None:
    got = npu_plan.designs_from_config(cfg, cfg.get("_tag", ""), cfg.get("_fused_gu", True))
    if got is None:
        return None
    return [f"{d['role']} {d['M']} {d['K']} {d['N']} {d['elf']}" for d in got]


# The matrix: dense configs only, in the shapes the derivation's rules branch on -- the 0.6B's own dims, a
# padded hidden (896 -> K 1024, the case engine#845 exists for), q|gate doubling, a split gate/up, odd head
# counts, and a b8 family suffix.
DENSE = [
    {"name": "qwen3-0.6b", "hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
     "num_key_value_heads": 8, "head_dim": 128},
    {"name": "0.6b split gate/up", "hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
     "num_key_value_heads": 8, "head_dim": 128, "_fused_gu": False},
    {"name": "padded hidden 896", "hidden_size": 896, "intermediate_size": 4864, "num_attention_heads": 14,
     "num_key_value_heads": 2, "head_dim": 64},
    {"name": "attn_output_gate", "hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
     "num_key_value_heads": 8, "head_dim": 128, "attn_output_gate": True},
    {"name": "no head_dim key (derived)", "hidden_size": 512, "intermediate_size": 2048,
     "num_attention_heads": 8, "num_key_value_heads": 8},
    {"name": "b8 family", "hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
     "num_key_value_heads": 8, "head_dim": 128, "_tag": "_b8"},
]

for case in DENSE:
    name = case.pop("name")
    want = engine_designs(case)
    got = plan_designs(case)
    check(got is not None, f"{name}: the generator states a set")
    if got is None:
        continue
    check(got == want, f"{name}: generator set == engine set\n    engine: {want}\n    plan:   {got}")

# Families the tool does not model yet: NO section, so the engine keeps deriving and the model still loads.
OMIT = [
    ("mla", {"hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
             "num_key_value_heads": 1, "head_dim": 128, "q_lora_rank": 1536, "kv_lora_rank": 512}),
    ("gdn layer mix", {"hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
                       "num_key_value_heads": 8, "head_dim": 128,
                       "linear_num_key_heads": 4, "linear_key_head_dim": 128,
                       "linear_num_value_heads": 8, "linear_value_head_dim": 128,
                       "layer_types": ["linear_attention", "full_attention"]}),
    ("second attention type", {"hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
                               "num_key_value_heads": 8, "head_dim": 128, "global_head_dim": 256}),
    ("per-layer embeddings", {"hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
                              "num_key_value_heads": 8, "head_dim": 128, "ple_dim": 32}),
    ("double-wide mlp", {"hidden_size": 1024, "intermediate_size": 3072, "num_attention_heads": 16,
                         "num_key_value_heads": 8, "head_dim": 128, "use_double_wide_mlp": True}),
]
for name, cfg in OMIT:
    check(plan_designs(cfg) is None, f"{name}: no section is emitted (the engine keeps deriving)")

print("npu_plan_designs_parity: " + ("ok" if not failures else f"{len(failures)} failure(s)"))
sys.exit(1 if failures else 0)
