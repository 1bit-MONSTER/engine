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
"""Pin the registry gap guard's firing behavior (tools/registry_build.py, ctest registry_gaps).

The ctest `registry_gaps` runs `--check-gaps` on the committed registry, which always has zero
violations, so a regression that made `gap_violations` stop firing would pass silently. This
test drives the guard against synthetic registries to prove it still fires, still honors a
recorded `mapped_ok` reason, and that the committed registry is clean - i.e. it maps no reviewed
class without one. A reviewed class that the pinned backend starts to implement legitimately
leaves `registry/significant.json`'s unmapped set (DeepSeek-V4.1-Flash did, at pin 5e6388a5a),
which is why the committed-state check is about reasons, not about absence.
"""
import json
import os
import subprocess
import sys
import tempfile

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)
import registry_build as rb  # noqa: E402

SIG = {"DeepseekV41ForCausalLM": {"model_type": "deepseek_v41", "family": "DeepSeek V4.1"}}
MAPPED = {"architectures": {"DeepseekV41ForCausalLM": {"gguf": "x.gguf", "backends": ["hrx"]}}}

failures = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        failures.append(name)


# 1. Fires when a reviewed class is mapped without a reason.
bad = rb.gap_violations(MAPPED, SIG)
check("mapped-without-reason -> one violation", bad == [("DeepseekV41ForCausalLM", MAPPED["architectures"]["DeepseekV41ForCausalLM"])])

# 2. Honors a recorded reason.
ok_sig = {"DeepseekV41ForCausalLM": {**SIG["DeepseekV41ForCausalLM"], "mapped_ok": "upstream llama.cpp added real support"}}
check("mapped with a recorded mapped_ok -> no violation", rb.gap_violations(MAPPED, ok_sig) == [])

# 3. Vacuous reasons do not exempt.
for vacuous in ({"mapped_ok": "   "}, {"mapped_ok": ""}, {"mapped_ok": True}, {"mapped_ok": None}):
    v_sig = {"DeepseekV41ForCausalLM": {**SIG["DeepseekV41ForCausalLM"], **vacuous}}
    check(f"vacuous mapped_ok {vacuous!r} -> still a violation", len(rb.gap_violations(MAPPED, v_sig)) == 1)

def excused(entry):
    reason = entry.get("mapped_ok")
    return isinstance(reason, str) and bool(reason.strip())


# 4. The committed registry is clean: no violations, and every reviewed class it maps has a reason.
committed = json.load(open(os.path.join(rb.ROOT, "registry/architectures.json")))
check("committed registry -> no violations", rb.gap_violations(committed) == [])
real_sig = json.load(open(os.path.join(rb.ROOT, "registry/significant.json")))["classes"]
unexcused = [cls for cls in real_sig
             if cls in committed["architectures"] and not excused(real_sig[cls])]
check("every reviewed class the committed registry maps carries a mapped_ok reason", unexcused == [])

# 5. The CLI exits 1 and names the class against a synthetic registry - for a reviewed class that
# carries no reason, so the CLI reads the committed significant.json (a class with mapped_ok is
# deliberately exempt, which is what #4 checks).
plain = next((c for c, e in real_sig.items() if not excused(e)), None)
check("there is a reviewed class with no mapped_ok to drive the CLI case", plain is not None)
synth_mapped = {"architectures": {plain: {"gguf": "x.gguf", "backends": ["hrx"]}}}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(synth_mapped, f)
    synth = f.name
p = subprocess.run([sys.executable, os.path.join(rb.ROOT, "tools", "registry_build.py"),
                    "--check-gaps", "--out", synth], capture_output=True, text=True)
os.unlink(synth)
check(f"--check-gaps exits 1 and names {plain} on a synthetic mapped registry",
      p.returncode == 1 and plain in p.stdout)

if failures:
    print(f"{len(failures)} failure(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS")
