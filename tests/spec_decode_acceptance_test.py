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
"""tools/spec_decode_acceptance.py -- its own self-test, and the fact that its refusals are in the tool.

The harness measures acceptance against a live llama-server, which CI cannot provide. What CI *can* check is the part that
is wrong silently: the accepted-prefix rule and the two units that sit next to each other in the JSON (tokens/round vs
accepted-drafts/round). Both are exercised here through the tool's `--selftest`, and the file also asserts that a run with
no servers fails loudly instead of printing a zero-acceptance "measurement".
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "spec_decode_acceptance.py")

fails = 0


def check(cond, what):
    global fails
    print(f"{'PASS' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


r = subprocess.run([sys.executable, TOOL, "--selftest"], capture_output=True, text=True)
check(r.returncode == 0, f"the tool's own selftest passes (rc={r.returncode})")
for want in ("a shared prefix of 2 is accepted", "a first-token miss accepts nothing",
             "a draft longer than the target's continuation is capped by the target",
             "tokens/round is emitted/rounds", "accepted drafts/round is a different number"):
    check(want in r.stdout, f"selftest covers: {want}")

# No server: the tool must fail, not report 0 acceptance as if it were a result.
p = os.path.join(ROOT, "tests", ".spec_decode_prompt.txt")
with open(p, "w") as fh:
    fh.write("hello\n")
try:
    r2 = subprocess.run([sys.executable, TOOL, "--target", "http://127.0.0.1:1", "--draft", "http://127.0.0.1:1",
                         "--prompt-file", p, "--tokens", "4", "--k", "2"], capture_output=True, text=True, timeout=120)
    check(r2.returncode == 3, f"an unreachable server exits 3 rather than printing a measurement (rc={r2.returncode})")
    check("did not answer" in r2.stderr, f"and says why: {r2.stderr.strip()[:90]}")
finally:
    os.unlink(p)

# A run with no arguments prints usage and a non-zero code, so `... --tokens 8` alone cannot look like success.
r3 = subprocess.run([sys.executable, TOOL], capture_output=True, text=True)
check(r3.returncode == 2, f"no arguments exits 2 and prints the usage (rc={r3.returncode})")

print(f"\n{'all checks passed' if not fails else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
