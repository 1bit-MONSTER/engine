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
"""tools/oga_pack_probe.sh's ticket guard, tested without a device (engine#1371 / #1611).

The guard exists because gate 2 opens the NPU, and a probe that runs unticketed is the shape of every collision this fleet has
had. Its first two versions were both wrong in ways a test would have caught:

  * matching each ancestor's argv[0] against `npu-hold|gpu-hold` can never match -- the helpers are bash scripts, so argv[0] is
    the interpreter and the script path is argv[1];
  * matching the whole command line instead let a shell whose *text* mentioned npu-hold (a patch script, an editor, a grep) pass
    the guard on its own search string.

The version that works reads the holder's own statement from the token file (`gpu-hold <label> pid <PID> <date>`) and asks
whether an ancestor of the probe is that pid. That is testable here: a fake token file naming the pid of the shell that runs the
probe is exactly the positive case, and a token file naming nobody is the negative one.

Arms:
  1. no token file at all            -> warning, gate 2 skipped, exit 3 (gate 1 only), no device touched
  2. REQUIRE_TICKET=1, no ticket     -> refusal, exit 2, and it never even imports
  3. token naming an ancestor's pid  -> ticket detected, no warning, gate 2 attempted (exit 4 with a stub binary, not 3)
  4. basename "."                    -> refused, exit 2, before any `rm -rf`
  5. no binary on PATH, BIN unset    -> refused by name, exit 2 (was: rc 127 from the shell, reported as "the pack stops at gate 1")
  6. a GPU ticket and no NPU token   -> NOT a ticket: gate 2 is NPU work and runs only on the NPU token
"""
import os
import shutil
import subprocess
import sys
import tempfile

BASH = shutil.which("bash") or "/bin/bash"   # arm 5 empties PATH, so the interpreter has to be named absolutely
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "oga_pack_probe.sh")

fails = 0


def check(cond, what):
    global fails
    print(f"{'PASS' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def write_stub():
    """A stub 1bit: it 'imports' (creates $OUT/model.q4nx) and is not a server, so gate 1 passes and gate 2 fails at health."""
    path = TMP + "/stub-1bit"
    with open(path, "w") as fh:
        fh.write('#!/bin/bash\n'
                 'if [ "${1:-}" = npu-pack ]; then\n'
                 '  out=""; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && { out=$2; shift; }; shift; done\n'
                 '  [ -n "$out" ] && { mkdir -p "$out"; : > "$out/model.q4nx"; }\n'
                 '  exit 0\n'
                 'fi\n'
                 'exit 1\n')
    os.chmod(path, 0o755)
    return path


def run(env_extra, pack=".", stub_bin=None, timeout=120):
    """Run the probe with a stub binary: gate 1 'succeeds' only if the stub makes it so, so what is asserted is the guard."""
    if stub_bin is None:
        stub_bin = STUB
    env = dict(os.environ)
    env.update({"BIN": stub_bin, "SCRATCH": TMP + "/scratch", "TOKEN_FILES": TMP + "/no-such-token"})
    env.update(env_extra)
    return subprocess.run(["bash", TOOL, pack, "19959"], capture_output=True, text=True, timeout=timeout, env=env)


TMP = tempfile.mkdtemp(prefix="oga-probe-test.")
STUB = write_stub()
os.makedirs(TMP + "/scratch", exist_ok=True)
pack = os.path.join(TMP, "Qwen3-4B_rai_1.9.0_npu")
os.makedirs(pack, exist_ok=True)

# 1. no token: gate 1 only, exit 3, warning names the correct invocation
r = run({}, pack=pack)
check(r.returncode == 3, f"no ticket -> gate 1 only, exit 3 (rc={r.returncode})")
check("GATE 2 WILL NOT RUN" in r.stderr, "and the warning says so")
check("npu-hold -t 4" in r.stderr, "and names the invocation that would run both gates")
check("gate 2: forward init" not in r.stdout, "gate 2 was not attempted")

# 2. REQUIRE_TICKET=1: refusal before the import
r = run({"REQUIRE_TICKET": "1"}, pack=pack)
check(r.returncode == 2, f"REQUIRE_TICKET=1 + no ticket -> exit 2 (rc={r.returncode})")
check("gate 1: import" not in r.stdout, "and it refuses before importing anything")

# 3. a token naming the pid of the shell that runs the probe: detected (the positive case that the argv-matching versions missed)
tok = TMP + "/NPU-TOKEN"
script = (
    'printf "gpu-hold test-probe pid %d 2026-11-09T00:00:00\\n" $$ > "$1"\n'
    'TOKEN_FILES="$1" BIN="$5" SCRATCH="$2" bash "$3" "$4" 19958; echo "rc=$?"\n'
)
r = subprocess.run(["bash", "-c", script, "x", tok, TMP + "/scratch", TOOL, pack, STUB],
                   capture_output=True, text=True, timeout=120)
check("GATE 2 WILL NOT RUN" not in r.stderr, f"a token naming an ancestor's pid is detected as a ticket (stderr={r.stderr.strip()[:80]})")
check("gate 2: forward init" in r.stdout, "and gate 2 is attempted (exit 4 with the stub, not the gate-1 exit 3)")
check("rc=4" in r.stdout, f"with the stub failing at gate 2, rc=4: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'no output'}")

# 4. a pack path whose basename is "." must be refused, never `rm -rf`'d
r = subprocess.run(["bash", TOOL, TMP + "/.", "19957"], capture_output=True, text=True, timeout=120,
                   env={**os.environ, "BIN": "/bin/true", "SCRATCH": TMP + "/scratch", "TOKEN_FILES": TMP + "/no-such-token"})
check(r.returncode == 2, f"a pack path whose basename is '.' is refused (rc={r.returncode})")
check("refusing a pack path" in r.stderr, "with the reason, before any rm -rf")

# 5. no binary anywhere: the verdict must name the missing binary, not the pack (the shell's rc 127 used to be read as gate 1)
r = subprocess.run([BASH, TOOL, pack, "19956"], capture_output=True, text=True, timeout=120,
                   env={**os.environ, "BIN": "", "ONEBIT_BIN": "", "PATH": TMP + "/empty-bin",
                        "SCRATCH": TMP + "/scratch", "TOKEN_FILES": TMP + "/no-such-token"})
check(r.returncode == 2, f"no 1bit binary -> exit 2 (rc={r.returncode})")
check("no 1bit binary" in r.stderr, "and it says so by name")
check("gate 1" not in r.stdout and "no container" not in r.stdout,
      "and it never reaches gate 1, so the pack is not blamed for the tool")

# 6. a GPU ticket is not an NPU ticket: gate 2 opens the NPU, so a GPU-only hold must not authorise it
home = TMP + "/home-gpu-only"
os.makedirs(home, exist_ok=True)
gpu_tok = home + "/GPU-TOKEN"
script = (
    'printf "gpu-hold test-gpu pid %d 2026-11-09T00:00:00\\n" $$ > "$1"\n'
    'HOME="$2" BIN="$5" SCRATCH="$3" bash "$4" "$6" 19955; echo "rc=$?"\n'
)
r = subprocess.run(["bash", "-c", script, "x", gpu_tok, home, TMP + "/scratch", TOOL, STUB, pack],
                   capture_output=True, text=True, timeout=120)
check("GATE 2 WILL NOT RUN" in r.stderr, "a GPU token alone is NOT a ticket for the NPU gate")
check("rc=3" in r.stdout, "so only gate 1 runs (rc=3), and no NPU work happens under a GPU ticket")

import shutil
shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'all checks passed' if not fails else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
