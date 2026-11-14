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
"""Run the add-on f16 audit against the pinned gpu-kernels tree -- and say so loudly when it cannot.

engine#2416: `tools/ci/hip_f16_audit.py` existed with a pin gate (engine#2411) and a self-test, and **nothing invoked it**:
`grep -rn hip_f16_audit` found one `add_python_test(... --self-test)` and no audit. The instruction engine#2406 put on the ratchet's
green line -- *"audit those on a pin bump"* -- stayed a thing a person had to remember, which is how engine#2409 read *"0 narrowing(s)"*
on files that had six casts.

The audit needs a gpu-kernels checkout and the engine checks out none of its own (submodules are hrx-system, llama.cpp, xdna-driver),
so a runner has nothing to point `--dir` at. This wrapper is the invoker, in the same shape `tools/ci/hrx_f16_narrowing.py` already
uses: find a tree, audit it, and when there is none **exit 77 with a message that names the consequence**. 77 is CTest's skip code,
wired with `SKIP_RETURN_CODE` in the row that calls this.

Two properties, both from the issue, both deliberately louder than a quiet pass:

* the skip says **the audit did NOT run**, not "ok" -- a silent skip is the whole failure mode;
* when a tree *is* present the pin gate applies (#2411), so it cannot audit a commit that is not the one `config/gpu-kernels.config`
  names; `--expect-sha` is for a candidate pin (`#2413`).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = 77


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="engine checkout (default: two levels above this file)")
    ap.add_argument("--dir", default=None, help="gpu-kernels checkout (default: $GPU_KERNELS_DIR, then third_party/gpu-kernels)")
    ap.add_argument("--expect-sha", default=None, help="audit THIS commit instead of the pinned one (a candidate pin)")
    args = ap.parse_args()

    engine_root = args.root or os.path.dirname(os.path.dirname(HERE))
    kernels = args.dir or os.environ.get("GPU_KERNELS_DIR") or os.path.join(engine_root, "third_party", "gpu-kernels")
    if not os.path.isdir(os.path.join(kernels, "addons", "hip-hrx", "kernels")):
        print("SKIPPED: the add-on f16 audit did NOT run -- no gpu-kernels checkout at %s" % kernels)
        print("         set GPU_KERNELS_DIR (or clone the pinned tree into third_party/gpu-kernels);")
        print("         the instruction this enforces is engine#2406's 'audit those on a pin bump'.")
        return SKIP

    cmd = [sys.executable, os.path.join(HERE, "hip_f16_audit.py"), "--dir", kernels, "--config-root", engine_root]
    if args.expect_sha:
        cmd += ["--expect-sha", args.expect_sha]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
