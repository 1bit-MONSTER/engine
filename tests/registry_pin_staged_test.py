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
"""Pin the staged-pin behavior of `tools/registry_build.py --check-pins` (ctest registry_pins_staged).

ctest `registry_pins` runs on a clean checkout, where the index and HEAD agree, so it cannot tell the two apart.
A lane bumping the pin stages the gitlink and runs the check *before* committing; reading only HEAD reported the
pin being replaced and passed a registry whose `sources` pin was stale, so the failure surfaced only after the
commit, as ctest `registry_pins` on a ten-minute CI run (engine#1738).

So this drives the real tool in a throwaway repo with a fake gitlink: committed-and-staged agreeing must pass,
and a bump staged under a stale `sources` pin must fail with the two pins named.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), "tools", "registry_build.py")
OLD = "8f124f906c67bd74f0f0a712836ef997938e92e0"
NEW = "60ae4c935893564267076fa7d84f17c5552cef46"


def main():
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo, "tools"))
        os.makedirs(os.path.join(repo, "registry"))
        shutil.copy(TOOL, os.path.join(repo, "tools", "registry_build.py"))
        with open(os.path.join(repo, "registry", "architectures.json"), "w") as f:
            json.dump({"sources": {"llama.cpp (hrx)": NEW}}, f)

        def git(*args):
            return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)

        git("init", "-q")
        git("update-index", "--add", "--cacheinfo", f"160000,{NEW},third_party/llama.cpp")
        git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "pin")
        if git("rev-parse", "HEAD").returncode != 0:
            print("could not build the fixture repository", file=sys.stderr)
            return 1

        def check_pins():
            return subprocess.run([sys.executable, os.path.join(repo, "tools", "registry_build.py"),
                                   "--check-pins"], capture_output=True, text=True)

        # The pin the registry records is the committed one: nothing to report.
        out = check_pins()
        if out.returncode != 0:
            fails.append("a registry that records the committed pin was refused: " + out.stdout.strip())

        # The lane stages the bump and runs the check before committing: the registry still records the old pin.
        git("update-index", "--cacheinfo", f"160000,{OLD},third_party/llama.cpp")
        out = check_pins()
        if out.returncode == 0:
            fails.append("a pin staged under a stale `sources` entry passed --check-pins (the pre-commit miss)")
        elif OLD[:9] not in out.stdout or NEW[:9] not in out.stdout:
            fails.append("the refusal does not name both pins: " + out.stdout.strip())

    for f in fails:
        print("FAIL " + f)
    if fails:
        return 1
    print("staged pin bumps are checked before the commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
