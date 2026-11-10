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
"""Pin tools/registry_build.py's pin() reading the INDEX, not just HEAD (engine#1745 review).

Two states a lane actually hits, and one it must not answer silently:

  * a staged gitlink bump -- the index holds the NEW pin while HEAD still holds the old one. Reading HEAD is what made
    the pins check say the registry matched while the bump sat staged (engine#1738: ten minutes of CI to learn locally);
  * an UNMERGED index after a submodule conflict -- "git ls-files -s" then lists stages 1/2/3, and matching on the mode
    alone returns the merge BASE: a third pin nobody chose. That is the same staleness one state over, so this test pins
    the refusal rather than the value.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

A = "1" * 40
B = "2" * 40
Z = "0" * 40


def run(git, *args, stdin=None):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=git, input=stdin,
                          capture_output=True, text=True, check=False)


def load(script):
    spec = importlib.util.spec_from_file_location("registry_build_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if not shutil.which("git"):
        print("registry_pin_index: SKIP (no git)")
        return 0
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "registry_build.py")
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails += 1

    d = tempfile.mkdtemp(prefix="registry-pin-")
    try:
        os.makedirs(os.path.join(d, "tools"))
        # ROOT is the script's grandparent, so the copy makes this temp repo the tool's root.
        shutil.copy(src, os.path.join(d, "tools", "registry_build.py"))
        run(d, "init", "-q")
        path = "third_party/llama.cpp"

        # 1. a staged bump: HEAD has no entry at all, the index has the new pin
        run(d, "update-index", "--add", "--cacheinfo", "160000,%s,%s" % (A, path))
        mod = load(os.path.join(d, "tools", "registry_build.py"))
        check(mod.pin(path) == A, "a staged gitlink is read from the index (got %s)" % mod.pin(path))

        # 2. an unmerged index (stages 1/2/3 only, as a real conflict leaves it): base A, ours A, theirs B
        run(d, "update-index", "--force-remove", path)
        run(d, "update-index", "--index-info",
            stdin="160000 %s 1\t%s\n160000 %s 2\t%s\n160000 %s 3\t%s\n" % (A, path, A, path, B, path))
        try:
            got = mod.pin(path)
            raised = False
        except SystemExit as e:
            got, raised = str(e), True
        check(raised, "an unmerged gitlink is REFUSED rather than returning the merge base (got %r)" % (got,))
        check(raised and "unmerged" in got and "stages 1, 2, 3" in got,
              "and the refusal names the unmerged stages: %s" % (got,))

        # 3. stage 0 again: the ordinary resolved state reads one pin, no refusal
        run(d, "update-index", "--index-info", stdin="160000 %s 0\t%s\n" % (A, path))
        check(mod.pin(path) == A, "a resolved index reads one pin again (got %s)" % mod.pin(path))
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("registry_pin_index: %s" % ("ok" if not fails else "%d failure(s)" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
