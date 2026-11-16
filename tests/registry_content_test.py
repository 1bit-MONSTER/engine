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
"""Pin the firing of ctest `registry_content`: `tools/registry_build.py --check` (engine#2698).

The registry has two axes a pin bump can move, and only one of them was checked. `--check-pins` (ctest
`registry_pins`, the row in CI) compares `sources["llama.cpp (hrx)"]` with the gitlink and never reads the
fork, so a hand-edited CONTENT was green in every PR job: measured on main 3e591b1, deleting one entry from
`gguf_architectures.hrx` -- the list CMakeLists.txt compiles into hrx_archs.h, which is what `1bit serve`
refuses architectures against -- left `--check-pins` at rc 0 while `--check` said "is stale".

So this drives the real tool against a fake fork and a fake gitlink in a throwaway repo, and asserts the
three things the new row has to be:

  1. a freshly generated registry passes `--check` (rc 0);
  2. a CONTENT edit that keeps `sources` -- the mutation every other row passes -- fails `--check` (rc 1)
     while `--check-pins` stays rc 0, which is the defect this row exists for;
  3. a checkout without the fork skips (rc 77) rather than passing, because the content cannot be known there.

A fork checked out at a commit other than the pin is arm 4: rc 1, because the content would come from a tree
the registry does not name.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), "tools", "registry_build.py")

CONSTANTS = '''MODEL_ARCH_NAMES = {
    MODEL_ARCH.TEST: "testarch",
}
'''

CONVERTER = '''class TestModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.TEST

    @ModelBase.register("TestForCausalLM")
    def __init__(self, *args):
        pass
'''

ARCH_CPP = '''static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_TEST, "testarch" },
};
'''

MODEL_CPP = '''llama_model * llama_model_mapping(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_TEST:
            break;
    }
}
'''


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def main():
    fails = []
    checks = []

    def check_that(cond, what):
        checks.append(what)
        if not cond:
            fails.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "engine")
        fork = os.path.join(repo, "third_party", "llama.cpp")
        os.makedirs(os.path.join(repo, "tools"))
        shutil.copy(TOOL, os.path.join(repo, "tools", "registry_build.py"))
        # the reviewed not-an-alias list the tool reads beside the check; empty here, the gaps row owns it
        write(os.path.join(repo, "registry", "significant.json"), json.dumps({"classes": {}}) + "\n")

        write(os.path.join(fork, "gguf-py", "gguf", "constants.py"), CONSTANTS)
        write(os.path.join(fork, "convert_hf_to_gguf.py"), CONVERTER)
        write(os.path.join(fork, "src", "llama-arch.cpp"), ARCH_CPP)
        write(os.path.join(fork, "src", "llama-model.cpp"), MODEL_CPP)
        git(fork, "init", "-q")
        git(fork, "add", "-A")
        git(fork, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "fork at the pin")
        pinned = git(fork, "rev-parse", "HEAD").stdout.strip()
        if not pinned:
            print("could not build the fixture fork", file=sys.stderr)
            return 1

        # the gitlink, staged only: pin() reads the index first, which is how a lane runs this before committing
        git(repo, "init", "-q")
        staged = git(repo, "update-index", "--add", "--cacheinfo", f"160000,{pinned},third_party/llama.cpp")
        if staged.returncode != 0:
            print(f"could not stage the gitlink: {staged.stderr}", file=sys.stderr)
            return 1

        tool = os.path.join(repo, "tools", "registry_build.py")
        registry = os.path.join(repo, "registry", "architectures.json")

        def run(*flags):
            return subprocess.run([sys.executable, tool, *flags], capture_output=True, text=True)

        wrote = run()
        check_that(wrote.returncode == 0 and os.path.exists(registry),
                   "the fixture registry generates (the arm is only meaningful on a real file)")
        check_that(pinned in open(registry).read(),
                   "the fixture registry records the indexed gitlink as its pin")

        arm1 = run("--check")
        check_that(arm1.returncode == 0, "arm 1: a freshly generated registry passes --check (rc 0)")

        # arm 2: the mutation. Remove the one architecture from the COMPILED list and keep `sources` as it is --
        # exactly what a hand edit looks like, and what every row but this one passed on main.
        before = json.load(open(registry))
        after = json.loads(json.dumps(before))
        dropped = after["gguf_architectures"]["hrx"].pop(0)
        write(registry, json.dumps(after, indent=1) + "\n")
        pins = run("--check-pins")
        content = run("--check")
        check_that(pins.returncode == 0,
                   f"arm 2: --check-pins still passes after dropping {dropped!r} from gguf_architectures.hrx "
                   "(the gap this row closes)")
        check_that(content.returncode == 1,
                   f"arm 2: --check FAILS on the same edit (rc {content.returncode})")
        check_that("stale" in content.stdout, "arm 2: --check names the generated file, not just a return code")

        # a per-architecture edit too: the other content axis, still green for --check-pins
        write(registry, json.dumps(before, indent=1) + "\n")
        after = json.loads(json.dumps(before))
        name = sorted(after["architectures"])[0]
        after["architectures"][name]["backends"] = []
        write(registry, json.dumps(after, indent=1) + "\n")
        check_that(run("--check-pins").returncode == 0,
                   "arm 2b: --check-pins passes after clearing an architecture's backends")
        check_that(run("--check").returncode == 1,
                   "arm 2b: --check fails on the per-architecture edit")

        # arm 3: no fork checked out -> 77 (skip), never 0
        write(registry, json.dumps(before, indent=1) + "\n")
        shutil.rmtree(os.path.join(fork, "src"))
        skipped = run("--check")
        check_that(skipped.returncode == 77,
                   f"arm 3: without the fork --check skips (rc {skipped.returncode}), it does not pass")
        check_that("not checked out" in skipped.stdout + skipped.stderr,
                   "arm 3: the skip says which checkout it needs")

        # arm 4: the fork is checked out, but not at the pin the registry names
        write(os.path.join(fork, "src", "llama-arch.cpp"), ARCH_CPP)
        write(os.path.join(fork, "src", "llama-model.cpp"), MODEL_CPP)
        # a real second commit (an added marker file), so HEAD is a different tree from the pinned one
        write(os.path.join(fork, "MARKER-LATER-COMMIT"), "later\n")
        git(fork, "add", "-A")
        moved_commit = git(fork, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm",
                           "a later fork commit")
        check_that(moved_commit.returncode == 0 and git(fork, "rev-parse", "HEAD").stdout.strip() != pinned,
                   "arm 4: the fixture fork really moved off the pin (an empty commit would prove nothing)")
        moved = run("--check")
        check_that(moved.returncode == 1,
                   f"arm 4: a fork at another commit than the pin fails (rc {moved.returncode})")
        check_that("before regenerating" in moved.stdout,
                   "arm 4: the failure says the content would come from the wrong tree")

    for c in checks:
        print(f"ok   {c}")
    for f in fails:
        print(f"FAIL {f}", file=sys.stderr)
    print(f"registry_content_test: {len(checks) - len(fails)}/{len(checks)} "
          f"{'passed' if not fails else 'FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
