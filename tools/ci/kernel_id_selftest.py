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
"""kernel_id_selftest.py — run gpu-kernels' own kernel-id self-test from the engine's pre-PR run.

Why this exists. gpu-kernels has no CI (gpu-kernels#102 says so in its README), so
`tools/check-kernel-ids.py --self-test` — the four fixtures that pin the id reader, wired into
`tools/check.sh` by gpu-kernels#99 — runs when a person remembers to run it. An arm that never
runs is not an arm, and the engine is the repo that does have CI, so it runs it here.

The tree is found exactly as tools/ci/hip_f16_audit.py finds it ($GPU_KERNELS_DIR, else
third_party/gpu-kernels) — imported from there rather than re-derived, so the two checks cannot
disagree about which tree is the pinned one. No tree, no check: SKIPPED (77), the ctest
convention, so a plain engine checkout without the submodule is not a failure.

Usage:
    tools/ci/kernel_id_selftest.py --check --find <engine-root>
    tools/ci/kernel_id_selftest.py --check --dir <gpu-kernels-tree>
    tools/ci/kernel_id_selftest.py --self-test

Exit: 0 the gpu-kernels self-test passed; 1 it failed; 77 no gpu-kernels tree or the tree has no
check-kernel-ids.py; 2 usage.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hip_f16_audit import SKIPPED, find_tree  # noqa: E402  (same directory, same discovery)

TOOL = "tools/check-kernel-ids.py"


def self_test_of(tree: str) -> tuple[int, str]:
    """(rc, output) of the tree's own --self-test. CPU only: fixtures, no device, no llvm-objdump."""
    tool = Path(tree) / TOOL
    if not tool.is_file():
        return SKIPPED, f"{TOOL} is not in {tree} (not a gpu-kernels checkout?)"
    p = subprocess.run([sys.executable, str(tool), "--self-test"], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def check(tree: str | None) -> int:
    if tree is None:
        print("SKIPPED: kernel_id_selftest found no gpu-kernels tree ($GPU_KERNELS_DIR unset, no "
              "third_party/gpu-kernels) - gpu-kernels' kernel-id self-test did NOT run; set GPU_KERNELS_DIR "
              "to a gpu-kernels checkout at the pin in config/gpu-kernels.config to run it")
        return SKIPPED
    rc, out = self_test_of(tree)
    if rc == SKIPPED:
        print(f"SKIPPED: {out}")
        return SKIPPED
    if rc != 0:
        print(f"kernel_id_selftest: {tree}/{TOOL} --self-test failed (rc {rc}):")
        for line in out.splitlines()[-20:]:
            print(f"  {line}")
        return 1
    print(f"ok   kernel_id_selftest: {tree}/{TOOL} --self-test ({out.splitlines()[-1] if out else 'no output'})")
    return 0


def _fake_tree(root: Path, body: str | None) -> Path:
    """A tree that looks like gpu-kernels for this check: tools/check-kernel-ids.py, or not."""
    t = root / "gpu-kernels"
    (t / "tools").mkdir(parents=True, exist_ok=True)
    if body is not None:
        (t / "tools" / "check-kernel-ids.py").write_text("#!/usr/bin/env python3\n" + body)
    return t


def self_test() -> int:
    bad = 0

    def chk(cond: bool, msg: str) -> None:
        nonlocal bad
        if not cond:
            bad += 1
        print(("ok   " if cond else "FAIL ") + msg)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ok_tree = _fake_tree(root / "ok", 'print("self-test: 4 case(s), 0 failure(s)")\nraise SystemExit(0)\n')
        bad_tree = _fake_tree(root / "bad", 'print("FAIL expect rc 2, got 0")\nraise SystemExit(1)\n')
        bare_tree = _fake_tree(root / "bare", None)

        chk(check(str(ok_tree)) == 0, "a tree whose self-test passes: rc 0")
        chk(check(str(bad_tree)) == 1, "a tree whose self-test fails: rc 1 (the failure is the check's)")
        chk(check(str(bare_tree)) == SKIPPED, "a tree without check-kernel-ids.py: SKIPPED, not a failure")
        chk(check(None) == SKIPPED, "no tree at all: SKIPPED, not a failure")

        # the discovery is hip_f16_audit's, not a second opinion
        chk(find_tree(str(root / "ok"), env={"GPU_KERNELS_DIR": str(ok_tree)}) == str(ok_tree),
            "find_tree honours $GPU_KERNELS_DIR")
        sub = root / "sub"; (sub / "third_party").mkdir(parents=True)
        sub_tree = _fake_tree(sub / "third_party", None)
        chk(find_tree(str(sub), env={}) == str(sub_tree), "find_tree falls back to third_party/gpu-kernels")
        chk(find_tree(str(root / "nowhere"), env={}) is None, "find_tree: no tree is None, not a guess")

    print("kernel_id_selftest --self-test:", "ok" if bad == 0 else f"{bad} failure(s)")
    return 0 if bad == 0 else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="run the gpu-kernels kernel-id self-test")
    ap.add_argument("--find", help="the engine root to search for a gpu-kernels tree")
    ap.add_argument("--dir", help="a gpu-kernels checkout (must contain tools/check-kernel-ids.py)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        print("kernel_id_selftest: one of --check or --self-test is required", file=sys.stderr)
        return 2
    tree = a.dir or (find_tree(a.find) if a.find else None)
    return check(tree)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
