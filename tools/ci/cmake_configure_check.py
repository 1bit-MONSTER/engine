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
"""cmake_configure_check.py: the tree must CONFIGURE, not just parse as text.

engine#2445 is why this exists. Resolving a rebase conflict in `CMakeLists.txt` left one line ending
`…_test.py)ms)`, and every check in `pre-pr-checks` stayed green: they read files, run Python, and diff
refs -- none of them asks CMake to read `CMakeLists.txt` the way CI does. The CI `Configure` step caught
it in 63 seconds with

    CMake Error at CMakeLists.txt:474:
      Parse error.  Expected a newline, got identifier with text "ms".

and a `grep` for the row name could not see it either: the line STARTS correctly, which is the shape a
line-oriented check is blind to.

So this runs the configure and nothing else. It is not a build: no compiler is invoked, no target runs,
and the cost is a few seconds on a warm box. The output directory defaults to a directory under
`~/.cache` (never `/tmp`, which is tmpfs on this fleet) and is removed afterwards.

    cmake_configure_check.py --check              # configure this tree
    cmake_configure_check.py --check --root /path # configure another checkout
    cmake_configure_check.py --self-test          # the arms, no cmake needed

Exit: 0 the tree configures, 1 it does not (the first CMake error is printed), 2 cannot tell
(cmake missing, no CMakeLists.txt, or the configure timed out -- never treated as a pass).
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

# One line is enough to name the failure: CMake puts "CMake Error at <file>:<line>:" followed by the text.
ERROR_AT = re.compile(r"^CMake Error at ([^:]+):(\d+):", re.M)


def scratch_parent():
    """`~/.cache`, created if missing.

    CI's HOME (`/github/home`) has no `.cache` on a fresh runner, and `tempfile.mkdtemp(dir=…)` raises
    FileNotFoundError rather than creating it -- which is how this file's own self-test went red on its
    first CI run (measured: `FileNotFoundError: … /github/home/.cache/cfgres-…`).
    """
    base = pathlib.Path(os.path.expanduser("~")) / ".cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def configure(root, build_dir, timeout=300):
    """(rc, output). rc 2 is this tool's 'cannot tell', distinct from CMake's own failure."""
    if shutil.which("cmake") is None:
        return 2, "cannot tell: cmake is not on PATH"
    if not (root / "CMakeLists.txt").is_file():
        return 2, "cannot tell: no CMakeLists.txt under %s" % root
    cmd = ["cmake", "-S", str(root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A configure that never finishes is not a pass; say so rather than defaulting to green.
        return 2, "cannot tell: cmake configure did not finish within %ds" % timeout
    return (0 if p.returncode == 0 else 1), (p.stdout or "") + (p.stderr or "")


def inputs_digest(root):
    """A digest of the files whose content can change whether this tree configures.

    Cheap and sufficient: `CMakeLists.txt` and every `*.cmake` beside it. A configure is 25 s on this
    tree (measured), which is too slow to run on every keystroke of an unrelated change and fast enough
    to run whenever one of these files is touched -- so the answer is cached against exactly them.
    """
    h = hashlib.sha256()
    for name in ["CMakeLists.txt"] + sorted(p.name for p in root.glob("*.cmake")):
        path = root / name
        if path.is_file():
            h.update(name.encode())
            h.update(path.read_bytes())
    cmake_dir = root / "cmake"
    if cmake_dir.is_dir():
        for path in sorted(cmake_dir.rglob("*.cmake")):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def check(root, build_dir=None):
    tmp = None
    # NOTE: capture this BEFORE the temp dir replaces it -- the first version of the cache asked
    # `build_dir is None` after the substitution and so never hit (measured: a second run still took 17 s).
    may_cache = build_dir is None
    if build_dir is None:
        tmp = tempfile.mkdtemp(prefix="pre-pr-configure-", dir=str(scratch_parent()))
        build_dir = pathlib.Path(tmp)
    cache = None
    if may_cache:
        cache = pathlib.Path(os.path.expanduser("~")) / ".cache" / "pre-pr-configure" / (inputs_digest(root) + ".json")
        try:
            hit = json.loads(cache.read_text())
        except (OSError, ValueError):
            hit = None
        if hit is not None and hit.get("rc") == 0:
            print("cmake_configure_check: the tree configures (cached, %s unchanged since %s)"
                  % (root, hit.get("when", "an earlier run")))
            return 0
    try:
        rc, out = configure(root, build_dir)
        if rc == 0:
            if cache is not None:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"rc": 0, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
                except OSError:
                    pass  # a cache that cannot be written is not a failed check
            print("cmake_configure_check: the tree configures (%s)" % root)
            return 0
        if rc == 2:
            print("cmake_configure_check: %s" % out, file=sys.stderr)
            return 2
        # Name the site, not just the fact: the line is what a reader has to go and look at.
        hits = ERROR_AT.findall(out)
        if hits:
            for path, line in hits[:5]:
                print("cmake_configure_check: CMake error at %s:%s" % (path, line), file=sys.stderr)
        else:
            print("cmake_configure_check: cmake exited non-zero with no 'CMake Error at' line", file=sys.stderr)
        detail = [ln for ln in out.splitlines() if "Error" in ln or "error" in ln][:6]
        for ln in detail:
            print("  " + ln, file=sys.stderr)
        return 1
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def self_test():
    """The arms, on fixture trees: a parse error must be red, a good file green, and no file 'cannot tell'."""
    fails = 0

    def chk(label, got, want, detail=""):
        nonlocal fails
        if got == want:
            print("ok   " + label)
        else:
            print("FAIL %s (rc %s, wanted %s)%s" % (label, got, want, (" [" + detail.strip()[:160] + "]") if detail else ""))
            fails += 1

    def root_with(body):
        d = pathlib.Path(tempfile.mkdtemp(prefix="cfgres-", dir=str(scratch_parent())))
        (d / "CMakeLists.txt").write_text(body)
        return d

    if shutil.which("cmake") is None:
        print("cmake_configure_check: cmake is not on PATH, so the arms cannot run (rc 2)")
        return 2

    # 1. THE ARM THAT MATTERS: the exact breakage engine#2445 shipped. A text check cannot see it,
    #    because the line starts correctly and only the tail is wrong.
    broken = "cmake_minimum_required(VERSION 3.20)\nproject(x NONE)\nadd_test(NAME x COMMAND true)ms)\n"
    d = root_with(broken)
    rc, out = configure(d, d / "build")
    chk("a line with a stray tail is refused (the engine#2445 shape)", rc, 1, out)
    chk("  ... and the error names the file and line", bool(ERROR_AT.search(out)), True, out)
    shutil.rmtree(d, ignore_errors=True)

    # 2. the same file without the tail configures: the arm is about the tail, not about cmake hating us.
    d = root_with("cmake_minimum_required(VERSION 3.20)\nproject(x NONE)\n")
    rc, out = configure(d, d / "build")
    chk("a valid CMakeLists.txt configures", rc, 0, out)
    shutil.rmtree(d, ignore_errors=True)

    # 3. absence is not a pass: no CMakeLists.txt is rc 2, and the caller says so rather than green.
    d = pathlib.Path(tempfile.mkdtemp(prefix="cfgres-", dir=str(pathlib.Path(os.path.expanduser("~")) / ".cache")))
    rc, out = configure(d, d / "build")
    chk("no CMakeLists.txt is 'cannot tell' (2), never 0", rc, 2, out)
    shutil.rmtree(d, ignore_errors=True)

    # 4. the cache must answer only for the SAME inputs: same digest -> hit, different digest -> miss.
    #    The first version of the cache asked `build_dir is None` after the temp dir had replaced it, so it
    #    never hit and every run paid the 25 s configure; this arm is what would have caught that.
    d = root_with("cmake_minimum_required(VERSION 3.20)\nproject(x NONE)\n")
    before = inputs_digest(d)
    (d / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\nproject(x NONE)\n# one more line\n")
    chk("a changed CMakeLists.txt changes the digest (so the cache cannot serve a stale pass)", before != inputs_digest(d), True)
    shutil.rmtree(d, ignore_errors=True)

    # 5. the arm for the CI failure that this file's first run produced: a HOME with no ~/.cache at all.
    #    `tempfile.mkdtemp(dir=~/.cache)` raises FileNotFoundError there rather than creating it, so the
    #    self-test has to create the parent -- and an arm that points HOME at an empty directory is the
    #    only way to keep that true on a box whose ~/.cache already exists.
    saved = os.environ.get("HOME")
    fresh = tempfile.mkdtemp(prefix="home-", dir=str(scratch_parent()))
    try:
        os.environ["HOME"] = fresh
        chk("a HOME with no ~/.cache still yields a scratch dir", scratch_parent().is_dir(), True)
        d = pathlib.Path(tempfile.mkdtemp(prefix="cfgres-", dir=str(scratch_parent())))
        (d / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\nproject(x NONE)\n")
        rc, out = configure(d, d / "build")
        chk("  ... and an arm can configure from it", rc, 0, out)
        shutil.rmtree(d, ignore_errors=True)
    finally:
        os.environ["HOME"] = saved
        shutil.rmtree(fresh, ignore_errors=True)

    print("cmake_configure_check self-test: %s" % ("ok" if fails == 0 else "%d failure(s)" % fails))
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".", help="tree to configure (default: the current directory)")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.check:
        ap.error("one of --check or --self-test is required")
    return check(pathlib.Path(a.root).resolve())


if __name__ == "__main__":
    sys.exit(main())
