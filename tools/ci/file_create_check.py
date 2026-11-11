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
"""Refuse a new fopen(..., "w"/"a") in the tree's C and C++ sources (CodeQL cpp/world-writable-file-creation).

fopen() creates with 0666 & ~umask -- world-writable under a permissive umask -- and follows a symlink planted at the
name. CodeQL reports each such site, and a PR that only touches nearby code inherits the alert as new: #1830 sat on a
CHANGES verdict for two dump sites it did not write. #2057 moved every write site onto npu/forward/include/file_create.h
(0600 dumps, at most 0644 for the head-image cache, O_NOFOLLOW), and this check keeps it that way: the next dump someone
adds will not know about the header, and CodeQL would otherwise find it a week later.

A line may keep fopen for writing when it says why, on the same line: `// file-create-ok: <reason>` (a /proc
pseudo-file, the Windows fallback, a test fixture). An empty reason does not count. Matches inside a // comment are
ignored, so prose that quotes the call is fine.

usage:
  tools/ci/file_create_check.py --check [--root DIR]
  tools/ci/file_create_check.py --self-test
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

PATTERN = re.compile(r'\bfopen\s*\([^;]*?,\s*"(?:w|a)[^"]*"')
MARKER = re.compile(r"file-create-ok:\s*\S")
EXTS = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp")


def offending(text):
    """[(line_no, line)] of the write-mode fopen calls in `text` that carry no reason."""
    out = []
    for no, line in enumerate(text.splitlines(), 1):
        m = PATTERN.search(line)
        if not m:
            continue
        comment = line.find("//")
        if comment != -1 and comment < m.start():
            continue   # prose that quotes the call
        if line.lstrip().startswith(("*", "/*")):
            continue
        if MARKER.search(line):
            continue
        out.append((no, line.strip()))
    return out


def tracked_sources(root):
    r = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True)
    return [p for p in r.stdout.decode().split("\0") if p.endswith(EXTS)]


def check(root):
    bad = []
    for rel in tracked_sources(root):
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        bad += [(rel, no, line) for no, line in offending(text)]
    for rel, no, line in bad:
        print(f"{rel}:{no}: fopen for writing creates 0666 and follows a symlink: {line}")
    if bad:
        print(f"file_create_check: {len(bad)} site(s). Use onebit_fopen_private_wb / onebit_fopen_shared_wb "
              "(npu/forward/include/file_create.h), or say why on the line: // file-create-ok: <reason>")
        return 1
    print("file_create_check: no fopen for writing without a stated reason")
    return 0


def self_test():
    fails = 0

    def t(ok, what):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + what)
        fails += 0 if ok else 1

    t(len(offending('FILE* f = fopen(path, "wb");')) == 1, "a plain write-mode fopen is refused")
    t(len(offending('if (std::FILE* f = std::fopen(path, "wb")) {')) == 1, "std::fopen inside an if is refused")
    t(len(offending('FILE* f = fopen(p, "a");')) == 1, "append mode is refused (it creates too)")
    t(len(offending('FILE* f = fopen(p, "w+");')) == 1, "w+ is refused")
    t(offending('FILE* f = fopen(p, "rb");') == [], "a read is fine")
    t(offending('FILE* f = fopen(p, "r+");') == [], "r+ does not create, so it is fine")
    t(offending('f = fopen("/proc/self/clear_refs", "w");  // file-create-ok: a /proc pseudo-file') == [],
      "a stated reason is accepted")
    t(len(offending('f = fopen(p, "w");  // file-create-ok:')) == 1, "an empty reason is not")
    t(offending('// fopen(path, "w") creates with 0666') == [], "prose in a // comment is ignored")
    t(offending(' * fopen(path, "w") in a block comment') == [], "prose in a block comment is ignored")
    t(offending('FILE* f = onebit_fopen_private_wb(path);') == [], "the openers themselves are fine")
    # End to end on a throwaway repository: one bad file fails the check, the fixed file passes it.
    with tempfile.TemporaryDirectory() as d:
        def git(*a):
            subprocess.run(["git", "-C", d, *a], check=True, capture_output=True)
        git("init", "-q")
        with open(os.path.join(d, "x.c"), "w") as f:
            f.write('void f(const char* p) { FILE* o = fopen(p, "wb"); }\n')
        git("add", "x.c")
        t(check(d) == 1, "--check fails on a tracked file with the pattern")
        with open(os.path.join(d, "x.c"), "w") as f:
            f.write('void f(const char* p) { FILE* o = onebit_fopen_private_wb(p); }\n')
        t(check(d) == 0, "--check passes once it uses the opener")
    print("file_create_check self-test:", "ok" if fails == 0 else f"{fails} failed")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.check:
        return check(os.path.abspath(a.root))
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
