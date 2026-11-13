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
# Copyright 2026 The 1bit-MONSTER Authors.
"""Balance the braces of every C++ source CI cannot compile.

`CI` builds with `ONEBIT_NPU=OFF`, and a source added to a target only inside
`if(ONEBIT_NPU)` is therefore never compiled by it: `app/forward_serve.cpp`
appears exactly once in `CMakeLists.txt`, at `target_sources(onebit PRIVATE
app/forward_serve.cpp)` inside that block, and the CI build log of main contains
zero `Building CXX object .../forward_serve.cpp.o` lines while it does compile
`app/serve.cpp`, `app/main.cpp` and `app/embed_serve.cpp`.

That is intended (the lane needs the private NPU runtime), but it means a
**syntax** error in those files is invisible until someone builds on a device.
engine#2330 is exactly that: `app/forward_serve.cpp` was one `}` short after
#2327 and main did not compile for the lane until the fix-forward; every commit
in the window still reported `build: success`, because CI was right about the
configuration it builds.

A brace imbalance needs no headers, no runtime and no link to detect, so this
check closes that class cheaply. It reports; it does not judge a file it cannot
read (rc 2).

**Why not `npu_decl_check.strip_code`:** measured on this tree, 44 of the 54 raw
string literals under `app/` and `npu/` contain a `"`, and `strip_code` treats
`R"(a"b)"` as a string ending at the inner quote - `app/serve.cpp` then looks
like 2 opens and 0 closes. A raw string runs to `)delim"`, so the lexer below
tracks its delimiter.

A file whose `#if` conditions this cannot resolve is **named as not checked**, never
guessed at: a wrong guess here would either pass a broken file or fail a good one.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Every tree that holds a source added to a target inside `if(ONEBIT_NPU)`: measured from CMakeLists,
# 124 such sources live outside app/ and npu/ (moe/, laya/, tools/, tests/).
ROOTS = ("app", "npu", "moe", "laya", "tools", "tests")
EXTS = (".cpp", ".cc", ".c", ".h", ".hpp")


LANE_DEFINES = ("ONEBIT_NPU", "ONEBIT_MOE", "ONEBIT_LAYA", "ONEBIT_HRX_SERVER", "__linux__")

COND = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")


def _truth(expr: str, defines):
    """Return True/False for a simple `#if` expression, or None when unresolved."""
    e = expr.strip()
    while e.startswith("(") and e.endswith(")"):
        e = e[1:-1].strip()
    if e in ("0", "1"):
        return e == "1"
    m = re.fullmatch(r"(!?)\s*defined\s*\(?\s*([A-Za-z_]\w*)\s*\)?", e)
    if m:
        inside = m.group(2) in defines
        return (not inside) if m.group(1) else inside
    parts = re.split(r"\s*(&&|\|\|)\s*", e)
    if len(parts) == 1:
        return None
    vals = []
    for i, part in enumerate(parts):
        if i % 2:
            vals.append(part)
            continue
        v = _truth(part, defines)
        if v is None:
            return None
        vals.append(v)
    out = vals[0]
    for i in range(1, len(vals), 2):
        out = (out and vals[i + 1]) if vals[i] == "&&" else (out or vals[i + 1])
    return out


def active_lines(text: str, defines):
    """Blank inactive preprocessor branches; return (text, unresolved line or None)."""
    out, stack, active = [], [], True
    for lineno, line in enumerate(text.split("\n"), 1):
        m = COND.match(line)
        if not m:
            out.append(line if active else " " * len(line))
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("ifdef", "ifndef"):
            val = (rest in defines)
            if kind == "ifndef":
                val = not val
        else:
            val = _truth(rest, defines) if kind in ("if", "elif") else True
            if val is None:
                return None, lineno
        if kind in ("ifdef", "ifndef", "if"):
            stack.append((active, val))
            active = active and val
        elif kind == "elif":
            parent, taken = stack[-1]
            if taken:
                active = False
            else:
                active = parent and bool(val)
                stack[-1] = (parent, bool(val))
        elif kind == "else":
            parent, taken = stack[-1]
            active = parent and not taken
            stack[-1] = (parent, True)
        else:  # endif
            active = stack.pop()[0] if stack else True
        out.append(" " * len(line))
    return "\n".join(out), None


def tokens(text: str):
    """Yield (offset, char) for code characters only, tracking line numbers."""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif text.startswith('R"', i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            # R"delim( ... )delim"
            k = text.find("(", i + 2)
            if k < 0:
                i += 2
                continue
            delim = text[i + 2:k]
            close = ")" + delim + '"'
            j = text.find(close, k + 1)
            i = n if j < 0 else j + len(close)
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            i = min(j + 1, n)
        else:
            yield i, c
            i += 1


def imbalance(text: str):
    """Return (opens, closes, first-unmatched-open-line) after lexing."""
    depth = 0
    stack = []
    opens = closes = 0
    for off, ch in tokens(text):
        if ch == "{":
            opens += 1
            stack.append(off)
            depth += 1
        elif ch == "}":
            closes += 1
            depth -= 1
            if stack:
                stack.pop()
    line = text.count("\n", 0, stack[-1]) + 1 if stack else None
    return opens, closes, line


def check(root: str):
    bad, unreadable, checked, skipped = [], [], 0, []
    for top in ROOTS:
        for dirpath, _, files in os.walk(os.path.join(root, top)):
            for name in sorted(files):
                if not name.endswith(EXTS):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    text = open(path, encoding="utf-8", errors="replace").read()
                except OSError as exc:
                    unreadable.append((path, str(exc)))
                    continue
                filtered, unknown = active_lines(text, LANE_DEFINES)
                if filtered is None:
                    skipped.append((os.path.relpath(path, root), unknown))
                    continue
                checked += 1
                opens, closes, line = imbalance(filtered)
                if opens != closes:
                    bad.append((os.path.relpath(path, root), opens, closes, line))
    return bad, unreadable, checked, skipped


def self_test() -> int:
    arms = [
        ("balanced", "int f(){ if(x){ return 1; } return 0; }", True),
        ("one short", "int f(){ if(x){ return 1; }", False),
        ("brace in a string does not count", 'int f(){ const char* s = "{"; }', True),
        ("brace in a comment does not count", "int f(){ } // one more }", True),
        ("a raw string with a quote does not count", 'int f(){ auto s = R"(a"b{c)d)"; }', True),
        ("a delimited raw string does not count", 'int f(){ auto s = R"j(}{(})j"; }', True),
        ("char literal brace does not count", "int f(){ char c = '{'; }", True),
        ("an inactive #if 0 branch is skipped", "#if 0\nint broken(){\n#endif\nint f(){ return 0; }", True),
        ("the ONEBIT_NPU branch is the one counted", "#ifdef ONEBIT_NPU\nint f(){\n#endif\nint g(){ return 0; }", False),
    ]
    failed = 0
    for name, src, ok in arms:
        filtered, unknown = active_lines(src, LANE_DEFINES)
        if filtered is None:
            print(f"  FAIL {name}: unexpectedly unresolved at line {unknown}")
            failed += 1
            continue
        opens, closes, _ = imbalance(filtered)
        got = opens == closes
        if got != ok:
            print(f"  FAIL {name}: expected balanced={ok}, got {got} ({opens}/{closes})")
            failed += 1
        else:
            print(f"  ok   {name}")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    ap.add_argument("--self-test", action="store_true", help="check the lexer's arms and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    bad, unreadable, checked, skipped = check(args.root)
    for path, err in unreadable:
        print(f"lane_syntax: cannot read {path}: {err}")
    if bad:
        for path, opens, closes, line in bad:
            where = f" first unclosed '{{' at line {line}" if line else ""
            print(f"lane_syntax: {path}: {opens} '{{' vs {closes} '}}'{where}")
        print(f"lane_syntax: {len(bad)} of {checked} file(s) are brace-imbalanced")
        return 1
    for path, line in skipped:
        print(f"lane_syntax: not checked {path}: unresolved #if condition at line {line}")
    print(f"lane_syntax: {checked} file(s) balanced, {len(skipped)} not checked")
    return 2 if unreadable else 0


if __name__ == "__main__":
    sys.exit(main())
