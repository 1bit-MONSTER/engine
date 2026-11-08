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
"""Refuse a member of Q4nxNpuForward that is declared in the header and never defined (engine#1573, #1575).

`main` was green while a full `cmake --build` failed at 92 %: `step_batch_embeds` had two overloads, only the
`HybridInputs` one was defined, and `npu/forward/embeds_equiv_probe.cpp` calls the other.  CI cannot see it --
`.github/workflows/ci.yml` builds `--target onebit` only, because a full build links the NPU forward and wants the private
`ONEBIT_NPU_RT_DIR` -- and `-fsyntax-only` cannot see it either, because a missing definition is a link-time fact, not a
syntax one.

So this pairs the two lists by hand, at (name, arity), which is enough to catch that class and cheap enough to run on every
PR with no runtime, no device and no build:

* declarations  -- the direct members of `class Q4nxNpuForward` in the header that need a definition out of line: a body
  written in the header (`{ ... }`), `= delete`, `= default` and `operator` are skipped, and so are the members of a nested
  `struct`/`class`, which belong to that type rather than to the forward;
* definitions   -- every `Q4nxNpuForward::<name>(` in the tree, from `git ls-files`, so a definition in a second
  translation unit still counts.

Overloads are why the key carries the parameter count: `step_batch_embeds(3)` missing while `step_batch_embeds(4)` exists is
exactly the shape that broke, and a bare-name check would have called it defined.

usage:
  tools/ci/npu_decl_check.py --check [--root DIR]
  tools/ci/npu_decl_check.py --self-test
"""
import argparse
import os
import re
import subprocess
import sys

HEADER = "npu/forward/include/q4nx_forward.h"
CLASS = "Q4nxNpuForward"

SKIP_STATEMENT = ("using ", "typedef ", "friend ", "template", "static_assert", "enum ", "class ", "struct ",
                  "union ", "operator")


def strip_code(text):
    """Blank out comments and string/char literals, keeping every newline so line numbers survive."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def matching(text, start, open_ch, close_ch):
    """Index of the bracket that closes the one at `start`, or -1."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def arity(params):
    """How many parameters a parameter list holds: top-level commas + 1, with `()` and `(void)` being none."""
    params = params.strip()
    if not params or re.fullmatch(r"void\s*", params):
        return 0
    depth, count = 0, 1
    for ch in params:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def class_body(code):
    """The text between `class Q4nxNpuForward {` and its closing brace, or None."""
    m = re.search(r"\bclass\s+" + CLASS + r"\b[^;{]*\{", code)
    if not m:
        return None
    start = m.end() - 1
    end = matching(code, start, "{", "}")
    return None if end < 0 else code[start + 1:end]


def declarations(code):
    """[(name, arity, line)] for the direct members of the class that need an out-of-line definition."""
    body = class_body(code)
    if body is None:
        return None
    offset = code.index(body) if body else 0
    found, depth, current = [], 0, []
    for idx, ch in enumerate(body):
        if ch == "{":
            depth += 1
            current.append(ch)
        elif ch == "}":
            depth -= 1
            current.append(ch)
        elif ch == ";" and depth == 0:
            found.extend(_declaration("".join(current), offset + idx - len(current) + 1))
            current = []
        else:
            current.append(ch)
    return found


def _declaration(statement, line_pos):
    """One statement at class scope -> [(name, arity, line)] (empty when it is not a function declaration)."""
    text = " ".join(statement.split())
    if not text or "(" not in text:
        return []
    if any(text.startswith(k) or text.startswith("static " + k.strip()) for k in SKIP_STATEMENT) or "operator" in text:
        return []
    paren = text.index("(")
    if "=" in text[:paren] or "{" in text or "delete" in text or "default" in text:
        return []  # a data member with an initialiser, or a definition written in the header
    close = matching(text, paren, "(", ")")
    if close < 0:
        return []
    name = re.findall(r"[A-Za-z_]\w*", text[:paren])
    if not name:
        return []
    return [(name[-1], arity(text[paren + 1:close]), 0)]


def definitions(code):
    """{(name, arity)} for every `Q4nxNpuForward::<name>(` the text defines or declares out of line."""
    found = set()
    for m in re.finditer(r"\b" + CLASS + r"::([A-Za-z_]\w*)\s*\(", code):
        open_at = m.end() - 1
        close = matching(code, open_at, "(", ")")
        if close > 0:
            found.add((m.group(1), arity(code[open_at + 1:close])))
    return found


def line_of(code, needle, hits):
    """Line number of the n-th occurrence of `needle` (1-based), for the message."""
    start = 0
    for _ in range(max(hits, 1)):
        at = code.find(needle + "(", start)
        if at < 0:
            return "?"
        start = at + 1
    return code.count("\n", 0, at) + 1


def check(root):
    """The gate: print a refusal per declared-but-undefined member, and return the number of them."""
    header = os.path.join(root, HEADER)
    try:
        code = strip_code(open(header).read())
    except OSError as e:
        print(f"npu_decl_check: cannot read {HEADER}: {e}")
        return 1
    declared = declarations(code)
    if declared is None:
        print(f"npu_decl_check: no `class {CLASS}` in {HEADER}")
        return 1

    sources = subprocess.run(["git", "-C", root, "ls-files", "*.cpp", "*.cc"], capture_output=True, text=True)
    if sources.returncode != 0:
        print(f"npu_decl_check: git ls-files failed in {root}")
        return 1
    defined = set()
    for rel in sources.stdout.split():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue  # a submodule (third_party/llama.cpp) is a gitlink, not source we parse
        try:
            defined |= definitions(strip_code(open(path, errors="replace").read()))
        except OSError:
            continue

    missing = [(name, count) for name, count, _ in declared if (name, count) not in defined]
    for name, count in missing:
        plural = "" if count == 1 else "s"
        print(f"npu_decl_check: {CLASS}::{name} ({count} parameter{plural}) is declared in {HEADER} and never defined"
              f" -- a full build links it and fails, and CI does not link it at all (engine#1573)")
    if not missing:
        print(f"npu_decl_check: {len(declared)} declared member(s), all defined")
    return len(missing)


SELF_HEADER = """class Q4nxNpuForward {
public:
    bool plain(int a, int b);                     // needs a definition: 2 parameters
    bool heavy(int a, int b, std::vector<float>& c);  // needs one: 3 (the #1573 shape)
    void inline_body() const { return; }          // written in the header: never looked for
    void removed() = delete;                      // no definition expected
    struct Nested { size_t bytes() const; };      // belongs to Nested, not to the forward
};
"""


def self_test():
    """Fixture pairs: the gate must be red on the #1573 shape and green once the definitions exist."""
    import tempfile
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "npu/forward/include"))
        os.makedirs(os.path.join(tmp, "npu/forward/src"))
        open(os.path.join(tmp, HEADER), "w").write(SELF_HEADER)
        subprocess.run(["git", "init", "-q", tmp], check=False)

        def write_sources(body, second=""):
            open(os.path.join(tmp, "npu/forward/src/q4nx_forward.cpp"), "w").write(body)
            open(os.path.join(tmp, "npu/forward/src/other.cpp"), "w").write(second)
            # --check reads `git ls-files`, so the fixture has to be in the index (same set CI sees)
            subprocess.run(["git", "-C", tmp, "add", "-A"], check=False, capture_output=True)

        # (a) only the 3-parameter overload defined -> exactly `plain (2 parameters)` refused
        write_sources("bool Q4nxNpuForward::heavy(int a, int b, std::vector<float>& c) { return true; }\n")
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = check(tmp)
        printed = buf.getvalue()
        if rc != 1 or "plain (2 parameters)" not in printed or "heavy" in printed:
            print(f"npu_decl_check --self-test: (a) expected one refusal for plain (2 parameters), got rc={rc}: {printed.strip()}")
            bad += 1

        # (b) the definition moved to a second translation unit -> green
        write_sources("", "bool Q4nxNpuForward::plain(int a, int b) { return a == b; }\n"
                          "bool Q4nxNpuForward::heavy(int a, int b, std::vector<float>& c) { return true; }\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = check(tmp)
        if rc != 0:
            print(f"npu_decl_check --self-test: (b) a definition in another .cpp must count, got rc={rc}: {buf.getvalue().strip()}")
            bad += 1

        # (c) a default argument and a nested `<...>` type in the signature still count their parameters right
        open(os.path.join(tmp, HEADER), "w").write(
            "class Q4nxNpuForward {\n  bool f(std::map<int, std::vector<float>> m, int n = -1);\n};\n")
        write_sources("bool Q4nxNpuForward::f(std::map<int, std::vector<float>> m, int n) { return n > 0; }\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = check(tmp)
        if rc != 0:
            print(f"npu_decl_check --self-test: (c) two parameters with a comma inside <> must pair, got rc={rc}: {buf.getvalue().strip()}")
            bad += 1
    print(f"npu_decl_check --self-test: {bad} failure(s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="pair the header's declarations with the tree's definitions")
    ap.add_argument("--root", default=".", help="repository root for --check (default: .)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.check:
        return 1 if check(os.path.abspath(args.root)) else 0
    ap.error("--check or --self-test")


if __name__ == "__main__":
    sys.exit(main())
