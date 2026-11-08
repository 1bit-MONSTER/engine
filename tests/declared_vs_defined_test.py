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
"""declared_vs_defined_test.py -- every member a forward class DECLARES must have a definition in the tree (engine#1575).

Why: `main` stayed green while a full `cmake --build` failed at 92 % with
`undefined reference to Q4nxNpuForward::step_batch_embeds(std::vector<float> const&, int, std::vector<float>&)` (#1573). CI links
only `--target onebit`, so a declaration with no definition costs nothing until someone builds the targets that odr-use it — and the
NPU targets cannot be linked in CI at all, because they need the private runtime.

So the check is static, and it is **arity-aware** on purpose: the failure that motivated it was a *three*-argument overload declared
(`npu/forward/include/q4nx_forward.h:154`) while only the *two*-argument `HybridInputs` overload had a body
(`npu/forward/src/q4nx_forward.cpp:3026`). A name-only check would have called that defined.

Scope, stated rather than implied:
  * only the classes in CLASSES below are checked (a new forward class must be added here, which is the point);
  * matching is by name + parameter count, so a wrong *type* with the right arity is not caught — that still needs a link;
  * members that are inline (`{ … }` on the declaration), `= default`, `= delete`, pure virtual, templates, or constructors /
    destructors are skipped, because they are either defined at the declaration or not functions in the class's own budget;
  * definitions are searched in the class's own source directory (npu/forward/src), so a definition in the file is what counts.

    tests/declared_vs_defined_test.py            # check the tree this file lives in
    tests/declared_vs_defined_test.py --root DIR # check another checkout (what --inject uses)
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# class -> (declaration header, directory of definitions)
CLASSES = {
    "Q4nxNpuForward": ("npu/forward/include/q4nx_forward.h", "npu/forward/src"),
}


def split_top_level(s):
    """Split a parameter list on commas that are not inside <>, (), [] or {}. Good enough for C++ declarations in this tree."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


def arity(params):
    """Parameter count, ignoring a trailing `= default` value and a lone `void`."""
    p = params.strip()
    if not p or p == "void":
        return 0
    parts = split_top_level(p)
    n = 0
    for part in parts:
        # a default argument's VALUE may contain commas inside <> or () — those are already protected by the splitter
        n += 1
    return n


DECL = re.compile(r"^\s*(?:static\s+|inline\s+|virtual\s+|explicit\s+|constexpr\s+)*"
                  r"(?:[\w:<>,\s\*&]+?)\b(\w+)\s*\(([^;{}]*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?([;{=])")


def class_body(text, cls):
    """The text between `class <cls> {` (or its forward-declared body) and the matching closing brace."""
    m = re.search(r"\bclass\s+" + re.escape(cls) + r"\b[^;{]*\{", text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def declarations(root, cls, header):
    """Member declarations made DIRECTLY in `cls`, as [(name, arity)] -- nested classes are skipped, not flattened.

    A first cut joined every line of the class body and picked up `size_t bytes() const;` from the nested StateSnapshot class,
    then reported it as a missing `Q4nxNpuForward::bytes`. That is the kind of false positive that gets a guard switched off, so
    the scan is character-level and brace-aware: only text at depth 0 of the class body is a declaration of THIS class.
    """
    path = os.path.join(root, header)
    if not os.path.exists(path):
        return None, f"{header}: not found"
    raw = open(path, errors="replace").read()
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)          # block comments
    raw = re.sub(r"//[^\n]*", "", raw)                        # line comments: an accumulated declaration must not start with one
    body = class_body(raw, cls)
    if body is None:
        return None, f"{header}: no `class {cls}` body"
    decls = []
    depth = 0
    acc = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "{":
            if depth == 0:
                # an inline body: whatever accumulated is a declaration that defines itself, nothing is owed
                acc = []
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                acc = []
        elif depth == 0 and ch == ";":
            # the terminator is not in `acc` (we branch on it), so put it back: DECL's tail group keys off it
            text = " ".join(x.strip() for x in "".join(acc).splitlines()) + ";"
            m = DECL.search(text)
            if m:
                name, params = m.group(1), m.group(2)
                if name != cls and name != "~" + cls:
                    decls.append((name, arity(params)))
            acc = []
        elif depth == 0:
            acc.append(ch)
        i += 1
    return decls, None


def definitions(root, cls, srcdir):
    """set of (name, arity) defined as `Class::name(...)` anywhere under srcdir."""
    out = set()
    d = os.path.join(root, srcdir)
    for dirpath, _dirs, files in os.walk(d):
        for f in files:
            if not f.endswith((".cpp", ".cc", ".cxx")):
                continue
            text = open(os.path.join(dirpath, f), errors="replace").read()
            for m in re.finditer(r"\b" + re.escape(cls) + r"::(\w+)\s*\(", text):
                # read the parameter list from the opening paren
                i = m.end() - 1
                depth = 0
                for j in range(i, min(len(text), i + 4000)):
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                        if depth == 0:
                            out.add((m.group(1), arity(text[i + 1:j])))
                            break
    return out


def check(root):
    problems = []
    for cls, (header, srcdir) in CLASSES.items():
        decls, err = declarations(root, cls, header)
        if err:
            problems.append(err)
            continue
        defs = definitions(root, cls, srcdir)
        for name, n in decls:
            if (name, n) not in defs:
                # a name defined with a DIFFERENT arity is the exact defect this check exists for: say so
                near = sorted({a for (nm, a) in defs if nm == name})
                if near:
                    problems.append(f"{cls}::{name} declared with {n} parameter(s) but only defined with "
                                    f"{', '.join(str(a) for a in near)} -- an overload with no body")
                else:
                    problems.append(f"{cls}::{name} declared with {n} parameter(s) and never defined in {srcdir}")
    return problems


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT, help="checkout to check (default: this file's tree)")
    a = ap.parse_args(argv)
    problems = check(a.root)
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        print(f"\n{len(problems)} declared-but-undefined member(s): CI links only --target onebit, so nothing else will tell you "
              f"until a target that odr-uses the symbol is linked (engine#1575, #1573).")
        return 1
    print("declared-vs-defined: every member of " + ", ".join(CLASSES) + " with a declaration has a definition with the same arity")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
