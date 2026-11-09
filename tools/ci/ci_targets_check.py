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
"""Every target ci.yml asks ninja to build must exist in the configuration ci.yml creates.

Why this exists (engine#1591's neighbour, engine#1583): a PR added one name to
`ci.yml`'s `cmake --build build --target …` list for a target defined inside
`if(ONEBIT_NPU)`. CI configures without the private NPU runtime, so ninja got

    ninja: error: unknown target 'embeds_equiv_probe'

and that is not "one target is missing" -- ninja rejects the whole command line
before building anything, so `onebit` stopped being compiled too and every PR's
build step went red.

A configure would answer this exactly, but a configure here needs the
third_party submodules (177 MB + 189 MB) and 20-30 s; this reads the two files
instead. It tracks the CMake `if()`/`endif()` nesting around each
`add_executable`/`add_library`, reads the `-D` flags CI configures with, and
fails a target whose guard CI does not satisfy -- naming the option, which is
the sentence the ninja error does not give.

    tools/ci/ci_targets_check.py --check [--root .]
    tools/ci/ci_targets_check.py --self-test
"""
import argparse
import os
import re
import sys

CI = ".github/workflows/ci.yml"
CMAKE = "CMakeLists.txt"

# `add_executable(name …)` / `add_library(name …)`, and CMake's if()/endif() nesting.
ADD = re.compile(r"^\s*add_(?:executable|library)\s*\(\s*([A-Za-z0-9_.+-]+)", re.IGNORECASE)
IF = re.compile(r"^\s*if\s*\(\s*([^)]*)\)", re.IGNORECASE)
ENDIF = re.compile(r"^\s*endif\s*\(", re.IGNORECASE)
DEFINE = re.compile(r"-D([A-Za-z0-9_]+)=([^\s\"]*)")


def continued_lines(text):
    """The lines of a YAML `run:` block, with backslash continuations joined."""
    out, buf = [], ""
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        out.append(buf + stripped)
        buf = ""
    if buf:
        out.append(buf)
    return out


def ci_targets(ci_text):
    """The target names in ci.yml's `cmake --build … --target …` command."""
    names = []
    for line in continued_lines(ci_text):
        if "cmake --build" not in line or "--target" not in line:
            continue
        words = line.split("--target", 1)[1].split()
        for w in words:
            if w in ("-j",) or w.startswith("-") or not w:
                continue
            names.append(w)
    return names


def ci_defines(ci_text):
    """The -D flags ci.yml configures with, as {NAME: VALUE}."""
    out = {}
    for line in ci_text.splitlines():
        if "cmake -B" in line or "cmake -S" in line:
            for k, v in DEFINE.findall(line):
                out[k] = v
    return out


def cmake_targets(cmake_text):
    """{target: [guard conditions enclosing it]} for every add_executable/add_library.

    A paren-aware scan, because the real tree has both shapes this has to survive:
    a condition spanning lines (`if(ONEBIT_X AND\n    ONEBIT_Y)`) and `else()` /
    `elseif()` inside a block. `else()` marks the entry unknown rather than
    negating it: a target in an else-branch is guarded by the negation, and this
    check would rather leave that to a human than guess (the message prints the
    guard it saw, so a false positive costs one line to read).
    """
    targets = {}
    stack = []            # each entry: {"cond": str|None, "buf": str|None}
    pending = None        # accumulating a multi-line if()/elseif()
    for raw in cmake_text.splitlines():
        line = raw.strip()
        if pending is not None:
            pending[1] += " " + line
            if pending[1].count("(") <= pending[1].count(")"):
                _open_guard(stack, pending[0], pending[1])
                pending = None
            continue
        m = re.match(r"^(if|elseif|else|endif)\s*\((.*)$", line, re.IGNORECASE)
        if m:
            kw, rest = m.group(1).lower(), m.group(2)
            if kw in ("if", "elseif"):
                if rest.count(")") == 0:
                    pending = [kw, rest]
                else:
                    _open_guard(stack, kw, rest)
            elif kw == "else":
                if stack:
                    stack[-1] = {"cond": None}
            else:                                   # endif()
                if stack:
                    stack.pop()
            continue
        m = ADD.match(line)
        if m:
            guards = [e["cond"] for e in stack if e["cond"]]
            unknown = any(e["cond"] is None for e in stack)
            targets.setdefault(m.group(1), {"guards": guards, "unknown": unknown})
    return targets


def _open_guard(stack, kw, text):
    """Push/replace the top of the guard stack for if()/elseif()."""
    body = text.split("(", 1)[1] if "(" in text else text
    body = body.rsplit(")", 1)[0].strip()
    if kw == "elseif" and stack:
        stack[-1] = {"cond": body}
    else:
        stack.append({"cond": body})


def option_defaults(cmake_text):
    """{OPTION: "ON"|"OFF"} for `option(NAME "doc" ON|OFF)`; other defaults are left out.

    A target under `if(OPTION)` is fine when the option defaults ON, and CI does
    not have to pass -D for that: ONEBIT_NPU_HOST defaults ON except on APPLE
    (its default is `${_onebit_npu_host_default}`, so it is *not* in this map and
    `satisfied()` treats it as unknown-and-therefore-fine). ONEBIT_NPU is a plain
    OFF, which is the case engine#1583 tripped over.
    """
    out = {}
    for m in re.finditer(r"^\s*option\s*\(\s*([A-Za-z0-9_]+)\s+\"[^\"]*\"\s+([A-Za-z0-9_]+)\s*\)",
                         cmake_text, re.MULTILINE):
        name, default = m.group(1), m.group(2).upper()
        if default in ("ON", "OFF"):
            out[name] = default
    return out


def satisfied(guard, defines, defaults):
    """True when CI's -D flags satisfy one `if()` condition, or the check cannot tell.

    Known shapes only, and anything else is treated as satisfied so this check
    never fails a tree over a condition it does not model:
        ONEBIT_X          -> needs -DONEBIT_X=ON (or any non-OFF value)
        NOT ONEBIT_X      -> needs ONEBIT_X absent or OFF
    """
    guard = guard.strip()
    negated = False
    if guard.upper().startswith("NOT "):
        negated = True
        guard = guard[4:].strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", guard):
        return True, ""
    value = defines.get(guard)
    if value is None:
        # no -D in ci.yml: the option's own default decides, and a default this
        # scan cannot see (a ${variable}) counts as satisfied -- see option_defaults().
        value = defaults.get(guard)
        if value is None:
            return True, guard
    configured = value.upper() not in ("OFF", "0", "NO", "FALSE", "")
    ok = (not configured) if negated else configured
    return ok, guard


def check(root):
    ci_path = os.path.join(root, CI)
    cmake_path = os.path.join(root, CMAKE)
    for p in (ci_path, cmake_path):
        if not os.path.isfile(p):
            print(f"ci_targets_check: no {p}", file=sys.stderr)
            return 2
    names = ci_targets(open(ci_path, encoding="utf-8").read())
    defines = ci_defines(open(ci_path, encoding="utf-8").read())
    cmake_text = open(cmake_path, encoding="utf-8").read()
    targets = cmake_targets(cmake_text)
    defaults = option_defaults(cmake_text)
    if not names:
        print("ci_targets_check: ci.yml names no targets (the parse found nothing)", file=sys.stderr)
        return 2

    bad = 0
    for name in names:
        info = targets.get(name)
        if info is None:
            print(f"ci_targets_check: ci.yml builds '{name}', which no add_executable/add_library defines")
            bad += 1
            continue
        if info["unknown"]:
            continue                                # an else()-branch: not modelled, see cmake_targets()
        for guard in info["guards"]:
            ok, option = satisfied(guard, defines, defaults)
            if not ok:
                print(f"ci_targets_check: ci.yml builds '{name}', which is inside if({guard}) and ci.yml does not "
                      f"configure {option} -- ninja rejects the whole --target list before building anything")
                bad += 1
    if bad:
        return 1
    print(f"ci_targets_check: all {len(names)} target(s) ci.yml builds exist in the configuration it creates")
    return 0


SELF_CI = """
      - name: Configure
        run: cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-16
      - name: Build
        run: |
          cmake --build build --target onebit \\
            plain_test \\
            hidden_test -j
"""

SELF_CMAKE = """
option(ONEBIT_NPU "x" OFF)
add_executable(onebit main.cpp)
add_executable(plain_test plain.cpp)
if(ONEBIT_NPU)
    add_executable(hidden_test hidden.cpp)
endif()
"""


def self_test():
    import tempfile
    fails = 0

    def case(what, ci, cmake, want):
        nonlocal fails
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, ".github", "workflows"))
            open(os.path.join(d, CI), "w", encoding="utf-8").write(ci)
            open(os.path.join(d, CMAKE), "w", encoding="utf-8").write(cmake)
            got = check(d)
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}  (rc {got}, wanted {want})")
        if not ok:
            fails += 1

    # The shape #1583 shipped: a target under a guard CI does not configure.
    case("a target under if(ONEBIT_NPU) fails when ci.yml does not set it",
         SELF_CI, SELF_CMAKE, 1)
    # … and passes once CI configures the option.
    case("the same target passes when ci.yml sets -DONEBIT_NPU=ON",
         SELF_CI.replace("-DCMAKE_C_COMPILER", "-DONEBIT_NPU=ON -DCMAKE_C_COMPILER"), SELF_CMAKE, 0)
    # A name nobody defines: the other way the list can be wrong.
    case("a name with no add_executable is reported",
         SELF_CI.replace("plain_test", "no_such_target"), SELF_CMAKE, 1)
    # An option that defaults ON needs no -D: ONEBIT_NPU_HOST is that shape in the real tree.
    case("a target under an option that defaults ON passes without a -D",
         SELF_CI, SELF_CMAKE.replace("if(ONEBIT_NPU)", "if(ONEBIT_HOST)") + "option(ONEBIT_HOST \"x\" ON)\n", 0)
    # Unrelated guards must not fail a tree.
    case("a target under a non-option guard is left alone",
         SELF_CI, SELF_CMAKE.replace("if(ONEBIT_NPU)", "if(NOT WIN32)"), 0)
    # And the clean case.
    case("a plain target list passes",
         SELF_CI.replace("hidden_test ", ""), SELF_CMAKE, 0)

    print(f"ci_targets_check --self-test: {fails} failure(s)")
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv[1:])
    if a.self_test:
        return self_test()
    if a.check:
        return check(a.root)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
