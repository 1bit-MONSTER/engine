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

**Direction B (engine#1595) is the reverse gap.** `need` is the set of `add_executable` targets that
some `add_test`'s COMMAND names -- a test whose target CI never builds reports "Not Run" and is a test
in name only. `have` is the `--target` list above. The difference is a ratchet rather than a wall, because
35 of them are already there: `tools/ci/add_test_gaps.txt` records today's names, and the check **fails
when the difference grows**, printing only the new names. A shrinking baseline is progress, not a failure,
so a rename does not have to be argued with -- a count would pass it.

A script test names its interpreter (`add_test(… COMMAND bash script.sh)`), and `bash` is not an
`add_executable`, so a self-test runner costs nothing here: that is why the release check's suite can live
in an `add_test` (engine#1596) without raising the drift it measures.

    tools/ci/ci_targets_check.py --check [--root .]
    tools/ci/ci_targets_check.py --self-test
    tools/ci/ci_targets_check.py --where ROW[,ROW...]    where each ctest row is registered (file:line)
"""
import argparse
import glob
import os
import re
import shutil
import sys

CI = ".github/workflows/ci.yml"
CMAKE = "CMakeLists.txt"

# `add_executable(name …)` / `add_library(name …)`, and CMake's if()/endif() nesting.
ADD = re.compile(r"^\s*add_(?:executable|library)\s*\(\s*([A-Za-z0-9_.+-]+)", re.IGNORECASE)
IF = re.compile(r"^\s*if\s*\(\s*([^)]*)\)", re.IGNORECASE)
ENDIF = re.compile(r"^\s*endif\s*\(", re.IGNORECASE)
INCLUDE = re.compile(r'^\s*include\(\s*"?([^\s")]+\.cmake)"?\s*\)', re.M)
# `file(GLOB VAR … <dir>/*.cmake)` … `foreach(… IN LISTS VAR)` … `include(${…})`: the one-file-per-test
# shape engine#1696 landed. The include argument is a VARIABLE, so INCLUDE above cannot match it, and a
# reader that only followed literal includes would stop seeing every registration at once - the count
# would fall to zero and read as "the gap closed".
# engine#1979: the pattern main uses is `cmake/tests-*.cmake` (a prefix before the star), not `<dir>/*.cmake`; a
# reader that wanted the star right after the slash matched nothing there, and every such file went unread.
GLOB = re.compile(r'file\s*\(\s*GLOB\s+(\w+)[^)]*?([^\s")]+/[^\s"/)]*\*[^\s"/)]*\.cmake)', re.IGNORECASE)
INCLUDE_VAR = re.compile(r'include\(\s*\$\{(\w+)\}\s*\)')
FOREACH_LISTS = re.compile(r'foreach\s*\(\s*(\w+)\s+IN\s+LISTS\s+(\w+)', re.IGNORECASE)
DEFINE = re.compile(r"-D([A-Za-z0-9_]+)=([^\s\"]*)")


COMMENT = re.compile(r"(^|\s)#.*$")


def continued_lines(text):
    r"""The lines of a YAML `run:` block, comment-stripped, with backslash continuations joined.

    Two things have to happen in this order, and getting it wrong is how this gate invented two
    target names in one day:

      * strip the comment **per physical line, before joining** -- a `#` that follows a
        continuation is no longer at the start of the logical line once the lines are joined, so a
        full-line check cannot see it (engine#1601's review: a `# notatarget trailingprose` inside
        a `\`-continued list produced `ci_targets_check: ci.yml builds 'trailingprose'`);
      * decide the **continuation from the original line**, not the stripped one -- a line that is
        only a comment *and* ends in `\` still continues the command, so stripping it first would
        end the logical line early and silently drop every target after it.

    The `(^|\s)` in COMMENT keeps a `#` inside a quoted string (``"a#b"``) from being eaten.
    """
    out, buf = [], ""
    for line in text.splitlines():
        cont = line.rstrip().endswith("\\")
        core = COMMENT.sub("", line).rstrip()
        if cont:
            buf += (core[:-1] if core.endswith("\\") else core) + " "
            continue
        out.append(buf + core)
        buf = ""
    if buf.strip():
        out.append(buf)
    return out


def ci_targets(ci_text):
    """The target names in ci.yml's `cmake --build … --target …` command."""
    names = []
    for line in continued_lines(ci_text):
        # A YAML `run:` block can carry `#` comments, and they talk about the very command this scans
        # ("adding an unknown target to `cmake --build --target ...` is a hard ninja error"), so a comment
        # naming the phrase would otherwise contribute its prose as target names. engine#1583's second
        # attempt did exactly that: the scan reported targets called 'What' and 'this'.
        if line.lstrip().startswith("#"):
            continue
        if "cmake --build" not in line or "--target" not in line:
            continue
        words = line.split("--target", 1)[1].split()
        for w in words:
            if w in ("-j",) or w.startswith("-") or not w:
                continue
            names.append(w)
    return names


# `add_executable(name …)` and `add_test(NAME n COMMAND …)`, for direction B (engine#1595).
ADD_EXE = re.compile(r"^\s*add_executable\s*\(\s*([A-Za-z0-9_.+-]+)", re.IGNORECASE)
ADD_TEST = re.compile(r"^\s*add_test\s*\(\s*NAME\s+([A-Za-z0-9_.+-]+)\s+COMMAND\s+(.*)$", re.IGNORECASE)
BASELINE = "tools/ci/add_test_gaps.txt"
# `add_dependencies(<target> <dep>…)`: the second way a target gets built (engine#1595 review).
ADD_DEPS = re.compile(r"add_dependencies\s*\(\s*([A-Za-z0-9_.+-]+)([^)]*)\)", re.IGNORECASE)


def build_closure(cmake_text, listed, defines=None, defaults=None):
    """`listed`, plus every target reachable from it through `add_dependencies`.

    `ci.yml`'s `--target` list is not the only way a target gets built: this tree documents
    `add_dependencies(onebit onebit_npu_forward_syntax)` precisely so a gate target is compiled without being
    named on the command line, and the comment there says why ("a check that depends on the order of another
    step's target list is the kind that goes quiet when that list is edited"). A `have` that reads only the
    list calls every such test "never built", so the first PR to use the documented hook reds direction B.
    """
    deps = {}
    for _n, line, guards, unknown in scan_guards(cmake_text):
        m = ADD_DEPS.search(line)
        if not m:
            continue
        # engine#1595 review: an `add_dependencies` inside `if(ONEBIT_NPU)` does not exist in the configuration
        # CI creates, so counting it lets one dead-guard line clear a test that still never reaches ctest.
        if defines is not None and not unknown and any(not satisfied(g, defines, defaults or {})[0] for g in guards):
            continue
        for d in m.group(2).split():
            if re.fullmatch(r"[A-Za-z0-9_.+-]+", d):
                deps.setdefault(m.group(1), set()).add(d)
    built, frontier = set(listed), list(listed)
    while frontier:
        for d in deps.get(frontier.pop(), ()):
            if d not in built:
                built.add(d)
                frontier.append(d)
    return built


def add_test_blocks(cmake_text):
    """Every `add_test( … )` directive, paren-joined across lines: (line number, text).

    The real tree writes 39 of them across lines, so a line-at-a-time scan misses them - and a missed
    add_test is a missed gap, which for a ratchet means silence rather than a wrong red.
    """
    blocks, buf, start = [], "", None
    for i, raw in enumerate(cmake_text.splitlines(), 1):
        line = raw.strip()
        if buf:
            buf += " " + line
            if buf.count("(") <= buf.count(")"):
                blocks.append((start, buf))
                buf, start = "", None
            continue
        if re.match(r"^add_test\s*\(", line, re.IGNORECASE):
            if line.count("(") > line.count(")"):
                buf, start = line, i
            else:
                blocks.append((i, line))
    return blocks


def add_test_need(cmake_text):
    """{target: [test names]} for every add_executable that some add_test's COMMAND names.

    Direction B, engine#1595. A target is named two ways in this tree - bare, as `COMMAND plain_test`,
    and through a generator expression, as `$<TARGET_FILE:plain_test>` - so both count. A script test
    names its interpreter (`COMMAND bash script.sh`), which is not an add_executable, and that is why a
    self-test runner costs nothing here (engine#1596).
    """
    exes = set()
    for line in cmake_text.splitlines():
        m = ADD_EXE.match(line)
        if m:
            exes.add(m.group(1))
    need = {}
    for lineno, block in add_test_blocks(cmake_text):
        m = re.search(r"\bNAME\s+([A-Za-z0-9_.+-]+)", block)
        test_name = m.group(1) if m else f"line {lineno}"
        named = set(re.findall(r"\$<TARGET_FILE:([A-Za-z0-9_.+-]+)>", block))
        for tok in re.split(r"[\s()]+", block):
            if tok in exes:
                named.add(tok)
        for target in sorted(named & exes):
            need.setdefault(target, []).append(test_name)
    return need


def read_baseline(path):
    """The names already known to drift, or None when there is no baseline file (which fails closed)."""
    if not os.path.isfile(path):
        return None
    names = set()
    for line in open(path, encoding="utf-8"):
        name = line.split("#", 1)[0].strip()
        if name:
            names.add(name)
    return names


def ci_defines(ci_text):
    """The -D flags ci.yml configures with, as {NAME: VALUE}."""
    out = {}
    for line in ci_text.splitlines():
        if "cmake -B" in line or "cmake -S" in line:
            for k, v in DEFINE.findall(line):
                out[k] = v
    return out


def scan_guards(cmake_text):
    """(lineno, line, guards, unknown) for every line that is not itself an if()/elseif()/else()/endif().

    A paren-aware scan, because the real tree has both shapes this has to survive: a condition spanning
    lines (`if(ONEBIT_X AND\n    ONEBIT_Y)`) and `else()` / `elseif()` inside a block. `else()` marks the
    enclosing scope unknown rather than negating it: a target in an else-branch is guarded by the negation,
    and this check would rather leave that to a human than guess.

    Shared since engine#1595's review: `add_test` drift and `add_dependencies` edges need the same guards
    `add_executable` gets, and a second walker would be a second place for the paren handling to drift.
    """
    stack = []            # each entry: {"cond": str|None}
    pending = None        # accumulating a multi-line if()/elseif()
    lines = cmake_text.splitlines()
    for i, raw in enumerate(lines):
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
        while line.endswith("\\") and i + 1 < len(lines):
            i += 1
            line = line[:-1].strip() + " " + lines[i].strip()
        if line:
            yield i + 1, line, [e["cond"] for e in stack if e["cond"]], any(e["cond"] is None for e in stack)


def cmake_targets(cmake_text):
    """{target: {"guards", "unknown"}} for every add_executable/add_library."""
    targets = {}
    for _n, line, guards, unknown in scan_guards(cmake_text):
        m = ADD.match(line)
        if m:
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


def family_files(root, cmake_text):
    """[(relpath, text)] for every cmake file CMakeLists.txt actually include()s, in include order.

    engine#2129 follow-up (a release verifier grepped CMakeLists.txt for two ctest rows and found zero,
    because both live in `cmake/tests-*.cmake`): the paths were discarded here, so nothing could answer
    *where* a row is registered and a CMakeLists-only grep looked like proof of absence. Keeping the path
    costs nothing and makes that question answerable from the same reader -- the #2012 lesson (a checker
    whose coverage did not follow the refactor) applied to the human verifying a release.
    """
    out = []
    seen = set()
    for gvar, pattern in GLOB.findall(cmake_text):
        loop_vars = [v for v, g in FOREACH_LISTS.findall(cmake_text) if g == gvar]
        if not loop_vars or loop_vars[0] not in INCLUDE_VAR.findall(cmake_text):
            continue
        pat = pattern.replace("${CMAKE_SOURCE_DIR}/", "").replace("${PROJECT_SOURCE_DIR}/", "")
        for path in sorted(glob.glob(os.path.join(root, pat))):
            if os.path.isfile(path) and os.path.realpath(path) not in seen:
                seen.add(os.path.realpath(path))
                out.append((os.path.relpath(path, root), open(path, encoding="utf-8").read()))
    for m in INCLUDE.finditer(cmake_text):
        rel = m.group(1).replace("${CMAKE_SOURCE_DIR}/", "").replace("${PROJECT_SOURCE_DIR}/", "")
        path = os.path.join(root, rel)
        if os.path.isfile(path) and os.path.realpath(path) not in seen:
            seen.add(os.path.realpath(path))
            out.append((rel, open(path, encoding="utf-8").read()))
    return out


def family_text(root, cmake_text):
    """The text of every cmake file CMakeLists.txt actually include()s, in include order.

    engine#1595 review: a glob over `cmake/tests-*.cmake` reads a file nothing includes, so a family file
    that was moved out of the build is still counted (and its tests still ratcheted), while a family file
    pulled in under a guard is counted as if it were unconditional. Following include() is the only reading
    that matches what CMake configures; the guard on the include is left to cmake_targets(), which already
    tracks a guard stack across the text this returns.

    One implementation of the walk: `family_files()` holds it, because a caller that only wants the text
    loses the path and then cannot say where a row lives.
    """
    return "\n".join(text for _, text in family_files(root, cmake_text))



def check(root):
    ci_path = os.path.join(root, CI)
    cmake_path = os.path.join(root, CMAKE)
    for p in (ci_path, cmake_path):
        if not os.path.isfile(p):
            print(f"ci_targets_check: no {p}", file=sys.stderr)
            return 2
    names = ci_targets(open(ci_path, encoding="utf-8").read())
    defines = ci_defines(open(ci_path, encoding="utf-8").read())
    # CMakeLists.txt plus any included family file. A `add_test`/`add_executable` pair moved into
    # `cmake/tests-*.cmake` (engine#1696: one family out of the contiguous add_test block, so two test-adding
    # PRs stop conflicting by construction) is still a test this ratchet must see: a family file it does not
    # read is a family whose gaps are silently unratcheted, which looks exactly like "no new gaps".
    cmake_text = open(cmake_path, encoding="utf-8").read()
    cmake_text += "\n" + family_text(root, cmake_text)
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

    # ── direction B (engine#1595): a test whose target CI never builds never reaches ctest ───────────
    need = add_test_need(cmake_text)
    # `built`, not `names`: add_dependencies is the tree's documented second route (see build_closure).
    drift = set(need) - build_closure(cmake_text, names, defines, defaults)
    # engine#1595 review: a test declared inside a guard CI's configure makes false is not drift. CI is *supposed*
    # to skip it -- an NPU-only test under `if(ONEBIT_NPU)` with ONEBIT_NPU=OFF -- and telling its author to
    # "build it or drop the test" was a false positive. Only a test CI would register and build, and does not, is
    # the defect this ratchets.
    # engine#1810 review: three PRs in one shift added a test `add_executable` and stopped here, because the message
    # named only two routes and neither was available -- ci.yml's build list is a `.github/workflows/` file, which a lane
    # identity cannot push (no `workflow` scope). Folding the cases into a target ci.yml already builds needs no scope
    # at all, so the message names it.
    line_guard = {n: (g, u) for n, _l, g, u in scan_guards(cmake_text)}
    by_design = set()
    for name in sorted(drift):
        # Both sites count: the add_test block that registers the test, and the add_executable it names. The tree
        # writes NPU tests as `if(ONEBIT_NPU) add_executable(..) add_test(..) endif()`, so either side alone would
        # miss the half that matters on some pack.
        # `need` is {target: [test names]}, so the blocks to look at are the ones naming this target's TESTS -- an
        # add_test's NAME is the test, not the executable, and matching the target there silently finds nothing.
        guards, unknown = [], False
        blocks = add_test_blocks(cmake_text)
        for test_name in need[name]:
            for lineno, block in blocks:
                if re.search(r"\bNAME\s+" + re.escape(test_name) + r"\b", block):
                    g, u = line_guard.get(lineno, ([], False))
                    guards += g
                    unknown = unknown or u
        for t in need[name]:
            info = targets.get(t, {})
            guards += info.get("guards", [])
            unknown = unknown or info.get("unknown", False)
        if not unknown and any(not satisfied(g, defines, defaults)[0] for g in guards):
            by_design.add(name)
    drift -= by_design
    if by_design:
        print(f"ci_targets_check: {len(by_design)} test(s) are declared only under guards ci.yml's configure turns "
              f"off, so CI never registers them by design ({', '.join(sorted(by_design)[:6])}"
              f"{', …' if len(by_design) > 6 else ''}) - not drift")
    baseline = read_baseline(os.path.join(root, BASELINE))
    if baseline is None:
        # No baseline is not "nothing known": treating it as empty would let the first run of a run
        # without the file fail on 35 names a reader could not tell from regressions. Fail closed at 35.
        print(f"ci_targets_check: no {BASELINE}, so the {len(drift)} target(s) whose tests CI never builds "
              f"cannot be told apart from new ones", file=sys.stderr)
        return 1
    new = sorted(drift - baseline)
    for name in new:
        guards = sorted({g for t in need[name] for g in targets.get(t, {}).get("guards", [])})
        where = f" under if({' AND '.join(guards)})" if guards else ""
        print(f"ci_targets_check: add_test names '{name}'{where}, and ci.yml never builds "
              f"{'/'.join(sorted(set(need[name])))} -- that test never reaches ctest in the configuration CI "
              f"creates (it is not registered, or registered and not built). Routes, of which only the first needs a "
              f"ci.yml edit: name the target in ci.yml's build list (a `.github/workflows/` edit, so it needs "
              f"`workflow` scope); make it a build dependency of a target ci.yml builds "
              f"(`add_dependencies(onebit <target>)`, which this checker already follows); fold these cases into such "
              f"a target (the tests/npu_pack_import_test.cpp pattern); or drop the test")
    if new:
        return 1
    # A baseline entry that no longer drifts is progress; say so rather than staying silent, so the
    # file can be shrunk by whoever reads the line.
    # A baseline entry can leave `drift` for two reasons, and only one of them is good news:
    #   * the target IS still named by an add_test and ci.yml now builds it / a dependency builds it - progress;
    #   * nothing names it any more - the test was deleted (fine, but then its line goes too) OR the reader
    #     stopped seeing it (engine#1696's one-file-per-test move did exactly this to this tool: 179 files
    #     went unread and the gap "closed"). A number may only fall while its subject is still visible.
    vanished = sorted(baseline - set(need))
    built = sorted((baseline & set(need)) - drift)
    if built:
        print(f"ci_targets_check: {len(built)} baseline entr(y/ies) no longer drift ({', '.join(built[:6])}"
              f"{', …' if len(built) > 6 else ''}) - drop them from {BASELINE} to keep it honest")
    if vanished:
        for name in vanished:
            print(f"ci_targets_check: '{name}' is in {BASELINE} but no add_test names it any more - if the "
                  f"test was deleted, remove its line in this change; if it was not, the reader has stopped "
                  f"seeing it and every count below is about less than it claims")
        return 1
    print(f"ci_targets_check: {len(drift)} target(s) whose test never reaches ctest under ci.yml "
          f"(baseline {len(baseline)}), and none new")
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




def all_rows(root, cmake_path):
    """[(row, site, line)] for every `add_test`/`add_python_test` row the reader can see.

    engine#2160 review: `--where` matches EXACTLY, which is right (a substring match would answer for
    `tmp_path_check_self` when asked about `tmp_path_check`).  But the miss and the typo read identically:
    `--where submodule_pin` says NOT REGISTERED while the row is `submodule_pin_check`.  This lists the rows
    so a miss can name the near ones without ever calling a partial match a registration.
    """
    files = [(os.path.relpath(cmake_path, root), open(cmake_path, encoding="utf-8").read())]
    files += family_files(root, files[0][1])
    rows = []
    for rel, text in files:
        for i, line in enumerate(continued_lines(text), 1):
            m = ROW.match(line)
            if m and (m.group(1), rel, i) not in rows:
                rows.append((m.group(1), rel, i))
    return rows


def where_rows(root, cmake_path, wanted):
    """[(row, site, line)] for each wanted row name, searched in CMakeLists.txt then every included file.

    engine#2129 follow-up: a release verifier grepped `CMakeLists.txt` for `submodule_pin` and `tmp_path`,
    got 0, and nearly raised a false alarm against its own list - both are registered in `cmake/tests-*.cmake`,
    which #1983's glob include()s. A CMakeLists-only grep is the wrong instrument for a family file, so this
    answers the question with the same reader that follows the globs and includes.
    """
    files = [(os.path.relpath(cmake_path, root), open(cmake_path, encoding="utf-8").read())]
    files += family_files(root, files[0][1])
    hits = []
    for rel, text in files:
        for i, line in enumerate(continued_lines(text), 1):
            m = ROW.match(line)
            if m and m.group(1) in wanted and (m.group(1), rel, i) not in hits:
                hits.append((m.group(1), rel, i))
    return hits


ROW = re.compile(r'^\s*(?:add_test|add_python_test)\s*\(\s*(?:NAME\s+)?(\w+)')


def where_rows_self_test(tmp):
    """engine#2129 follow-up: the answer must come from the family files too, not only CMakeLists.txt."""
    import pathlib as _p
    root = _p.Path(tmp)
    (root / "cmake").mkdir(parents=True, exist_ok=True)
    (root / "CMakeLists.txt").write_text(
        'file(GLOB V CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cmake/tests-*.cmake)\n'
        'list(SORT V)\n'
        'foreach(v IN LISTS V)\n    include(${v})\nendforeach()\n'
        'add_python_test(plain_row x.py)\n', encoding="utf-8")
    (root / "cmake" / "tests-family.cmake").write_text(
        'add_python_test(globbed_row y.py)\n', encoding="utf-8")
    return where_rows(str(root), str(root / "CMakeLists.txt"), ["plain_row", "globbed_row", "absent_row"])


def self_test():
    import tempfile
    fails = 0

    def case(what, ci, cmake, want, baseline="auto", family=None, family_rel="cmake/tests/zz_family.cmake"):
        nonlocal fails
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, ".github", "workflows"))
            open(os.path.join(d, CI), "w", encoding="utf-8").write(ci)
            open(os.path.join(d, CMAKE), "w", encoding="utf-8").write(cmake)
            if family is not None:
                fam_path = os.path.join(d, family_rel)
                os.makedirs(os.path.dirname(fam_path), exist_ok=True)
                open(fam_path, "w", encoding="utf-8").write(family)
            if baseline == "auto":
                # The direction-A fixtures below are not about direction B, so the baseline records
                # whatever they drift - a direction-B case must then say what its baseline is.
                drift = sorted(set(add_test_need(cmake)) - set(ci_targets(ci)))
                baseline = "".join(n + "\n" for n in drift)
            if baseline is not None:
                bpath = os.path.join(d, BASELINE)
                os.makedirs(os.path.dirname(bpath), exist_ok=True)
                open(bpath, "w", encoding="utf-8").write(baseline)
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
    # A comment inside the run block that names the command must not be parsed as targets (engine#1583's
    # second attempt did: the scan reported targets 'What' and 'this' from the prose).
    case("a # comment mentioning cmake --build --target contributes no target names",
         SELF_CI.replace("hidden_test ", "").replace("      - name: Build",
                         "      # NB: adding an unknown target to `cmake --build --target ...` is a hard ninja error\n"
                         "      - name: Build"), SELF_CMAKE, 0)
    # … and a comment AFTER a continuation is the same prose, one hop in: the continuations are
    # joined first, so it is no longer at the start of the logical line (found in #1601's review).
    case("a # comment after a continuation contributes no target names",
         SELF_CI.replace("hidden_test ", "").replace(
             "            plain_test \\\n",
             "            # notatarget trailingprose \\\n            plain_test \\\n"),
         SELF_CMAKE, 0)
    # Unrelated guards must not fail a tree.
    case("a target under a non-option guard is left alone",
         SELF_CI, SELF_CMAKE.replace("if(ONEBIT_NPU)", "if(NOT WIN32)"), 0)
    # ── direction B (engine#1595): a test whose target CI never builds ─────────────────────────────────
    # hidden_test is direction A's subject, so these fixtures drop it; `unbuilt_test` is direction B's.
    # The case that says the fixture really drifts is the second one - the same tree with an empty
    # baseline must fail - otherwise "passes when the baseline records it" could be passing on nothing.
    B_CI = SELF_CI.replace("hidden_test ", "")
    B_CMAKE = SELF_CMAKE + """
add_executable(unbuilt_test unbuilt.cpp)
add_test(NAME unbuilt_suite COMMAND unbuilt_test)
"""
    case("a test whose target ci.yml never builds passes when the baseline records it",
         B_CI, B_CMAKE, 0, baseline="unbuilt_test\n")
    case("the same tree with an empty baseline fails, naming the target it never builds",
         B_CI, B_CMAKE, 1, baseline="")
    case("a missing baseline fails closed rather than passing vacuously",
         B_CI, B_CMAKE, 1, baseline=None)
    case("a baseline entry that no longer drifts is progress, not a failure",
         B_CI.replace("--target onebit", "--target onebit unbuilt_test"), B_CMAKE, 0,
         baseline="unbuilt_test\n")
    # engine#1596's shape: a script test names `bash`, which is not an add_executable, so it adds nothing.
    # engine#1696's one-file-per-test shape: `file(GLOB V … cmake/tests/*.cmake)` + `foreach(v IN LISTS V)`
    # + `include(${v})`. The include argument is a VARIABLE, so a reader that only follows literal
    # includes stops seeing every registration at once - the count falls to 0 and reads as "closed".
    GLOB_CMAKE = SELF_CMAKE + """
file(GLOB ONEBIT_TEST_REGISTRATIONS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cmake/tests/*.cmake)
foreach(_onebit_reg IN LISTS ONEBIT_TEST_REGISTRATIONS)
    include(${_onebit_reg})
endforeach()
"""
    case("an add_test in a glob-included family file is still read",
         B_CI, GLOB_CMAKE, 1, baseline="",
         family="add_executable(glob_fam g.cpp)\nadd_test(NAME glob_suite COMMAND glob_fam)\n")
    # engine#1979: the shape main has, a prefix before the star (`cmake/tests-*.cmake`). With the GLOB pattern that
    # wanted `<dir>/*.cmake` this arm returned 0: the file was configured by CMake and read by nobody here.
    PREFIX_GLOB_CMAKE = GLOB_CMAKE.replace("cmake/tests/*.cmake", "cmake/tests-*.cmake")
    case("an add_test in a file included by a cmake/tests-*.cmake glob is still read",
         B_CI, PREFIX_GLOB_CMAKE, 1, baseline="",
         family="add_executable(glob_fam g.cpp)\nadd_test(NAME glob_suite COMMAND glob_fam)\n",
         family_rel="cmake/tests-zz.cmake")
    # ... and a file the glob does not match (one directory down) is not configured, so it is not read.
    case("a file the cmake/tests-*.cmake glob does not match is not read",
         B_CI, PREFIX_GLOB_CMAKE, 0, baseline="",
         family="add_executable(glob_fam g.cpp)\nadd_test(NAME glob_suite COMMAND glob_fam)\n",
         family_rel="cmake/tests-zz/inner.cmake")
    # ... and a baseline name nothing names any more is a FAILURE, not "progress": the two reasons a
    # count can fall are "now built" (good) and "no longer seen" (the reader narrowed).
    case("a baseline name no add_test names any more fails instead of reading as progress",
         B_CI, SELF_CMAKE, 1, baseline="vanished_test\n")
    case("a script test (COMMAND bash …) adds nothing to the drift",
         B_CI, SELF_CMAKE + "add_test(NAME script_suite COMMAND bash ${CMAKE_SOURCE_DIR}/x.sh --self-test)\n",
         0, baseline="")
    # engine#1595 review: a target CI builds only through `add_dependencies` is built, so its test does not drift.
    D_CMAKE = B_CMAKE + """
add_executable(hoisted_test hoisted.cpp)
add_test(NAME hoisted_suite COMMAND hoisted_test)
add_dependencies(onebit hoisted_test)
"""
    case("a target built through add_dependencies is not drift, so its test passes",
         B_CI, D_CMAKE, 0, baseline="unbuilt_test\n")
    case("the same tree without the add_dependencies line drifts that test, naming it",
         B_CI, B_CMAKE + """
add_executable(hoisted_test hoisted.cpp)
add_test(NAME hoisted_suite COMMAND hoisted_test)
""", 1, baseline="unbuilt_test\n")
    # engine#1595 review, both halves of the false-positive/false-negative pair, end to end.
    # B_CI, not SELF_CI: SELF_CI's --target list names hidden_test, which lives behind the same dead guard, so
    # direction A would fail this arm for the other reason and the arm would pass or fail on the wrong check.
    case("a test declared inside a guard ci.yml turns off is not drift (CI is meant to skip it)",
         B_CI, SELF_CMAKE + """
if(ONEBIT_NPU)
    add_executable(npu_only_test npu.cpp)
    add_test(NAME npu_only COMMAND npu_only_test)
endif()
""", 0, baseline="")
    case("an add_dependencies inside a dead guard does not clear a live test",
         B_CI, B_CMAKE + """
add_executable(live_test live.cpp)
add_test(NAME live_suite COMMAND live_test)
if(ONEBIT_NPU)
    add_dependencies(onebit live_test)
endif()
""", 1, baseline="unbuilt_test\n")
    case("...but the same edge outside the guard does clear it",
         B_CI, B_CMAKE + """
add_executable(live_test live.cpp)
add_test(NAME live_suite COMMAND live_test)
add_dependencies(onebit live_test)
""", 0, baseline="unbuilt_test\n")
    # engine#1696: a test that lives in an included family file is still ratcheted. `case` writes only
    # .github/workflows/CI and CMakeLists.txt, so this arm builds its tree by hand.
    def family_tree(included):
        """A tree whose family file holds one drifting test, included or merely present.

        engine#1595 review: the arm used to be a bare glob, so a file nothing includes was still read. The
        fixture now says which of the two it is building, because include() is the whole point.
        """
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, ".github", "workflows"))
        os.makedirs(os.path.join(d, "cmake"))
        open(os.path.join(d, CI), "w", encoding="utf-8").write(B_CI)
        inc = "include(cmake/tests-family.cmake)\n" if included else ""
        open(os.path.join(d, CMAKE), "w", encoding="utf-8").write(B_CMAKE + inc)
        open(os.path.join(d, "cmake", "tests-family.cmake"), "w", encoding="utf-8").write("""
add_executable(family_test family.cpp)
add_test(NAME family_suite COMMAND family_test)
""")
        bpath = os.path.join(d, BASELINE)
        os.makedirs(os.path.dirname(bpath), exist_ok=True)
        open(bpath, "w", encoding="utf-8").write("unbuilt_test\n")
        return d

    d = family_tree(included=True)
    got = check(d)
    shutil.rmtree(d, ignore_errors=True)
    ok = got == 1
    print(f"  {'ok  ' if ok else 'FAIL'}  a test moved into an include()d family file is still ratcheted  "
          f"(rc {got}, wanted 1)")
    if not ok:
        fails += 1
    # ...and a family file that nothing include()s is not part of the tree CMake configures, so it adds nothing.
    d = family_tree(included=False)
    got = check(d)
    shutil.rmtree(d, ignore_errors=True)
    ok = got == 0
    print(f"  {'ok  ' if ok else 'FAIL'}  a family file nothing include()s is not read  (rc {got}, wanted 0)")
    if not ok:
        fails += 1

    # And the clean case.
    case("a plain target list passes",
         SELF_CI.replace("hidden_test ", ""), SELF_CMAKE, 0)

    # engine#2129 follow-up: --where must find a row in a glob-included family file, not only in
    # CMakeLists.txt -- that is the reading a release verifier needs and the one a bare grep gets wrong.
    with tempfile.TemporaryDirectory() as td:
        hits = where_rows_self_test(td)
        got = {(row, rel) for row, rel, _ in hits}
        if ("plain_row", "CMakeLists.txt") not in got:
            print("  FAIL --where does not find a row registered in CMakeLists.txt"); fails += 1
        else:
            print("  ok    --where finds a row registered in CMakeLists.txt")
        if not any(row == "globbed_row" and rel.startswith("cmake") for row, rel in got):
            print("  FAIL --where does not find a row registered in a GLOB-included family file"); fails += 1
        else:
            print("  ok    --where finds a row registered in a GLOB-included family file")
        if any(row == "absent_row" for row, _ in got):
            print("  FAIL --where reports a row that is registered nowhere"); fails += 1
        else:
            print("  ok    --where reports an unregistered row as absent")
        # engine#2160 review: a miss and a typo must not read the same. `--where submodule_pin` is a MISS on the
        # exact rule and a PREFIX of the real row; the hint names the near rows without calling them a match.
        allr = all_rows(td, os.path.join(td, "CMakeLists.txt"))
        names = [r for r, _, _ in allr]
        if not any(n.startswith("globbed") for n in names):
            print("  FAIL the prefix hint has no rows to suggest"); fails += 1
        else:
            near = [n for n in names if n.startswith("globbed") and n != "globbed"]
            if not near:
                print("  FAIL a row that only STARTS WITH the query is not offered as a hint"); fails += 1
            else:
                print("  ok    a partial name is offered as a hint when the exact name is absent")
        if any(n.startswith("absent_row") for n in names):
            print("  FAIL a hint was invented for a name nothing starts with"); fails += 1
        else:
            print("  ok    and no hint is invented when nothing starts with the query")

    print(f"ci_targets_check --self-test: {fails} failure(s)")
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--where", metavar="ROWS", help="print where each ctest row is registered (file:line)")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv[1:])
    if a.self_test:
        return self_test()
    if a.where:
        wanted = [w.strip() for w in a.where.split(",") if w.strip()]
        root = os.path.abspath(a.root)
        hits = where_rows(root, os.path.join(root, "CMakeLists.txt"), wanted)
        found = {row for row, _, _ in hits}
        hits_all = all_rows(root, os.path.join(root, "CMakeLists.txt"))
        for row, rel, line in hits:
            print(f"{row}\t{rel}:{line}")
        for row in wanted:
            if row not in found:
                near = [r for r, _, _ in all_rows(root, os.path.join(root, "CMakeLists.txt"))
                        if r != row and r.startswith(row)]
                hint = ""
                if near:
                    where = {r: f"{rel}:{line}" for r, rel, line in hits_all}
                    hint = ("\n\t  but %d row(s) START WITH it -- the row name may be what is wrong, not the "
                            "registration: %s" % (len(near), ", ".join(f"{r} ({where.get(r, '?')})" for r in near)))
                print(f"{row}\tNOT REGISTERED (searched CMakeLists.txt and every file it include()s or globs){hint}")
        return 0 if found else 1
    if a.check:
        return check(a.root)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
