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
"""engine#2916: the two pinned submodules have a PAIRED C API, so a one-sided bump breaks the build.

`ggml-hrx`'s Loom JIT calls `loomc_*` symbols that only `hrx-system` declares. The two
are separate gitlinks, so moving one without the other produces a compile error in a
third repository:

    ggml/src/ggml-hrx/loom-jit.cpp:223: error: use of undeclared identifier
        'loomc_link_index_builder_release'

Measured on 2026-11-20: the fork renamed `loomc_link_index_builder_free` ->
`loomc_link_index_builder_release` in hrx-system commit 771711d49be, while the engine
pinned an hrx-system *before* that commit. The current pin PAIR was consistent; only a
one-sided bump broke it, and nothing in the engine referenced the symbol, so the fix was
a lockstep bump of both gitlinks -- which took ten comments to establish.

This asks the pair, not each side. It needs no build and no device:

    tools/ci/loomc_pair_check.py --llama ~/wt/hl-engine/third_party/llama.cpp \
                                 --hrx-system ~/wt/hl-engine/third_party/hrx-system

Exit codes follow the repo convention: 0 every symbol used is declared, 1 a symbol the
fork uses is not declared by the pinned hrx-system (naming it), 2 cannot tell (a tree is
absent, so the answer would be a guess).
"""

import argparse
import re
import sys
from pathlib import Path

SYMBOL = re.compile(r"\bloomc_[A-Za-z0-9_]+\b")

# The last call's all-conditional uses, so the report can name them without
# changing the exit code: a guard that is always excluded is not a breakage.
CONDITIONAL = set()


def fork_dir(path: Path) -> Path:
    """The `ggml-hrx` source dir, given either the llama.cpp root or that dir itself.

    Same reason the header search is a glob rather than a fixed include path: a
    caller that passes the repository root and one that passes the module directory
    are both asking the same question, and a tool that answers only one of them
    reports "cannot tell" for the other -- which is how this check first failed on
    the very tree it was written for.
    """
    nested = path / "ggml/src/ggml-hrx"
    return nested if nested.is_dir() else path


def source_files(fork: Path):
    """Every `*.cpp`/`*.h` under the fork's ggml-hrx, not only the ones directly in it.

    engine#2916 review (mine): the first version globbed one directory, so the 216 sources
    in `runtime/`, `dispatch_registration/`, `graph/` and `hip/` were never read -- and
    `runtime/loom-kernel-jit.cpp` is exactly where a `loomc_` call would go next. A tree
    with a use there answered `ok` about a tree it had not looked at. Dotted and build
    directories are skipped: an in-tree build writes generated sources, and a census of
    those is a census of the compiler, not of the fork.
    """
    out = []
    for path in sorted(fork.rglob("*")):
        if path.suffix not in (".cpp", ".h") or not path.is_file():
            continue
        if any(part.startswith(".") or part.startswith("build") for part in path.relative_to(fork).parts[:-1]):
            continue
        out.append(path)
    return out


def symbols_used(fork: Path):
    """`loomc_*` names the fork's ggml-hrx sources mention.

    Comments are stripped first: a symbol named only in a comment is not a link
    dependency, and counting it would invent a breakage.

    Preprocessor conditionals are tracked too (engine#2916, found in use): a use
    inside `#if defined(SOMETHING)` is not a link dependency when SOMETHING is never
    defined. Measured here -- `loomc_amdgpu_emit_options_t` and
    `loomc_amdgpu_runtime_global_flags_t` appear ONLY inside
    `#if defined(LOOMC_AMDGPU_RUNTIME_GLOBALS_KNOWN)`, which nothing defines, so both
    were reported as missing until this. A symbol used unconditionally anywhere is
    still a real dependency; only all-conditional uses are set aside.

    `#else`/`#elif` are the other half (engine#2916 review, mine): a conditional with
    an alternate branch always compiles one of its sides, so a use in it is a
    dependency whatever the macro is. Treating the whole block as guarded hid
    `loomc_artifact_kind_t` -- used only in the `#else` of
    `#if defined(LOOMC_ARTIFACT_ROLE_COMPILE_REPORT)` at loom-jit.cpp:380, which is the
    side the compiler reads -- and made this check answer `ok` about a symbol it had
    stopped counting. So a use is set aside only when some enclosing guard has NO
    alternate branch; that is the shape `LOOMC_AMDGPU_RUNTIME_GLOBALS_KNOWN` really is.
    """
    fork = fork_dir(fork)
    unconditional, conditional = set(), set()
    for f in source_files(fork):
        text = f.read_text(errors="replace")
        text = re.sub(r"//[^\n]*", " ", text)
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        # One frame per open conditional: True once it has an `#else`/`#elif`, because
        # then one of its branches is compiled whatever the condition says.
        guards = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                directive = re.match(r"#\s*(\w+)", stripped)
                keyword = directive.group(1) if directive else ""
                if keyword in ("if", "ifdef", "ifndef"):
                    guards.append(False)
                elif keyword in ("elif", "else"):
                    if guards:
                        guards[-1] = True
                elif keyword == "endif":
                    if guards:
                        guards.pop()
                continue
            names = SYMBOL.findall(line)
            if not names:
                continue
            # Set a use aside only when an enclosing guard cannot compile any branch
            # (`False` = no alternate). A guard with an alternate always compiles one
            # side, so its uses are dependencies.
            if any(closed is False for closed in guards):
                conditional.update(names)
            else:
                unconditional.update(names)
    CONDITIONAL.clear()
    CONDITIONAL.update(conditional - unconditional)
    return unconditional


def symbols_declared(hrx: Path):
    """`loomc_*` names any header under the hrx-system tree declares.

    Every header, not one include directory: the C binding's headers moved once
    already, and a check that hard-codes their path reports the whole surface as
    missing the day they move again.
    """
    declared = set()
    for f in hrx.rglob("*.h"):
        declared |= set(SYMBOL.findall(f.read_text(errors="replace")))
    return declared


def check(fork: Path, hrx: Path):
    """-> (missing sorted, used count, declared count), or raises CannotTell."""
    for tree, what in ((fork, "llama.cpp"), (hrx, "hrx-system")):
        if not tree.is_dir():
            raise CannotTell("%s is not checked out at %s" % (what, tree))
    used = symbols_used(fork)
    if not used:
        raise CannotTell(
            "%s (or its ggml/src/ggml-hrx) has no *.cpp mentioning a loomc_ symbol, "
            "so this is not the fork's ggml-hrx" % fork)
    declared = symbols_declared(hrx)
    if not declared:
        raise CannotTell("%s has no *.h declaring a loomc_ symbol" % hrx)
    return sorted(used - declared), len(used), len(declared)


class CannotTell(Exception):
    pass


def self_test():
    """Arms for the parse, the pair, and each way the answer could be a guess."""
    import contextlib
    import io
    import tempfile

    bad = 0

    def chk(cond, what):
        nonlocal bad
        if not cond:
            print("FAIL %s" % what)
            bad += 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fork = root / "fork"
        hrx = root / "hrx"
        fork.mkdir()
        (hrx / "loom/binding/c/include").mkdir(parents=True)
        (fork / "loom-jit.cpp").write_text(
            "using B = LoomHandle<loomc_link_index_builder_t, loomc_link_index_builder_release>;\n")
        (hrx / "loom/binding/c/include/loomc.h").write_text(
            "void loomc_link_index_builder_release(...);\nvoid loomc_link_index_builder_t(...);\n")
        missing, used, declared = check(fork, hrx)
        chk(missing == [], "a matched pair reports nothing missing")
        chk(used == 2 and declared == 2, "both counts are the symbols actually present")

        # the exact #2916 shape: fork wants _release, the pinned side has _free
        (hrx / "loom/binding/c/include/loomc.h").write_text(
            "void loomc_link_index_builder_free(...);\nvoid loomc_link_index_builder_t(...);\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == ["loomc_link_index_builder_release"],
            "a one-sided rename is reported by name, not as a count")

        # a symbol named only in a comment is not a dependency
        (fork / "loom-jit.cpp").write_text(
            "// loomc_ghost_symbol is coming\nusing A = loomc_real_one;\n")
        (hrx / "loom/binding/c/include/loomc.h").write_text("void loomc_real_one(void);\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == [], "a symbol mentioned only in a comment is not counted as used")

        # a missing tree is cannot-tell, never "nothing missing"
        try:
            check(root / "absent", hrx)
            chk(False, "a missing fork tree must not pass")
        except CannotTell as e:
            chk("not checked out" in str(e), "a missing fork tree says what is missing")

        # a tree that is not the fork's ggml-hrx must not report a clean pair
        (fork / "loom-jit.cpp").write_text("int main(void) { return 0; }\n")
        try:
            check(fork, hrx)
            chk(False, "a tree with no loomc_ use must not pass")
        except CannotTell as e:
            chk("not the fork's ggml-hrx" in str(e), "an empty fork side says it cannot tell")

        # headers found anywhere, not only one include dir
        (hrx / "loom/binding/c/include/loomc.h").unlink()
        (hrx / "elsewhere").mkdir()
        (hrx / "elsewhere/loomc_moved.h").write_text("void loomc_real_one(void);\n")
        (fork / "loom-jit.cpp").write_text("using A = loomc_real_one;\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == [], "a declaration found in a moved header still counts")

        # a use inside an `#if defined(...)` that nothing defines is not a dependency
        (fork / "loom-jit.cpp").write_text(
            "using A = loomc_real_one;\n"
            "#if defined(NEVER_DEFINED_ANYWHERE)\n"
            "using G = loomc_guarded_symbol;\n"
            "#endif\n")
        (hrx / "loom/binding/c/include/loomc.h").write_text("void loomc_real_one(void);\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == [], "a guarded use adds no missing symbol")
        chk(CONDITIONAL == {"loomc_guarded_symbol"}, "and it is reported separately, by name")
        # …but an unconditional use of the same symbol is still real
        (fork / "loom-jit.cpp").write_text(
            "using A = loomc_real_one;\nusing G = loomc_guarded_symbol;\n"
            "#if defined(NEVER_DEFINED_ANYWHERE)\nusing H = loomc_guarded_symbol;\n#endif\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == ["loomc_guarded_symbol"],
            "the same symbol used unconditionally IS missing, guard or no guard")
        chk("loomc_guarded_symbol" not in CONDITIONAL,
            "…and it is NOT also listed as conditional: one unconditional use decides it")

        # an `#else` branch always compiles one of the two sides, so a use in it is a
        # dependency whatever the macro is (engine#2916 review, mine: this is
        # `loomc_artifact_kind_t` at loom-jit.cpp:380, the `#else` of
        # `#if defined(LOOMC_ARTIFACT_ROLE_COMPILE_REPORT)`, and the tool answered `ok`
        # about it while it had stopped counting it)
        (fork / "loom-jit.cpp").write_text(
            "using A = loomc_real_one;\n"
            "#if defined(NEVER_DEFINED_ANYWHERE)\nusing G = loomc_guarded_symbol;\n"
            "#else\nusing E = loomc_else_symbol;\n#endif\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == ["loomc_else_symbol"],
            "a use in an #else branch is missing when it is undeclared: that branch compiles")
        chk(CONDITIONAL == {"loomc_guarded_symbol"},
            "…and only the no-alternate side is set aside")

        # an `#elif` chain too: the elif branch compiles when the first condition does not
        (fork / "loom-jit.cpp").write_text(
            "using A = loomc_real_one;\n"
            "#if defined(NEVER_DEFINED_ANYWHERE)\nusing G = loomc_guarded_symbol;\n"
            "#elif defined(ALSO_NEVER)\nusing L = loomc_elif_symbol;\n#endif\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == ["loomc_elif_symbol"], "an #elif branch compiles too")
        chk(CONDITIONAL == {"loomc_guarded_symbol"}, "…and the branch with no alternate is still the guarded one")

        # and the guard has to END: a use after `#endif` is unconditional (a counter that
        # never decrements passed the old arms; measured, this arm fails without the pop)
        (fork / "loom-jit.cpp").write_text(
            "using A = loomc_real_one;\n"
            "#if defined(NEVER_DEFINED_ANYWHERE)\nusing G = loomc_guarded_symbol;\n#endif\n"
            "using T = loomc_after_the_guard;\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == ["loomc_after_the_guard"],
            "a use AFTER #endif is unconditional again, not swallowed by the guard")
        chk(CONDITIONAL == {"loomc_guarded_symbol"}, "…and the conditional list holds only the guarded one")

        # a use one directory down is still a use: the 216 sources in runtime/, graph/,
        # dispatch_registration/ and hip/ were invisible to the first version (engine#2916
        # review, mine), and `runtime/loom-kernel-jit.cpp` is where such a call would go next
        (fork / "runtime").mkdir(exist_ok=True)
        (fork / "runtime" / "loom-kernel-jit.cpp").write_text("using N = loomc_nested_symbol;\n")
        (fork / "loom-jit.cpp").write_text("using A = loomc_real_one;\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == ["loomc_nested_symbol"], "a use one directory down is read, not skipped")
        chk(CONDITIONAL == set(), "…and it is not mistaken for a guarded one")
        # …while an in-tree build's generated sources are not the fork's sources
        (fork / "build-hip").mkdir(exist_ok=True)
        (fork / "build-hip" / "generated.cpp").write_text("using B = loomc_generated_symbol;\n")
        missing, _, _ = check(fork, hrx)
        chk(missing == ["loomc_nested_symbol"], "a build directory's generated sources are skipped")
        (fork / "runtime" / "loom-kernel-jit.cpp").unlink()
        (fork / "build-hip" / "generated.cpp").unlink()

        # the llama.cpp ROOT is as valid an argument as the module dir: this check
        # first failed on the real tree because it accepted only the latter
        root_form = root / "cp"
        (root_form / "ggml/src/ggml-hrx").mkdir(parents=True)
        (root_form / "ggml/src/ggml-hrx/loom-jit.cpp").write_text("using A = loomc_real_one;\n")
        missing, _, _ = check(root_form, hrx)
        chk(missing == [], "the llama.cpp ROOT and the ggml-hrx dir both answer")

    print("%s" % ("ok" if bad == 0 else "%d FAIL" % bad))
    return 1 if bad else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llama", required=False, help="path to the pinned llama.cpp checkout")
    ap.add_argument("--hrx-system", required=False, help="path to the pinned hrx-system checkout")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.llama or not a.hrx_system:
        print("loomc_pair_check: cannot tell: both --llama and --hrx-system are required", file=sys.stderr)
        return 2
    try:
        missing, used, declared = check(Path(a.llama), Path(a.hrx_system))
    except CannotTell as e:
        print("loomc_pair_check: cannot tell: %s" % e, file=sys.stderr)
        return 2
    if CONDITIONAL:
        print("loomc_pair_check: %d symbol(s) are used only inside a preprocessing guard, so they are "
              "not counted as missing:" % len(CONDITIONAL))
        for c in sorted(CONDITIONAL):
            print("  %s" % c)
    if missing:
        print("loomc_pair_check: %d of %d loomc_* symbol(s) used by the fork are NOT declared by "
              "the pinned hrx-system:" % (len(missing), used), file=sys.stderr)
        for m in missing:
            print("  %s" % m, file=sys.stderr)
        print("  the two pins are out of step: move both gitlinks, not one "
              "(engine#2916)", file=sys.stderr)
        return 1
    print("loomc_pair_check: ok -- %d loomc_* symbol(s) used, all %d declared by the pinned hrx-system"
          % (used, declared))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
