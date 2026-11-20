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


def symbols_used(fork: Path):
    """`loomc_*` names the fork's ggml-hrx sources mention.

    Comments are stripped first: a symbol named only in a comment is not a link
    dependency, and counting it would invent a breakage.
    """
    used = set()
    fork = fork_dir(fork)
    for f in sorted(fork.glob("*.cpp")) + sorted(fork.glob("*.h")):
        text = f.read_text(errors="replace")
        text = re.sub(r"//[^\n]*", " ", text)
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        used |= set(SYMBOL.findall(text))
    return used


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
