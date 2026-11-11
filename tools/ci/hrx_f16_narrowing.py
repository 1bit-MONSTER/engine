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

"""hrx_f16_narrowing.py -- keep the engine#1962 enumeration from drifting while it is being finished.

engine#1962 is "a dense MUL_MAT returns NaN or wrong values when an activation exceeds the f16 range".
Its repairs land on the *producers* (a plan records a 2^-6 prescale and the consumer scales back), and
the question the issue still holds open is: **which of the narrowing consumers are protected**. That
cannot be answered by reading the consumers -- the guard is not in them -- so the enumeration is only
worth anything if it cannot silently change underneath the person answering it.

This is the ratchet:

  * every .loom body that narrows f32 to f16 -- by the `fptrunc` op, the scalar `truncf` spelling, or an
    `f32 to f16` type line, which covers both the scalar and the `vector<Nxf32> to vector<Nxf16>` forms -- is
    listed in
    tools/ci/hrx_f16_narrowing.txt, and a body that appears without a line, or a line whose body is gone,
    fails the check -- the two arms a ratchet needs (a list that can only grow cannot see a rename,
    engine#1728's lesson);
  * the producer-side guard markers (`scale_log2`, `prescale`) are counted per body and reported, but
    **not** asserted as zero: moving the guard from the producer to the consumer is the issue's own
    option 2, and when that happens the count must change and the baseline must be updated deliberately.

The corpus is a submodule, so the check skips (rc 77) without third_party/llama.cpp checked out -- the
data file is in the engine tree, so the recorded set is visible in review even where the check cannot run.

usage: hrx_f16_narrowing.py --check [--root .]     rc 1 on drift, rc 77 without the corpus
       hrx_f16_narrowing.py --self-test            fixtures only, no corpus
       hrx_f16_narrowing.py --write                rewrite the baseline from the corpus
"""
import argparse
import pathlib
import sys

CORPUS = pathlib.Path("third_party/llama.cpp/ggml/src/ggml-hrx/kernel-corpus/kernels")
BASELINE = pathlib.Path("tools/ci/hrx_f16_narrowing.txt")
# engine#2219: `truncf` and `f32 to f16` between them miss `vector.fptrunc %v : vector<4xf32> to vector<4xf16>`
# -- the vector spelling contains neither `truncf` (it is `fptrunc`) nor `f32 to f16` (the types are wrapped in
# `vector<Nx...>`). At pin 2b7960bae that form is 132 of the 233 fptrunc lines and 27 of the 52 narrowing bodies,
# so the ratchet was listing 25 of 52 and reporting "none new, none stale". `fptrunc` is LLVM's narrowing op and
# covers both spellings; the other markers stay because a hand-written or vendor file may use them without it.
NARROW = ("fptrunc", "truncf", "f32 to f16")
GUARD = ("scale_log2", "prescale")


def baseline_paths(text):
    return {line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def narrowing_bodies(corpus):
    """Every .loom under the corpus that narrows f32 to f16, as corpus-relative POSIX paths."""
    found = set()
    for path in corpus.rglob("*.loom"):
        body = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in body for marker in NARROW):
            found.add(path.relative_to(corpus).as_posix())
    return found


def guarded(bodies, corpus):
    """Bodies that also carry a producer-side guard marker -- reported, never asserted."""
    out = set()
    for rel in bodies:
        body = (corpus / rel).read_text(encoding="utf-8", errors="replace")
        if any(marker in body for marker in GUARD):
            out.add(rel)
    return out


def drift(baseline, found):
    return sorted(found - baseline), sorted(baseline - found)


def check(root):
    corpus = root / CORPUS
    if not corpus.is_dir() or not any(corpus.rglob("*.loom")):
        print(f"skip: no .loom corpus at {CORPUS} (check out third_party/llama.cpp)", file=sys.stderr)
        return 77
    base_text = (root / BASELINE).read_text(encoding="utf-8")
    baseline = baseline_paths(base_text)
    found = narrowing_bodies(corpus)
    new, stale = drift(baseline, found)
    guard = guarded(found, corpus)
    for rel in new:
        print(f"hrx_f16_narrowing: NEW narrowing body {rel} -- it narrows an f32 value to f16 and the "
              f"baseline does not name it; add the line and say what feeds it (engine#1962)")
    for rel in stale:
        print(f"hrx_f16_narrowing: {rel} is in the baseline but no longer narrows -- drop the line, the "
              f"list should shrink")
    if new or stale:
        print(f"hrx_f16_narrowing: FAIL ({len(new)} new, {len(stale)} stale of {len(baseline)})")
        return 1
    print(f"hrx_f16_narrowing: {len(baseline)} narrowing body(ies), none new, none stale; "
          f"{len(guard)} of them carry a guard marker" +
          (f" ({', '.join(sorted(guard))})" if guard else " (the guard sits on the producer, engine#1962)"))
    return 0


def write(root):
    corpus = root / CORPUS
    if not corpus.is_dir():
        print(f"skip: no .loom corpus at {CORPUS}", file=sys.stderr)
        return 77
    head = "".join(line for line in (root / BASELINE).read_text(encoding="utf-8").splitlines(True)
                   if line.lstrip().startswith("#"))
    bodies = sorted(narrowing_bodies(corpus))
    (root / BASELINE).write_text(head + "\n".join(bodies) + "\n", encoding="utf-8")
    print(f"hrx_f16_narrowing: wrote {len(bodies)} body(ies) to {BASELINE}")
    return 0


def self_test():
    import tempfile
    failures = 0

    def check_that(ok, what):
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            failures += 1

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        corpus = root / CORPUS
        corpus.mkdir(parents=True)
        (root / BASELINE).parent.mkdir(parents=True, exist_ok=True)
        (corpus / "a.loom").write_text("  %x = scalar.fptrunc %y : f32 to f16\n")
        (corpus / "b.loom").write_text("  %x = scalar.constant 1.0 : f32\n")
        (root / BASELINE).write_text("# comment\n# another\na.loom\n")
        check_that(not drift(baseline_paths((root / BASELINE).read_text()), narrowing_bodies(corpus))[0],
                   "an unchanged corpus has no drift")
        (corpus / "c.loom").write_text("  %z = scalar.fptrunc %w : f32 to f16\n")
        new, stale = drift(baseline_paths((root / BASELINE).read_text()), narrowing_bodies(corpus))
        check_that(new == ["c.loom"] and not stale, "a new narrowing body is drift, and only it")
        (corpus / "c.loom").write_text("  %z = scalar.constant 0.0 : f32\n")
        (corpus / "a.loom").rename(corpus / "renamed.loom")
        new, stale = drift(baseline_paths((root / BASELINE).read_text()), narrowing_bodies(corpus))
        check_that(stale == ["a.loom"],
                   "a rename leaves a stale baseline line (the engine#1728 lesson)")
        (corpus / "renamed.loom").write_text("  %x = scalar.fptrunc %y : f32 to f16  # scale_log2\n")
        check_that(guarded({"renamed.loom"}, corpus) == {"renamed.loom"},
                   "a guard marker is reported, not denied")
        (corpus / "renamed.loom").write_text("  %x = scalar.fptrunc %y : f32 to f16\n")
        check_that(guarded({"renamed.loom"}, corpus) == set(),
                   "and an unguarded body reports no guard")
        # The comments in the baseline are not bodies.
        (root / BASELINE).write_text("# f32 to f16 is the marker\nrenamed.loom\n")
        check_that(baseline_paths((root / BASELINE).read_text()) == {"renamed.loom"},
                   "a '#' line in the baseline is not a path (the trailing-comment trap)")
        # A missing corpus is a skip, not a pass.
        empty = root / "empty"
        empty.mkdir()
        check_that(check(empty) == 77, "a missing corpus returns 77, not 0")

    print(f"hrx_f16_narrowing self-test: {'ok' if not failures else f'{failures} failure(s)'}")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    if args.self_test:
        return self_test()
    if args.write:
        return write(root)
    if args.check:
        return check(root)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
