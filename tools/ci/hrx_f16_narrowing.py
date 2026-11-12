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

  * every .loom body that narrows f32 to f16 (scalar `f32 to f16` or vector `vector<NxF32> to vector<NxF16>`)
    is listed in
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
import hashlib
import pathlib
import re
import sys

CORPUS = pathlib.Path("third_party/llama.cpp/ggml/src/ggml-hrx/kernel-corpus/kernels")
BASELINE = pathlib.Path("tools/ci/hrx_f16_narrowing.txt")
# engine#2186 item 1 / engine#2167: a THIRD shape loses a value to half precision and no truncation scan can see it --
# an f16 *accumulator* updated by a multiply. There is no `fptrunc` anywhere in it, because the accumulator already IS
# f16, so the body list above is blind to it by construction. Same mechanism as that list: a committed set of the files
# that have one, so a new one is drift.
ACC_BASELINE = pathlib.Path("tools/ci/hrx_f16_accumulators.txt")
ACC_INIT = re.compile(r"vector\.fragment<init>[^\n]*vector<(\d+)xf16>")
ACC_MUL = re.compile(r"vector\.mulf|template\.apply<@[^>]*wmma[^>]*>|vector\.(fma|contract|dot|outerproduct)\b|wmma\.(mma|load|store)")
# Measured (engine#2167): at 120 lines past the init, `mul_mat_id_f16_f16_wmma_core`'s four accumulators read as
# unaffected, because their update sits ~250 lines below in a long Q4_K/Q6 loop body. 250 finds them.
ACC_WINDOW = 250


def corpus_fingerprint(corpus):
    """sha256 over the corpus: sorted (relative path, bytes), so a moved file changes it too.

    A narrowing count is only comparable to another run if the tree it read is identified. Two
    checkouts of third_party/llama.cpp at different commits both answer "9 bodies" until one of them
    does not, and without this the only way to tell is to read the submodule sha by hand.
    """
    h = hashlib.sha256()
    n = 0
    for f in sorted(corpus.rglob("*.loom")):
        h.update(str(f.relative_to(corpus)).encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
        n += 1
    return f"{h.hexdigest()[:16]} ({n} .loom file(s))"
# The vector form is most of the corpus: `vector.fptrunc %x : vector<4xf32> to vector<4xf16>` contains neither
# "f32 to f16" nor "truncf", and at llama.cpp 2b7960bae 132 of the 233 narrowing lines are written that way.
# Matching the scalar spelling alone listed 25 bodies; the corpus has 51 (engine#2204's review). A bare `truncf` is
# not a narrowing to f16: `scalar.truncf %v : f32` rounds within f32 (unary_f32_apply.loom), and every truncf that
# does narrow names its f16 type, which the two patterns below already match (engine#2220's review).
NARROW = re.compile(r"f32 to f16|vector<[0-9x]*f32>\s+to\s+vector<[0-9x]*f16>")
GUARD = ("scale_log2", "prescale")


def baseline_entries(text):
    """`path` or `path count` per line -> {path: count or None}. A path-only line is still accepted, so an older
    baseline keeps working; a line with a count is the stronger form and is what `--write` emits."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        out[parts[0]] = int(parts[1]) if len(parts) > 1 else None
    return out


def baseline_paths(text):
    return set(baseline_entries(text))


def narrowing_lines(corpus):
    """{path: number of lines that narrow f32 to f16}. The body list alone cannot see a new narrowing written inside a
    body it already names -- 51 of the corpus's bodies are listed, so that is the common case, not the rare one."""
    found = {}
    for path in corpus.rglob("*.loom"):
        text = path.read_text(encoding="utf-8", errors="replace")
        n = sum(1 for line in text.splitlines() if NARROW.search(line))
        if n:
            found[path.relative_to(corpus).as_posix()] = n
    return found


def narrowing_bodies(corpus):
    """Every .loom under the corpus that narrows f32 to f16, as corpus-relative POSIX paths."""
    found = set()
    for path in corpus.rglob("*.loom"):
        body = path.read_text(encoding="utf-8", errors="replace")
        if NARROW.search(body):
            found.add(path.relative_to(corpus).as_posix())
    return found


# Every `fptrunc` narrows to something, so one whose type pair neither NARROW nor KNOWN_NON_F16 recognises is a
# spelling this tool does not know yet. It fails rather than passing silently, the way the vector form did
# (engine#2219): a new spelling is classified by extending one of the two patterns, never by omission.
FPTRUNC = re.compile(r"\bfptrunc\b")
KNOWN_NON_F16 = re.compile(r"\bbf16\b|xbf16>")


def unclassified_truncations(corpus):
    """`fptrunc` lines that are neither an f32 -> f16 narrowing nor a known non-f16 target, as (path:line, text)."""
    out = []
    for path in sorted(corpus.rglob("*.loom")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if FPTRUNC.search(line) and not NARROW.search(line) and not KNOWN_NON_F16.search(line):
                out.append((f"{path.relative_to(corpus).as_posix()}:{number}", line.strip()))
    return out


def accumulator_counts(corpus):
    """{path: number of mul-updated f16 fragment accumulators}; measured at the pin: 14 across 7 bodies."""
    found = {}
    for path in corpus.rglob("*.loom"):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        n = 0
        for i, line in enumerate(lines):
            if not ACC_INIT.search(line):
                continue
            if any(ACC_MUL.search(later) for later in lines[i:i + ACC_WINDOW]):
                n += 1
        if n:
            found[path.relative_to(corpus).as_posix()] = n
    return found


def accumulator_bodies(corpus):
    """Every .loom with an f16 fragment accumulator updated by a multiply or a WMMA/contract accumulate.

    The window matters more than the matcher: the update of `mul_mat_id_f16_f16_wmma_core`'s four `vector<8xf16>`
    accumulators sits ~250 lines below their `vector.fragment<init>`, so a short window reports them as unaffected --
    the bucket a reader skims past -- and engine#2167 is exactly about those four.
    """
    found = set()
    for path in corpus.rglob("*.loom"):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if not ACC_INIT.search(line):
                continue
            if any(ACC_MUL.search(later) for later in lines[i:i + ACC_WINDOW]):
                found.add(path.relative_to(corpus).as_posix())
                break
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


def check(root, corpus=None):
    corpus = pathlib.Path(corpus) if corpus else root / CORPUS
    if not corpus.is_dir() or not any(corpus.rglob("*.loom")):
        print(f"skip: no .loom corpus at {corpus} (check out third_party/llama.cpp, or pass --corpus)", file=sys.stderr)
        return 77
    # Printed before the verdict, not after it: a FAIL is when the identity of the tree matters most.
    print(f"CORPUS       {corpus.resolve()}")
    print(f"FINGERPRINT  {corpus_fingerprint(corpus)}")
    base_text = (root / BASELINE).read_text(encoding="utf-8")
    baseline = baseline_paths(base_text)
    found = narrowing_bodies(corpus)
    new, stale = drift(baseline, found)
    entries = baseline_entries(base_text)
    counts = narrowing_lines(corpus)
    grew, shrank = [], []
    for rel, want in entries.items():
        if want is None or rel not in counts:
            continue
        if counts[rel] > want:
            grew.append((rel, want, counts[rel]))
        elif counts[rel] < want:
            shrank.append((rel, want, counts[rel]))
    # engine#2186 item 1 / engine#2167: the accumulator class, which no truncation scan can see.
    acc_base_file = root / ACC_BASELINE
    if not acc_base_file.is_file():
        print(f"hrx_f16_narrowing: {ACC_BASELINE} is missing -- run `--write` once to record the accumulators this "
              f"corpus has (engine#2186)")
        return 1
    acc_baseline = baseline_paths(acc_base_file.read_text(encoding="utf-8"))
    acc_found = accumulator_bodies(corpus)
    acc_new, acc_stale = drift(acc_baseline, acc_found)
    acc_entries = baseline_entries(acc_base_file.read_text(encoding="utf-8"))
    acc_counts = accumulator_counts(corpus)
    acc_grew, acc_shrank = [], []
    for rel, want in acc_entries.items():
        if want is None or rel not in acc_counts:
            continue
        if acc_counts[rel] > want:
            acc_grew.append((rel, want, acc_counts[rel]))
        elif acc_counts[rel] < want:
            acc_shrank.append((rel, want, acc_counts[rel]))
    guard = guarded(found, corpus)
    unknown = unclassified_truncations(corpus)
    for where, text in unknown:
        print(f"hrx_f16_narrowing: UNCLASSIFIED fptrunc at {where}: {text} -- its type pair matches neither the f16 "
              f"patterns nor a known non-f16 target; extend NARROW or KNOWN_NON_F16 (engine#2219)")
    for rel in new:
        print(f"hrx_f16_narrowing: NEW narrowing body {rel} -- it narrows an f32 value to f16 and the "
              f"baseline does not name it; add the line and say what feeds it (engine#1962)")
    for rel in stale:
        print(f"hrx_f16_narrowing: {rel} is in the baseline but no longer narrows -- drop the line, the "
              f"list should shrink")
    for rel in acc_new:
        print(f"hrx_f16_narrowing: NEW f16 accumulator {rel} -- an f16 fragment is updated by a multiply, so the sum "
              f"overflows f16 long before any single input does and no fptrunc is involved; add the line and say what "
              f"the accumulator holds (engine#2186, engine#2167)")
    for rel in acc_stale:
        print(f"hrx_f16_narrowing: {rel} is in the accumulator baseline but has no mul-updated f16 fragment -- drop "
              f"the line, the list should shrink")
    for rel, want, got in grew:
        print(f"hrx_f16_narrowing: MORE narrowing LINES in {rel} ({want} -> {got}) -- the body is already listed, so "
              f"the body check alone would pass (engine#1962)")
    for rel, want, got in shrank:
        print(f"hrx_f16_narrowing: {rel} narrows fewer lines than the baseline ({want} -> {got}) -- drop the count")
    for rel, want, got in acc_grew:
        print(f"hrx_f16_narrowing: MORE f16 accumulators in {rel} ({want} -> {got}) -- a second accumulator inside a "
              f"known body (engine#2186, engine#2167)")
    for rel, want, got in acc_shrank:
        print(f"hrx_f16_narrowing: {rel} has fewer f16 accumulators ({want} -> {got}) -- drop the count")
    if new or stale or unknown or acc_new or acc_stale or grew or shrank or acc_grew or acc_shrank:
        print(f"hrx_f16_narrowing: FAIL ({len(new)} new, {len(stale)} stale of {len(baseline)}, "
              f"{len(unknown)} unclassified fptrunc; {len(acc_new)} new, {len(acc_stale)} stale of "
              f"{len(acc_baseline)} accumulator(s))")
        return 1
    print(f"hrx_f16_narrowing: {len(baseline)} narrowing body(ies), none new, none stale; "
          f"{len(acc_baseline)} f16 accumulator body(ies), none new, none stale; "
          f"{len(guard)} of the narrowing bodies carry a guard marker" +
          (f" ({', '.join(sorted(guard))})" if guard else " (the guard sits on the producer, engine#1962)"))
    return 0


def write(root, corpus=None):
    corpus = pathlib.Path(corpus) if corpus else root / CORPUS
    if not corpus.is_dir():
        print(f"skip: no .loom corpus at {corpus}", file=sys.stderr)
        return 77
    head = "".join(line for line in (root / BASELINE).read_text(encoding="utf-8").splitlines(True)
                   if line.lstrip().startswith("#"))
    counts = narrowing_lines(corpus)
    lines = [f"{rel}\t{n}" for rel, n in sorted(counts.items())]
    (root / BASELINE).write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    bodies = sorted(counts)
    acc_counts = accumulator_counts(corpus)
    accs = [f"{rel}\t{n}" for rel, n in sorted(acc_counts.items())]
    acc_head = "".join(line for line in (root / ACC_BASELINE).read_text(encoding="utf-8").splitlines(True)
                       if line.lstrip().startswith("#")) if (root / ACC_BASELINE).is_file() else ""
    (root / ACC_BASELINE).write_text(acc_head + "\n".join(accs) + "\n", encoding="utf-8")
    print(f"hrx_f16_narrowing: wrote {len(bodies)} body(ies) to {BASELINE} and {len(accs)} to {ACC_BASELINE}")
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
        (corpus / "v.loom").write_text("  %h = vector.fptrunc %v : vector<4xf32> to vector<4xf16>\n")
        (corpus / "w.loom").write_text("  %h = vector.fptrunc %v : vector<4xf32> to vector<4xbf16>\n")
        new, _ = drift(baseline_paths((root / BASELINE).read_text()), narrowing_bodies(corpus))
        check_that(new == ["c.loom", "v.loom"],
                   "a vector<4xf32> to vector<4xf16> narrowing is found, and a bf16 one is not")
        (corpus / "t.loom").write_text("  %t = scalar.truncf %v : f32\n  %u = vector.truncf %w : vector<4xf32>\n")
        new, _ = drift(baseline_paths((root / BASELINE).read_text()), narrowing_bodies(corpus))
        check_that("t.loom" not in new, "a truncf that stays in f32 is not a narrowing")
        check_that(unclassified_truncations(corpus) == [],
                   "an f16 and a bf16 fptrunc are both classified")
        (corpus / "x.loom").write_text("  %h = vector.fptrunc %v : vector<[4]xf32> to vector<[4]xf16>\n")
        check_that([w for w, _ in unclassified_truncations(corpus)] == ["x.loom:1"],
                   "an fptrunc spelling the patterns do not know fails as unclassified, not silently")
        (corpus / "x.loom").unlink()
        (corpus / "v.loom").unlink()
        (corpus / "w.loom").unlink()
        (corpus / "t.loom").unlink()
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
        # engine#2186 item 1 / engine#2167: the accumulator class. A mul-updated f16 fragment is found; an add-only one
        # is not; a WMMA/contract update counts as a multiply; and an update far below the init is still found -- the
        # last is the fixture that would have caught the 120-line window, which reported four real accumulators as
        # unaffected.
        accdir = root / "acc"
        accdir.mkdir()
        (accdir / "mul.loom").write_text(
            "  %init = vector.fragment<init> %z shape [%m, %n] : vector<8xf16>\n"
            "    %next = vector.mulf %acc, %k : vector<8xf16>\n")
        check_that(accumulator_bodies(accdir) == {"mul.loom"}, "a mul-updated f16 accumulator is found")
        (accdir / "add.loom").write_text(
            "  %init = vector.fragment<init> %z shape [%m, %n] : vector<16xf16>\n"
            "    %next = vector.addf %acc, %k : vector<16xf16>\n")
        check_that("add.loom" not in accumulator_bodies(accdir), "an add-only f16 accumulator is not this class")
        (accdir / "wmma.loom").write_text(
            "  %init = vector.fragment<init> %z shape [%m, %n] : vector<16xf16>\n"
            "    %next = template.apply<@ggml.some_wmma_core>(%acc) : (vector<16xf16>) -> (vector<16xf16>)\n")
        check_that("wmma.loom" in accumulator_bodies(accdir), "a WMMA/contract update counts as a multiply (engine#2167)")
        (accdir / "far.loom").write_text(
            "  %init = vector.fragment<init> %z shape [%m, %n] : vector<16xf16>\n"
            + "  // pad\n" * 200 +
            "    %next = vector.mulf %acc, %k : vector<16xf16>\n")
        check_that("far.loom" in accumulator_bodies(accdir), "an update 200 lines below the init is found")

    # The corpus identity: same bytes -> same fingerprint, an edit -> a different one, a move -> a
    # different one (the relative path is hashed), and the count it reports.
    with tempfile.TemporaryDirectory() as d:
        c = pathlib.Path(d)
        (c / "a.loom").write_text("one\n"); (c / "s").mkdir(); (c / "s" / "b.loom").write_text("two\n")
        fp = corpus_fingerprint(c)
        check_that(fp.endswith("(2 .loom file(s))"), "fingerprint: counts the .loom files it read")
        check_that(fp == corpus_fingerprint(c), "fingerprint: the same corpus twice is the same value")
        (c / "s" / "b.loom").write_text("two changed\n")
        check_that(fp != corpus_fingerprint(c), "fingerprint: a one-byte edit changes it")
        (c / "a.loom").rename(c / "s" / "a.loom")
        check_that(fp != corpus_fingerprint(c), "fingerprint: a move changes it too (the path is hashed)")
    with tempfile.TemporaryDirectory() as tmp:
        root2 = pathlib.Path(tmp)
        c2 = root2 / CORPUS
        c2.mkdir(parents=True)
        (c2 / "a.loom").write_text("  %x = scalar.fptrunc %y : f32 to f16\n")
        check_that(narrowing_lines(c2) == {"a.loom": 1}, "the narrowing-line count of a listed body is counted")
        (c2 / "a.loom").write_text("  %x = scalar.fptrunc %y : f32 to f16\n  %z = scalar.fptrunc %w : f32 to f16\n")
        check_that(narrowing_lines(c2) == {"a.loom": 2},
                   "a second narrowing INSIDE an already-listed body raises the count (the body check alone would pass)")
        (root2 / BASELINE).parent.mkdir(parents=True, exist_ok=True)
        (root2 / BASELINE).write_text("a.loom\t2\n")
        check_that(baseline_entries((root2 / BASELINE).read_text()) == {"a.loom": 2},
                   "a 'path count' baseline line parses to both")
        (root2 / BASELINE).write_text("a.loom\n")
        check_that(baseline_entries((root2 / BASELINE).read_text()) == {"a.loom": None},
                   "a path-only line still parses, so an older baseline keeps working")
        c2b = root2 / "acc2"
        c2b.mkdir()
        (c2b / "one.loom").write_text(
            "  %init = vector.fragment<init> %z shape [%m, %n] : vector<8xf16>\n"
            "    %next = vector.mulf %acc, %k : vector<8xf16>\n")
        check_that(accumulator_counts(c2b) == {"one.loom": 1}, "an accumulator count of one is counted")
        (c2b / "one.loom").write_text(
            "  %init = vector.fragment<init> %z shape [%m, %n] : vector<8xf16>\n"
            "    %next = vector.mulf %acc, %k : vector<8xf16>\n"
            "  %init2 = vector.fragment<init> %z2 shape [%m, %n] : vector<16xf16>\n"
            "    %next2 = vector.mulf %acc2, %k2 : vector<16xf16>\n")
        check_that(accumulator_counts(c2b) == {"one.loom": 2},
                   "a SECOND accumulator inside a listed body raises the count (the body check alone would pass)")

    print(f"hrx_f16_narrowing self-test: {'ok' if not failures else f'{failures} failure(s)'}")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--corpus", default=None,
                    help="the .loom corpus to scan (default: the pinned llama.cpp checkout's kernel-corpus)")
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    corpus = pathlib.Path(args.corpus).resolve() if args.corpus else None
    if args.self_test:
        return self_test()
    if args.write:
        # The baseline is a file in the tree it describes, so rewriting it from a corpus that lives
        # somewhere else records a corpus this checkout does not have -- and the next --check then
        # reports every one of its own bodies as "in the baseline but no longer narrows". Refused
        # rather than warned, because the damage is a committed baseline that no longer describes
        # this tree (found by doing exactly that with a two-line fixture corpus).
        if corpus is not None and root not in corpus.parents and corpus != root:
            print(f"hrx_f16_narrowing: refusing --write with --corpus {corpus} outside --root {root}: "
                  f"the baseline is written into the root and would describe a corpus this tree does not have",
                  file=sys.stderr)
            return 2
        return write(root, corpus)
    if args.check:
        return check(root, corpus)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
