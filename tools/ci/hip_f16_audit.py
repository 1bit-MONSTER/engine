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
"""hip_f16_audit.py -- the f16 properties `hrx_f16_narrowing` enforces, over the HIP add-on's kernels.

`tools/ci/hrx_f16_narrowing.py` walks the Loom corpus (`corpus.rglob("*.loom")`) and cannot see the add-on,
whose kernels live in the gpu-kernels repo as `.hip`/`.h`. A gpu-kernels pin bump therefore adds kernels that
no check reads -- engine#2406 put that on the ratchet's green line as *"audit those on a pin bump"*, which was
done by hand twice on 2026-11-14 (engine#2405's 1,210 new lines, gpu-kernels#76's `q41_decode.hip`). This is
that audit, mechanised.

Two properties, the same two the Loom ratchet counts:

  narrowing    -- an f32 value written into f16 (`__float2half*`, `(half)x`, `half(x)`)
  accumulator  -- an f16 accumulator (`half acc`, `__half acc`, `half sum`, `half_t acc`)

A narrowing is not automatically a defect: writing a block's `d` scale into a q4_0 block is the format, and
`set_rows_q4_0.hip` does exactly that. An **f16 accumulator is** the defect the Loom ratchet exists to prevent,
because it folds rounding into a sum that grows with the key count. So narrowing is reported per file and an
accumulator fails the audit.

Usage:
    hip_f16_audit.py --dir <gpu-kernels checkout>
    hip_f16_audit.py --dir … --json
    hip_f16_audit.py --self-test

Exit 0 no f16 accumulator (the table is still printed), 1 one or more, 2 the tree could not be read (which is
not a pass -- the same rule as the ratchet: `cannot tell` never reads as clean).
"""

import argparse
import json
import re
import sys
from pathlib import Path

KERNEL_SUBDIR = Path("addons/hip-hrx/kernels")

# `__float2half`, `__float2half_rn`, `(half)v`, `half(v)`. Deliberately not `half x = ...`, which declares.
NARROWING = re.compile(r"__float2half\w*\s*\(|\(\s*(?:__)?half\s*\)\s*[A-Za-z_(]|\b(?:__)?half\s*\(\s*[A-Za-z_]")
# an accumulator declared in f16: `half acc`, `__half acc =`, `half_t acc`, and `acc` declared as such.
ACCUMULATOR = re.compile(r"\b(?:__)?half(?:_t)?\s+(?:acc|accum|sum|total|dot|partial)\w*")


COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def audit_file(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Blank the comments rather than deleting them, so reported line counts stay true. Measured: the first run
    # of this tool flagged `kquant_tokens_q8.hip:181`, which is the sentence "sends half of them and keeps the
    # other half (by bit M of its lane ...)" -- prose, matched by the `half (` pattern.
    text = COMMENTS.sub(lambda m: " " * len(m.group(0)), raw)
    narrowings = [m.group(0).strip() for m in NARROWING.finditer(text)]
    accumulators = sorted(set(m.group(0).strip() for m in ACCUMULATOR.finditer(text)))
    return {"file": path.name, "narrowings": len(narrowings), "accumulators": accumulators,
            "lines": text.count("\n") + 1}


def audit(root):
    kernels = Path(root) / KERNEL_SUBDIR
    if not kernels.is_dir():
        return None
    rows = [audit_file(p) for p in sorted(kernels.glob("*.hip"))] + \
           [audit_file(p) for p in sorted(kernels.glob("*.h"))]
    return rows


def report(rows, as_json):
    if as_json:
        print(json.dumps(rows, indent=1))
    else:
        for row in rows:
            acc = ", ".join(row["accumulators"]) if row["accumulators"] else "-"
            print(f"  {row['file']:34} {row['lines']:5} lines  narrowings {row['narrowings']:3}  "
                  f"f16 accumulators {acc}")
    bad = [row for row in rows if row["accumulators"]]
    return bad


def self_test():
    import tempfile
    problems = []

    def chk(want, got, what):
        if want != got:
            problems.append(f"{what}: want {want!r}, got {got!r}")

    with tempfile.TemporaryDirectory() as tmp:
        kernels = Path(tmp) / KERNEL_SUBDIR
        kernels.mkdir(parents=True)
        # the format case: one narrowing, no accumulator (set_rows_q4_0.hip's block scale)
        (kernels / "format.hip").write_text(
            "const float d = vmax / -8.0f;\n"
            "*reinterpret_cast<uint16_t *>(dst) = __half_as_ushort(__float2half_rn(d));\n")
        # the clean case: f32 accumulation
        (kernels / "clean.hip").write_text(
            "float acc[4] = {0, 0, 0, 0};\nacc[k] = fmaf(d, q, acc[k]);\n")
        # the defect case
        (kernels / "f16acc.hip").write_text(
            "__half acc = 0;\nfor (int i = 0; i < n; ++i) acc = acc + __float2half(v[i]);\n")
        rows = audit(tmp)
        chk(3, len(rows), "three kernels audited")
        by = {row["file"]: row for row in rows}
        chk(1, by["format.hip"]["narrowings"], "the block scale is one narrowing")
        chk([], by["format.hip"]["accumulators"], "a narrowing alone is not an accumulator")
        chk(0, by["clean.hip"]["narrowings"], "fmaf accumulation narrows nothing")
        chk([], by["clean.hip"]["accumulators"], "a float accumulator is not f16")
        chk(["__half acc"], by["f16acc.hip"]["accumulators"], "the actual defect is found")
        bad = [row["file"] for row in rows if row["accumulators"]]
        chk(["f16acc.hip"], bad, "exactly the one defect file fails")

        # prose is not a narrowing: the first real run flagged a comment saying "half (by bit M ...)"
        (kernels / "comment.hip").write_text(
            "// sends half of them and keeps the other half (by bit M of its lane id)\n"
            "/* a block comment mentioning half (too) */\nfloat x = 0;\n")
        rows = audit(tmp)
        by = {row["file"]: row for row in rows}
        chk(0, by["comment.hip"]["narrowings"], "a comment mentioning 'half (' is not a narrowing")
        chk(4, len(rows), "the comment file is still audited")

        chk(None, audit("/nonexistent-path-xyz"), "an unreadable tree is None, not an empty pass")

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print("hip_f16_audit self-test: ok")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", help="a gpu-kernels checkout (must contain addons/hip-hrx/kernels)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir:
        print("hip_f16_audit: --dir is required (the gpu-kernels checkout to audit)", file=sys.stderr)
        return 2
    rows = audit(a.dir)
    if rows is None:
        print(f"hip_f16_audit: {Path(a.dir) / KERNEL_SUBDIR} is not a directory - cannot tell, "
              f"and that is not a pass", file=sys.stderr)
        return 2
    bad = report(rows, a.json)
    if bad:
        for row in bad:
            print(f"FAIL hip_f16_audit: {row['file']} accumulates in f16 ({', '.join(row['accumulators'])}) -- "
                  f"the Loom ratchet refuses this in a .loom body and it cannot see this file",
                  file=sys.stderr if a.json else sys.stdout)
        return 1
    if a.json:
        return 0            # --json is machine-readable: the summary would make the output unparseable
    print(f"hip_f16_audit: {len(rows)} kernel file(s), no f16 accumulator; "
          f"{sum(row['narrowings'] for row in rows)} narrowing(s), which are format writes unless a file says otherwise")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
