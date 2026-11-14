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

  narrowing    -- an f32 value written into f16 (`__float2half*`, `(half)x`, `half(x)`,
                  `(_Float16)x`). The cast arms require an identifier or `(` after the cast, so a cast
                  of a NUMERIC LITERAL (`(_Float16) 0.0f`) is not counted: that converts a constant, not
                  an f32 value
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
import os
import re
import sys
from pathlib import Path

KERNEL_SUBDIR = Path("addons/hip-hrx/kernels")

# `__float2half`, `__float2half_rn`, `(half)v`, `half(v)`. Deliberately not `half x = ...`, which declares.
NARROWING = re.compile(r"__float2half\w*\s*\(|\(\s*(?:(?:__)?half|_Float16)\s*\)\s*[A-Za-z_(]"   # (half)x, (_Float16)x
                       r"|\b(?:__)?half\s*\(\s*[A-Za-z_]")                                                   # half(x)
# an accumulator declared in f16: `half acc`, `__half acc =`, `half_t acc`, and `acc` declared as such.
# `half`, `__half`, `half_t`, `_Float16`, and the vector spellings these kernels actually use
# (`half16`/`half8`/`half2`, `typedef _Float16 half16 __attribute__((ext_vector_type(16)))`). Measured on
# gpu-kernels 96593660718d: the first version matched only the scalar spellings, and `glm_mla_decode.hip`
# declares its operands as `half16` -- an accumulator written that way would have been missed entirely.
ACCUMULATOR = re.compile(r"\b(?:(?:__)?half\w*|_Float16)\s+(?:acc|accum|sum|total|dot|partial)\w*")


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


def sha_matches(got, want):
    """True when `got` (a full HEAD sha) is the commit `want` names, which may be abbreviated.

    Compared prefix-wise in both directions for the same reason the fleet's verdict checker does: a 7-character sha
    in a PR body and a 40-character sha from git name the same commit, and refusing that pair would make the gate
    fire on the one case it is meant to allow — auditing a candidate pin copied out of a PR description.
    """
    if not got or not want:
        return False
    got, want = got.lower(), want.lower()
    n = min(len(got), len(want))
    return got[:n] == want[:n]


def pin_sha(config_root):
    """The gpu-kernels sha `config/gpu-kernels.config` pins, or None with the reason printed."""
    cfg = Path(config_root) / "config" / "gpu-kernels.config"
    if not cfg.is_file():
        print(f"hip_f16_audit: {cfg} does not exist - cannot tell what the pin is", file=sys.stderr)
        return None
    for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    print(f"hip_f16_audit: {cfg} has no sha= line - cannot tell what the pin is", file=sys.stderr)
    return None


def head_sha(tree):
    """The commit a gpu-kernels checkout is at, or None when it is not a git tree."""
    import subprocess
    got = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
    return got.stdout.strip() if got.returncode == 0 else None


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
        # the `_Float16` spelling the add-on actually uses, and the numeric-literal cast beside it
        (kernels / "f116.hip").write_text("_Float16 h = (_Float16) value[k];\n")
        (kernels / "f116lit.hip").write_text("_Float16 h = (_Float16) 0.0f;\n")
        rows = audit(tmp)
        chk(5, len(rows), "five kernels audited")
        by = {row["file"]: row for row in rows}
        chk(1, by["format.hip"]["narrowings"], "the block scale is one narrowing")
        chk([], by["format.hip"]["accumulators"], "a narrowing alone is not an accumulator")
        chk(0, by["clean.hip"]["narrowings"], "fmaf accumulation narrows nothing")
        # `_Float16` is the spelling the add-on's kernels actually use (engine#2405: every cast in its three new
        # kernels is `(_Float16)`), so a pattern that only knew `half`/`__half` reported zero of them.
        chk(1, by["f116.hip"]["narrowings"], "(_Float16)x is a narrowing")
        chk(0, by["f116lit.hip"]["narrowings"], "(_Float16) 0.0f is a constant conversion, not a value narrowing")
        chk([], by["clean.hip"]["accumulators"], "a float accumulator is not f16")
        chk(["__half acc"], by["f16acc.hip"]["accumulators"], "the actual defect is found")
        bad = [row["file"] for row in rows if row["accumulators"]]
        chk(["f16acc.hip"], bad, "exactly the one defect file fails")

        # the vector spelling: glm_mla_decode.hip declares its operands as `half16`, and an accumulator written
        # that way must fail too -- the first version of this regex matched only the scalar spellings.
        (kernels / "vecacc.hip").write_text(
            "typedef _Float16 half16 __attribute__((ext_vector_type(16)));\n"
            "half16 acc = (half16) 0.0f;\nacc = acc + v;\n")
        rows = audit(tmp)
        by = {row["file"]: row for row in rows}
        chk(["half16 acc"], by["vecacc.hip"]["accumulators"], "a half16 accumulator is found")
        # and an operand declared half16 is NOT an accumulator: the real kernel is correct
        (kernels / "vecoperand.hip").write_text(
            "float8 s = (float8) 0.0f;\nhalf16 a = load();\ns = wmma(a, b, s);\n")
        rows = audit(tmp)
        by = {row["file"]: row for row in rows}
        chk([], by["vecoperand.hip"]["accumulators"], "a half16 operand with a float8 accumulator is clean")

        # prose is not a narrowing: the first real run flagged a comment saying "half (by bit M ...)"
        (kernels / "comment.hip").write_text(
            "// sends half of them and keeps the other half (by bit M of its lane id)\n"
            "/* a block comment mentioning half (too) */\nfloat x = 0;\n")
        rows = audit(tmp)
        by = {row["file"]: row for row in rows}
        chk(0, by["comment.hip"]["narrowings"], "a comment mentioning 'half (' is not a narrowing")
        chk(8, len(rows), "every file in the tree is audited (4 originals + 2 vector cases + 2 _Float16 cases)")

        chk(None, audit("/nonexistent-path-xyz"), "an unreadable tree is None, not an empty pass")

        # the pin gate: an audit of a tree that is not the pinned commit is the false clean this tool exists for
        import subprocess as _sp
        root = Path(tmp) / "engine"
        (root / "config").mkdir(parents=True)
        tree = Path(tmp) / "kernels-repo"
        tree.mkdir()
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"]):
            _sp.run(cmd, cwd=tree, capture_output=True)
        at = _sp.run(["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True).stdout.strip()
        (root / "config" / "gpu-kernels.config").write_text(f"sha={at}\n")
        chk(at, pin_sha(root), "the pin is read from config/gpu-kernels.config")
        chk(at, head_sha(tree), "the tree's commit is read")
        (root / "config" / "gpu-kernels.config").write_text("sha=" + "f" * 40 + "\n")
        chk("f" * 40, pin_sha(root), "a different pin is read as different")
        chk(None, head_sha(Path(tmp) / "not-a-repo") if (Path(tmp) / "not-a-repo").mkdir() is None else None,
            "a non-git directory has no commit")
        chk(None, pin_sha(Path(tmp) / "nowhere"), "an engine checkout with no config/ is None, not a guess")

        # --expect-sha: the candidate-pin case, where the commit is in a PR rather than in config/ yet (engine#2411)
        chk(True, sha_matches(at, at), "a full sha matches itself")
        chk(True, sha_matches(at, at[:7]), "a 7-character sha from a PR names the same commit")
        chk(True, sha_matches(at, at.upper()), "case does not change the commit")
        chk(False, sha_matches(at, "f" * 40), "a different sha does not match")
        chk(False, sha_matches(None, at), "a tree with no commit never matches, not even a guess")
        chk(False, sha_matches(at, ""), "an empty expectation is not a wildcard")

    # --find (engine#2416): where the ctest row looks, and that "nothing found" is a skip, never a clean pass
    with tempfile.TemporaryDirectory() as tmp:
        engine = Path(tmp) / "engine"
        (engine / "third_party").mkdir(parents=True)
        explicit = Path(tmp) / "gk"
        explicit.mkdir()
        chk(None, find_tree(engine, env={}), "no $GPU_KERNELS_DIR and no submodule: nothing found")
        chk(str(explicit), find_tree(engine, env={"GPU_KERNELS_DIR": str(explicit)}), "$GPU_KERNELS_DIR is used")
        chk(None, find_tree(engine, env={"GPU_KERNELS_DIR": str(Path(tmp) / "missing")}),
            "a $GPU_KERNELS_DIR that is not a directory is not a tree")
        (engine / "third_party" / "gpu-kernels").mkdir()
        chk(str(engine / "third_party" / "gpu-kernels"), find_tree(engine, env={}), "the submodule path is used")
        chk(str(explicit), find_tree(engine, env={"GPU_KERNELS_DIR": str(explicit)}),
            "$GPU_KERNELS_DIR wins over the submodule")
        empty = Path(tmp) / "empty-engine"
        empty.mkdir()
        saved = os.environ.pop("GPU_KERNELS_DIR", None)
        try:
            chk(SKIPPED, main(["--find", str(empty)]), "--find with no tree exits 77 (skipped), not 0")
        finally:
            if saved is not None:
                os.environ["GPU_KERNELS_DIR"] = saved

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print("hip_f16_audit self-test: ok")
    return 0


SKIPPED = 77   # ctest SKIP_RETURN_CODE: the same convention as hrx_f16_narrowing's "skip: no .loom corpus"


def find_tree(engine_root, env=None):
    """The gpu-kernels checkout --find audits: $GPU_KERNELS_DIR, else <engine>/third_party/gpu-kernels. None when
    neither is a directory -- the engine has no gpu-kernels submodule, so a plain checkout has none (engine#2416)."""
    env = os.environ if env is None else env
    for candidate in (env.get("GPU_KERNELS_DIR"), str(Path(engine_root) / "third_party" / "gpu-kernels")):
        if candidate and Path(candidate).is_dir():
            return candidate
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", help="a gpu-kernels checkout (must contain addons/hip-hrx/kernels)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--config-root", help="an engine checkout: audit the tree the gpu-kernels pin names, and refuse "
                                          "a tree at any other commit (an audit of the wrong tree reads as clean)")
    ap.add_argument("--expect-sha", help="audit THIS commit (7..40 hex characters), for a candidate pin that has not "
                                         "merged yet; overrides --config-root. An audit of the wrong tree reads as "
                                         "clean, so the commit audited has to be asserted, not remembered")
    ap.add_argument("--find", metavar="ENGINE_ROOT",
                    help="the ctest row (engine#2416): audit $GPU_KERNELS_DIR or ENGINE_ROOT/third_party/gpu-kernels "
                         "against ENGINE_ROOT's pin; with neither, say the audit did NOT run and exit 77 (skipped)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.find:
        tree = find_tree(a.find)
        if tree is None:
            print("SKIPPED: hip_f16_audit found no gpu-kernels tree ($GPU_KERNELS_DIR unset, no "
                  f"{Path(a.find) / 'third_party' / 'gpu-kernels'}) - the add-on f16 audit did NOT run; set "
                  "GPU_KERNELS_DIR to a gpu-kernels checkout at the pin in config/gpu-kernels.config to run it")
            return SKIPPED
        a.dir = tree
        if not a.expect_sha:
            a.config_root = a.find
    if not a.dir:
        print("hip_f16_audit: --dir is required (the gpu-kernels checkout to audit)", file=sys.stderr)
        return 2
    if a.expect_sha or a.config_root:
        # engine#2406's instruction is "audit those on a pin bump": the audit is of THE PINNED TREE, and a checkout
        # at another commit is exactly the false clean this tool exists to prevent.
        #
        # --expect-sha exists because the pin is configured in the engine's working tree, and the audit a pin BUMP
        # needs happens BEFORE that bump merges: the candidate commit is in the PR, not yet in config/. Without an
        # explicit sha the gate refuses the one audit it was built for, and the escape it prints ("drop --config-root
        # and say which commit you audited") records nothing, which reopens the hole. engine#2411.
        if a.expect_sha:
            want, where = a.expect_sha.strip(), "--expect-sha"
            if not (7 <= len(want) <= 40) or any(c not in "0123456789abcdef" for c in want.lower()):
                print(f"hip_f16_audit: --expect-sha {a.expect_sha!r} is not a 7..40 character hex sha",
                      file=sys.stderr)
                return 2
        else:
            want = pin_sha(a.config_root)
            if want is None:
                return 2
            where = f"the pin in {Path(a.config_root) / 'config' / 'gpu-kernels.config'}"
        got = head_sha(a.dir)
        if got is None:
            print(f"hip_f16_audit: {a.dir} is not a git checkout - cannot tell which commit is audited",
                  file=sys.stderr)
            return 2
        if not sha_matches(got, want):
            print(f"hip_f16_audit: this tree is at {got[:12]}, {where} says {want[:12]} - audit the pinned tree "
                  f"or say which commit you audited with --expect-sha", file=sys.stderr)
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
