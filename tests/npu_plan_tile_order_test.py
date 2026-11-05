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
"""tools/npu_plan.py: the plan's `container.tile_order` is a measurement, not a constant (engine#1110).

`npu/plan/plan.h` refuses any tile order but "engine-1", and that is the only thing between a
container whose tiles are another order and a clean 200 with noise in it.  The plan generator used
to write "engine-1" for every directory it was pointed at, so the guard was green on exactly the
containers it exists for -- including a directory `scripts/repack_gguf.py` had just refused
(measured: repack said `matches NO layout`, then `npu_plan.py` wrote `tile_order="engine-1"` for the
same directory and the loader accepted it).

Now the repack stamps what it measured (also on its refusing path) and the generator reads that
stamp.  This pins the mapping and the two halves of it with no model, no numpy, no device:

  * a stamp saying `engine-1` claims engine-1 (and says it was measured);
  * `unverified` still claims engine-1 (the loader accepts nothing else) but says so out loud;
  * any other id -- `unknown` from "matches NO layout", or a named layout like `flm` -- is FATAL:
    no plan is written, so the repack's refusal cannot be undone by planning the directory anyway;
  * no stamp keeps the old silent default, because those directories predate the field.

usage: tests/npu_plan_tile_order_test.py        (stdlib only: no numpy, no gguf-py, no model, no device)
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, os.pardir, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("npu_plan", "tools/npu_plan.py")
# The stamp writer lives in the repack; imported so the id it stamps and the ids the planner accepts
# cannot drift apart (the same reason repack_tile_order_test imports the policy instead of copying it).
try:
    R = _load("repack_gguf", "scripts/repack_gguf.py")
except Exception:                                          # pragma: no cover - repack is stdlib too
    R = None

FAIL = []


def check(ok, what, why=""):
    print(("ok   " if ok else "FAIL ") + what + (("  [" + why + "]") if not ok and why else ""))
    if not ok:
        FAIL.append(what)


def stamp(d, text):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "repack-stamp.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return d


def claim_of(d):
    """(claim, note) from the stamp, or ('FATAL', message) when it refuses."""
    try:
        return P.tile_order_claim(d) + ("",)
    except SystemExit as e:
        return ("FATAL", "", str(e))


def main():
    tmp = tempfile.mkdtemp(prefix="npu-plan-tile-")
    try:
        base = os.path.join(tmp, "d")
        os.makedirs(base)

        # 1. Measured, and measured as ours: the claim is justified and the note says where it came from.
        d = stamp(os.path.join(tmp, "ours"),
                  "tile_order=engine-1\ntile_check=matches layout ours (168 tensor(s) exactly)\n")
        claim, note, _ = claim_of(d)
        check(claim == "engine-1", "a stamp that measured ours claims engine-1", claim)
        check("measured at repack" in note and "168" in note, "and says it was measured, quoting the verdict", note)

        # 2. Nothing comparable: the loader still only accepts engine-1, so claim it -- but never silently.
        d = stamp(os.path.join(tmp, "unverified"), "tile_order=unverified\ntile_check=tile check unavailable (no numpy)\n")
        claim, note, _ = claim_of(d)
        check(claim == "engine-1", "unverified still claims engine-1 (it is the only id the loader takes)", claim)
        check("NOT measured" in note, "but the note says nothing checked the tiles", note)

        # 3. The measured-other case, both spellings of it: a named layout, and "matches NO layout".
        #    These are the two verdict lines that reach the repack's refusing path.
        for order, line in (("flm", "matches layout flm (worst max|err| 2.474e+00)"),
                            ("unknown", "matches NO layout: closest is flm with worst max|err| 2.474e+00")):
            d = stamp(os.path.join(tmp, "bad-" + order), f"tile_order={order}\ntile_check={line}\n")
            claim, note, msg = claim_of(d)
            check(claim == "FATAL", f"tile_order={order} is fatal: the plan may not claim engine-1", claim)
            check(order in msg and "engine-1" in msg, f"and the refusal names {order!r} and what it would have claimed", msg)

        # 4. A stamp written BEFORE the field existed is no evidence, not evidence against -- and this is
        #    not hypothetical: both repack-cache directories on the box carry verbatim this stamp, so getting
        #    it wrong means refusing two directories that work today (it did, before the sweep caught it).
        old = stamp(os.path.join(tmp, "pre-field"),
                    "converter_path=/home/bcloud/wt/npu-rt-753-pr/third_party/FLM_Q4NX_Converter\n"
                    "converter_commit=81be119e\narch=qwen3\nmodel_type=qwen3\nq4nx_default_tensor_type=Q4_1\n"
                    "repack_script=/home/bcloud/wt/onebp-zyphra/scripts/repack_gguf.py\n")
        claim, note, _ = claim_of(old)
        check(claim == "engine-1", "a pre-field stamp (no tile_order key) keeps the default", claim)
        check(note == "", "and is silent: it says nothing about the tiles, so there is nothing to report", note)

        # 5. No stamp: unchanged behaviour, and quiet (directories that predate the field).
        os.makedirs(os.path.join(tmp, "nostamp"))
        claim, note, _ = claim_of(os.path.join(tmp, "nostamp"))
        check(claim == "engine-1" and note == "", "a directory with no stamp keeps the old silent default", claim + "/" + note)

        # 6. The id the repack stamps on its refusing path is one the planner refuses.  This is the
        #    joint that was open: the two files must agree on the vocabulary or the stamp is wasted.
        if R is not None:
            for line, want in (("matches layout flm (worst max|err| 2.474e+00)", "flm"),
                               ("matches NO layout: closest is flm with worst max|err| 2.474e+00", "unknown")):
                got = R.tile_order_id(line)
                check(got == want, f"repack stamps {want!r} for the line it refused on", got)
                d = stamp(os.path.join(tmp, "joint-" + want), f"tile_order={got}\ntile_check={line}\n")
                check(claim_of(d)[0] == "FATAL", f"and the planner refuses a stamp of {got!r} (the joint)", "")

        # 7. A real plan is not written for the fatal case: `generate` has to raise, not just warn.
        #    Synthesised minimally so this stays stdlib: the refusal happens before any tensor is read.
        bad = stamp(os.path.join(tmp, "gen-fatal"), "tile_order=unknown\ntile_check=matches NO layout\n")
        with open(os.path.join(bad, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"hidden_size": 1536, "num_hidden_layers": 2, "intermediate_size": 8960,
                       "model_type": "zamba2"}, f)
        raised = ""
        try:
            P.generate(bad, bad)
        except SystemExit as e:
            raised = str(e)
        check("tile_order='unknown'" in raised, "generate() raises before building a plan for a measured-other dir", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"npu_plan_tile_order_test: {len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
