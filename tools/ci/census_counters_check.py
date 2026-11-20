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
"""census_counters_check.py -- the census snapshot's own counters must match the registry it ships with.

Two of `registry/census.json`'s counters are a pure function of the repo tree, not of Hugging Face:

    declined_architectures  the classes in registry/significant.json with `"watch": "report"` and no backends in
                            registry/architectures.json (tools/census.py:125)
    declined_models         the census' own counts for those classes (tools/census.py:128)

Everything else in that file is a measurement of HF's population and cannot be recomputed without a sweep. These two
can, with no network at all -- which is why they are the two that drift: engine#2850 shipped 2026-11-17 with
`declined_architectures: 23` and `declined_models: 465` while the tree it committed alongside them computes 30 and
1014, and `registry/significant.json` had not changed since 2026-11-16 23:22.

WHY IT RUNS IN THE CENSUS JOB, NOT IN PR CI: a `"watch": "report"` entry added by any PR makes the committed counters
stale the moment it merges, so a PR gate would red-light exactly the PR that reviewed a class. The job that regenerates
the file is the place where "the counters describe the tree" is checkable and where it can be fixed in the same run.

Exit: 0 both counters match, 1 a counter is wrong (named with both numbers), 2 cannot tell (a file this needs is
missing or the snapshot has no counts to sum).
"""
import argparse
import json
import sys
from pathlib import Path

REG = Path("registry")


def load(root: Path):
    """(census, significant classes, architectures) or (None, reason)."""
    try:
        census = json.loads((root / REG / "census.json").read_text(encoding="utf-8"))
        sig = json.loads((root / REG / "significant.json").read_text(encoding="utf-8"))["classes"]
        archs = json.loads((root / REG / "architectures.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        return None, f"no {e.filename}"
    except (KeyError, ValueError) as e:
        return None, f"cannot read the registry ({e})"
    return (census, sig, archs), None


def declined_classes(sig, archs):
    """tools/census.py:124-125 exactly: `watch == "report"` and the class has no backend.

    The committed registry/architectures.json nests its classes under "architectures" (tools/census.py:80 unwraps it
    before the same predicate, as do tools/census_watch.py:249 and tools/registry_build.py:223); a flat lookup here
    found no entry for ANY class, so every `watch: report` class counted as declined and the row disagreed with a
    correct snapshot every day (engine#3041). Both shapes are accepted so a fixture and the committed file can differ.
    """
    table = archs.get("architectures") or archs
    return {hf for hf, entry in sig.items() if entry.get("watch") == "report" and not (table.get(hf) or {}).get("backends")}


def check(root: Path):
    """-> (failures, cannot_tell). Each failure is (counter, committed, computed)."""
    loaded, why = load(root)
    if why:
        return [], why
    census, sig, archs = loaded
    coverage = census.get("coverage") or {}
    counts = (census.get("raw") or {}).get("counts")
    declined = declined_classes(sig, archs)
    if counts is None:
        return [], "the snapshot carries no raw.counts, so declined_models cannot be recomputed"
    expected = {
        "declined_architectures": len(declined),
        "declined_models": sum(n for hf, n in counts.items() if hf in declined),
    }
    failures = []
    for counter, want in expected.items():
        got = coverage.get(counter)
        if got != want:
            failures.append((counter, got, want))
    return failures, None


def self_test() -> int:
    """Hermetic: three trees by hand, so no checkout and no network is needed."""
    import tempfile

    fails = 0

    def arm(name, ok, detail=""):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
        if not ok:
            fails += 1

    def build(tmp, coverage, sig, archs, counts):
        root = tmp / f"repo{len(list(tmp.iterdir()))}"
        (root / "registry").mkdir(parents=True)
        (root / "registry" / "census.json").write_text(
            json.dumps({"date": "2026-11-17", "coverage": coverage, "raw": {"counts": counts}}), encoding="utf-8")
        (root / "registry" / "significant.json").write_text(json.dumps({"classes": sig}), encoding="utf-8")
        # the committed file nests its classes; the fixture has to, or the row cannot see a lookup that forgets it
        (root / "registry" / "architectures.json").write_text(json.dumps({"architectures": archs}), encoding="utf-8")
        return root

    # ReportedButMappedForCausalLM is the arm for engine#3041: a class that IS reviewed-for-report but HAS a backend.
    # It is not declined, and only a lookup that unwraps architectures.json can tell -- a flat lookup counts it, which
    # is what the committed file did to every class for as long as the row existed.
    sig = {"ReviewedForCausalLM": {"watch": "report"}, "ReportedButMappedForCausalLM": {"watch": "report"}}
    archs = {"MappedForCausalLM": {"backends": ["hrx"]}, "ReportedButMappedForCausalLM": {"backends": ["hrx"]}}
    counts = {"ReviewedForCausalLM": 7, "MappedForCausalLM": 100, "ReportedButMappedForCausalLM": 5}

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, {"declined_architectures": 1, "declined_models": 7}, sig, archs, counts)
        got, why = check(root)
        arm("a consistent snapshot passes", got == [] and why is None, str(why or got))

        # A flat file (no "architectures" level) is still read: the tolerance is one expression, and a fixture that
        # predates the nesting should not start failing.
        root = build(tmp, {"declined_architectures": 1, "declined_models": 7}, sig, archs, counts)
        (root / "registry" / "architectures.json").write_text(json.dumps(archs), encoding="utf-8")
        got, why = check(root)
        arm("a flat architectures.json is read too", got == [] and why is None, str(why or got))

        root = build(tmp, {"declined_architectures": 23, "declined_models": 465}, sig, archs, counts)
        got, why = check(root)
        arm("a carried counter fails, both numbers named",
            [c for c, _g, _w in got] == ["declined_architectures", "declined_models"] and got[0][1:] == (23, 1), str(got))

        # Coverage that would be right if empty backends counted as MAPPED. The check must disagree, because
        # `a_get` is `bool(a and a.get("backends"))` -- presence is not mapping (tools/census.py:96).
        root = build(tmp, {"declined_architectures": 0, "declined_models": 0},
                     sig, {"ReviewedForCausalLM": {"backends": []}}, counts)
        got, why = check(root)
        arm("a class with EMPTY backends still counts as declined (a_get is truthiness, not presence)",
            [c for c, _g, _w in got] == ["declined_architectures", "declined_models"], str(got))

        root = build(tmp, {"declined_architectures": 0, "declined_models": 0}, {"ReviewedForCausalLM": {"watch": "alert"}}, archs, counts)
        got, why = check(root)
        arm("watch=alert is a claim, not a decline", got == [] and why is None, str(why or got))

        root = build(tmp, {"declined_architectures": 0, "declined_models": 0}, sig, archs, counts)
        (root / "registry" / "census.json").write_text(json.dumps({"coverage": {}}), encoding="utf-8")
        got, why = check(root)
        arm("no raw.counts is cannot-tell, not a pass", got == [] and why is not None, str(why))

        root = tmp / "empty"
        (root / "registry").mkdir(parents=True)
        got, why = check(root)
        arm("a missing registry file is cannot-tell", got == [] and why is not None, str(why))

    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".", help="the repo root (default: .)")
    ap.add_argument("--self-test", action="store_true", help="hermetic fixtures, no checkout needed")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.check:
        ap.print_usage()
        return 2
    failures, why = check(Path(a.root))
    if why:
        print(f"census-counters: cannot tell: {why}", file=sys.stderr)
        return 2
    for counter, got, want in failures:
        print(f"  {counter}: the snapshot says {got}, this tree computes {want}\n"
              f"     it is a function of registry/significant.json and registry/architectures.json, not of the sweep, so"
              f" the file was carried from another tree rather than regenerated (engine#2850)")
    if failures:
        return 1
    print("census-counters: both tree-derived counters match the registry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
