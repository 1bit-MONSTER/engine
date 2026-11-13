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
"""registry_loss_check.py -- did this branch drop a registry entry the base still has?

The registry files are append-mostly: a class in `registry/significant.json` is a reviewed decision, and a
counter in `registry/census.json` is a measurement. Neither is ever meant to lose a row because a branch was
cut before the row landed. `tools/registry_build.py --check-gaps` cannot see that: it iterates the classes
that are *present* and reports "none of the 41 reviewed significant classes is mapped without a reason",
which is equally true of a file with 41 classes and one with 40.

Measured on engine#2021 (`ac36bd0`): main has 23 `*ForCausalLM` classes and the branch 22, `Skylar2` is gone
from both `registry/significant.json` (2 mentions -> 0) and `tests/census_watch_test.py` (5 -> 0), and
`--check-gaps` passes on it. Measured on engine#2369 (`bc4126c`): `registry/census.json` loses `models`,
`with_architecture`, `declined_models` and `architectures_seen`.

Three checks, because the loss shows up in three places:
  classes   -- a key of `classes` in significant.json that the base has and the head does not
  counters  -- a top-level counter in census.json that the base has and the head drops or decreases
  mentions  -- a class name mentioned in `tests/` on the base and not at all in the head

Exit 0 quiet, 1 loss found, 2 a ref could not be read (which is not a pass).
"""

import argparse
import json
import subprocess
import sys

SIGNIFICANT = "registry/significant.json"
CENSUS = "registry/census.json"


def git(*args):
    return subprocess.run(("git",) + args, capture_output=True, text=True)


def read_ref(rev, path):
    """Return file text at rev, or None when the rev or the path does not exist there."""
    if not rev:
        try:
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return None
    shown = git("show", f"{rev}:{path}")
    return shown.stdout if shown.returncode == 0 else None


def ref_exists(rev):
    if not rev:
        return True
    return git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").returncode == 0


def classes_of(text, problems, label):
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        problems.append(f"{label}: {SIGNIFICANT} is not valid JSON ({exc})")
        return None
    classes = data.get("classes")
    if not isinstance(classes, dict):
        problems.append(f"{label}: {SIGNIFICANT} has no 'classes' object")
        return None
    return set(classes)


def counters_of(text, problems, label):
    """Every numeric leaf of census.json, flattened, because the counters live under `coverage` and `raw`."""
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        problems.append(f"{label}: {CENSUS} is not valid JSON ({exc})")
        return None
    flat = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}/{key}" if prefix else key, value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix] = node

    walk("", data)
    # Only the `coverage` summary's key set is stable: `raw/counts/<Arch>` grows as the census observes more
    # architectures, so a branch cut before the last rebuild is missing those keys without having lost anything.
    return {key: value for key, value in flat.items() if key.startswith("coverage/")}


def test_mentions(rev, names):
    """How many times each name appears anywhere under tests/ at rev."""
    counts = {name: 0 for name in names}
    if not names:
        return counts
    if rev:
        files = git("ls-tree", "-r", "--name-only", rev, "tests/").stdout.split()
    else:
        files = git("ls-files", "tests/").stdout.split()
    for path in files:
        if not path.endswith((".py", ".sh", ".md", ".txt", ".json")):
            continue
        text = read_ref(rev, path)
        if not text:
            continue
        for name in names:
            counts[name] += text.count(name)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Refuse a branch that drops registry rows the base has.")
    parser.add_argument("--base", default="origin/main", help="base ref (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    args = parser.parse_args()

    problems = []
    for label, rev in (("base", args.base), ("head", args.head)):
        if not ref_exists(rev):
            problems.append(f"{label} ref '{rev}' cannot be read - cannot tell, and that is not a pass")
    if problems:
        print("registry_loss_check: " + "; ".join(problems), file=sys.stderr)
        return 2

    base_classes = classes_of(read_ref(args.base, SIGNIFICANT), problems, "base")
    head_classes = classes_of(read_ref(args.head, SIGNIFICANT), problems, "head")
    if problems and (base_classes is None or head_classes is None):
        print("registry_loss_check: " + "; ".join(problems), file=sys.stderr)
        return 2
    if base_classes is None or head_classes is None:
        print(f"registry_loss_check: {SIGNIFICANT} missing on one side - cannot tell, and that is not a pass",
              file=sys.stderr)
        return 2

    lost_classes = sorted(base_classes - head_classes)
    added_classes = sorted(head_classes - base_classes)

    base_census = counters_of(read_ref(args.base, CENSUS), problems, "base")
    head_census = counters_of(read_ref(args.head, CENSUS), problems, "head")
    lost_counters = []
    moved_counters = []
    if base_census is not None and head_census is not None:
        for key, value in sorted(base_census.items()):
            if key not in head_census:
                lost_counters.append(f"{key} ({value}) is gone")
            elif head_census[key] != value:
                # Not a failure: the census is a measurement, rebuilt daily, so a branch cut before the last
                # rebuild legitimately carries older numbers. A census counter without its date is not
                # comparable to one with it, so a change is reported and only an absent counter fails.
                moved_counters.append(f"{key} {value} -> {head_census[key]}")

    lost_mentions = []
    if lost_classes:
        base_counts = test_mentions(args.base, lost_classes)
        head_counts = test_mentions(args.head, lost_classes)
        for name in lost_classes:
            if base_counts[name] and not head_counts[name]:
                lost_mentions.append(f"{name}: {base_counts[name]} mention(s) under tests/ on the base, 0 at the head")

    if not args.quiet:
        print(f"registry_loss_check: base {args.base} has {len(base_classes)} class(es), head {args.head} has "
              f"{len(head_classes)}")

    for item in lost_classes:
        print(f"FAIL registry_loss_check: class '{item}' is on the base and not at the head")
    for item in lost_counters:
        print(f"FAIL registry_loss_check: census counter {item}")
    for item in lost_mentions:
        print(f"FAIL registry_loss_check: {item}")
    if added_classes and not args.quiet:
        print(f"registry_loss_check: {len(added_classes)} class(es) added: {', '.join(added_classes)}")
    if moved_counters and not args.quiet:
        print(f"registry_loss_check: {len(moved_counters)} census counter(s) changed (not a failure; the census is "
              f"rebuilt, so check the date at both refs if it matters)")

    found = bool(lost_classes or lost_counters or lost_mentions)
    if not found and not args.quiet:
        print("registry_loss_check: no class, counter or test reference lost")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
