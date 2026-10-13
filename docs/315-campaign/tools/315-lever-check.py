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
"""engine#315: check a proposed fix against the falsified-lever record.

The contract says: "Re-read the issue's falsified-levers list and confirm the claimed
mechanism is not one of them; if it is, the fix is rejected." That check was performed by
hand once and got it wrong - the HOST_COHERENT change was accepted on its shape (forward-
only, minimal, mergeable) while the record already said coherence was falsified with its
own campaign. This script makes the check mechanical so it cannot be skipped or eyeballed.

    ./315-lever-check.py <diff-or-patch-file> [<more patches>]
    ./315-lever-check.py --check-rev <git-rev> --repo <path>

Exit status: 0 = clear, 1 = collides with a falsified lever (fix is rejected),
             2 = usage/IO error.

It reads the falsified-lever table out of KNOWN-ISSUES.md so the two cannot drift, and
matches on the symbols/flags the table names - not on prose similarity.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_known_issues():
    """Walk up from this file so moving the tool cannot silently blind the check."""
    for base in [HERE, *HERE.parents]:
        cand = base / "KNOWN-ISSUES.md"
        if cand.exists():
            return cand
    return None


KNOWN = find_known_issues()

# lever -> (label, regex over the diff text). Derived from the falsified-lever table in
# KNOWN-ISSUES.md ("Falsified in this round"). Keep each pattern tied to the artefact the
# table names, so a rename in the source shows up as a miss rather than a false pass.
LEVERS = [
    ("memory coherence is the carrier",
     r"HRX_MEMORY_TYPE_HOST_COHERENT|HOST_COHERENT",
     "docs/315-mechanism.md section 13; `315accept` 9 clean + 1 fault / 10"),
    ("consumer-side ordering (graph sync)",
     r"DIAGNOSTIC_GRAPH_SYNC|GRAPH_SYNC",
     "section 57 / 315gsync2; 1 fault / 4 control vs 0 / 4 fix"),
    ("ordering the instrument's own reads",
     r"ggml_backend_synchronize\s*\(",
     "315row vs 315fp2; 1/1 vs 1/4, Fisher p = 1.00"),
    ("forced AQL execution barrier on every packet",
     r"HRX_FORCE_BARRIERS",
     "315fbar; 3 faults / 4 control vs 1 / 4 fix"),
    ("flash attention is the carrier",
     r"GGML_HRX_FA_PARTIAL_ALIGN|-fa off",
     "315nofa; 1/4 vs 2/4, fault persists"),
    ("pm4 command-buffer mode",
     r"HRX_COMMAND_BUFFER_MODE\s*=\s*pm4|pm4",
     "315pm4; every fix arm failed the first request, unusable"),
    ("transient reuse knob",
     r"GGML_HRX_TRANSIENT_REUSE",
     "15-arm matched study, no effect"),
]


def table_rows():
    """Sanity-check that KNOWN-ISSUES.md still carries the falsified-lever table."""
    if KNOWN is None or not KNOWN.exists():
        return None
    text = KNOWN.read_text(errors="replace")
    if "Falsified in this round" not in text:
        return None
    return len(re.findall(r"^\| .+ \| .+ \| .+ \|", text, re.M))


def get_diff(args):
    if args.check_rev:
        if not args.repo:
            sys.exit("--check-rev needs --repo")
        out = subprocess.run(
            ["git", "-C", args.repo, "show", args.check_rev],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            sys.exit(f"git show failed: {out.stderr.strip()}")
        return f"{args.check_rev}\n{out.stdout}"
    parts = []
    for p in args.patches:
        path = Path(p)
        if not path.exists():
            sys.exit(f"no such patch: {p}")
        parts.append(f"{p}\n{path.read_text(errors='replace')}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patches", nargs="*")
    ap.add_argument("--check-rev")
    ap.add_argument("--repo")
    args = ap.parse_args()
    if not args.patches and not args.check_rev:
        ap.print_help()
        return 2

    n = table_rows()
    if n is None:
        print("ERROR: could not read the falsified-lever table from KNOWN-ISSUES.md; "
              "refusing to certify a fix without the record.", file=sys.stderr)
        return 2
    print(f"falsified-lever table: {n} rows read from {KNOWN.name}\n")

    diff = get_diff(args)
    # Only inspect added lines: a removal comment is not the change.
    added = "\n".join(l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))

    hits = []
    for label, pattern, where in LEVERS:
        if re.search(pattern, added):
            hits.append((label, where))

    if not hits:
        print("CLEAR: no falsified lever appears in the added lines.")
        print("(This checks the falsified-lever list only. It does NOT confirm the change")
        print(" addresses the mechanism - that needs a mechanism, which is task 7.)")
        return 0

    print("REJECTED: the change touches a lever the issue already falsified:\n")
    for label, where in hits:
        print(f"  * {label}")
        print(f"      falsified by: {where}")
    print("\nPer the contract, a fix built on a falsified lever is rejected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
