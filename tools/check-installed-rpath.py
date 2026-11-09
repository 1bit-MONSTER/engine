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
# Copyright 2026 The 1bit-MONSTER Authors.
# SPDX-License-Identifier: Apache-2.0
"""Every ELF in an INSTALLED tree must carry `$ORIGIN` only (engine #1411).

The rule from the 2026-11-08 incident (an installed engine's RUNPATH into `~/wt/release-rc/work/build`
let a rebuild swap its libraries under two running services: coder-llm crash-looped 154x,
1bit-canonical 755x) is enforced for the *shipped* side by the release check (#1403). It cannot reach
copies that are already installed - that is what this sweeps.

An installed package finds its own libraries beside itself (`$ORIGIN`). Any other entry either points
into a tree a rebuild overwrites (the poison) or points at a toolchain directory - and whether that
directory is harmless depends on the day: engine #1615 found nine entries that "resolved nothing" on
2026-11-08 and resolved `libomp.so` on 2026-11-09, after the SDK was reinstalled, with no system copy to
fall back on. A stable count therefore says nothing, so each offending entry is now CLASSIFIED:

    LIVE  a library this ELF actually needs is found in that directory (the entry is load-bearing; the
          line names the library and the real path it resolves to, symlinks followed)
    dead  nothing this ELF needs is there, so the entry only needs to be dropped

Both are still failures - neither is something a package should carry - but only LIVE changes what the
binary can start without. Exit codes are unchanged.

usage: check-installed-rpath.py [ROOT...]     (default: ~/.local/share/1bit-*)
       exit 0 every RUNPATH is $ORIGIN; 1 some ELF is not; 2 a ROOT does not exist
"""
import glob
import os
import re
import subprocess
import sys

DYNAMIC = re.compile(r'\((?:RUNPATH|RPATH)\)[^\[]*\[([^\]]*)\]')
RPATH = re.compile(r'\(RPATH\)')
NEEDED = re.compile(r'\(NEEDED\)[^\[]*\[([^\]]*)\]')
MAX_SHOWN = 6


def runpaths(path):
    """((kind, value) for each DT_RUNPATH/DT_RPATH entry, the DT_NEEDED sonames) from one readelf pass."""
    out = subprocess.run(["readelf", "-d", path], capture_output=True, text=True).stdout
    return ([(m.group(0)[1:].split(")")[0], m.group(1)) for m in DYNAMIC.finditer(out)],
            NEEDED.findall(out))


def live_in(entry, elf_path, needed):
    """[(soname, realpath)] that a non-`$ORIGIN` entry would actually supply to this ELF.

    `$ORIGIN` is expanded so a relative entry can be checked too; `$LIB`/`$PLATFORM` are left
    unexpanded and therefore claim nothing (this is a sweep, not ld.so).
    """
    origin = os.path.dirname(os.path.abspath(elf_path))
    hits = []
    for directory in (e.replace("$ORIGIN", origin) for e in entry.split(":") if e):
        if "$" in directory:
            continue
        for soname in needed:
            candidate = os.path.join(directory, soname)
            if os.path.isfile(candidate):
                hits.append((soname, os.path.realpath(candidate)))
    return hits


def verdict(elvs):
    """The failures among [(path, kind, value, needed)], one per offending entry, deduplicated by value."""
    bad, seen = [], set()
    for path, kind, value, needed in elvs:
        entries = [e for e in value.split(":") if e]
        # `$ORIGIN` and anything under it (`$ORIGIN/xrt/lib`) is still "beside itself"; what counts is
        # whether the entry LEAVES the package. RPATH with escaping `..` is not chased: this is a sweep,
        # not a sandbox.
        offending = [e for e in entries if not (e == "$ORIGIN" or e.startswith("$ORIGIN/"))]
        if kind == "RPATH" and not offending:
            offending = ["(RPATH, not RUNPATH)"]  # a baked-in search path wins over LD_LIBRARY_PATH
        for entry in offending:
            if (path, entry) not in seen:
                seen.add((path, entry))
                bad.append((path, kind, entry, value, needed))
        if not entries and kind == "RPATH":
            bad.append((path, kind, "(empty RPATH)", value, needed))
    return bad


def main(argv):
    # the default glob also matches 1bit-weekly.log and friends, so it keeps only directories;
    # an explicit ROOT is taken as given and a typo in it still fails loudly below.
    roots = argv[1:] or sorted(d for d in glob.glob(os.path.expanduser("~/.local/share/1bit-*"))
                               if os.path.isdir(d))
    if not roots:
        print("check-installed-rpath: no installed 1bit package under ~/.local/share/1bit-* "
              "and no ROOT given", file=sys.stderr)
        return 2
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        print("check-installed-rpath: not a directory: " + ", ".join(missing), file=sys.stderr)
        return 2
    total_elfs = total_bad = total_live = 0
    for root in roots:
        elvs, bad = [], []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "rb") as f:
                        if f.read(4) != b"\x7fELF":
                            continue
                except OSError:
                    continue
                found, needed = runpaths(path)
                total_elfs += 1
                elvs.extend((path, kind, value, needed) for kind, value in found)
        bad = verdict(elvs)
        total_bad += len(bad)
        if bad:
            live = [(p, e, hits) for p, _k, e, _v, needed in bad
                    if (hits := live_in(e, p, needed))]
            print(f"{root}: {len(bad)} RUNPATH entr{'y' if len(bad) == 1 else 'ies'} not $ORIGIN "
                  f"({len(live)} LIVE, {len(bad) - len(live)} dead)")
            for path, kind, entry, value, needed in bad[:MAX_SHOWN]:
                hits = live_in(entry, path, needed)
                if hits:
                    print(f"  {os.path.relpath(path, root)}  {kind}={value}")
                    for soname, real in hits:
                        print(f"      LIVE {soname} -> {real}")
                else:
                    print(f"  {os.path.relpath(path, root)}  {kind}={value}  (dead)")
            if len(bad) > MAX_SHOWN:
                print(f"  ... and {len(bad) - MAX_SHOWN} more")
            total_live += len(live)
        else:
            print(f"{root}: every RUNPATH is $ORIGIN")
    if total_bad:
        load_bearing = (f", {total_live} of them LIVE: a library is found through the entry, so the "
                        "binary cannot start if that directory is rebuilt, moved or removed"
                        if total_live else ", none of them currently resolve anything")
        print(f"FAIL {total_bad} offending RUNPATH entr{'y' if total_bad == 1 else 'ies'} across "
              f"{total_elfs} ELF(s){load_bearing} - an installed package must find its libraries "
              "beside itself (a path into a build tree swaps the libraries under a running service "
              "when it rebuilds)")
        return 1
    print(f"PASS every RUNPATH in {len(roots)} installed tree(s) ({total_elfs} ELF(s)) is $ORIGIN")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
