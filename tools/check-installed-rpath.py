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
    dead  nothing this ELF needs is there, so the entry only needs to be dropped - read as "nothing
          this sweep can see is here", not as "certified safe": `$LIB`/`$PLATFORM` are not expanded
          (see below), so a dead entry may still be live on a real loader

Both are still failures - neither is something a package should carry - but only LIVE changes what the
binary can start without. Exit codes are unchanged.

usage: check-installed-rpath.py [--self-test | ROOT...]     (default: ~/.local/share/1bit-*)
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


BUILD_TREE_RE = re.compile(r"^/work(?:-[A-Za-z0-9_.-]+)?(?:/|$)")   # /work, /work-c22, /work-c33 -- not /workshop
def resolved_kind(real):
    """What a LIVE entry resolves THROUGH: a build tree, a declared SDK, or something else.

    The distinction is the whole point of the LIVE count. `44 LIVE` is not actionable because it mixes
    a catastrophic case with a normal one (engine#1411, decided there):

      BUILD TREE    /work-c22/hrx-toolchain, ~/wt/build, /tmp/... -- exists because we built there.
                    Deleting or rebuilding it stops an installed package starting: this is the hazard.
      DECLARED SDK  /opt/rocm-therock/... -- the installed ROCm SDK. The entry is a dependency on a
                    system SDK, the same coupling every ROCm consumer here has. Making the package
                    `$ORIGIN`-only would vendor a SECOND libomp and ship an artifact that is not the
                    one that was tested, so it is reported and not failed.

    A function, not an inline test, so the classification has an arm that does not need a real ELF.
    """
    home = os.path.expanduser("~")
    # segment-aware: `/work` must match `/work/...` and `/work` itself, not `/workshop` (the arm below
    # caught exactly that on the first run of this function)
    if BUILD_TREE_RE.match(real) or real.startswith(("/tmp/", "/dev/shm/")) \
            or real.startswith(home + "/wt/") or real == home + "/wt" \
            or real.startswith(home + "/lb/") \
            or "/.cache/scratch-" in real:
        return "BUILD TREE"
    if real.startswith("/opt/rocm"):
        return "DECLARED SDK"
    return "OTHER"


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


def hazard_fails(build_tree_live, other_live):
    """True when a LIVE entry resolves somewhere this check can vouch for: a build tree, or a path it
    cannot classify. A DECLARED SDK entry does not fail -- it is a dependency on an installed SDK, and
    the decision to accept that is recorded on engine#1411. An UNCLASSIFIED path does fail: "I do not
    know what that is" may not pass a gate (the first version of this change let /work-c22 fall into
    OTHER and returned 0 with six build-tree entries still load-bearing -- measured)."""
    return bool(build_tree_live or other_live)


def self_test():
    """The classifier's three answers, and the one that makes the exit code mean something.

    No ELF is built: `resolved_kind` is pure, which is the reason it is a function. The real-box run is
    the other half of the evidence -- 6 LIVE entries resolve through /work-c22 and 36 through the
    installed ROCm SDK, so the same FAIL line used to conflate a hazard with a dependency.
    """
    home = os.path.expanduser("~")
    cases = [
        ("/work-c22/hrx-toolchain/lib/python3.14/site-packages/x/libomp.so", "BUILD TREE"),
        (home + "/wt/release-rc/build/lib/libggml.so", "BUILD TREE"),
        (home + "/lb/k74-ab/proof/pkg-B/1bit/hrx/llama/bin/libhrx.so.0", "BUILD TREE"),
        ("/tmp/scratch/lib/libomp.so", "BUILD TREE"),
        ("/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/lib/libomp.so",
         "DECLARED SDK"),
        ("/usr/lib/x86_64-linux-gnu/libz.so.1", "OTHER"),
        ("/opt/otherthing/lib/libx.so", "OTHER"),
    ]
    fails = 0
    for real, want in cases:
        got = resolved_kind(real)
        if got != want:
            print(f"FAIL {real} -> {got}, expected {want}")
            fails += 1
    # a prefix that merely LOOKS like a build tree must not match: /workshop is not /work
    if resolved_kind("/workshop/lib/libx.so") == "BUILD TREE":
        print("FAIL /workshop matched the /work prefix")
        fails += 1
    # ... and the exit policy: a build tree fails, an unknown path fails, a declared SDK does not
    for args, want in (((6, 0), True), ((0, 1), True), ((0, 0), False)):
        if hazard_fails(*args) != want:
            print(f"FAIL hazard_fails{args} -> {hazard_fails(*args)}, expected {want}")
            fails += 1
    print(f"check-installed-rpath self-test: {fails} failure(s)")
    return 1 if fails else 0


def main(argv):
    # the default glob also matches 1bit-weekly.log and friends, so it keeps only directories;
    # an explicit ROOT is taken as given and a typo in it still fails loudly below.
    if len(argv) > 1 and argv[1] == "--self-test":
        return self_test()
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
    total_build_tree = total_other_live = 0
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
                        print(f"      LIVE {soname} -> {real}  [{resolved_kind(real)}]")
                else:
                    print(f"  {os.path.relpath(path, root)}  {kind}={value}  (dead)")
            if len(bad) > MAX_SHOWN:
                print(f"  ... and {len(bad) - MAX_SHOWN} more")
            if len(bad) > len(live):
                print(f"  (dead = nothing this sweep can see is needed here; $LIB/$PLATFORM are not "
                      "expanded, so dead is not a promise that the entry can be dropped)")
            total_live += len(live)
            for _p, _e, hits in live:
                for _soname, _real in hits:
                    _k = resolved_kind(_real)
                    if _k == "BUILD TREE":
                        total_build_tree += 1
                    elif _k != "DECLARED SDK":
                        total_other_live += 1
        else:
            print(f"{root}: every RUNPATH is $ORIGIN")
    if total_bad:
        if hazard_fails(total_build_tree, total_other_live):
            where = (f"{total_build_tree} through a BUILD TREE (deleting or rebuilding that directory "
                     "stops the installed package starting)")
            if total_other_live:
                where += f", {total_other_live} through an UNCLASSIFIED absolute path (a dependency this "
                where += "check cannot vouch for)"
            print(f"FAIL {total_bad} offending RUNPATH entr{'y' if total_bad == 1 else 'ies'} across "
                  f"{total_elfs} ELF(s), {total_live} of them LIVE: {where}")
            return 1
        if total_live:
            print(f"WARN {total_bad} offending RUNPATH entr{'y' if total_bad == 1 else 'ies'} across "
                  f"{total_elfs} ELF(s); {total_live} LIVE, all through a DECLARED SDK ({total_other_live} "
                  "through something else) - a dependency on an installed SDK, not a path into a build "
                  "tree, so it does not stop the package starting")
        else:
            print(f"WARN {total_bad} offending RUNPATH entr{'y' if total_bad == 1 else 'ies'} across "
                  f"{total_elfs} ELF(s); none of them currently resolve anything")
        return 0
    print(f"PASS every RUNPATH in {len(roots)} installed tree(s) ({total_elfs} ELF(s)) is $ORIGIN")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
