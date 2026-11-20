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
"""bench_stamp.py -- say where a build tree came from, and refuse to compare it blind.

engine#2858: `~/.cache/hrx-build/bench` is built by CI on this box and then used to report device
throughput. On 11-18 that tree reported IQ1_S prefill at **11.22 tok/s** where four independent local
builds of the same source on the same box measured **56.5-56.6**, with identical node placement, PPL and
kernel-name set -- and nothing in the tree said which lowering had produced its code objects, so the
number could not be told apart from a real one.

So a build tree carries a stamp, and a comparison asks the stamp first:

    bench_stamp.py write --tree DIR            # what CI does after a build
    bench_stamp.py check --tree DIR --expect-engine <sha> --expect-host <name>
      rc 0  every expected value matches, and the lowering is known
      rc 1  a value differs (the reason names the field, the stamp's value and the expected one)
      rc 2  cannot tell: no stamp, or a field the stamp itself records as unknown

`loom` is the lowering revision (the IREE/loom that emitted the HSA code objects). A tree whose stamp
says `loom=unknown` is NOT comparable with a local build, because that is exactly the variable nobody has
recorded -- so `check` answers rc 2 rather than 0. A check that cannot look is not a pass.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys

STAMP = "bench-stamp.txt"
FIELDS = ("engine", "llama_cpp", "hrx_system", "loom", "host", "built_at")


def git_rev(path, *args):
    """The revision at `path`, or "" when it is not its OWN checkout.

    `git -C <missing dir>` walks UP to the enclosing repository, so an uninitialised submodule used to
    report the parent checkout's sha -- a stamp that names the engine three times and looks complete. The
    toplevel has to be the path itself.
    """
    path = pathlib.Path(path)
    if not path.is_dir():
        return ""
    top = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0 or pathlib.Path(top.stdout.strip()).resolve() != path.resolve():
        return ""
    p = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


HSACO_COMMENT = re.compile(r"\(\s*(https?://\S+?)\s+([0-9a-f]{7,40})\s*\)")


def parse_comment(text):
    """(url, sha) from a `.comment` dump, or None. Split out from the readelf call so the rule has an arm."""
    m = HSACO_COMMENT.search(text.replace("\n", " "))
    return (m.group(1), m.group(2)) if m else None


def hsaco_rev(tree, limit=40):
    """The lowering that emitted this tree's code objects, read from the objects themselves.

    `LOOM_REV` was read in three places and written by nothing, so every stamp said `loom=unknown` and `check` could
    never answer 0 -- the decisive field was the one nobody recorded. The objects carry it: every .hsaco has the
    emitting compiler in its `.comment` section (`AMD clang version 24.0.0git (… e095f1b3becf)`). Reading one costs
    a `readelf`, and `unknown` then means what it should: this tree has no code objects at all.
    """
    seen = {}
    for path in pathlib.Path(tree).rglob("*.hsaco"):
        p = subprocess.run(["readelf", "-p", ".comment", str(path)], capture_output=True, text=True)
        if p.returncode != 0:
            continue
        found = parse_comment(p.stdout)
        if found:
            seen.setdefault(found[1][:12], {"url": found[0], "objects": 0})["objects"] += 1
        if len(seen) > 1 or sum(v["objects"] for v in seen.values()) >= limit:
            break
    if len(seen) == 1:
        rev, info = next(iter(seen.items()))
        return rev, info["url"]
    if len(seen) > 1:
        # more than one lowering in one tree is worth saying out loud, not smoothing over
        return "mixed:" + ",".join(sorted(seen)), ""
    return "unknown", ""


def read_stamp(tree):
    p = pathlib.Path(tree) / STAMP
    if not p.is_file():
        return None
    out = {}
    for line in p.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def write_stamp(tree, root=".", loom=None, host=None, engine=None, llama_cpp=None, hrx_system=None):
    """Write the stamp for a build tree. Missing facts are written as `unknown`, never guessed.

    The three `*` overrides exist because a CI checkout is not the tree a lane has: actions/checkout gives a PR run
    a synthetic merge commit, so deriving `engine` from git inside the job stamps a sha that no lane's `git
    rev-parse HEAD` will ever equal -- and then every `check --expect-engine` answers rc 1 on a tree that is exactly
    the pin, which is a stamp that reports noise instead of provenance.
    """
    root = pathlib.Path(root)
    loom_url = ""
    if not loom:
        loom, loom_url = hsaco_rev(tree)
    stamp = {
        "engine": engine or git_rev(root, "rev-parse", "--short=12", "HEAD") or "unknown",
        "llama_cpp": llama_cpp or git_rev(root / "third_party" / "llama.cpp", "rev-parse", "--short=12", "HEAD") or "unknown",
        "hrx_system": hrx_system or git_rev(root / "third_party" / "hrx-system", "rev-parse", "--short=12", "HEAD") or "unknown",
        "loom": loom or os.environ.get("LOOM_REV") or "unknown",
        "host": host or os.uname().nodename,
        "built_at": subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip(),
    }
    if loom_url:
        stamp["loom_url"] = loom_url
    p = pathlib.Path(tree) / STAMP
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{k}={stamp[k]}\n" for k in FIELDS + (("loom_url",) if loom_url else ())))
    return stamp


def check(tree, expect):
    """(rc, [reasons]). rc 0 comparable, 1 a value differs, 2 cannot tell."""
    stamp = read_stamp(tree)
    if stamp is None:
        return 2, [f"{tree}: no {STAMP}, so this tree cannot say where it came from (a build tree is not a "
                   f"measurement until it can)"]
    reasons, differs = [], False
    for key, want in expect.items():
        if not want:
            continue
        got = stamp.get(key, "")
        # shas are compared as prefixes: the stamp writes 12 characters, `git rev-parse HEAD` gives 40, and a
        # comparison that says "the field moved" because of the length is the noise rc 1 must never become
        same = got == want
        # …but a prefix only says anything if it is long enough to name a revision. `llama_cpp=8` is a prefix of a
        # pin beginning `8d26add2c…`, and on 2026-11-20 that satisfied the fork-ctest ratchet (engine#2761) well
        # enough for it to report "18 known failure(s), no new ones" about a build tree nobody could identify.
        # Seven is git's own floor: a shorter short-sha does not resolve. bench_stamp.py writes 12.
        if (not same and key != "host" and got and len(got) >= 7
                and (got.startswith(want) or want.startswith(got))):
            same = True
        if not same:
            differs = True
            reasons.append(f"{key}: the tree says {got or '(absent)'}, this comparison expects {want}")
    # a moved field is the thing you CAN tell, so it wins: reporting rc 2 ("cannot tell") for a tree whose engine
    # clearly changed sends a caller down the "no information" branch when it had information
    if differs:
        return 1, reasons
    loom = stamp.get("loom", "unknown")
    if loom.startswith("unknown") or loom.startswith("mixed:"):
        # `mixed:` is a tree whose code objects came from more than one compiler -- a rebuild that reused objects.
        # It must not answer 0 either: "comparable" names ONE lowering, and this tree does not have one.
        why = ("records it as unknown" if loom.startswith("unknown")
               else "holds objects from more than one lowering (" + loom[6:] + ")")
        return 2, [f"loom: the stamp {why}, so this tree's numbers cannot be told apart from a differently-lowered "
                   f"build's (engine#2858: 11.22 vs 56.5 tok/s on one box)"]
    return 0, []


def self_test():
    """Arms: a matching stamp passes; a moved checkout, a different host and an unknown lowering do not."""
    import tempfile

    bad = 0

    def chk(cond, what):
        nonlocal bad
        if not cond:
            print(f"FAIL {what}")
            bad += 1

    with tempfile.TemporaryDirectory() as td:
        tree = pathlib.Path(td) / "bench"
        tree.mkdir()
        write_stamp(tree, root=td, loom="c0ffee", host="box-a")
        stamped = read_stamp(tree)["engine"]

        chk(check(tree, {"host": "box-a"})[0] == 0, "the matching host passes")
        rc, why = check(tree, {"host": "box-b"})
        chk(rc == 1 and any("host" in r for r in why), "a different host is rc 1 and names the field")
        rc, why = check(tree, {"engine": "deadbeef"})
        chk(rc == 1 and any("engine" in r for r in why), "a different engine sha is rc 1 and names the field")

        # a 12-character stamp matches a full 40-character expectation: `git rev-parse HEAD` gives 40, the stamp
        # writes 12, and rc 1 for a length difference is the noise rc 1 must never become
        chk(check(tree, {"engine": stamped + "0" * 28})[0] == 0,
            "a short stamp matches a full-length expectation")
        chk(check(tree, {"engine": stamped[:-1] + ("0" if stamped[-1] != "0" else "1")})[0] == 1,
            "one character different is still a moved field")

        # an explicit sha wins over the checkout's HEAD: CI's merge commit is not a sha any lane can name
        write_stamp(tree, root=td, loom="c0ffee", host="box-a", engine="111111111111")
        chk(check(tree, {"engine": "111111111111"})[0] == 0, "an explicit --engine override is what the stamp records")
        chk(check(tree, {"engine": stamped})[0] == 1 or stamped == "111111111111",
            "and it does NOT silently fall back to the checkout's HEAD")

        # the lowering is read from the code objects, so "comparable" is reachable at all
        line = ("String dump of section '.comment':\n"
                "  [     1]  AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git e095f1b3becf0c1bfdc5e7647003d60976a2460b)\n"
                "  [    71]  Linker: AMD LLD 24.0.0 (https://github.com/ROCm/llvm-project.git e095f1b3becf0c1bfdc5e7647003d60976a2460b)\n")
        parsed = parse_comment(line)
        chk(parsed is not None and parsed[1].startswith("e095f1b3"),
            "the emitting compiler is read out of a .comment dump")
        chk(hsaco_rev(tree)[0] == "unknown", "a tree with no code objects records unknown, which is then true")
        chk(parse_comment("Linker: AMD LLD, no parenthesised revision here") is None,
            "a comment with no revision is not a revision")

        # a tree whose lowering is unrecorded cannot be compared, and that is rc 2, not 0
        write_stamp(tree, root=td, loom="unknown", host="box-a")
        rc, why = check(tree, {"host": "box-a"})
        chk(rc == 2 and any("loom" in r for r in why),
            "a tree whose lowering is unknown is 'cannot tell', not a pass")
        # but a moved field is something you CAN tell, so it wins over the unknown lowering
        rc, why = check(tree, {"engine": "deadbeef"})
        chk(rc == 1 and any("engine" in r for r in why),
            "a moved engine is rc 1 even when the lowering is unknown (knowable beats cannot-tell)")

        # a tree holding objects from two compilers is not comparable either, and must not pass as one lowering
        (tree / STAMP).write_text("engine=abc\nloom=mixed:aaa,bbb\nhost=box-a\n")
        rc, why = check(tree, {"host": "box-a"})
        chk(rc == 2 and any("more than one lowering" in r for r in why),
            "a mixed-lowering tree is rc 2 and says which lowerings it holds")

        # a tree with no stamp at all is rc 2 as well
        chk(check(pathlib.Path(td) / "nope", {})[0] == 2, "a tree with no stamp is rc 2, not a pass")

        # the stamp is a file the checker reads, so a hand-edited one is read as written
        (tree / STAMP).write_text("engine=abc\nloom=c0ffee\nhost=box-a\n")
        chk(check(tree, {"engine": "abc"})[0] == 0, "a hand-written stamp with a known lowering passes")

    print("ok" if bad == 0 else f"{bad} FAIL")
    return 1 if bad else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("write", help="stamp a build tree")
    w.add_argument("--tree", required=True)
    w.add_argument("--root", default=".", help="the checkout the tree was built from (default: .)")
    w.add_argument("--loom", default=None, help="the lowering revision, if the build recorded one")
    w.add_argument("--host", default=None)
    w.add_argument("--engine", default=None, help="the sha to record instead of this checkout's HEAD (CI's merge sha)")
    w.add_argument("--llama-cpp", default=None)
    w.add_argument("--hrx-system", default=None)
    c = sub.add_parser("check", help="compare a build tree's stamp with this checkout and host")
    c.add_argument("--tree", required=True)
    c.add_argument("--expect-engine", default="")
    c.add_argument("--expect-llama-cpp", default="")
    c.add_argument("--expect-hrx-system", default="")
    c.add_argument("--expect-loom", default="")
    c.add_argument("--expect-host", default="")
    a = ap.parse_args(argv)
    if getattr(a, "self_test", False):
        return self_test()
    if not a.cmd:
        ap.print_help()
        return 2
    if a.cmd == "write":
        stamp = write_stamp(a.tree, a.root, a.loom, a.host, a.engine, a.llama_cpp, a.hrx_system)
        for k in FIELDS:
            print(f"{k}={stamp[k]}")
        return 0
    rc, why = check(a.tree, {"engine": a.expect_engine, "llama_cpp": a.expect_llama_cpp,
                             "hrx_system": a.expect_hrx_system, "loom": a.expect_loom, "host": a.expect_host})
    for r in why:
        print(f"bench_stamp: {r}", file=sys.stderr if rc else sys.stdout)
    if rc == 0:
        print(f"bench_stamp: {a.tree} matches this checkout and host, and its lowering is recorded")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
