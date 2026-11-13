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
"""moving_ref_check.py -- a build input names a commit or a digest, never a branch or a tag.

A release should be a pin bump, not a scramble. That only holds if every input a build or CI run takes is named
by something that cannot move. The other pin checks cover their own files:
  - check_pins.py and submodule_pin_check.py cover submodule gitlinks;
  - config_pin_check.py covers config/*.config;
  - the os/v2 snapshot and the os/v3 ISO sha cover the OS bases.
Nothing looked at the inputs written inline in workflows and build scripts, and two of them had moved:
  - CI's `container: ubuntu:26.04` was sha256:6a06111c... on 2026-11-09 and sha256:065aa862... on 2026-11-13,
    so CI built on two different images with no change in this tree;
  - os/v3/puppy fetched vercmp.c from initrd_progs `master` and built from whatever woof-CE commit the clone
    happened to be on.

Rules. A line breaks one when it names:
  uses        a GitHub Action `uses: owner/repo@ref` whose ref is not a 40-hex commit;
  image       a container `image:` with no `@sha256:` digest;
  docker      a `docker|podman run|create|pull ... name:tag` with no `@sha256:` digest;
  raw-ref     a raw.githubusercontent.com / codeload / `archive/` URL whose ref is not a 40-hex commit;
  latest      a `releases/latest` URL, or a `:latest` image;
  pip         a workflow `pip install` of a requirement with no `==` version.

A line that reads a moving ref ON PURPOSE carries `moving-ref: <why>` in a comment on that line or the line above.
The bump bots are an example: they read the tip in order to propose a pin, and do not consume it. The reason is
printed in the report, so an exemption is visible, not silent.

Scope: .github/workflows/*.yml, scripts/, os/ and tools/ shell scripts, and CMakeLists.txt / cmake/*.cmake.
Probes and tests that talk to 127.0.0.1 are not inputs, and no rule matches them.

Baseline: tools/ci/moving_ref_baseline.txt lists findings known and owed, one `path<TAB>rule<TAB>text<TAB>why` per
line. A baselined finding is printed and does not fail. A NEW finding fails, and so does a STALE entry (one that no
longer matches anything), so the baseline can only shrink. It exists for one reason: the workflow files can only be
changed by a push with the `workflow` scope, so their fixes wait on the owner.

Usage:
  moving_ref_check.py --check [--root .] [--baseline FILE]
  moving_ref_check.py --self-test
Exit: 0 clean, 1 a moving ref (each one is printed as path:line: rule: text), 2 usage, 3 self-test failed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

SHA40 = re.compile(r"^[0-9a-f]{40}$")
PRAGMA = re.compile(r"moving-ref:\s*(\S.*)")

USES = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)")
IMAGE = re.compile(r"^\s*image:\s*([^\s#]+)")
# `docker run ... ubuntu:24.04 bash x` -- the first token that looks like name[:tag] after the subcommand's
# options is hard to find in general, so match any registry-shaped `name:tag` on a docker/podman line.
DOCKER = re.compile(r"\b(docker|podman)\s+(run|create|pull)\b")
IMAGE_TOKEN = re.compile(r"(?<![\w$/{.-])((?:[a-z0-9.-]+/)*[a-z0-9][a-z0-9._-]*):([A-Za-z0-9][A-Za-z0-9._-]*)(@sha256:[0-9a-f]{64})?(?![\w:/])")
RAW = re.compile(
    r"https?://(?:raw\.githubusercontent\.com/[^/\s]+/[^/\s]+/([^/\s\"']+)/"
    r"|codeload\.github\.com/[^/\s]+/[^/\s]+/(?:tar\.gz|zip)/([^/\s\"']+)"
    r"|github\.com/[^/\s]+/[^/\s]+/archive/(?:refs/(?:heads|tags)/)?([^/\s\"']+?)(?:\.tar\.gz|\.zip)\b)")
LATEST = re.compile(r"releases/latest\b|:latest\b")
PIP = re.compile(r"\bpip3?\s+install\b(.*)")


def pip_findings(args: str) -> list[str]:
    """The requirements on a `pip install` line that carry no `==` version."""
    bad = []
    skip_next = False
    for tok in args.split():
        if tok.startswith("#") or tok in ("|", "&&", ";", "||"):
            break
        if skip_next:
            skip_next = False
            continue
        if tok in ("-r", "--requirement", "-c", "--constraint", "-e", "--editable", "--index-url", "-i",
                   "--extra-index-url", "-f", "--find-links", "--target", "-t", "--prefix", "--root"):
            skip_next = True
            continue
        if tok.startswith("-") or tok.startswith((".", "/", "$", '"$', "'")) or "/" in tok:
            continue
        if "==" not in tok:
            bad.append(tok)
    return bad


def scan_text(path: str, text: str) -> list[tuple[int, str, str, str]]:
    """[(line, rule, text, exemption-or-'')] for one file's text."""
    out = []
    raw = text.splitlines()
    # A shell command continued with `\` is one command: `docker run ... \` on one line and the image ten lines down
    # is still a `docker run` of that image (os/v3/puppy/build.sh is written exactly so). Join each continued line
    # into its first line, keep the first line's number, and leave the continuation lines empty so numbers stay put.
    lines = list(raw)
    i = 0
    while i < len(lines):
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines) and not lines[j].lstrip().startswith("#"):
            j += 1
        if j > i:
            lines[i] = " ".join(l.rstrip().rstrip("\\") for l in raw[i:j + 1])
            for k in range(i + 1, j + 1):
                lines[k] = ""
        i = j + 1
    is_workflow = path.replace(os.sep, "/").startswith(".github/workflows/")
    for i, line in enumerate(lines):
        if "127.0.0.1" in line or "localhost" in line:
            continue
        pragma = PRAGMA.search(line) or (PRAGMA.search(lines[i - 1]) if i else None)
        why = pragma.group(1).strip() if pragma else ""
        code = line.split(" #", 1)[0] if not line.lstrip().startswith("#") else ""
        if not code.strip():
            continue
        hits = []
        if is_workflow:
            m = USES.match(code)
            if m and not m.group(1).startswith(("./", "docker://")):
                ref = m.group(1).rpartition("@")[2] if "@" in m.group(1) else ""
                if not SHA40.match(ref):
                    hits.append(("uses", m.group(1)))
            m = IMAGE.match(code)
            if m and "@sha256:" not in m.group(1):
                hits.append(("image", m.group(1)))
            m = PIP.search(code)
            if m:
                bad = pip_findings(m.group(1))
                if bad:
                    hits.append(("pip", " ".join(bad) + " (no ==version)"))
        if DOCKER.search(code):
            for t in IMAGE_TOKEN.finditer(code):
                if not t.group(3) and not t.group(1).startswith("-"):
                    hits.append(("docker", t.group(0)))
        for m in RAW.finditer(code):
            ref = next(g for g in m.groups() if g is not None)
            if "$" not in ref and not SHA40.match(ref):
                hits.append(("raw-ref", m.group(0)))
        if LATEST.search(code):
            hits.append(("latest", LATEST.search(code).group(0)))
        for rule, what in hits:
            out.append((i + 1, rule, what, why))
    return out


def in_scope(rel: str) -> bool:
    rel = rel.replace(os.sep, "/")
    if rel.startswith(".github/workflows/") and rel.endswith((".yml", ".yaml")):
        return True
    if rel == "CMakeLists.txt" or (rel.startswith("cmake/") and rel.endswith(".cmake")):
        return True
    if rel.startswith(("scripts/", "os/", "tools/")) and not rel.startswith("os/licenses/"):
        return rel.endswith(".sh") or os.path.basename(rel) in ("Dockerfile", "Containerfile")
    return False


def scan_tree(root: str) -> list[tuple[str, int, str, str, str]]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "third_party", "build", "node_modules")]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if not in_scope(rel):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            for line, rule, what, why in scan_text(rel, text):
                found.append((rel, line, rule, what, why))
    return sorted(found)


def read_baseline(path: str) -> dict[tuple[str, str, str], str]:
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows[(parts[0], parts[1], parts[2])] = parts[3] if len(parts) > 3 else ""
    return rows


def check(root: str, baseline_path: str | None = None) -> int:
    found = scan_tree(root)
    if baseline_path is None:
        baseline_path = os.path.join(root, "tools", "ci", "moving_ref_baseline.txt")
    base = read_baseline(baseline_path)
    seen = set()
    moving = []
    for rel, line, rule, what, why in found:
        if why:
            print(f"allowed {rel}:{line}: {rule}: {what}  (moving-ref: {why})")
        elif (rel, rule, what) in base:
            seen.add((rel, rule, what))
            print(f"owed    {rel}:{line}: {rule}: {what}  (baseline: {base[(rel, rule, what)]})")
        else:
            moving.append((rel, line, rule, what))
    stale = [k for k in base if k not in seen]
    for rel, line, rule, what in moving:
        print(f"MOVING  {rel}:{line}: {rule}: {what}")
    for rel, rule, what in stale:
        print(f"STALE   {rel}: {rule}: {what}  (baselined, no longer found: delete the row)")
    if moving or stale:
        print(f"moving_ref: {len(moving)} new moving input(s), {len(stale)} stale baseline row(s); pin a commit or a digest, "
              "or mark a deliberate read with a `moving-ref: <why>` comment")
        return 1
    print(f"moving_ref: every inline build input is pinned ({sum(1 for f in found if f[4])} deliberate read(s) marked, "
          f"{len(seen)} baselined and owed)")
    return 0


def self_test() -> int:
    sha = "a" * 40
    dig = "@sha256:" + "0" * 64
    cases = [
        # (path, text, expected rules)
        (".github/workflows/x.yml", f"      - uses: actions/checkout@{sha}\n", []),
        (".github/workflows/x.yml", "      - uses: actions/checkout@v4\n", ["uses"]),
        (".github/workflows/x.yml", "      - uses: ./.github/actions/local\n", []),
        (".github/workflows/x.yml", "      image: ubuntu:26.04\n", ["image"]),
        (".github/workflows/x.yml", f"      image: ubuntu:26.04{dig}\n", []),
        (".github/workflows/x.yml", "        run: python3 -m pip install numpy gguf\n", ["pip"]),
        (".github/workflows/x.yml", "        run: python3 -m pip install numpy==2.5.4 gguf==0.19.0\n", []),
        (".github/workflows/x.yml", "        run: pip install -r requirements.txt -e .\n", []),
        ("os/b.sh", "    -e COMPAT=x ubuntu:24.04 bash /recipe/c.sh\nexec docker run --rm ubuntu:24.04 bash\n", ["docker"]),
        ("os/b.sh", f"exec docker run --rm ubuntu:24.04{dig} bash\n", []),
        ("os/b.sh", 'exec docker run --rm -v "$a":/w "$image" bash\n', []),
        ("os/b.sh", "curl -fsSL https://raw.githubusercontent.com/o/r/master/f.c | gcc -x c -\n", ["raw-ref"]),
        ("os/b.sh", f"curl -fsSL https://raw.githubusercontent.com/o/r/{sha}/f.c | gcc -x c -\n", []),
        ("os/b.sh", 'curl -fsSL "https://raw.githubusercontent.com/o/r/$ref/f.c"\n', []),
        ("scripts/f.sh", "curl -L https://github.com/o/r/archive/refs/heads/main.tar.gz | tar xz\n", ["raw-ref"]),
        ("scripts/f.sh", "tag=$(gh api repos/o/r/releases/latest --jq .tag_name)\n", ["latest"]),
        ("scripts/f.sh", "# moving-ref: the bump bot proposes the pin\ntag=$(gh api repos/o/r/releases/latest)\n", []),
        ("os/probe.sh", "curl -fsS http://127.0.0.1:8000/releases/latest\n", []),
        ("os/b.sh", "# docker run ubuntu:24.04 in a comment is prose\n", []),
        # the build.sh shape: the image ten continuation lines below the `docker run`
        ("os/b.sh", "exec docker run --rm \\\n    -v a:/b \\\n    -e X=1 ubuntu:24.04 bash /r.sh\n", ["docker"]),
        ("os/b.sh", f"exec docker run --rm \\\n    -v a:/b \\\n    -e X=1 ubuntu:24.04{dig} bash /r.sh\n", []),
    ]
    bad = 0
    for path, text, want in cases:
        got = [r for _, r, _, why in scan_text(path, text) if not why]
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'} {path}: {text.strip().splitlines()[-1][:70]!r} -> {got} (want {want})")
    # the exemption is reported, not dropped
    ex = scan_text("scripts/f.sh", "tag=$(gh api repos/o/r/releases/latest)  # moving-ref: bump bot\n")
    ok = len(ex) == 1 and ex[0][3] == "bump bot"
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL'} a marked read is kept with its reason: {ex}")
    # scope: a tree with one moving workflow ref and the same text out of scope
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, ".github/workflows"))
        os.makedirs(os.path.join(d, "docs"))
        with open(os.path.join(d, ".github/workflows/ci.yml"), "w") as f:
            f.write("    container:\n      image: ubuntu:26.04\n")
        with open(os.path.join(d, "docs/x.md"), "w") as f:
            f.write("image: ubuntu:26.04\n")
        rows = scan_tree(d)
        ok = [(r[0], r[2]) for r in rows] == [(".github/workflows/ci.yml", "image")]
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'} scope: workflows are read, docs are not: {rows}")
        # baseline: a listed finding passes, a new one fails, a stale row fails
        bl = os.path.join(d, "bl.txt")
        with open(bl, "w") as f:
            f.write(".github/workflows/ci.yml\timage\tubuntu:26.04\tneeds workflow scope\n")
        r1 = check(d, bl)
        with open(bl, "a") as f:
            f.write("scripts/gone.sh\tlatest\t:latest\tfixed long ago\n")
        r2 = check(d, bl)
        with open(bl, "w") as f:
            f.write("")
        r3 = check(d, bl)
        ok = (r1, r2, r3) == (0, 1, 1)
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'} baseline: listed passes, stale row fails, unlisted fails: {(r1, r2, r3)}")
    print(f"self-test: {'ok' if not bad else f'{bad} FAILED'}")
    return 3 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    return self_test() if a.self_test else check(a.root, a.baseline)


if __name__ == "__main__":
    sys.exit(main())
