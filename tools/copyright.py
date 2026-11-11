#!/usr/bin/env python3
"""Check or add the copyright and Apache-2.0 notice on every tracked file.

usage: tools/copyright.py --check   # exit 1 and list files missing the notice
       tools/copyright.py --fix     # add the notice where it is missing

The notice is the one Apache-2.0's appendix says to attach to each file.
Files that cannot carry a comment (binaries, JSON, test data) are listed in
EXEMPT and skipped.
"""
import argparse
import fnmatch
import re
import subprocess
import sys

NOTICE = """Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.""".split("\n")

# Comment style by file name pattern.
LINE_STYLES = [
    (("*.cpp", "*.h", "*.inc", "*.hpp", "*.c", "*.cc", "*.rs", "*.hip"), "//"),
    (("*.py", "*.sh", "*.yml", "*.yaml", "*.cff", "*.cmake", "*.cmake.in", "CMakeLists.txt", ".gitmodules",
      ".gitignore", ".gitattributes", ".clang-format", "*.toml", "*.config", "*.fragment", "*.example",
      "*.service", "*.timer", "*.conf", "*.env", "lemonade-hrx", "packages-extra", "05onboard-data", "10lemonade", "zz-onboard-ports", "20onboard-ssh",
      # engine#1474: the BASE layer's keep-list is a '#'-comment data file read by scripts/base-modes.sh, listed by name
      # like packages-extra above because it has no extension
      "base-modes-keep",
      # engine#1595: the direction-B ratchet's baseline is a '#'-comment data file too, and `.txt` has no style of
      # its own (a blanket "*.txt" would demand a notice in files that are data, not code).
      "add_test_gaps.txt",
      # engine#1820: the other sweep's allow-list, same shape and the same reason -- `#`-comment data, `.txt` has no
      # style, and the header it wants is already at the top of the file.
      "unreferenced_tests.txt",
      # engine#1946: the ctest skip baseline, the same shape and the same reason -- a '#'-comment data file whose
      # is the notice it would otherwise be asked for. (CI caught this on the PR: the tool refuses an unknown sty
      # than passing it, which is the behaviour worth having.)
      "ctest_skips.txt",
      # engine#2018: the test-registration baseline of `tests/*_test.py`, same shape and the same reason -- data,
      "unreferenced_py_tests.txt",
            "hrx_f16_narrowing.txt",
      # engine#2027: the `#`-comment baseline for hardcoded /tmp paths, same shape and the same reason --
      # data, `.txt` has no style, and the notice it wants is already at the top of the file.
      "tmp_paths.txt",
      "21serial-console", "90onboard-limits", "Dockerfile", "*.Dockerfile",
      # cycle 17: Puppy's switch that skips its first-run Quick Setup is an empty marker file whose only content
      # is this notice - Puppy tests for its existence, so a comment is free
      "personal_settings_popup_disabled",
      # os/v2's packaged files: systemd units of kinds the list did not name yet, and the two '#'-comment
      # files a v2 build reads (the snapshot pin, and the board's power profile at /etc/1bit/power)
      "*.mount", "*.preset", "*.network", "power", "pin.txt",
      # os/v3 Puppy image, cycle 16: XDG desktop entries are ini-shaped with '#' line comments. The autostart
      # that opens the appliance's chat screen is one, and CI is right to want a notice on it (it runs a
      # program as root).
      "*.desktop",
      # cycle 30: udev rules are '#' comments (the NPU keepawake rule is the first one the image ships)
      "*.rules",
      # engine#1279: the packaging recipes. A PKGBUILD is bash makepkg sources and an rpm spec's
      # header is '#!' preprocessor lines, so both take '#' comments
      "*.spec", "PKGBUILD"), "#"),
]
BLOCK_STYLES = [(("*.md", "*.html", "*.svg"), ("<!--", "-->")), (("*.css", "*.js"), ("/*", "*/"))]

EXEMPT = ["third_party/*", "os/licenses/*", "config/draft-vocab/*", "patches/*", "LICENSE", "NOTICE", "LICENSE-BINARY", "NOTICE-BINARY", "*.gguf", "tests/golden/*.bin", "tests/golden/*/*.bin", "tests/golden/*/*.f32", "*.json", "*.tsv", "tests/golden/*/tokens.txt", "*.embed", "*/Cargo.lock", "*.png", "*.ico",
          # engine#315 evidence artefacts that cannot carry a comment: diffs must stay byte-applicable
          # and data files have no comment syntax. Matches the existing "patches/*" exemption.
          "*.patch", "*.csv", "*.cfg",
          # engine#1146, generalised by engine#805: an acceptance ticket under tests/fixtures/ is DATA that a parser
          # reads byte-verbatim (like tests/golden/*/tokens.txt), so a notice comment would make the fixture no longer
          # the log it claims to be; each fixture directory's README carries the notice instead. By kind, not by
          # directory: the clause-55 window needed this exemption first and the tau=0.6 window then failed the check
          # for the same reason one directory over.
          "tests/fixtures/*/*.txt",
          # os/v2 packages: dpkg control files, apt/dpkg list files and the console banner are data
          # with no comment syntax - the same exemption patches/* and *.cfg get
          "os/v2/debian/*/control", "os/v2/debian/*/conffiles", "*.list",
          "os/v2/debian/*/files/etc/issue", "os/v2/debian/*/files/etc/motd",
          # os/v3 Puppy image: an ssh authorized_keys file is one key per line and a comment in it would be
          # parsed as an option string, so it is data, not code (a build that ships it is what U7 asks for)
          "*/authorized_keys",
          # os/v3 Puppy image, cycle 13: the presets a headless boot reads instead of asking. /etc/keymap is
          # one keymap name ("us") and /etc/localtime is a symlink into /usr/share/zoneinfo - neither can
          # carry a comment, and both are read by keymap-set and glibc verbatim
          "os/v3/puppy/embed/etc/keymap", "os/v3/puppy/embed/etc/localtime",
          # cycle 17: /etc/timezone is one zone name, read verbatim by every Debian-family tool (and by
          # 08onboard-presets, which pins /etc/localtime to it) - a comment line would BE the zone
          "os/v3/puppy/embed/etc/timezone",
          # os/v3 Puppy image, cycle 15: the appliance's hostname is one word and is read verbatim by
          # hostname(1), rc.sysinit and /etc/hosts - a comment line would BE the hostname
          "os/v3/puppy/embed/etc/hostname",
          # os/v3 Puppy image, cycle 18: the console TUI's rendered frames ARE the regression fixtures - a notice
          # line in one would change the frame it exists to freeze
          "os/v3/puppy/console-tui/tests/snapshots/*.txt"]


def _match(name, patterns):
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _line_block(prefix):
    return "".join((prefix + " " + line).rstrip() + "\n" for line in NOTICE)


def _shebang(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readline().startswith("#!")
    except OSError:
        return False


def notice_block(path):
    """The notice formatted as comments for `path`, or None if no style is known."""
    name = path.rsplit("/", 1)[-1]
    for patterns, prefix in LINE_STYLES:
        if _match(name, patterns):
            return _line_block(prefix)
    # A file with no extension has no pattern to match (scripts/gpu-hold): the shebang decides, and a script
    # that starts with #! takes the '#' line style.
    if "." not in name and _shebang(path):
        return _line_block("#")
    for patterns, (open_, close) in BLOCK_STYLES:
        if _match(name, patterns):
            return open_ + "\n" + "\n".join(NOTICE) + "\n" + close + "\n"
    return None


def exempt(path):
    return _match(path, EXEMPT) or _match(path.rsplit("/", 1)[-1], EXEMPT)


NOTICE_WINDOW = 20      # the lines has_notice() looks at; duplicated_notice() uses the same window

BUILD_DIR = re.compile(r"(^|/)(build[^/]*|cmake-build[^/]*)/")


def under_build_dir(path):
    """True when a path sits in a build tree, tracked or not.

    A build tree holds generated sources (`build*/generated/recipes_default.h` and friends) that carry no
    notice by design, and it is never repo content until someone commits it. Before this, an untracked
    generated `.h` was not skipped the way an unknown-style file is -- `.h` IS a known style, so it went
    straight to missing[] -- and a developer clone with a build dir read 36 "missing copyright notice"
    lines while `main` was clean. Only the untracked case is skipped: a TRACKED file in a build tree is
    still an error, because then it is repo content (engine#1564's argument, applied to a known style).
    """
    return BUILD_DIR.search(path) is not None


def has_notice(text):
    head = "\n".join(text.splitlines()[:NOTICE_WINDOW])
    return NOTICE[0] in head and NOTICE[1] in head and "Licensed under the Apache License, Version 2.0" in head


def short_notice(text):
    """True when the header carries Copyright + SPDX but not the licence block.

    engine#2175: `notice_block()` returns the 16-line form for a known style, so a header that keeps only
    `// Copyright ...` and `// SPDX-License-Identifier: Apache-2.0` and then jumps to the file's own
    description reads as MISSING -- and the message says "missing copyright notice", which sends the reader
    looking for a notice that is on line 1. Measured on engine#2175 / #2173 (their audio headers) against
    #2133 (the same work, full block): copyright present, SPDX present, "Licensed under the Apache License"
    absent. Naming the shape is the difference between a 15-line paste and a hunt.
    """
    head = "\n".join(text.splitlines()[:NOTICE_WINDOW])
    return (NOTICE[0] in head and NOTICE[1] in head
            and "Licensed under the Apache License, Version 2.0" not in head)


def duplicated_notice(text):
    """True when the Copyright + SPDX pair appears twice inside the header window.

    Present-in-the-head was all this gate ever asked, so a file whose header block was pasted twice passed —
    which is how #1535 and #1544 shipped one, and why #1547 lists 20 files. The window is deliberate: the
    three legitimate second occurrences in the tree (a heredoc emitting a licence header in
    scripts/mk-kernel-packages.sh, the NOTICE string a test asserts on in
    tests/copyright_check_untracked_test.py, and the NOTICE template in tools/npu_leaderboard.py) all sit
    past line 20, because they are *data*, not the file's own header. A duplicate inside the window is the
    accidental shape; anything further down is somebody's deliberate string.
    """
    head = "\n".join(text.splitlines()[:NOTICE_WINDOW])
    if head.count(NOTICE[0]) > 1 and head.count(NOTICE[1]) > 1:
        return True
    # The same defect in a different spelling: the canonical notice plus a near-miss of our own holder.
    if NOTICE[0] in head and any(near_miss_holder(l) for l in head.splitlines()):
        return True
    return False


_COPY_LINE = re.compile(r"^\W*Copyright\s+(?:\(c\)\s*)?\d{4}\s+(.+?)\s*$", re.IGNORECASE)


def _holder(line):
    """The holder a copyright line names, normalised: 'bong-water-bong' -> 'bongwaterbong'."""
    m = _COPY_LINE.match(line)
    return re.sub(r"[^a-z0-9]", "", m.group(1).lower()) if m else ""


_CANON_HOLDER = _holder(NOTICE[0])

# How much of a holder two copyright lines must share, normalised, to be treated as the same claim misspelled
# instead of a different attribution. This is a DECISION, not a value that arrived:
#
#   too small  -> a real second attribution gets deleted, which is silent data loss in a file someone can read;
#   too large  -> a misspelling survives as a second notice, which is what engine#1606 shipped.
#
# Six is chosen so that `bongwaterbong` (this repo, misspelled) matches `bongwaterwaterbong` (canonical) - the
# shape that actually occurred - while a distinct holder like `the1bitmonsterauthors` (which the tree carries
# beside the notice in tests/installed_rpath_test.sh and tools/check-installed-rpath.py) shares nothing and is
# left alone. The residual cost of being wrong is one visible line removed from a file a human can inspect,
# which is why the failure mode is chosen in that direction; the dual-attribution case is pinned by a test so a
# future change to this number cannot quietly start deleting `The 1bit-MONSTER Authors.`.
HOLDER_NEAR_MISS_PREFIX = 6


def near_miss_holder(line):
    """True when a copyright line names a holder that is a TYPO of this repo's, not a different attribution.

    engine#1606's build failed on exactly this: a hand-written `Copyright 2026 bong-water-bong` beside the
    canonical block, and `--fix` inserted a second notice rather than replacing it. It cannot be "any second
    copyright line" - the tree legitimately carries dual attribution (`Copyright 2026 The 1bit-MONSTER
    Authors.` for derived work) inside the same header window - so the test is the holder: normalised
    holders that share a long prefix and differ are the same claim misspelled; holders that share nothing
    are a different claim and are left alone.
    """
    h = _holder(line)
    if not h or h == _CANON_HOLDER:
        return False
    n = 0
    while n < len(h) and n < len(_CANON_HOLDER) and h[n] == _CANON_HOLDER[n]:
        n += 1
    return n >= HOLDER_NEAR_MISS_PREFIX


# The licence block's own blank comment lines (`#`, `//`) reduce to "", so a set built from every NOTICE line held the
# empty string, and canonical_notice_line() said yes to every blank or bare-comment line in a file. add_notice() then
# deleted each one in the header window (the separator after the description, the blank lines between #include groups
# and after `namespace {`), and repair_notice()'s doubled-block dedupe deleted every blank line after the first in the
# whole file. Only lines with words in them can be a claim, so only those are canonical.
_CANON_LINES = {k for k in (re.sub(r"^\W+", "", n).strip() for n in NOTICE) if k}


def canonical_notice_line(line):
    """True when the line IS one of the canonical notice lines, spelled exactly.

    engine#2129, met while fixing a new file's header: an SPDX two-liner without the licence block is "missing" to
    has_notice(), so --fix prepends the block -- and the two-liner it just merged into that block stayed behind, so the
    file --fix had just written failed --check with "duplicated inside the header window". near_miss_holder() cannot
    cover it by construction: it returns False for an exact holder, because an exact holder next to the canonical block
    is the *repaired* state. The block already contains both lines, so a copy inside the window is not a second claim.
    """
    key = re.sub(r"^\W+", "", line).strip()
    return bool(key) and key in _CANON_LINES


def repair_notice(text, block):
    """The file already HAS the canonical notice, plus a near-miss of our holder: delete the misspelling.

    This is the state engine#1606 shipped - canonical block present, hand-written `bong-water-bong` beside it -
    and it needed its own path because it is neither "notice missing" (has_notice() is true, so --fix skipped
    the file) nor "duplicate canonical pair" (duplicated_notice() only reported it, even under --fix). Dropping
    the copyright line and the SPDX line directly beneath it keeps a near-miss pair from leaving an orphan.

    It also keeps the FIRST canonical notice and drops later copies of the same line, which is the doubled-block shape:
    --fix used to answer `repaired:` for that state while `--check` went on reporting the duplicate, i.e. the one mode
    that could fix it claimed to and did not.
    """
    out, drop_next_spdx = [], False
    seen_canonical = set()
    for line in text.splitlines(keepends=True):
        if near_miss_holder(line):
            drop_next_spdx = True
            continue
        if drop_next_spdx and re.match(r"^\W*SPDX-License-Identifier\s*:", line, re.IGNORECASE):
            drop_next_spdx = False
            continue
        drop_next_spdx = False
        # A SECOND canonical notice is the same defect as a near-miss beside the first: the block pasted twice. `--check`
        # has always reported it (`duplicated_notice`); `--fix` used to print "repaired" and leave the copy in place, so the
        # file it had just written still failed the very check that sent it there. The first occurrence is kept and later
        # ones dropped, which is what "delete the second copy" means.
        if canonical_notice_line(line):
            key = re.sub(r"^\W+", "", line).strip()
            if key in seen_canonical:
                continue
            seen_canonical.add(key)
        out.append(line)
    repaired = "".join(out)
    return repaired if has_notice(repaired) else add_notice(repaired, block)


def strip_short_header(text):
    """Drop a short header (Copyright + SPDX and nothing after it) so add_notice can put the block there.

    engine#2175: `--fix` on a short header prepended the canonical block and LEFT the two stub lines, so the
    file came out with the Copyright/SPDX pair twice, `duplicated_notice()` fired, and `--check` was still
    red -- the tool's own suggested remedy could not fix the shape it told four branches about. The stub is
    exactly the two lines `short_notice()` looks for; if the file's head does not match that shape, this
    returns the text unchanged and the caller's behaviour is what it was.
    """
    lines = text.splitlines(keepends=True)
    # A script's short header starts on line 2, behind its `#!`: without this, the stub is not found, the block
    # is prepended after the shebang, and the file ends up with two notices again (measured: short.sh).
    at = 1 if lines and lines[0].startswith("#!") else 0
    if len(lines) >= at + 2 and NOTICE[0] in lines[at] and NOTICE[1] in lines[at + 1]:
        head, rest = lines[:at], lines[at + 2:]
        if rest and rest[0].strip() == "" and len(rest) > 1:
            rest = rest[1:]
        return "".join(head + rest)
    return text


def add_notice(text, block):
    text = strip_short_header(text)
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    # A near-miss of this repo's holder is REPLACED, not stacked beside: `--fix` used to insert the canonical
    # block and leave the hand-written one in place, which is how a file ended up with two notices and a green
    # `--check` (engine#1606). Dual attribution with a different holder is untouched - only the window is read.
    head = lines[:NOTICE_WINDOW]
    kept = [l for l in head if not near_miss_holder(l) and not canonical_notice_line(l)]
    lines = kept + lines[NOTICE_WINDOW:]
    return "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    # Tracked AND untracked-but-not-ignored files. `git ls-files` alone cannot see a file that has
    # not been staged yet -- which is exactly the state a new file is in when a lane runs this check
    # before `git add`, so the check would report green on the file most likely to be missing its
    # notice (this happened: onnx/tests/ort_run_generic.sh passed --check and failed CI on the same
    # head). Ignored files stay out: a build artifact is not source.
    def listing(*args):
        return subprocess.run(["git", "ls-files", *args], capture_output=True, text=True,
                              check=True).stdout.split("\n")
    tracked = set(filter(None, listing()))
    untracked = set(filter(None, listing("--others", "--exclude-standard")))
    files = sorted(tracked | untracked)
    missing, unknown, duplicated = [], [], []
    for path in filter(None, files):
        if exempt(path):
            continue
        if path in untracked and under_build_dir(path):
            continue
        block = notice_block(path)
        if block is None:
            # A file this tool cannot attach a notice to. If it is IN the repository that is a real gap -- a source
            # file whose style nobody taught the tool, which would ship without a licence header. If it is merely
            # untracked, it is not repo content yet: run output, a probe's dump, a scratch file. Failing on that is
            # the shape that teaches people to skip the gate (engine#1564: `.hrx123/logs/*.jsonl` reddened a
            # developer clone while `main` was clean), so an untracked unknown-style file is skipped and a tracked
            # one is still an error.
            if path not in untracked:
                unknown.append(path)
            continue
        # git ls-files lists a path that was staged and then deleted before it is committed; opening it used
        # to raise FileNotFoundError and take the whole gate down with a traceback (found while testing the
        # near-miss fix below). A missing path is skipped: it is not repo content, and `git status` is where a
        # half-deleted file belongs, not a licence gate.
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            continue
        if duplicated_notice(text):
            if args.fix:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(repair_notice(text, block))
                print("repaired:", path)
            else:
                duplicated.append(path)
            continue
        if has_notice(text):
            continue
        if args.fix:
            with open(path, "w", encoding="utf-8") as f:
                f.write(add_notice(text, block))
            print("added:", path)
        else:
            missing.append(path)

    for path in duplicated:
        print("copyright notice duplicated inside the header window (delete the second copy):", path)
    for path in unknown:
        print("no comment style known (add one to LINE_STYLES/BLOCK_STYLES or EXEMPT):", path)
    for path in missing:
        # engine#2175: a short header is not a missing notice. Say which, so the fix is a paste rather than a
        # search -- and keep the exit code the same (this is still a failure: the licence block is required).
        if short_notice(open(path, encoding="utf-8", errors="replace").read()):
            print("copyright + SPDX present but the licence block is not (paste the 16-line form):", path)
        else:
            print("missing copyright notice:", path)
    if missing or unknown or duplicated:
        print("run: python3 tools/copyright.py --fix", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
