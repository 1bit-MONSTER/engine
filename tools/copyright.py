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


def has_notice(text):
    head = "\n".join(text.splitlines()[:20])
    return NOTICE[0] in head and NOTICE[1] in head and "Licensed under the Apache License, Version 2.0" in head


def add_notice(text, block):
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
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
    files = sorted(set(filter(None, listing())) | set(filter(None, listing("--others", "--exclude-standard"))))
    missing, unknown = [], []
    for path in filter(None, files):
        if exempt(path):
            continue
        block = notice_block(path)
        if block is None:
            unknown.append(path)
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if has_notice(text):
            continue
        if args.fix:
            with open(path, "w", encoding="utf-8") as f:
                f.write(add_notice(text, block))
            print("added:", path)
        else:
            missing.append(path)

    for path in unknown:
        print("no comment style known (add one to LINE_STYLES/BLOCK_STYLES or EXEMPT):", path)
    for path in missing:
        print("missing copyright notice:", path)
    if missing or unknown:
        print("run: python3 tools/copyright.py --fix", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
