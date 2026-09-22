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
    (("*.cpp", "*.h", "*.inc", "*.hpp", "*.c", "*.cc"), "//"),
    (("*.py", "*.sh", "*.yml", "*.yaml", "*.cmake", "CMakeLists.txt",
      ".gitignore", ".gitattributes", ".clang-format", "*.toml"), "#"),
]
BLOCK_STYLES = [(("*.md",), ("<!--", "-->"))]

EXEMPT = ["LICENSE", "NOTICE", "*.gguf", "*.json", "*.tsv", "tests/golden/*/tokens.txt"]


def _match(name, patterns):
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def notice_block(path):
    """The notice formatted as comments for `path`, or None if no style is known."""
    name = path.rsplit("/", 1)[-1]
    for patterns, prefix in LINE_STYLES:
        if _match(name, patterns):
            return "".join((prefix + " " + line).rstrip() + "\n" for line in NOTICE)
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

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
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
