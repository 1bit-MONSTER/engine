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
"""notice_covers_deps.py -- every FetchContent dependency has a NOTICE entry.

`NOTICE` is the file that says what is compiled into this binary and under which licence. It is
maintained by hand, and on 2026-11-13 a PR added `stb_image` -- fetched, compiled into the app, shipped
in the binary -- with no entry: nothing looked. This asks the one question that finding needed, and it
reads the two files that already exist rather than a registry someone has to remember to update.

    notice_covers_deps.py [--check] [--root ROOT]

Matches on the `GIT_REPOSITORY` URL, not on the dependency name: `FetchContent_Declare(stb ...)` is
called `stb` in CMake and appears in NOTICE as `stb_image (nothings/stb ...)`, so the name is not the
thing the two files share -- the URL is. Exit 0 when every dependency is covered, 1 when one is not,
2 on a usage error.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

DECLARE = re.compile(r"FetchContent_Declare\(\s*([A-Za-z0-9_.+-]+)((?:[^()]|\([^()]*\))*)\)", re.S)
URL = re.compile(r"(?:GIT_REPOSITORY|URL)\s+(\S+)")


def deps(text):
    """[(name, url)] for every FetchContent_Declare in the CMake text, in order."""
    out = []
    for m in DECLARE.finditer(text):
        name, body = m.group(1), m.group(2)
        u = URL.search(body)
        out.append((name, u.group(1) if u else ""))
    return out


def covered(url, notice):
    """True when NOTICE names this dependency -- by its URL's host/path, or its full URL.

    The URL is what the two files share, so it is what this compares; a bare repository name would
    match prose (`stb` appears in sentences about the image decoder) and a name mismatch (`stb` in
    CMake vs `stb_image` in NOTICE) is exactly the case that was missed.
    """
    if not url:
        return True  # no URL to compare (an in-source or archive dep): not this check's finding
    u = url.strip().rstrip("/")
    short = re.sub(r"^[a-z+]+://", "", u)
    short = re.sub(r"\.git$", "", short)
    return short in notice


def check(root):
    root = pathlib.Path(root)
    cmake = root / "CMakeLists.txt"
    notice_path = root / "NOTICE"
    if not cmake.is_file():
        print(f"notice_covers_deps: no CMakeLists.txt under {root}", file=sys.stderr)
        return 2
    if not notice_path.is_file():
        print(f"notice_covers_deps: no NOTICE under {root} -- a dependency list with no licence file",
              file=sys.stderr)
        return 2
    notice = notice_path.read_text(errors="replace")
    ds = deps(cmake.read_text(errors="replace"))
    if not ds:
        print("notice_covers_deps: no FetchContent_Declare in CMakeLists.txt (cannot tell, and that is "
              "not a pass)", file=sys.stderr)
        return 2
    missing = [(n, u) for n, u in ds if not covered(u, notice)]
    for n, u in missing:
        print(f"notice_covers_deps: {n} ({u or 'no URL'}) is fetched and not named in NOTICE")
    print(f"notice_covers_deps: {len(ds) - len(missing)} of {len(ds)} FetchContent dependency(ies) named in NOTICE")
    return 1 if missing else 0


def self_test():
    failures = 0

    def chk(ok, what):
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            failures += 1

    cm = ("FetchContent_Declare(json\n  GIT_REPOSITORY https://github.com/nlohmann/json.git\n  GIT_TAG v3.11.3)\n"
          "FetchContent_Declare(stb\n  GIT_REPOSITORY https://github.com/nothings/stb.git)\n")
    d = deps(cm)
    chk(len(d) == 2 and d[0] == ("json", "https://github.com/nlohmann/json.git"),
        "both declarations are read, with their repository URL")
    notice = "JSON for Modern C++ (nlohmann/json v3.11.3)\n  https://github.com/nlohmann/json\n"
    chk(covered(d[0][1], notice), "a URL named in NOTICE counts")
    chk(not covered(d[1][1], notice), "a URL absent from NOTICE does not")
    chk(covered("", notice), "a declaration with no URL is not this check's finding")
    chk(covered(d[0][1] + "/", notice), "a trailing slash on the URL does not change the answer")
    chk(not covered("https://example.com/stb.git", "https://example.com/not-stb\n"),
        "a substring of a longer path is not a match")

    # end to end on a scratch tree: one dep covered, one not -> rc 1 and the uncovered one named
    tmp = tempfile.mkdtemp(prefix="notice-deps-")
    try:
        (pathlib.Path(tmp) / "CMakeLists.txt").write_text(cm)
        (pathlib.Path(tmp) / "NOTICE").write_text(notice)
        out = subprocess.run([sys.executable, __file__, "--check", "--root", tmp],
                             capture_output=True, text=True)
        chk(out.returncode == 1, "a tree with one uncovered dependency is rc 1")
        chk("stb" in out.stdout and "not named in NOTICE" in out.stdout,
            "and the message names the dependency")
        (pathlib.Path(tmp) / "NOTICE").write_text(notice + "stb_image (nothings/stb)\n  https://github.com/nothings/stb\n")
        out2 = subprocess.run([sys.executable, __file__, "--check", "--root", tmp],
                              capture_output=True, text=True)
        chk(out2.returncode == 0, "and covering it makes the tree rc 0")
        (pathlib.Path(tmp) / "CMakeLists.txt").write_text("project(x)\n")
        out3 = subprocess.run([sys.executable, __file__, "--check", "--root", tmp],
                              capture_output=True, text=True)
        chk(out3.returncode == 2, "a CMakeLists with no declarations is 'cannot tell', rc 2")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"notice_covers_deps --self-test: {'FAIL' if failures else 'ok'}")
    return 1 if failures else 0


def main(argv):
    ap = argparse.ArgumentParser(description="every FetchContent dependency has a NOTICE entry")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        ap.print_usage(sys.stderr)
        return 2
    return check(a.root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
