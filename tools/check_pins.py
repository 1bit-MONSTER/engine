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
"""Fail a pull request that moves a submodule pin backwards or sideways.

usage: tools/check_pins.py <base-commit>

For every submodule whose gitlink differs between <base-commit> and HEAD, GitHub's compare API
must say the new commit is ahead of the old one. A pin that is behind or has diverged drops
commits the engine already shipped, as #154 did to third_party/llama.cpp (it lost ZAYA1-VL);
a deliberate rollback is merged with the "pin rollback" label, which skips this check.
Repositories the token cannot read (private ones) are reported and skipped.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def gitlink(commit, path):
    out = git("ls-tree", commit, "--", path).split()
    return out[2] if len(out) >= 3 and out[1] == "commit" else None


def compare(repo, old, new):
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/compare/{old}...{new}",
                                 headers={"Accept": "application/vnd.github+json"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["GITHUB_TOKEN"])
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    cfg = git("config", "-f", ".gitmodules", "--get-regexp", r"submodule\..*\.(path|url)")
    subs = {}
    for line in cfg.splitlines():
        key, value = line.split(" ", 1)
        name, field = key[len("submodule."):].rsplit(".", 1)
        subs.setdefault(name, {})[field] = value
    bad = 0
    for sub in subs.values():
        path, url = sub.get("path"), sub.get("url", "")
        old, new = gitlink(base, path), gitlink("HEAD", path)
        if not old or not new or old == new:
            continue
        m = re.match(r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", url)
        if not m:
            print(f"skip {path}: {url} is not on GitHub")
            continue
        try:
            status = compare(m.group(1), old, new)["status"]
        except urllib.error.HTTPError as e:
            print(f"skip {path}: cannot compare in {m.group(1)} (HTTP {e.code})")
            continue
        ok = status in ("ahead", "identical")
        print(f"{'ok  ' if ok else 'FAIL'} {path}: {old[:9]} -> {new[:9]} is {status}")
        bad += not ok
    if bad:
        print("a pin moved backwards or sideways: point it at a commit that contains the old one "
              "(git submodule update --remote), or label the PR \"pin rollback\" if that is intended")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
