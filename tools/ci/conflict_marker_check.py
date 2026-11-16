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
"""Merge-conflict markers committed to the tree.

Engine#2625 merged with its own conflict markers still in `docs/nvme-iommu-fault-triage.md` -- `<<<<<<< HEAD` at
line 137 and `>>>>>>> 2b1faf1b2 …` at line 157, both **inside a ```bash block**, so the numbered recipe was split by
paragraphs of prose, the fence closed in the wrong place, and the page other lanes reason from did not render. The
merge kept the markers because the conflict was resolved by hand in the wrong place and the file was never read back:
the same shape as engine#2622 (a parser returning `None` for every real line), #2630 (`0 unmaps` indistinguishable
from a dead probe) and llama.cpp#406 (counters that do not sum) -- an output nobody checked.

The check is deliberately narrow. It looks for a marker at the **start of a line** (`<<<<<<<`, `=======`, `>>>>>>>`,
`|||||||`), which is what git writes; prose is free to quote markers mid-line, and this file is such prose.

Usage:
  tools/ci/conflict_marker_check.py --check [--root .]   # rc 1 on any marker, 0 when clean
  tools/ci/conflict_marker_check.py --self-test          # rc 1 unless the arms below behave
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

# `=======` is only a conflict separator INSIDE a conflict block. On its own it is a Markdown setext underline or a
# shell rule, and the first version of this check flagged 10 of those across 4 files on a tree with no conflicts at
# all -- a check that fails on clean input is one nobody keeps. So the scan is stateful: `open` counts the markers
# git writes, and `=======` is reported only while a `<<<<<<<` is still open.
MARKERS = {
    "<<<<<<<": "conflict start",
    "|||||||": "conflict base",
    ">>>>>>>": "conflict end",
}
EQUALS = "======="

SKIP_DIRS = {".git", "build", "build-host", "build-shp", "__pycache__", "third_party", "vendor"}
SKIP_SUFFIX = {".onnx", ".gguf", ".q4nx", ".fconst", ".bin", ".so", ".a", ".o", ".hsaco", ".png", ".jpg", ".pdf"}


def scan(root: pathlib.Path) -> list[tuple[str, int, str]]:
    """[(relpath, line number, marker kind)] for every marker at the start of a line in a text file."""
    hits: list[tuple[str, int, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            if p.suffix.lower() in SKIP_SUFFIX:
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            if "\0" in text[:4096]:      # binary that slipped past the suffix list
                continue
            open_block = False
            for i, line in enumerate(text.splitlines(), 1):
                head = line[:8]
                if head.startswith("<<<<<<<"):
                    open_block = True
                    hits.append((str(p.relative_to(root)), i, "conflict start"))
                elif head.startswith("|||||||"):
                    hits.append((str(p.relative_to(root)), i, "conflict base"))
                elif head.startswith(">>>>>>>"):
                    open_block = False
                    hits.append((str(p.relative_to(root)), i, "conflict end"))
                elif open_block and head.startswith(EQUALS):
                    hits.append((str(p.relative_to(root)), i, "conflict separator"))
    return hits


def main(argv: list[str]) -> int:
    g = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g.add_argument("--check", action="store_true")
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--root", default=".")
    a = g.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        g.print_help()
        return 2
    root = pathlib.Path(a.root)
    if not root.is_dir():
        print(f"conflict_marker: no such root {root}")
        return 2
    hits = scan(root)
    if hits:
        for rel, n, what in hits[:40]:
            print(f"  {rel}:{n}: {what}")
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more")
        print(f"conflict_marker: {len(hits)} marker(s) committed in {len({h[0] for h in hits})} file(s)")
        return 1
    print("conflict_marker: 0 marker(s) at the start of any line")
    return 0


def self_test() -> int:
    import tempfile

    ok = True

    def chk(cond: bool, what: str) -> None:
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "docs").mkdir()
        (root / "build").mkdir()

        clean = (
            "# Title\n\n"
            "A paragraph that QUOTES the markers mid-line, as this tool's own docstring does: `<<<<<<< HEAD` is what\n"
            "git writes, and a `=======` rule inside a sentence is not a conflict.\n\n"
            "```\n# not a conflict either\n=======\n```\n"
        )
        # 1. a document that only mentions the markers mid-line is clean -- otherwise this check would fail on its
        #    own docstring, and a check that cannot pass on a clean tree is a check nobody keeps.
        (root / "docs" / "clean.md").write_text(clean)
        chk(scan(root) == [], "markers quoted mid-line, and a bare ======= line with no open block, are not conflicts")
        (root / "docs" / "rule.md").write_text("Title\n=====\n\nsome prose\n\n=======\n\nmore prose\n")
        chk(scan(root) == [], "a setext underline and a rule that look like markers are not conflicts either")
        (root / "docs" / "rule.md").unlink()

        # 2. the shape engine#2625 shipped: markers at the start of a line, inside a fence
        (root / "docs" / "broken.md").write_text("```bash\n# 1. pick the IOVA\n<<<<<<< HEAD\n=======\nprose\n>>>>>>> 2b1faf1b2 (msg)\n# 2. arm it\n```\n")
        hits = scan(root)
        kinds = {h[2] for h in hits}
        chk(len(hits) == 3 and kinds == {"conflict start", "conflict separator", "conflict end"},
            "the #2625 shape yields start, separator and end (3 markers)")
        chk(any(h[0].endswith("broken.md") and h[1] == 3 for h in hits), "... and the line number is the marker's own")

        # 3. build dirs and binaries are not scanned: a build tree holds no committed text, and a binary can carry
        #    bytes that look like a marker.
        (root / "build" / "x.md").write_text("<<<<<<< HEAD\n")
        (root / "blob.bin").write_bytes(b"<<<<<<< HEAD\n")
        hits = scan(root)
        chk(len(hits) == 3, "build/ and .bin are skipped")

        # 4. rc follows: 0 clean, 1 dirty, 2 no root
        chk(main(["--check", "--root", str(root)]) == 1, "--check returns 1 when a marker is committed")
        (root / "docs" / "broken.md").unlink()
        chk(main(["--check", "--root", str(root)]) == 0, "--check returns 0 on the clean tree")
        chk(main(["--check", "--root", str(root / "nope")]) == 2, "--check returns 2 for a root that is not a directory")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
