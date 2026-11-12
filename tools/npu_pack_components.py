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
"""List the components of a Ryzen AI OGA pack, including the sibling-directory ones.

A partitioned pack is not one flat component: the pack root holds the *text* component
(`text.bin`, `dd_metastate_Llm_Token_*`, `cache/`) and a vision component ships as a
sibling directory (`partitioned/`, `partitioned_patch1_sdgemm_leftover/`, `vit_*`), each a
complete component set of its own. The engine has no notion of these directories yet, so
this tool exists to make the rule checkable on the real files before the loader learns it
(engine #1503, plan W4).

Discovery rule: a directory is a component when it holds a runnable description of one graph -
either compiled metastate (`dd_metastate_*.fconst`) or an ONNX graph (`*.onnx`, counting the
`*.onnx.data` sidecar). Both forms occur in the same pack: on `Qwen3.5_4B_rai_1.9.0_npu`,
`partitioned/` is the compiled vision component (`Llm_Vision_Vision_node_conv3d`) while
`partitioned_patch1_sdgemm_leftover/` is the same node as `vision.onnx` + `.onnx.data` +
`vision_strategy.yaml`. The pack root is always the `text` component. No manifest is invented
where the pack has none, and variant selection stays the caller's decision: this tool prints
every component, with its form, rather than silently picking one.
"""

import argparse
import os
import sys


def components(root):
    """(name, path, files, total_bytes, form) for the root and every sibling component."""
    out = []
    for name, path in [("text", root)] + sorted(
        (entry, os.path.join(root, entry))
        for entry in os.listdir(root)
        if os.path.isdir(os.path.join(root, entry))
    ):
        entries = sorted(os.listdir(path))
        fconsts = [f for f in entries if f.startswith("dd_metastate_") and f.endswith(".fconst")]
        onnx = [f for f in entries if f.endswith(".onnx") or f.endswith(".onnx.data")]
        if not fconsts and not onnx:
            continue
        form = "compiled" if fconsts else "onnx"
        files = fconsts or onnx
        total = sum(os.path.getsize(os.path.join(path, f)) for f in files)
        out.append((name, path, files, total, form))
    return out


def selftest():
    """Known-answer check on a synthetic layout: three components, both forms, plus the negative control."""
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # text (compiled) at the root, a compiled vision sibling, an ONNX vision sibling, and a decoy dir with no graph.
        for d, names in ((tmp, ["dd_metastate_Llm_Token_x.fconst"]),
                         (os.path.join(tmp, "partitioned"), ["dd_metastate_Llm_Vision_Y.fconst"]),
                         (os.path.join(tmp, "partitioned_patch1_x"), ["vision.onnx", "vision.onnx.data"]),
                         (os.path.join(tmp, "decoy"), ["readme.txt"])):
            os.makedirs(d, exist_ok=True)
            for n in names:
                with open(os.path.join(d, n), "wb") as f:
                    f.write(b"x" * 16)
        got = {n: m for n, _p, _f, _b, m in components(tmp)}
        want = {"text": "compiled", "partitioned": "compiled", "partitioned_patch1_x": "onnx"}
        if got != want:
            print(f"FAIL discovery: got {got} want {want}"); ok = False
        if "decoy" in got:
            print("FAIL discovery: a directory with no graph was reported as a component"); ok = False
        # negative control: an empty directory has no components at all.
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty, exist_ok=True)
        if components(empty):
            print("FAIL control: components found in an empty directory"); ok = False
    print("selftest: " + ("PASS (3 components, both forms, decoy ignored, empty refused)" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack", nargs="?", help="pack directory (omit with --selftest)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--selftest", action="store_true", help="build a synthetic pack layout and check the rule")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.pack:
        print("pack directory required (or --selftest)", file=sys.stderr)
        return 2

    if not os.path.isdir(args.pack):
        print(f"not a directory: {args.pack}", file=sys.stderr)
        return 2
    comps = components(args.pack)
    if not comps:
        print("no components found (no dd_metastate_*.fconst and no *.onnx in the root or a sibling directory)", file=sys.stderr)
        return 1

    if args.json:
        import json

        print(json.dumps([{"name": n, "dir": os.path.basename(p), "form": m, "files": f, "bytes": b} for n, p, f, b, m in comps], indent=2))
        return 0

    print(f"pack: {args.pack}")
    for name, path, files, total, form in comps:
        print(f"  {name:<8} {os.path.basename(path) or '.':<34} {form:<8} {len(files):>2} file  {total / 2**20:>9.1f} MiB")
        for f in files:
            print(f"      {f}")
    if len(comps) > 1:
        # Two components with the same role are variants: the caller must choose, never a glob order.
        print(f"  note: {len(comps)} components; the root is 'text' and the rest are siblings - "
              f"a sibling whose role matches another sibling is a VARIANT and must be selected explicitly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
