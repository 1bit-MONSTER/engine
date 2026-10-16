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
"""Per-function version of the binding-access audit.

The first pass analysed whole .loom files, so a store in one template could be attributed to a *different*
export from the same file - which inflates the finding set. This version restricts the store/load analysis to
the body of the function that generates the export's own symbol: it locates `template.def<@...> device @<symbol>(`
and stops at the next top-level `template.def` / `kernel.def` / `template.decl`.

Dangerous direction (reported): a binding the manifest calls `read` whose function body stores to it, or one it
calls `write` whose body loads from it - both mean the recorder has no edge where the kernel needs one.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
view_re = re.compile(r"(%\w+)\s*=\s*buffer\.view\s+(%\w+)")
alias_re = re.compile(r"\s*(%\w+(?:\s*,\s*%\w+)*)\s*=\s*buffer\.assume\.(?:alignment|noalias)\s+(%\w+(?:\s*,\s*%\w+)*)")
top_re = re.compile(r"^(template\.def|template\.decl|kernel\.def)")


def func_body(lines, symbol):
    start = None
    for i, line in enumerate(lines):
        # two spellings in the corpus: `template.def<@...> device @symbol(` and
        # `kernel.def target(...) export("symbol")`
        if line.startswith("template.def") and ("@" + symbol + "(") in line:
            start = i
            break
        if line.startswith("kernel.def") and ('export("' + symbol + '")') in line:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if top_re.match(lines[j]):
            end = j
            break
    return lines[start:end]


def analyse(body):
    root = {}
    for line in body:
        m = view_re.search(line)
        if m:
            root[m.group(1)] = m.group(2)
        m = alias_re.search(line)
        if m:
            lhs = [x.strip() for x in m.group(1).split(",")]
            rhs = [x.strip() for x in m.group(2).split(",")]
            for a, b in zip(lhs, rhs):
                root[a] = root.get(b, b)

    def rootof(v):
        seen = set()
        while v in root and v not in seen:
            seen.add(v)
            v = root[v]
        return v

    stores, loads = set(), set()
    for line in body:
        for m in re.finditer(r"(?:view|buffer)\.store[^\n]*?(%\w+)\[", line):
            stores.add(rootof(m.group(1)))
        for m in re.finditer(r"(?:view|buffer)\.load[^\n]*?(%\w+)\[", line):
            loads.add(rootof(m.group(1)))
    return stores, loads


def main():
    dangerous, conservative, checked, unresolved = [], [], 0, 0
    for manifest in sorted(ROOT.glob("*/manifest.json")):
        d = json.loads(manifest.read_text())
        for export in d.get("exports", []):
            src = manifest.parent / export["source"]
            if not src.exists():
                unresolved += 1
                continue
            body = func_body(src.read_text(errors="replace").splitlines(), export["symbol"])
            if body is None:
                unresolved += 1
                continue
            stores, loads = analyse(body)
            for name, access in zip(export.get("bindings", []), export.get("binding_access", [])):
                param = "%" + name
                wrote, read = param in stores, param in loads
                checked += 1
                if access == "read" and wrote:
                    dangerous.append((export["name"], name, "declared read, body stores"))
                elif access == "write" and read:
                    dangerous.append((export["name"], name, "declared write, body loads"))
                elif access == "read_write" and not (wrote and read):
                    conservative.append((export["name"], name, "read_write but body is one-sided"))
    print(f"bindings checked per function: {checked}   (unresolved exports: {unresolved})")
    print(f"DANGEROUS (no edge where the recorder needs one): {len(dangerous)}")
    seen = set()
    for name, binding, why in dangerous:
        if name in seen:
            continue
        seen.add(name)
        print(f"   {name}: {binding} - {why}")
    print(f"conservative (extra edges only): {len(conservative)}")


if __name__ == "__main__":
    sys.exit(main())
