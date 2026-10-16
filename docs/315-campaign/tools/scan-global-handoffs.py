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
"""Find kernels that write global memory and read it back inside the same dispatch, with no global fence.

Rationale: engine#123/#140 were exactly this - a producer wave writes global memory, a consumer wave
reads it back, and the barrier between them fenced LDS only (barrier<workgroup>), so the consumer could
take the previous contents of the slot. Wrong when the stale bytes are finite, NaN when they are not.
That class is invisible to the host-side dependency recorder, because both sides sit inside one dispatch.

Method, over loom-libs/ops/*.loom:
  * collect each template's parameters (buffer-typed) => global buffers;
  * resolve every buffer.view / buffer.assume.alignment chain to its root buffer;
  * a root buffer that is both stored to and loaded from (through any view) is an in-kernel handoff;
  * report the kernel's barrier<global> count next to it - zero means nothing fences that handoff
    against a concurrent wave, unless the handoff is LDS (the *_stage convention) rather than global.
"""
import pathlib
import re
import sys

OPS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

view_re = re.compile(r"(%\w+)\s*=\s*buffer\.view\s+(%\w+)")
alias_re = re.compile(r"(%\w+)\s*=\s*buffer\.assume\.alignment\s+(%\w+)")
param_re = re.compile(r"^template\.decl\s+\S+\((.*)\)\s*$")

rows = []
for path in sorted(OPS.glob("*.loom")):
    src = path.read_text(errors="replace").splitlines()

    # global buffers = buffer-typed template parameters
    globals_ = set()
    for line in src:
        m = param_re.match(line.strip())
        if not m:
            continue
        for part in m.group(1).split(","):
            if ": buffer" in part:
                name = part.strip().split(":", 1)[0].strip()
                if name.startswith("%"):
                    globals_.add(name)

    root = {}
    for line in src:
        m = view_re.search(line)
        if m:
            root[m.group(1)] = m.group(2)
        m = alias_re.search(line)
        if m:
            root[m.group(1)] = root.get(m.group(2), m.group(2))

    def rootof(v):
        seen = set()
        while v in root and v not in seen:
            seen.add(v)
            v = root[v]
        return v

    stores, loads = {}, {}
    for i, line in enumerate(src, 1):
        for m in re.finditer(r"(?:view|buffer)\.store[^\n]*?(%\w+)\[", line):
            stores.setdefault(rootof(m.group(1)), []).append(i)
        for m in re.finditer(r"(?:view|buffer)\.load[^\n]*?(%\w+)\[", line):
            loads.setdefault(rootof(m.group(1)), []).append(i)

    handoffs = [b for b in stores if b in loads]
    if not handoffs:
        continue
    n_global = sum(1 for line in src if "barrier<global>" in line)
    for b in handoffs:
        rows.append((path.name, b, b in globals_, n_global, stores[b][:4], loads[b][:4]))

print(f"{'kernel':56s} {'root buffer':24s} {'global?':8s} {'#barrier<global>':16s} stores / loads")
for name, buf, is_global, n_global, st, ld in rows:
    print(f"{name:56s} {buf:24s} {str(is_global):8s} {n_global:<16d} {st} / {ld}")

print()
print("candidates: a GLOBAL handoff (global? True) with zero barrier<global> in the kernel:")
for name, buf, is_global, n_global, st, ld in rows:
    if is_global and n_global == 0:
        print(f"  {name}: {buf} (stores {st[:3]}, loads {ld[:3]})")
