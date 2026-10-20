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
import re
import sys

for tag in ("315fbar", "315nofa", "315pm4", "315gsync2", "315row", "315fp2"):
    path = f"/home/bcloud/wt/{tag}-console.log"
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        print(f"{tag}: {e}")
        continue
    rows = []
    for line in fh:
        m = re.match(
            r"^\s*pair\s+(\d+)\s+arm\s+([cl]):\s+(\S+)\s+nan=(\d+)\s+req=(\d+)", line
        )
        if not m:
            continue
        g = re.search(r"gtt_start=(\d+)G", line)
        rows.append(
            dict(
                pair=int(m.group(1)),
                cell=m.group(2),
                v=m.group(3),
                nan=int(m.group(4)),
                req=int(m.group(5)),
                gtt=int(g.group(1)) if g else -1,
            )
        )
    print(f"=== {tag} ===")
    for c in ("c", "l"):
        sel = [r for r in rows if r["cell"] == c]
        inb = [r for r in sel if r["gtt"] >= 95]
        oor = [r for r in sel if r["gtt"] < 95]
        faults = [r for r in inb if r["v"] == "FAULT"]
        facea = [r for r in faults if r["nan"] > 0]
        unatt = [r for r in faults if r["nan"] == 0]
        print(
            f"  cell {c}: arms={len(sel)} in-regime={len(inb)} oor={len(oor)} "
            f"fault={len(faults)} faceA={len(facea)} unattributed={len(unatt)}"
        )
