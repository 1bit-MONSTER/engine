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
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""engine#315 task 3: fix the recorded counting defect in the dense-control script before re-running it.

The record (docs/315-campaign/evidence/315-crit4-control-result.md, "Eighth harness/parse defect") records
that this script counted responses with `grep -c '"content"'`, which counts LINES. /v1/chat/completions
responses are concatenated without newlines, so fifty responses sit on one line and the count read 1 - the
script printed "FAULT or incomplete" when the truth was 50 clean responses.

Counting is fixed to occurrences, and a distinct-content count is added, so this run's numbers are directly
comparable to the recorded baseline (50 of 50 responses, 1 distinct content).

Usage: python3 fix-dense-count.py <script>
"""
import sys

OLD = """RESP=$(grep -c '"content"' "$D/responses.txt" 2>/dev/null || echo 0)"""

NEW = """# engine#315: the recorded defect was that this counted LINES. /v1/chat/completions responses are
# concatenated without newlines, so fifty responses sat on one line and the old count read 1. Count
# OCCURRENCES instead, and also report how many distinct completions there were.
RESP=$(grep -o '"content"' "$D/responses.txt" 2>/dev/null | wc -l)
DISTINCT=$(grep -o '"content":"[^"]*"' "$D/responses.txt" 2>/dev/null | sort -u | wc -l)"""

OLD2 = """say "dense control: responses=$RESP nan_guard=$NAN hsa=$HSAE content_only=$CONTENT\""""

NEW2 = """say "dense control: responses=$RESP distinct_content=$DISTINCT nan_guard=$NAN hsa=$HSAE content_only=$CONTENT\""""


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    s = open(path).read()
    for name, old, new in (("response count", OLD, NEW), ("verdict line", OLD2, NEW2)):
        n = s.count(old)
        if n != 1:
            print("ANCHOR FAILED (%s): expected 1, found %d" % (name, n))
            return 1
        s = s.replace(old, new, 1)
        print("patched: %s" % name)
    open(path, "w").write(s)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
