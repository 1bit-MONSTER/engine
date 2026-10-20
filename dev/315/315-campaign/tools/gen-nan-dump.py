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
"""Generate the tail/NaN-only fingerprint patch from the live instrumented source.

Run on the box, inside ~/wt/llama-315copy.  Writes /tmp/315-nan-dump.patch.
Each anchor is asserted to occur exactly once, so a silent no-op is impossible.
"""
import pathlib
import subprocess
import sys

REL = "ggml/src/ggml-hrx/ggml-hrx.cpp"
src = pathlib.Path(REL)
orig = src.read_text()
new = orig

A_OLD = """                for (int i = 0; i < graph->n_nodes; ++i) {
                    ggml_tensor * node = graph->nodes[i];
                    if (node == nullptr || node->name[0] == '\\0' || std::strstr(want, node->name) == nullptr) {
                        continue;
                    }
"""

A_NEW = """                // engine#315: GGML_HRX_TENSOR_FP also accepts "tail:<N>", which selects the last N nodes
                // of the graph by position instead of by name. The stage that manufactures the NaN is not
                // known yet, so position is the only way to walk the head-end pipeline without guessing
                // tensor names; N bounds the extra device->host traffic per graph. GGML_HRX_TENSOR_FP_NANONLY
                // keeps only the NaN-bearing lines, which is what makes a position sweep readable.
                int tail_n = 0;
                if (std::strncmp(want, "tail:", 5) == 0) {
                    tail_n = std::atoi(want + 5);
                }
                const bool fp_nan_only = std::getenv("GGML_HRX_TENSOR_FP_NANONLY") != nullptr;
                const int  fp_first    = tail_n > 0 && graph->n_nodes > tail_n ? graph->n_nodes - tail_n : 0;
                for (int i = fp_first; i < graph->n_nodes; ++i) {
                    ggml_tensor * node = graph->nodes[i];
                    if (node == nullptr) {
                        continue;
                    }
                    if (tail_n == 0 && (node->name[0] == '\\0' || std::strstr(want, node->name) == nullptr)) {
                        continue;
                    }
"""

B_OLD = """                    std::fprintf(stderr,
                                 "[hrx-tensor-fp] call=%d name=%s bytes=%zu nan=%d of %zu hash=%016llx\\n",
                                 fp_calls, node->name, bytes, nan, n, (unsigned long long) hash);
"""

B_NEW = """                    if (fp_nan_only && nan == 0) {
                        continue;
                    }
                    std::fprintf(stderr,
                                 "[hrx-tensor-fp] call=%d name=%s op=%s bytes=%zu nan=%d of %zu hash=%016llx\\n",
                                 fp_calls, node->name, ggml_op_name(node->op), bytes, nan, n, (unsigned long long) hash);
"""

for tag, old, rep in (("A", A_OLD, A_NEW), ("B", B_OLD, B_NEW)):
    count = new.count(old)
    if count != 1:
        sys.exit(f"ABORT: anchor {tag} matched {count} times, expected 1")
    new = new.replace(old, rep)

tmp = pathlib.Path("/tmp/ggml-hrx.cpp.new")
tmp.write_text(new)
res = subprocess.run(
    ["diff", "-u", f"--label=a/{REL}", f"--label=b/{REL}", str(src), str(tmp)],
    capture_output=True, text=True,
)
if res.returncode != 1:
    sys.exit(f"ABORT: diff returned {res.returncode}: {res.stderr}")
patch = pathlib.Path("/tmp/315-nan-dump.patch")
patch.write_text(res.stdout)
print(f"wrote {patch} ({len(res.stdout)} bytes, {res.stdout.count(chr(10))} lines)")
