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
#
# engine#315: patch 2 of the tensor fingerprint. Section 53 of dev/315/315-mechanism.md withdrew the by-name
# per-node attribution because the fingerprint reads host-visible tensors through ggml_backend_tensor_get, which
# for those is a plain memcpy with no synchronisation - the reads are not a snapshot of one instant. This patch
# makes the reads ordered by synchronising the backend once per graph, before the read loop.
#
# The trade-off is explicit and is why this is a separate build and a separate run: a synchronise per graph is
# itself an ordering intervention, so this instrument can inhibit the very fault it is meant to locate. It is
# only worth building if the paired GRAPH_SYNC experiment (315gsync2) shows the fault SURVIVES a sync; if the
# fault disappears under synchronisation, the mechanism is the ordering and there is nothing left to locate.
#
# Idempotent: refuses to apply twice.
import hashlib
import pathlib
import sys

SRC = pathlib.Path.home() / "wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp"

ANCHOR = """    if (result.success()) {
        const char * want = std::getenv("GGML_HRX_TENSOR_FP");"""

REPLACEMENT = """    if (result.success()) {
        // engine#315: order the fingerprint's own reads. Everything the fingerprint names on the MoE path
        // lives in a host-visible, non-coherent arena (44048640 bytes; see the buffer trace), and
        // ggml_backend_tensor_get on a host-visible tensor is a plain memcpy - no iface call, no
        // synchronisation - so without this the per-node lines are not a snapshot of one instant and the
        // per-node attribution is invalid. This costs one wait per graph and is an intervention in its own
        // right: read the run's caveat before using its output.
        ggml_backend_synchronize(backend);
        const char * want = std::getenv("GGML_HRX_TENSOR_FP");"""

MARKER = "ggml_backend_synchronize(backend);\n        const char * want"


def main() -> int:
    text = SRC.read_text()
    if MARKER in text:
        print("already patched")
        return 0
    if text.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {text.count(ANCHOR)} times, expected 1")
        return 1
    text = text.replace(ANCHOR, REPLACEMENT)
    SRC.write_text(text)
    print(f"patched {SRC}")
    print(f"  synchronise-before-read: {text.count(MARKER)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
