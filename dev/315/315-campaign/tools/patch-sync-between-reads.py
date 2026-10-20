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
"""engine#315: the sync-between-reads instrument (task 1 of the re-scoped plan).

WHAT IT TESTS. The one measured fact this issue has not explained is that `result_norm` changes between two
CONSECUTIVE IDENTICAL plain reads in a faulting arm (`memcpy_stable=0` on 75/75 and 128/128, three reads
giving three hashes). Two explanations are already removed by measurement: the deferred host writeback (626
reads at the engine's logits read, 0 pending) and cache behaviour of a non-coherent persistent mapping (the
A/B at measured 0x52 and 0x56 diverges at 13.7% and 14.8%). What remains is that something WRITES the range,
and the question is whether that writer is ordered by the stream the harness already synchronises.

THE INSTRUMENT. The twin already reports three reads of the same tensor at the same call:
  hash            - through hrx_synchronous_d2h (the synchronising read)
  memcpy_hash     - a plain read (#1)
  memcpy2_hash    - a plain read taken immediately after #1 (#2, the control)
This adds a fresh `ggml_backend_synchronize(backend)` and then a FOURTH plain read, reported as
`hash_after_sync` with `after_sync_stable` = (hash_after_sync == memcpy2_hash).

PRE-REGISTERED READING (dev/315/315-campaign/evidence/315-PREREG-sync-between-reads.md):
  #1 != #2 and hash_after_sync == #2  -> the writer IS ordered by the stream: a fresh sync stops the
                                         movement, so the reads were racing work the sync covers. This
                                         makes GraphExecutor::execute returning without a wait the live
                                         candidate.
  #1 != #2 and hash_after_sync != #2  -> the writer is NOT ordered by context->stream at all; a fresh sync
                                         does not stop it, which kills that candidate and points at a second
                                         writer of the reused arena slot.
  #1 == #2                            -> uninformative for this question; record it.

It is a DIAGNOSTIC: gated on the same GGML_HRX_TENSOR_FP path as the twin, inert when unset.

Usage:  python3 patch-sync-between-reads.py <path-to-ggml-hrx.cpp>
Exit 0 and print what it did; exit 1 and print why if any anchor is missing.
"""
import sys

# Anchor: the end of the read-twice control block, inserted by patch-f32-read-twice-control.py.
A1_OLD = """                            if (std::isnan(fc[j])) {
                                ++nan_c;
                            }
                        }
                    }"""

A1_NEW = """                            if (std::isnan(fc[j])) {
                                ++nan_c;
                            }
                        }
                    }
                    // engine#315 (task 1): a FRESH synchronise, then a fourth plain read. The question is
                    // whether the writer that moves this range between two adjacent reads is ordered by
                    // the stream: if a fresh sync stops the movement, it was work the sync covers; if it
                    // does not, the writer is outside context->stream entirely.
                    ggml_backend_synchronize(backend);
                    uint64_t hash_after_sync = 1469598103934665603ull;
                    {
                        std::vector<uint8_t> host_s(bytes);
                        ggml_backend_tensor_get(node, host_s.data(), 0, bytes);
                        const float * fs = reinterpret_cast<const float *>(host_s.data());
                        for (size_t j = 0; j < n; ++j) {
                            uint32_t bits = 0;
                            std::memcpy(&bits, &fs[j], sizeof(bits));
                            hash_after_sync = (hash_after_sync ^ bits) * 1099511628211ull;
                        }
                    }
                    std::fprintf(stderr,
                                 "[hrx-f32-sync-after] call=%d name=%s bytes=%zu memcpy_hash=%016llx "
                                 "memcpy2_hash=%016llx after_sync_hash=%016llx after_sync_stable=%d\\n",
                                 fp_calls, node->name, bytes, (unsigned long long) hash,
                                 (unsigned long long) hash_c, (unsigned long long) hash_after_sync,
                                 hash_after_sync == hash_c ? 1 : 0);"""

EDITS = [
    ("fresh synchronise + fourth plain read after the control", A1_OLD, A1_NEW),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    try:
        src = open(path, "r", encoding="utf-8", errors="surrogateescape").read()
    except OSError as e:
        print("cannot read %s: %s" % (path, e))
        return 1

    if "hrx-f32-sync-after" in src:
        print("already applied: %s" % path)
        return 0

    if "memcpy_stable" not in src:
        print("REFUSING: the read-twice control is not present in %s." % path)
        print("Apply patch-input-f32-sync-twin.py then patch-f32-read-twice-control.py first.")
        return 1

    for name, old, new in EDITS:
        n = src.count(old)
        if n != 1:
            print("ANCHOR FAILED (%s): expected exactly 1 occurrence, found %d" % (name, n))
            print("---- anchor text was ----")
            print(old)
            return 1
        src = src.replace(old, new, 1)
        print("applied: %s" % name)

    open(path, "w", encoding="utf-8", errors="surrogateescape").write(src)
    print("wrote %s" % path)
    print()
    print("verify with:")
    print("  grep -c 'hrx-f32-sync-after' %s   # expect 1" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
