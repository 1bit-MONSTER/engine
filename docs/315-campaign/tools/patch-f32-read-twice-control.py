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
"""engine#315: the read-twice CONTROL for the f32 sync twin, as an idempotent source patch.

WHY THIS IS NEEDED, AND WHY IT MUST NOT BE APPLIED MID-RUN.

The f32 twin compares two reads of the same tensor at the same call:

  (1) the plain `ggml_backend_tensor_get` memcpy that a host-visible buffer gets - no iface call, no
      synchronisation;
  (2) the same bytes through `hrx_synchronous_d2h`.

If they disagree, that is read as "the read path is what (1) reported". But (1) and (2) are taken
SEQUENTIALLY, so a disagreement has two causes that this instrument cannot separate:

  (a) a stale host cache line - the path is the difference; or
  (b) the device wrote the buffer between the two reads - the buffer is the difference.

A second plain memcpy, taken immediately after the first and BEFORE the sync read, separates them:

  - memcpy#1 == memcpy#2  and  memcpy#1 != sync   -> path-dependent: (a), staleness;
  - memcpy#1 != memcpy#2                          -> the buffer moved between two identical reads: (b);
  - all three equal                               -> agreement, no defect on this tensor.

This is a pure addition to the diagnostic, inert unless the tensor fingerprint is enabled. It must be
applied only when no arm using the previous instrument is mid-flight, otherwise the run's records would
mix two instruments under one tag; the pre-registration for the previous run
(`315-f32-sync-twin-preregistration.md`) therefore stays as written and this becomes its own run.

Usage:  python3 patch-f32-read-twice-control.py <path-to-ggml-hrx.cpp>
Exit 0 and print what it did; exit 1 and print why if any anchor is missing.
"""
import sys

# --- anchor 1: insert the second plain read after the first read's hashes are known ----------------
A1_OLD = """                    std::fprintf(stderr, "[hrx-finite-fp] call=%d name=%s finite=%zu hash_finite=%016llx nan=%d\\n",
                                 fp_calls, node->name, finite_count, (unsigned long long) hash_finite, nan);"""

A1_NEW = """                    std::fprintf(stderr, "[hrx-finite-fp] call=%d name=%s finite=%zu hash_finite=%016llx nan=%d\\n",
                                 fp_calls, node->name, finite_count, (unsigned long long) hash_finite, nan);
                    // engine#315: the read-twice CONTROL for the f32 twin below. The twin compares a plain
                    // memcpy against hrx_synchronous_d2h, but those two reads are sequential, so a
                    // difference between them cannot by itself be attributed to the read PATH - the buffer
                    // may simply have changed in between. A second plain memcpy taken here, immediately
                    // after the first and before the sync read, separates the two: if the two memcpy reads
                    // differ, the buffer is moving under the host; if they agree and only the sync read
                    // differs, the difference is the path.
                    uint64_t hash_c = 1469598103934665603ull;
                    int      nan_c  = 0;
                    {
                        std::vector<uint8_t> host_c(bytes);
                        ggml_backend_tensor_get(node, host_c.data(), 0, bytes);
                        const float * fc = reinterpret_cast<const float *>(host_c.data());
                        for (size_t j = 0; j < n; ++j) {
                            uint32_t bits = 0;
                            std::memcpy(&bits, &fc[j], sizeof(bits));
                            hash_c = (hash_c ^ bits) * 1099511628211ull;
                            if (std::isnan(fc[j])) {
                                ++nan_c;
                            }
                        }
                    }"""

# --- anchor 2: report the control on the twin's own line ------------------------------------------
A2_OLD = '''                                             "memcpy_hash=%016llx nan_agree=%d hashes_equal=%d\\n",'''

A2_NEW = '''                                             "memcpy_hash=%016llx nan_agree=%d hashes_equal=%d "
                                             "memcpy2_hash=%016llx memcpy2_agree=%d memcpy_stable=%d\\n",'''

A3_OLD = """                                             (unsigned long long) hash_s, (unsigned long long) hash,
                                             nan_s == nan ? 1 : 0, hash_s == hash ? 1 : 0);"""

A3_NEW = """                                             (unsigned long long) hash_s, (unsigned long long) hash,
                                             nan_s == nan ? 1 : 0, hash_s == hash ? 1 : 0,
                                             (unsigned long long) hash_c, nan_c == nan ? 1 : 0,
                                             hash_c == hash ? 1 : 0);"""

EDITS = [
    ("second plain read (the control)", A1_OLD, A1_NEW),
    ("twin line: control fields", A2_OLD, A2_NEW),
    ("twin line: control args", A3_OLD, A3_NEW),
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

    if "memcpy_stable" in src:
        print("already applied: %s" % path)
        return 0

    if "hrx-f32-sync" not in src:
        print("REFUSING: the f32 sync twin is not present in %s." % path)
        print("Apply patch-input-f32-sync-twin.py first - this control reports on that twin's line.")
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
    print("  grep -c 'memcpy_stable' %s   # expect 1" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
