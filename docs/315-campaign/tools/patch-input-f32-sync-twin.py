#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""engine#315: the two instrument changes the next arm needs, as an idempotent source patch.

WHY THESE TWO, AND NOTHING ELSE.

(1) f32 sync twin. The i32 branch of the tensor fingerprint has always read each tensor twice - once by the
    plain `ggml_backend_tensor_get` memcpy that a host-visible buffer gets, and once through
    `hrx_synchronous_d2h` - and printed both so they could be compared. That twin is what established the
    record's "the two read paths agree" negative. The f32 branch has NO twin, and the f32 tensors are the ones
    that actually carry the NaN. So the single measurement that would settle whether the fault is a stale host
    read has never been taken on any tensor that matters. This adds the twin, and it compares a full content
    hash rather than only a NaN count, so a divergence confined to a tensor's interior is still visible.

(2) head:<N> selector. The graph input is `ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens)`
    and it is never given a name, so `GGML_HRX_TENSOR_FP`'s name list cannot match it and `tail:<N>` never
    reaches it - the input sits at the head of `graph->nodes`, not the tail. It is the one tensor upstream of
    everything, it has never been fingerprinted in any arm, and the record's own PR #90 was an input-path
    defect. `head:<N>` selects the first N nodes, symmetric with the existing `tail:<N>`.

Both are diagnostics under an env var; neither changes behaviour when the var is unset. Forward-only by
construction: the pin is untouched, and this is an addition to the fork-side ggml-hrx.

Usage:  python3 patch-input-f32-sync-twin.py <path-to-ggml-hrx.cpp>
Exit 0 and print what it did; exit 1 and print why if any anchor is missing.
"""
import sys

# --- anchor 1: the selector parse, so head:<N> sits beside tail:<N> -------------------------------
A1_OLD = """                int tail_n = 0;
                if (std::strncmp(want, "tail:", 5) == 0) {
                    tail_n = std::atoi(want + 5);
                }"""
A1_NEW = """                int tail_n = 0;
                if (std::strncmp(want, "tail:", 5) == 0) {
                    tail_n = std::atoi(want + 5);
                }
                // engine#315: "head:<N>" selects the first N nodes, symmetric with tail:<N>. The graph input
                // is created unnamed (ggml_new_tensor_2d with no ggml_set_name), so a name list cannot reach
                // it and tail never does - it is at the head. It is the only tensor upstream of everything.
                int head_n = 0;
                if (std::strncmp(want, "head:", 5) == 0) {
                    head_n = std::atoi(want + 5);
                }"""

# --- anchor 2: the loop bounds, currently tail-only ------------------------------------------------
A2_OLD = """                const int  fp_first    = tail_n > 0 && graph->n_nodes > tail_n ? graph->n_nodes - tail_n : 0;
                for (int i = fp_first; i < graph->n_nodes; ++i) {"""
A2_NEW = """                const int  fp_first    = tail_n > 0 && graph->n_nodes > tail_n ? graph->n_nodes - tail_n : 0;
                const int  fp_last     = head_n > 0 && graph->n_nodes > head_n ? head_n : graph->n_nodes;
                for (int i = fp_first; i < fp_last; ++i) {"""

# --- anchor 3: name-list filter must not reject head: ---------------------------------------------
A3_OLD = """                    if (tail_n == 0 && !hrx_fp_name_listed(want, node->name)) {
                        continue;
                    }
                    if (tail_n == 0 && node->type == GGML_TYPE_I32) {"""
A3_NEW = """                    if (tail_n == 0 && head_n == 0 && !hrx_fp_name_listed(want, node->name)) {
                        continue;
                    }
                    if (tail_n == 0 && head_n == 0 && node->type == GGML_TYPE_I32) {"""

# --- anchor 4: the f32 sync twin, inserted before the wavemask block ------------------------------
A4_OLD = """                    if (std::getenv("GGML_HRX_TENSOR_FP_WAVEMASK") != nullptr) {"""

A4_NEW = """                    // engine#315: the f32 twin of the i32 branch's (1)/(2) read pair. The plain memcpy
                    // above is what a host-visible buffer gets from ggml_backend_tensor_get - no iface call,
                    // no synchronisation - and the record's MoE buffers live in a host-visible NON-COHERENT
                    // arena. Read the same bytes through the synchronising path and compare a full content
                    // hash, so a divergence anywhere in the tensor is visible, not just one in the prefix.
                    {
                        ggml_backend_buffer_t node_buffer =
                            node->view_src != nullptr ? node->view_src->buffer : node->buffer;
                        if (node_buffer != nullptr) {
                            auto *       nctx = buffer_context(node_buffer);
                            const size_t noff = tensor_offset(nctx, node);
                            std::vector<uint8_t> hsync(bytes);
                            if (HRX_CHECK(hrx_synchronous_d2h(nctx->device->device, nctx->buffer, noff,
                                                              hsync.data(), bytes))) {
                                const float * g       = reinterpret_cast<const float *>(hsync.data());
                                uint64_t      hash_s  = 1469598103934665603ull;
                                int           nan_s   = 0;
                                int64_t       first_s = -1;
                                for (size_t j = 0; j < n; ++j) {
                                    uint32_t bits = 0;
                                    std::memcpy(&bits, &g[j], sizeof(bits));
                                    hash_s = (hash_s ^ bits) * 1099511628211ull;
                                    if (std::isnan(g[j])) {
                                        ++nan_s;
                                        if (first_s < 0) {
                                            first_s = (int64_t) j;
                                        }
                                    }
                                }
                                const int64_t ne0s = node->ne[0] > 0 ? node->ne[0] : 1;
                                std::fprintf(stderr,
                                             "[hrx-f32-sync] call=%d name=%s bytes=%zu nan=%d of %zu "
                                             "ne0=%lld first=%lld row=%lld hash=%016llx "
                                             "memcpy_hash=%016llx nan_agree=%d hashes_equal=%d\\n",
                                             fp_calls, node->name, bytes, nan_s, n, (long long) ne0s,
                                             (long long) first_s,
                                             (long long) (first_s >= 0 ? first_s / ne0s : -1),
                                             (unsigned long long) hash_s, (unsigned long long) hash,
                                             nan_s == nan ? 1 : 0, hash_s == hash ? 1 : 0);
                            }
                        }
                    }
                    if (std::getenv("GGML_HRX_TENSOR_FP_WAVEMASK") != nullptr) {"""

EDITS = [
    ("head:<N> selector parse", A1_OLD, A1_NEW),
    ("head:<N> loop bound", A2_OLD, A2_NEW),
    ("name-list filter excludes head:", A3_OLD, A3_NEW),
    ("f32 fp/sync twin with content hash", A4_OLD, A4_NEW),
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

    if "hrx-f32-sync" in src and "head_n" in src:
        print("already applied: %s" % path)
        return 0

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
    print("  grep -c 'hrx-f32-sync' %s        # expect 1" % path)
    print("  grep -c 'head_n' %s              # expect 5" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
