#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""engine#315: the discriminating instrument for the deferred-host-writeback candidate.

WHAT IT TESTS.

`315-candidate-deferred-host-writeback-publication.md` names a mechanism at source: a device->host download
is two steps - the device writes `download_buffer`, and the step that puts those bytes into the host-visible
destination is a QUEUED CPU `memcpy` (`GraphReplayStreamState::add_host_writeback`), executed only when the
stream is next marked synchronised. The upload direction guards exactly this hazard
(`command-program-executor.cpp:881`, "Uploading the same host memory before that has happened would read the
previous contents"); the read direction has no equivalent check.

The mechanism predicts one thing no surviving alternative predicts:

    AT THE MOMENT THE ENGINE ISSUES ITS LOGITS READ, A PENDING HOST WRITEBACK IS STILL QUEUED.
    In a faulting arm the queue is non-empty at that read; in a clean arm it is empty or already published.

WHERE, AND WHY NOT `buffer_get`.

The first attempt put this in `buffer_get` and the build rejected it: `buffer_get` only has the *buffer*
context, whose `device` is a `ggml_backend_hrx_device_context`, and that struct does **not** carry
`graph_replay_state` or `stream`. Those live on the *backend* context `ggml_backend_hrx_context`
(`backend-context.h:50-67`). So the synchronous read path cannot even see the publication machinery - which
is itself worth recording.

`backend_get_tensor_async` (`ggml-hrx.cpp:454`) is the right site for a second reason as well: it is the
path the sampling read uses (`llama-context.cpp:1488` issues the logits copy with `*_async` and returns,
per section 14), it holds `ggml_backend_hrx_context` so both fields are in scope, and **it does not
synchronise** - it enqueues `hrx_stream_copy_buffer` and returns. The probe reports the queue state at that
exact point.

It is a DIAGNOSTIC: gated on GGML_HRX_WRITEBACK_READ_PROBE, inert when unset, changes no ordering. It is
pointedly not a fix - publishing before reading is a correctness change and must follow the measurement.

Usage:  python3 patch-writeback-read-probe.py <path-to-ggml-hrx.cpp>
Exit 0 and print what it did; exit 1 and print why if any anchor is missing.
"""
import sys

A1_OLD = """    ggml::hrx::HostBufferRef destination = backend_context->device->host_buffers.find(data, size);
    if (destination.valid()) {"""

A1_NEW = """    if (std::getenv("GGML_HRX_WRITEBACK_READ_PROBE") != nullptr) {
        // engine#315: the discriminating probe. This is the engine's async logits read, and it does not
        // synchronise. If a device->host download is still queued here, the bytes the caller goes on to
        // read are the destination's PREVIOUS contents (see add_host_writeback / mark_stream_synchronized).
        static int wb_read_calls = 0;
        const bool pending = backend_context->graph_replay_state.has_pending_host_writebacks();
        std::fprintf(stderr,
                     "[hrx-wb-read] call=%d op=%s name=%s bytes=%zu registered=%d pending_writebacks=%d\\n",
                     wb_read_calls++, ggml_op_name(tensor->op), tensor->name, size,
                     0, pending ? 1 : 0);
    }
    ggml::hrx::HostBufferRef destination = backend_context->device->host_buffers.find(data, size);
    if (std::getenv("GGML_HRX_WRITEBACK_READ_PROBE") != nullptr) {
        static int wb_read_reg_calls = 0;
        std::fprintf(stderr, "[hrx-wb-read] call=%d name=%s host_dst=%p registered=%d\\n", wb_read_reg_calls++,
                     tensor->name, data, destination.valid() ? 1 : 0);
    }
    if (destination.valid()) {"""

EDITS = [
    ("backend_get_tensor_async: report pending host writebacks at the engine's logits read", A1_OLD, A1_NEW),
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

    if "hrx-wb-read" in src:
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
    print("  grep -c 'hrx-wb-read' %s   # expect 2" % path)
    print()
    print("Then BUILD AND SYNTAX-CHECK BEFORE TRUSTING IT. The two fields used - backend_context->")
    print("graph_replay_state (a value member of ggml_backend_hrx_context, backend-context.h:66) and")
    print("backend_context->stream - are the ones a first attempt got wrong by reading them off the device")
    print("context instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
