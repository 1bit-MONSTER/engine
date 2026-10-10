#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# engine#315: add a DOSE to the graph-sync diagnostic.
#
# The pin already has two forms, both binary in effect:
#   GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=<anything>                    -> synchronise after EVERY replay
#   GGML_HRX_DIAGNOSTIC_GRAPH_SYNC_PROGRAM_COMMANDS=<dispatch>   -> synchronise only when a replay has exactly
#                                                                   that many dispatches
# (runtime/command-program-executor.cpp:2034-2035; the sync itself is right below, and it is followed by
# mark_graph_replay_synchronized(context).)
#
# A binary cell can say "the sync removes the fault"; it cannot say the fault scales with how often the consumer
# waits. This patch adds
#   GGML_HRX_DIAGNOSTIC_GRAPH_SYNC_EVERY=N     -> synchronise every Nth replay (N=1 equals the existing knob,
#                                                 a large N is the no-knob cell)
# so the same run can carry several doses and the fault rate can be read as a function of the dose. Section 57 of
# docs/315-mechanism.md is the pre-commitment that says why this control is needed if the binary cell comes back
# clean, and nothing here changes the frozen harness or any registered criterion.
#
# Idempotent: refuses to apply twice.
import pathlib
import sys

SRC = pathlib.Path.home() / "wt/llama-315copy/ggml/src/ggml-hrx/runtime/command-program-executor.cpp"

ANCHOR = """    if (diagnostic_sync) {
        if (ErrorResult error = take_status(hrx_stream_synchronize(context.stream))) {"""

REPLACEMENT = """    // engine#315: dose. GGML_HRX_DIAGNOSTIC_GRAPH_SYNC_EVERY=N synchronises every Nth replay, so the fault
    // rate can be read as a function of how often the consumer waits instead of as a single binary cell.
    bool diagnostic_sync_dose = diagnostic_sync;
    if (!diagnostic_sync_dose) {
        const char * diagnostic_sync_every = std::getenv("GGML_HRX_DIAGNOSTIC_GRAPH_SYNC_EVERY");
        if (diagnostic_sync_every != nullptr && diagnostic_sync_every[0] != '\\0') {
            static std::atomic<uint64_t> diagnostic_replay_counter{0};
            const uint64_t every = std::strtoull(diagnostic_sync_every, nullptr, 0);
            const uint64_t n     = diagnostic_replay_counter.fetch_add(1) + 1;
            diagnostic_sync_dose = every > 0 && (n % every) == 0;
        }
    }
    if (diagnostic_sync_dose) {
        if (ErrorResult error = take_status(hrx_stream_synchronize(context.stream))) {"""

INCLUDE = "#include <atomic>"
MARKER = "diagnostic_sync_dose"


def main() -> int:
    text = SRC.read_text()
    if MARKER in text:
        print("already patched")
        return 0
    if text.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {text.count(ANCHOR)} times, expected 1")
        return 1
    text = text.replace(ANCHOR, REPLACEMENT)
    if INCLUDE not in text:
        # put it beside the first #include so the file stays in its own style
        first = text.find("#include")
        if first < 0:
            print("FAIL: no #include found")
            return 1
        text = text[:first] + INCLUDE + "\n" + text[first:]
    SRC.write_text(text)
    print(f"patched {SRC}")
    print(f"  dose branch: {text.count(MARKER)}  <atomic>: {text.count(INCLUDE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
