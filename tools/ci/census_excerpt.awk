# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# census_excerpt.awk -- select census-watch rows for the issue excerpt as BLOCKS: a row together with the "  -> "
# advice lines that follow it.
#
# Two rules pull in opposite directions, and both are already covered by arms in tests/census_watch_excerpt_test.py:
#
#   engine#2814  a row's advice must sit under the row. The old middle tier lifted EVERY "  -> " line out of its
#                class's block; five REMOTE-CODE classes share one advice sentence, so the 11-17 issue showed the
#                "!! UNCOVERED" row followed by five identical "-> custom modeling code …; alerts once 3 uploaders
#                use it" lines, while the rows they explained came 20 lines below. Adjacency is the only thing that
#                says which row an arrow belongs to, and that tier destroyed it.
#
#   engine#2711  a conclusion must survive the 40-line cap. A "  -> " line IS the whole finding for some conditions
#                (`census_watch.py:889` returns 1 for the Q4NX gap alone), so the arrows cannot be left to the tail of
#                the excerpt behind the per-row stream.
#
# The synthesis: emit blocks, and order blocks by whether they carry a conclusion -- alerts first, then rows that own
# advice, then rows that own none. Nothing is split from its row and no conclusion is behind the row stream.
#
# Usage: awk -v pat='<row pattern>' -v alert='<alert prefix>' -v want=alert|concl|rest -f census_excerpt.awk <log>
#   want=alert : the alert rows, each with its own advice
#   want=concl : the other rows that have advice of their own
#   want=rest  : the other rows that have none
#
function group(row, has_advice,    g) {
    if (row ~ alert) return "alert"
    return has_advice ? "concl" : "rest"
}
function flush(    g) {
    if (nrow == 0) return
    g = group(row[1], nrow > 1)
    if (g == want) for (i = 1; i <= nrow; i++) print row[i]
    nrow = 0
}
BEGIN { nrow = 0; want = (want == "" ? "rest" : want) }
/^ *-> / {
    # An arrow line belongs to the block that is open, if any; it can contain a row word (`  -> 3 Q4NX (NPU) pack(s)`)
    # so the shape is tested before the pattern, and an arrow with no open block is its own conclusion block.
    if (nrow == 0) { nrow = 1; row[1] = $0 } else row[++nrow] = $0
    next
}
{
    flush()
    if ($0 ~ pat) { nrow = 1; row[1] = $0 }
}
END { flush() }
