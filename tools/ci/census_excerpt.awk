# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# census_excerpt.awk -- select the census-watch rows for the issue excerpt, each WITH the "  -> " advice lines that
# follow it in the log.
#
# engine#2814: the workflow used to collect the rows in three hard tiers and the middle tier lifted EVERY "  -> " line
# out of its class's block. Five REMOTE-CODE classes share one advice sentence, so the 11-17 issue showed
#
#   !! UNCOVERED SimulatedForTestingClass (new class): 1 model(s) …
#      -> custom modeling code (…); alerts once 3 uploaders use it      <- belongs to a REMOTE-CODE row
#      -> custom modeling code (…); alerts once 3 uploaders use it      <- and so does this one
#      (… five identical lines …)
#   . REMOTE-CODE Qwen3LoopedForCausalLM: 14 model(s) …                 <- the rows they belonged to, 20 lines later
#
# which reads as a duplicated emission and as an alert contradicting its own "alerts once 3 uploaders" note. Neither
# was true: the block was split. Keeping each row and its advice together fixes the reading without changing what is
# selected or the order the tiers are emitted in.
#
# Usage: awk -v pat='<row pattern>' -v alert='<alert prefix>' -v want=alert|rest -f census_excerpt.awk <log>
#   want=alert : the alert rows only
#   want=rest  : the other selected rows (an alert row is skipped together with its advice, so nothing is emitted twice)
#
BEGIN { keep = 0 }
{
    is_row = ($0 ~ pat)
    if (is_row && (want == "alert" ? ($0 ~ alert) : !($0 ~ alert))) { print; keep = 1; next }
    if (is_row) { keep = 0; next }
    if (keep && $0 ~ /^ *-> /) { print; next }
    keep = 0
}
