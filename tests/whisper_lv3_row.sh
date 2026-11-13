#!/bin/sh
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
#
# tests/whisper_lv3_row.sh -- the whisper-large-v3 half of the transcription row (engine#1783).
#
# Why a second pack and not just the first: every constant that fit whisper-small was wrong for large-v3 -- the token window
# (128 vs 448), the vocabulary width (51865 vs 51866) and the <|transcribe|>/<|notimestamps|> ids that follow from it
# (50359/50363 vs 50360/50364). Each one was invisible until the tool was pointed at the second pack, and each produced a
# plausible-looking wrong answer rather than a crash. So the row runs BOTH families and requires the same ids from both.
#
# Usage: whisper_lv3_row.sh <whisper_transcribe> <ids.json>
#   ONEBIT_TEST_WHISPER_PACK_LV3 and ONEBIT_TEST_WHISPER_WAV_LV3 name the pack and the wav; unset means skip (77), because a
#   pack is a multi-GB download and cannot live in the tree.
set -u
bin=${1:?whisper_transcribe binary}
ids=${2:?expected ids json}
[ -n "${ONEBIT_TEST_WHISPER_PACK_LV3:-}" ] || {
    echo "SKIP: set ONEBIT_TEST_WHISPER_PACK_LV3 (and ONEBIT_TEST_WHISPER_WAV_LV3) to run the large-v3 arm"
    exit 77
}
[ -n "${ONEBIT_TEST_WHISPER_WAV_LV3:-}" ] || {
    echo "SKIP: ONEBIT_TEST_WHISPER_PACK_LV3 is set but ONEBIT_TEST_WHISPER_WAV_LV3 is not"
    exit 77
}
exec "$bin" "$ONEBIT_TEST_WHISPER_PACK_LV3" "$ONEBIT_TEST_WHISPER_WAV_LV3" "$ids"
