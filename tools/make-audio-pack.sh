#!/usr/bin/env bash
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
# make-audio-pack: lay out an audio pack with the head weight stored the way the matmul reads it (engine#2366).
#
# `fold_initializer_transpose.py` ships as a tool, and engine#2728 recorded the residual plainly: a tool that
# nothing calls leaves an unfurled pack paying 26 x 57.8 ms (host) or 26 x 101.3 ms (device) per transcription
# for a transpose of a weight that cannot change. This is the caller.
#
#   tools/make-audio-pack.sh --out DIR ENCODER.onnx DECODER.onnx [SIDECAR...]
#
# Refuses rather than writing a pack it cannot fold: a decoder whose fold would be skipped (no python `onnx`)
# is an error here, not a warning, because the whole point is that the pack on disk is the folded one.
# Sidecars (a whisper export's `.onnx.data`) are copied beside the graph they belong to; the fold rewrites the
# .onnx and leaves the external data alone, so a graph with a sidecar keeps it.

set -euo pipefail

usage() {
    echo "usage: make-audio-pack.sh --out DIR ENCODER.onnx DECODER.onnx [SIDECAR...]" >&2
    echo "       folds the decoder's initializer-fed Transpose before writing the pack (engine#2366)" >&2
    exit 2
}

out=
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out=${2:-}; shift 2 ;;
        -h|--help) usage ;;
        *) break ;;
    esac
done
[ -n "$out" ] && [ $# -ge 2 ] || usage

here=$(cd "$(dirname "$0")" && pwd)
fold=$here/fold_initializer_transpose.py
encoder=$1; decoder=$2; shift 2
sidecars=("$@")

command -v python3 >/dev/null || { echo "error: python3 is required to fold the decoder" >&2; exit 1; }
python3 -c "import onnx, numpy" 2>/dev/null || {
    echo "error: the python 'onnx' module is required: without it the pack would be written unfurled," >&2
    echo "       and an unfurled pack pays a 26x transpose per transcription (engine#2366)." >&2
    echo "       install it (pip install onnx numpy) or fold the decoder yourself with:" >&2
    echo "         $fold $decoder" >&2
    exit 1
}
for f in "$encoder" "$decoder" "${sidecars[@]+"${sidecars[@]}"}"; do
    [ -f "$f" ] || { echo "error: no such file: $f" >&2; exit 1; }
done

mkdir -p "$out"
cp -f "$encoder" "$out/encoder_model.onnx"
for f in "${sidecars[@]+"${sidecars[@]}"}"; do cp -f "$f" "$out/$(basename "$f")"; done

# The decoder is written folded, then read back: writing it and trusting the call is the mistake this script
# exists to prevent, so the check runs on the file that is on disk.
python3 "$fold" "$decoder" "$out/decoder_model.onnx"
if ! python3 "$fold" --check "$out/decoder_model.onnx"; then
    echo "error: the pack's decoder still carries an initializer-fed Transpose: $out/decoder_model.onnx" >&2
    exit 1
fi

echo "pack: $out (encoder + folded decoder${sidecars[0]+, $((${#sidecars[@]})) sidecar(s)})"
