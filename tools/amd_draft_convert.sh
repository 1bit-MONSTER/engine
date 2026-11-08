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
# amd_draft_convert.sh -- convert an AMD draft-model repo (PARD / PARD-2 / any standard-architecture checkpoint) to GGUF,
# without touching the shared Python environment.
#
# Why a wrapper at all: on this box HF->GGUF conversion is broken for reasons that have nothing to do with the model --
#   * every vendored llama.cpp `convert_hf_to_gguf.py` requires `huggingface-hub>=1.5,<2`, and the box has 2.4.0;
#   * `transformers` is not importable (its `~/.local` package directory is incomplete: no `models/` subdirectory);
#   * and the converter also needs `sentencepiece` for the sentencepiece vocabularies.
# Installing those into the shared environment would change every lane's Python, so this script builds them into a
# throwaway prefix under /tmp and runs the converter with PYTHONPATH pointed at it.  The engine tree and the user's
# site-packages stay exactly as they were.
#
# PARD / PARD-2 are ordinary causal LMs (`Qwen2ForCausalLM` / `Qwen3ForCausalLM` / `LlamaForCausalLM`) -- what makes them
# "parallel" drafters is the inference mask, not the architecture -- so they convert with the stock script and no vendor
# code.  Gumiho is the opposite case: a vendor hybrid head with Full Tree Attention and no config.json; it does NOT convert
# and needs a tree verifier, which our flat-batch loop does not have (docs/amd-drafters.md says why).
#
#   amd_draft_convert.sh <hf-repo-dir> <out.gguf> [quant]
#     e.g. amd_draft_convert.sh ~/models/amd-rai/pard/PARD-Qwen3-0.6B /tmp/pard-qwen3-0.6b-f16.gguf f16
#          (then: llama-quantize <out.gguf> <out-Q4_K_M.gguf> Q4_K_M)
#
# The repo must already be downloaded (`hf download <repo> --local-dir ...`); this script never fetches weights.
set -u
REPO=${1:?usage: amd_draft_convert.sh <hf-repo-dir> <out.gguf> [f16|bf16|q8_0 ...]}
OUT=${2:?usage: amd_draft_convert.sh <hf-repo-dir> <out.gguf> [f16|bf16|q8_0 ...]}
QUANT=${3:-f16}
PREFIX=${PREFIX:-/tmp/amd-draft-convert}
CONVERTER=${CONVERTER:-}
THERMAL=${THERMAL:-/home/bcloud/lb/thermal-run.sh}

[ -f "$REPO/config.json" ] || { echo "amd_draft_convert: no config.json in $REPO -- this is not a standard HF checkpoint."; \
    echo "  Gumiho is this case (vendor head, no config): see docs/amd-drafters.md. Refusing rather than guessing."; exit 2; }

if [ -z "$CONVERTER" ]; then
    CONVERTER=$(ls -t /home/bcloud/wt/*/third_party/llama.cpp/convert_hf_to_gguf.py /home/bcloud/wt/*/convert_hf_to_gguf.py 2>/dev/null | head -1)
fi
[ -n "$CONVERTER" ] && [ -f "$CONVERTER" ] || { echo "amd_draft_convert: no convert_hf_to_gguf.py found; set CONVERTER="; exit 2; }

if [ ! -d "$PREFIX/transformers" ]; then
    echo "== building the converter's python prefix in $PREFIX (not the shared env)"
    python3 -m venv --system-site-packages "$PREFIX-venv" >/dev/null 2>&1 || { echo "venv failed"; exit 3; }
    "$PREFIX-venv/bin/pip" install -q --no-user --target "$PREFIX" "transformers>=4.44" "huggingface-hub<2" sentencepiece \
        || { echo "pip install failed"; exit 3; }
fi
PY="$PREFIX-venv/bin/python"; [ -x "$PY" ] || PY=python3

echo "== converting $REPO -> $OUT ($QUANT)"
PYTHONPATH="$PREFIX" $THERMAL "$PY" "$CONVERTER" "$REPO" --outfile "$OUT" --outtype "$QUANT"
rc=$?
[ $rc = 0 ] && echo "== ok: $OUT ($(du -h "$OUT" | cut -f1))"
echo "== follow-up for a speed measurement:"
echo "   tools/spec_decode_speedup.sh <target.gguf> $OUT 64 4        # CPU:  -ngl 0   (cannot exceed 1x; engine#1423)"
echo "   BIN=~/wt/i1251/pincur/llama-server SRV_ARGS='-dev HRX0 -ngl 99 -fa on -devd HRX0' \\"
echo "     DRAFT_MAX_FLAG=--spec-draft-n-max tools/spec_decode_speedup.sh <target.gguf> $OUT 64 4   # device"
exit $rc
