#!/usr/bin/env bash
# verify.sh <tag> [depth ...]
# Persists every claim the goal makes: git rev, alignment literal actually in the
# source that is built, quiet-box proof, n, per-run verdicts, and the env used.
set -u
source "$(dirname "$0")/env.sh"
BIN="$HRX123_BIN"
MODEL="$HRX123_MODEL"
LLAMA="$HOME/1bit-engine/third_party/llama.cpp"
TAG="${1:?tag}"; shift
DEPTHS=("$@")
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(2100)
LOG="$(dirname "$0")/logs/verify-$TAG.log"
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

say "=== verify $TAG @ $(date -Is) ==="
say "llama.cpp rev: $(git -C "$LLAMA" rev-parse HEAD) ($(git -C "$LLAMA" branch --show-current))"
say "llama.cpp status: $(git -C "$LLAMA" status --short | tr '\n' ';' )"
say "alignment literal in built source:"
grep -nE "partial_(max|sum|output)\", \"common.decode.flash_attention|partial_scalar_bytes, (256|4096)" \
  "$LLAMA/ggml/src/ggml-hrx/dispatch_registration/common/dispatch-flash-attention.cpp" | tee -a "$LOG"
say "built libggml-hrx.so: $(stat -c '%y %n' "$BIN/libggml-hrx.so.0.18.0" 2>/dev/null)"
say "dispatch .o:        $(stat -c '%y %n' "$LLAMA/build-hrx-engine/ggml/src/ggml-hrx/CMakeFiles/ggml-hrx.dir/dispatch_registration/common/dispatch-flash-attention.cpp.o" 2>/dev/null)"
say "KFD holders: [$(fuser /dev/kfd 2>/dev/null || echo none)]  renderD128: [$(fuser /dev/dri/renderD128 2>/dev/null || echo none)]"
say "other llama on HRX0: [$(pgrep -af 'llama-bench|llama-server' | grep -v coder-llm | tr '\n' ';' || true)]"
say ""

run_oracle() {   # run_oracle <label> <env-prefix> <depth> <runs>
  local label="$1" prefix="$2" depth="$3" runs="$4"
  local faults=0
  for i in $(seq 1 "$runs"); do
    out="$(env $prefix "$BIN/llama-bench" -m "$MODEL" -dev HRX0 -p 0 -n 8 -d "$depth" -r 1 2>&1)"
    rc=$?
    if printf '%s' "$out" | grep -qiE "HSA_STATUS_ERROR_MEMORY_FAULT|failed to decode"; then v=FAULT; faults=$((faults+1)); else v=OK; fi
    say "  $label depth=$depth run $i/$runs -> $v (rc=$rc) $(date +%H:%M:%S)"
  done
  say "  TOTAL $label depth=$depth: $faults/$runs faults"
}

for d in "${DEPTHS[@]}"; do run_oracle "multipass" "" "$d" "${RUNS:-5}"; done
say ""
say "--- fallback: GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split ---"
for d in "${DEPTHS[@]}"; do run_oracle "fallback" "GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split" "$d" "${RUNS:-3}"; done
say ""
say "--- weights-buffer +8GB slack (masks the wild offset) ---"
for d in "${DEPTHS[@]}"; do run_oracle "pad8GB" "GGML_HRX_ALLOC_PAD_MB=8192" "$d" "${RUNS:-3}"; done
say "=== done $(date -Is) ==="
