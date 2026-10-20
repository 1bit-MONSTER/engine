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
# hw-gate.sh: the HRX correctness gate on real hardware (docs/hw-gate.md). It builds this
# checkout's HRX llama.cpp pair (third_party/llama.cpp + third_party/hrx-system) with the private
# HIP kernels compiled in, as the release does (the gpu-kernels add-on at the commit
# config/gpu-kernels.config pins, fetched by scripts/fetch-private.sh), and checks it on the Radeon
# iGPU against goldens recorded at the release pin:
#
#   ops     test-backend-ops -b HRX0 on the ops the HRX correctness bugs touched
#   kat     tools/kat.py check: known answers, determinism, cached prefix, varied lengths and drift
#           vs tests/golden/hw-gate/kat-<model>.json, on Qwen3-0.6B, ZAYA1-8B, Qwen3.8-27B
#           (K-quants), Qwen3-Coder-30B-A3B (MoE, Q4_K_M) and Qwen3-4B (IQ4_XS)
#   kld     llama-perplexity --kl-divergence vs stored base logits (512 tokens x 8 chunks)
#   needle  tests/hrx_long_prompt.py: the 320-record needle (7.1K tokens on Qwen3, 7.4K on ZAYA1;
#           #319, #369) on Qwen3-0.6B and ZAYA1-8B
#
#   scripts/hw-gate.sh [--bin DIR] [--phases build,ops,kat,kld,needle] [--models a,b] [--record]
#
#   --bin DIR   test the llama.cpp binaries in DIR (llama-server, test-backend-ops, llama-perplexity)
#               instead of building this checkout (the revert test uses it)
#   --record    record the goldens instead of checking them: kat-<model>.json under
#               tests/golden/hw-gate and the base logits under $HW_GATE_STORE/kld (manifest.json
#               keeps their sha256). Only at a release pin.
#
#   HW_GATE_PRIVATE=0  build the public HRX plumbing only, without the private kernels (a box with
#               no read access to gpu-kernels). The default builds them: a fetch failure is exit 2.
#
# Every GPU step holds ~/npu.lock (flock) only while it runs, inside scripts/mem-guard.sh and
# ~/lb/thermal-run.sh. Before each one it waits for the PR-Agent model (coder-llm, :8090) to be
# idle, and afterwards it reports any new amdgpu ring timeout (HRX dispatches starve that
# Vulkan service). Exit 0 = pass, 1 = a check failed, 2 = the box could not run the gate.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STORE=${HW_GATE_STORE:-$HOME/.cache/hw-gate}
MODELS=${HW_GATE_MODELS:-$HOME/models}
LOCK=${HW_GATE_LOCK:-$HOME/npu.lock}
LOCK_WAIT=${HW_GATE_LOCK_WAIT:-5400}
THERMAL=${HW_GATE_THERMAL:-$HOME/lb/thermal-run.sh}
MEM_FLOOR=${HW_GATE_MEM_FLOOR:-12}
CODER_URL=${HW_GATE_CODER_URL:-http://127.0.0.1:8090}
CODER_WAIT=${HW_GATE_CODER_WAIT:-900}
TOOLCHAIN=${ONEBIT_HRX_TOOLCHAIN:-/opt/rocm-therock}
PRIVATE=${HW_GATE_PRIVATE:-1}
TEXT=${HW_GATE_TEXT:-$MODELS/qwen38/wikitext-2-raw/wiki.test.raw}
PORT=${HW_GATE_PORT:-18471}
JOBS=${HW_GATE_JOBS:-4}
GOLDEN=$ROOT/tests/golden/hw-gate

TCTL=$(grep -l k10temp /sys/class/hwmon/*/name 2>/dev/null | head -n 1 | xargs -r dirname)/temp1_input
tctl() { echo $(( $(cat "$TCTL" 2>/dev/null || echo 0) / 1000 )); }
# thermal-run.sh kills its job at 93 C, so never start one on a box another job has heated: wait
# (up to HW_GATE_COOL_WAIT s) for Tctl <= HW_GATE_COOL_C first
cool() {
  local waited=0
  while [ "$(tctl)" -gt "${HW_GATE_COOL_C:-75}" ] && [ "$waited" -lt "${HW_GATE_COOL_WAIT:-600}" ]; do
    sleep 2; waited=$((waited + 2))
  done
  [ "$waited" -gt 0 ] && echo "hw-gate: waited ${waited}s for Tctl to fall to $(tctl) C" >&2
  return 0
}

# The ops the HRX bugs touched: matmul (#370, the kquant/dot4i work), MoE matmul (#308),
# flash attention (#313 prompt FA NaN, #310 MLA dk 576, #140 decode-split), the gated delta net and
# its conv (Qwen3.5/3.8, #315), GET_ROWS (the IQ quant ports), ROPE and the norms of the fused
# qwen.attention path (#318, #319).
OPS=${HW_GATE_OPS:-MUL_MAT,MUL_MAT_ID,FLASH_ATTN_EXT,GATED_DELTA_NET,SSM_CONV,GET_ROWS,ROPE,RMS_NORM,SOFT_MAX}
# Cases KNOWN-ISSUES.md lists as intermittent: a failure of only these is re-run once.
FLAKY_CASES=${HW_GATE_FLAKY:-"MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048"}

# name | file under $MODELS | kat --repeat | needle (1 = run tests/hrx_long_prompt.py on it)
# The first three are K-quants (ZAYA1-8B routes 1 of 16 experts). The Qwen3-Coder row covers
# MUL_MAT_ID with top-8 of 128 experts, the IQ row the IQ4_XS matmul and GET_ROWS paths. The
# needle runs on a small and a mid-size model.
MODEL_TABLE="qwen3-0.6b|Qwen3-0.6B-Q4_K_M.gguf|3|1
zaya1-8b|zaya1-8b-Q4_K_M.gguf|2|1
qwen3.8-27b|qwen38/UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf|2|0
qwen3-coder-30b-a3b|Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|2|0
qwen3-4b-iq4xs|Qwen3-4B-IQ4_XS.gguf|2|0"

# tools/mem_budget.py (through hw_gate_manifest.py): never start a GPU step that would push the box into
# the engine#315 regime (faults seen only at GTT >= 95 GiB) or past MemAvailable. "-" = no model.
memcheck() {
  python3 "$ROOT/tools/hw_gate_manifest.py" memcheck "$1" "$ROOT/third_party/llama.cpp"
}

BIN="" PHASES="build,ops,kat,kld,needle" ONLY_MODELS="" MODE=check SERVE_STEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN=$(cd "$2" && pwd); shift 2 ;;
    --phases) PHASES=$2; shift 2 ;;
    --models) ONLY_MODELS=$2; shift 2 ;;
    --record) MODE=record; shift ;;
    --cool)  # internal, inside the lock: wait for a cool box, check memory, then run the step
      shift; cool
      [ -n "${HW_GATE_STAMP:-}" ] && date +%s > "$HW_GATE_STAMP"   # the lock is held from here
      if [ "${1:-}" = --mem ]; then memcheck "$2" || exit 76; shift 2; fi
      exec "$@" ;;
    --serve-step) SERVE_STEP=1; shift; break ;;   # internal: one model's kat (+ needle) inside the lock
    -h|--help) sed -n '16,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "hw-gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

# tools/mem_budget.py: never start a GPU step that would push the box into the engine#315 regime
# (faults seen only at GTT >= 95 GiB) or past MemAvailable. "-" = no model (test-backend-ops).
LIBHSA=$(find "$TOOLCHAIN" -path '*/libhsa-runtime64.so.1' 2>/dev/null | sort | head -n 1)
export IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-$LIBHSA}

# ---- internal: one model in one lock hold: serve it, run kat (and the needle), stop the server,
# then llama-perplexity for the KLD. Exit bits: 1 kat failed, 4 needle failed, 8 perplexity failed;
# 3 = the server did not come up.
if [ "$SERVE_STEP" = 1 ]; then
  name=$1 model=$2 repeat=$3 out=$4 needle=$5 dokat=$6 dokld=$7 base=$8
  log=$out/server-$name.log
  rc=0
  if [ "$dokat" = 1 ] || [ "$needle" = 1 ]; then
    "$BIN/llama-server" -m "$model" --device HRX0 -ngl 99 --jinja -c 8192 -np 1 \
      --host 127.0.0.1 --port "$PORT" --no-webui > "$log" 2>&1 &
    srv=$!
    trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT
    for _ in $(seq 600); do
      curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
      kill -0 $srv 2>/dev/null || { echo "llama-server died while loading (see $log)"; tail -n 20 "$log"; exit 3; }
      sleep 1
    done
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 || { echo "llama-server not ready after 600 s"; exit 3; }
    date +%s > "$out/t-$name-loaded"
    if [ "$dokat" = 1 ]; then
      if [ "$MODE" = record ]; then
        python3 "$ROOT/tools/kat.py" run --url "http://127.0.0.1:$PORT" --repeat "$repeat" \
          --out "$GOLDEN/kat-$name.json" > "$out/kat-$name.txt" 2>&1 || rc=1
      else
        python3 "$ROOT/tools/kat.py" check --url "http://127.0.0.1:$PORT" --repeat "$repeat" \
          --golden "$GOLDEN/kat-$name.json" --out "$out/kat-$name.json" > "$out/kat-$name.txt" 2>&1 || rc=1
      fi
    fi
    if [ "$needle" = 1 ]; then
      python3 "$ROOT/tests/hrx_long_prompt.py" --url "http://127.0.0.1:$PORT" > "$out/needle-$name.txt" 2>&1 || rc=$((rc | 4))
    fi
    date +%s > "$out/t-$name-kat"
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; trap - EXIT
  fi
  if [ "$dokld" = 1 ]; then
    kargs=(--kl-divergence-base "$base"); [ "$MODE" = check ] && kargs+=(--kl-divergence)
    "$BIN/llama-perplexity" -m "$model" -f "$TEXT" -c 512 --chunks 8 -b 512 -ub 512 --device HRX0 -ngl 99 \
      "${kargs[@]}" > "$out/kld-$name.log" 2>&1 || rc=$((rc | 8))
  fi
  exit $rc
fi

OUT=${HW_GATE_OUT:-$STORE/runs/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT" "$STORE/kld"
RESULTS=$OUT/results.tsv
: > "$RESULTS"
START=$(date +%s)
infra=0

note() { printf '%s\n' "$*" | tee -a "$OUT/gate.log"; }
result() {  # check status secs detail
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS"
  note "$(printf '%-28s %-4s %5ss  %s' "$1" "$2" "$3" "$4")"
}
has() { case ",$PHASES," in *",$1,"*) return 0 ;; esac; return 1; }
want_model() { [ -z "$ONLY_MODELS" ] && return 0; case ",$ONLY_MODELS," in *",$1,"*) return 0 ;; esac; return 1; }

ring_timeouts() { journalctl -k --since "@$START" --no-pager 2>/dev/null | grep -c "ring .*timeout" || true; }

wait_coder_idle() {  # the PR-Agent's Vulkan model: never start an HRX step while it is generating
  local waited=0
  while [ "$waited" -lt "$CODER_WAIT" ]; do
    busy=$(curl -s --max-time 5 "$CODER_URL/slots" 2>/dev/null | python3 -c \
      'import json,sys
try: print(sum(1 for s in json.load(sys.stdin) if s.get("is_processing")))
except Exception: print(0)' 2>/dev/null)
    [ "${busy:-0}" = 0 ] && return 0
    [ "$waited" = 0 ] && note "waiting for coder-llm ($CODER_URL) to finish a request"
    sleep 15; waited=$((waited + 15))
  done
  note "::warning::coder-llm still busy after ${CODER_WAIT}s; running anyway"
}

# run "$@" on the GPU: wait for coder-llm, hold the lock, mem-guard + thermal-run
gpu() {
  wait_coder_idle
  local before after rc
  before=$(ring_timeouts)
  local tries=0 t0 t1
  t0=$(date +%s)
  export HW_GATE_STAMP=$OUT/.stamp
  rm -f "$HW_GATE_STAMP"
  while :; do
    flock -w "$LOCK_WAIT" -E 75 "$LOCK" "$0" --cool --mem "${GPU_MODEL:--}" "$ROOT/scripts/mem-guard.sh" "$MEM_FLOOR" "$THERMAL" "$@"
    rc=$?
    # 76: the memory pre-check said no; the lock is released, so other work can finish first
    [ $rc = 76 ] && [ $tries -lt "${HW_GATE_MEM_RETRIES:-10}" ] || break
    tries=$((tries + 1)); sleep 60
  done
  [ $rc = 76 ] && { note "::error::memory pre-check refused the step ${tries} times"; infra=1; }
  t1=$(date +%s)
  # wall time split for the summary: waiting (lock, heat, memory) vs holding the lock
  if [ -f "$HW_GATE_STAMP" ]; then
    LAST_HOLD=$(( t1 - $(cat "$HW_GATE_STAMP") )); LAST_WAIT=$(( $(cat "$HW_GATE_STAMP") - t0 ))
  else
    LAST_HOLD=0; LAST_WAIT=$(( t1 - t0 ))
  fi
  printf '%s\t%s\t%s\n' "$*" "$LAST_WAIT" "$LAST_HOLD" | cut -c1-400 >> "$OUT/timing.tsv"
  after=$(ring_timeouts)
  if [ "$after" -gt "$before" ]; then
    note "::warning::new amdgpu ring timeout(s) in the kernel log during: $*"
  fi
  if [ $rc = 75 ]; then note "::error::$LOCK still held after ${LOCK_WAIT}s"; infra=1; fi
  return $rc
}

note "hw-gate $MODE at $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null): llama.cpp $(git -C "$ROOT" ls-tree HEAD third_party/llama.cpp | awk '{print substr($3,1,12)}'), hrx-system $(git -C "$ROOT" ls-tree HEAD third_party/hrx-system | awk '{print substr($3,1,12)}'); logs $OUT"

# ---- build ---------------------------------------------------------------------------------
if [ -z "$BIN" ]; then
  src=$ROOT/third_party/llama.cpp
  # one persistent build tree per source path (and per add-on choice), so the runner's checkout
  # rebuilds incrementally
  key=$(printf '%s|private=%s' "$src" "$PRIVATE" | sha1sum | cut -c1-10)
  build=$STORE/build-$key
  BIN=$build/bin
  if has build; then
    # relative paths in ccache's hash: a checkout elsewhere (another worktree) reuses the objects
    export CCACHE_BASEDIR=$ROOT
    for s in llama.cpp hrx-system; do
      [ -f "$ROOT/third_party/$s/CMakeLists.txt" ] || { note "::error::third_party/$s is not checked out"; exit 2; }
    done
    t0=$(date +%s)
    # the private HIP kernels, as cmake/hrx.cmake (ONEBIT_GPU_PRIVATE) and scripts/weekly.sh build
    # them: the gpu-kernels add-on at the pinned commit, through llama.cpp's GGML_HRX_HIP_ADDON_DIR.
    # A llama.cpp pin and a gpu-kernels pin that do not compile together fail the gate here (the
    # 2026-10-20 break: the add-on used HipSetRowsArgs, which the pinned llama.cpp did not have).
    addon_args=() addon_pin=none
    if [ "$PRIVATE" = 1 ]; then
      addon_pin=$(sed -n 's/^sha=//p' "$ROOT/config/gpu-kernels.config" | tail -n 1)
      if ! addon=$("$ROOT/scripts/fetch-private.sh" "$ROOT/config/gpu-kernels.config" "$STORE/gpu-kernels" 2> "$OUT/private.log"); then
        note "::error::cannot fetch the private gpu-kernels add-on at ${addon_pin:0:12} (gh login with read access on this box?): $(tail -n 1 "$OUT/private.log")"
        note "  (HW_GATE_PRIVATE=0 gates the public plumbing only)"
        exit 2
      fi
      grep -q GGML_HRX_HIP_ADDON_DIR "$src/ggml/src/ggml-hrx/hip/ggml-hrx-hip.cmake" 2>/dev/null ||
        { note "::error::third_party/llama.cpp has no GGML_HRX_HIP_ADDON_DIR hook (needs the HIP plumbing)"; exit 2; }
      addon_args=(-DGGML_HRX_HIP_ADDON_DIR="$addon")
      note "private HIP kernels: gpu-kernels ${addon_pin:0:12} (addons/hip-hrx)"
    fi
    # configure once per add-on pin (a new pin can add kernel files the old configure never saw);
    # otherwise ninja re-runs cmake itself when a CMakeLists.txt changes
    [ "$(cat "$build/.gpu-kernels-pin" 2>/dev/null)" = "$addon_pin" ] || rm -f "$build/CMakeCache.txt"
    # the same configuration as cmake/hrx.cmake (llama_hrx), plus the two test tools
    : > "$OUT/build.log"
    built=1
    # thermal-run kills a build at 93 C (the box is shared); ninja resumes where it stopped
    for attempt in 1 2 3 4 5 6; do
      cool
      { [ -f "$build/CMakeCache.txt" ] || { "$THERMAL" cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER="$TOOLCHAIN/bin/amdclang" -DCMAKE_CXX_COMPILER="$TOOLCHAIN/bin/amdclang++" \
          -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
          -DHRX_SOURCE_DIR="$ROOT/third_party/hrx-system" "${addon_args[@]}" \
          -DGGML_HRX=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON \
          -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF >> "$OUT/build.log" 2>&1 &&
          echo "$addon_pin" > "$build/.gpu-kernels-pin"; }; } &&
      "$THERMAL" cmake --build "$build" -j "$JOBS" --target llama-server test-backend-ops llama-perplexity >> "$OUT/build.log" 2>&1 && { built=0; break; }
      grep -q "error:" <(tail -n 40 "$OUT/build.log") && break   # a compile error, not heat
      note "build stopped (attempt $attempt, Tctl $(tctl) C); resuming"
    done
    # (grep -c, not grep -q: under pipefail grep's early exit would fail nm's side of the pipe)
    if [ $built = 0 ] && [ "$PRIVATE" = 1 ] &&
       [ "$(nm -D --defined-only "$BIN/libggml-hrx.so" 2>/dev/null | grep -c ggml_hrx_hip_addon_register)" = 0 ]; then
      note "::error::the build passed but libggml-hrx.so does not export ggml_hrx_hip_addon_register: the private kernels are not in it"
      built=1
    fi
    if [ $built = 0 ]; then
      result build PASS $(( $(date +%s) - t0 )) "$build"
    else
      result build FAIL $(( $(date +%s) - t0 )) "see $OUT/build.log"
      # the compiler's error lines only, not its source excerpts: the job log is public and the
      # add-on's sources are not
      grep -E '(error|Error)[: ]' "$OUT/build.log" | grep -v -E '^ *[0-9]+ \|' | head -n 20
      exit 2
    fi
  fi
fi
for b in llama-server test-backend-ops llama-perplexity; do
  [ -x "$BIN/$b" ] || { note "::error::$BIN/$b missing"; exit 2; }
done

# ---- ops -----------------------------------------------------------------------------------
# One test-backend-ops run for all the ops: one lock hold instead of one per op (on a shared box
# every acquisition can wait minutes), with -j workers for the CPU reference.
if has ops; then
  t0=$(date +%s); log=$OUT/ops.log
  gpu "$BIN/test-backend-ops" test -b HRX0 -o "$OPS" -j "$JOBS" > "$log" 2>&1
  rc=$?
  sed -i 's/\x1b\[[0-9;]*m//g' "$log"   # test-backend-ops colours OK / FAIL
  fails=$(grep -E '^ +[A-Z_]+\(.*FAIL' "$log" | sed -E 's/^ +//; s/\): .*/)/' | sort -u)
  passed=$(grep -oE '[0-9]+/[0-9]+ tests passed' "$log" | tail -n 1)
  held=$LAST_HOLD
  if [ $rc = 0 ] && grep -q "Backend HRX0: OK" "$log"; then
    result ops PASS $(( $(date +%s) - t0 )) "${passed} ($OPS; held the lock ${held}s)"
  elif [ $rc = 75 ] || [ $rc = 76 ]; then
    result ops FAIL $(( $(date +%s) - t0 )) "could not run: lock or memory (rc $rc)"
  else
    unknown=$(printf '%s\n' "$fails" | grep -v -F "$FLAKY_CASES" | grep -c . || true)
    st=FAIL detail="${passed:-rc=$rc}: $(printf '%s' "$fails" | head -n 3 | tr '\n' ' ')"
    if [ -n "$fails" ] && [ "$unknown" = 0 ]; then
      # KNOWN-ISSUES.md engine#370: re-run only the intermittent case(s), three times
      note "ops: only the known-intermittent case failed; running it alone three times"
      ok=1
      while read -r c; do
        [ -n "$c" ] || continue
        for _ in 1 2 3; do
          op=${c%%(*}; params=${c#*(}; params=${params%)}
          re=$(python3 -c 'import re,sys; print("^" + re.escape(sys.argv[1]) + "$")' "$params")
          gpu "$BIN/test-backend-ops" test -b HRX0 -o "$op" -p "$re" > "$log.retry" 2>&1 || ok=0
          sed -i 's/\x1b\[[0-9;]*m//g' "$log.retry"
          grep -q "Backend HRX0: OK" "$log.retry" || ok=0
        done
      done <<< "$fails"
      [ $ok = 1 ] && st=PASS detail="${passed}; the known intermittent case (engine#370) failed once and passed 3/3 alone"
    fi
    result ops $st $(( $(date +%s) - t0 )) "$detail"
  fi
fi

# ---- kat + needle + KLD: one lock hold per model -------------------------------------------
if has kat || has needle || has kld; then
  [ -f "$TEXT" ] || { has kld && { result kld FAIL 0 "text missing: $TEXT"; infra=1; }; }
  while IFS='|' read -r name file repeat donee; do
    want_model "$name" || continue
    model=$MODELS/$file base=$STORE/kld/$name.kld
    if [ ! -f "$model" ]; then result "model $name" FAIL 0 "model missing: $model"; infra=1; continue; fi
    dokat=0; has kat && dokat=1
    needle=0; [ "$donee" = 1 ] && has needle && needle=1
    dokld=0; has kld && [ -f "$TEXT" ] && dokld=1
    if [ $dokld = 1 ] && [ "$MODE" = check ] &&
       ! python3 "$ROOT/tools/hw_gate_manifest.py" verify "$GOLDEN/manifest.json" "$name" "$base" "$model" "$TEXT" > "$OUT/kld-$name.verify" 2>&1; then
      result "kld $name" FAIL 0 "$(tail -n 1 "$OUT/kld-$name.verify")"; infra=1; dokld=0
    fi
    [ $dokat = 0 ] && [ $needle = 0 ] && [ $dokld = 0 ] && continue
    t0=$(date +%s)
    rec=(); [ "$MODE" = record ] && rec=(--record)
    GPU_MODEL=$model gpu "$0" --bin "$BIN" "${rec[@]}" --serve-step "$name" "$model" "$repeat" "$OUT" "$needle" "$dokat" "$dokld" "$base"
    rc=$?
    secs=$(( $(date +%s) - t0 )) held=$LAST_HOLD
    if [ $rc = 75 ] || [ $rc = 76 ]; then result "model $name" FAIL "$secs" "could not run: lock or memory (rc $rc)"; continue; fi
    if [ $rc = 3 ] || [ $rc -ge 16 ]; then
      result "model $name" FAIL "$secs" "server failed (rc $rc): $(tail -n 3 "$OUT/server-$name.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
      continue
    fi
    if [ $dokat = 1 ]; then
      st=PASS; [ $((rc & 1)) = 1 ] && st=FAIL
      [ "$MODE" = record ] && st=REC
      nw=$(grep -c '~ WARN' "$OUT/kat-$name.txt" || true)
      result "kat $name" $st "$held" "$(grep -cE '^(ok|warn|FAIL) ' "$OUT/kat-$name.txt") cases, ${nw} warnings; $(grep -E '^ *- ' "$OUT/kat-$name.txt" | head -n 3 | sed -E 's/^ *- //' | tr '\n' ';' | cut -c1-240)"
    fi
    if [ $needle = 1 ]; then
      st=PASS; [ $((rc & 4)) = 4 ] && st=FAIL
      result "needle $name" $st - "$(tail -n 1 "$OUT/needle-$name.txt")"
    fi
    if [ $dokld = 1 ]; then
      if [ "$MODE" = record ]; then
        if [ $((rc & 8)) = 0 ]; then
          python3 "$ROOT/tools/hw_gate_manifest.py" set "$GOLDEN/manifest.json" "$name" "$base" "$model" "$TEXT"
          result "kld $name" REC - "$(sha256sum "$base" | cut -c1-16) $(du -h "$base" | cut -f1)"
        else
          result "kld $name" FAIL - "record failed: $(tail -n 2 "$OUT/kld-$name.log" | tr '\n' ' ' | cut -c1-200)"
        fi
      else
        verdict=$(python3 "$ROOT/tools/hw_gate_manifest.py" judge "$GOLDEN/manifest.json" "$name" "$OUT/kld-$name.log")
        vrc=$?
        [ $((rc & 8)) = 8 ] && { verdict="llama-perplexity failed: $(tail -n 2 "$OUT/kld-$name.log" | tr '\n' ' ' | cut -c1-200)"; vrc=1; }
        st=PASS; [ $vrc != 0 ] && st=FAIL
        result "kld $name" $st - "$verdict"
      fi
    fi
  done <<< "$MODEL_TABLE"
fi

# ---- summary ---------------------------------------------------------------------------------
total=$(( $(date +%s) - START ))
nfail=$(awk -F'\t' '$2 == "FAIL"' "$RESULTS" | wc -l)
rings=$(ring_timeouts)
{
  echo "## HRX hardware gate: $([ "$nfail" = 0 ] && echo PASS || echo "FAIL ($nfail)")"
  echo
  echo "Commit \`$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)\`, llama.cpp \`$(git -C "$ROOT" ls-tree HEAD third_party/llama.cpp | awk '{print substr($3,1,12)}')\`, hrx-system \`$(git -C "$ROOT" ls-tree HEAD third_party/hrx-system | awk '{print substr($3,1,12)}')\`, binaries \`$BIN\`, ${total}s, Tctl at end $(( $(cat "$(grep -l k10temp /sys/class/hwmon/*/name | head -n 1 | xargs dirname)/temp1_input") / 1000 )) C, amdgpu ring timeouts during the run: $rings."
  echo
  echo "| check | result | secs | detail |"
  echo "|---|---|---|---|"
  awk -F'\t' '{ gsub(/\|/, "\\|", $4); printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4 }' "$RESULTS"
  if [ -s "$OUT/timing.tsv" ]; then
    echo
    echo "GPU steps: $(awk -F'\t' '{w+=$2; h+=$3} END {printf "%d s waiting (lock, heat, memory), %d s holding ~/npu.lock, longest hold %d s", w, h, m}' m="$(cut -f3 "$OUT/timing.tsv" | sort -n | tail -n 1)" "$OUT/timing.tsv")."
  fi
} > "$OUT/summary.md"
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && cat "$OUT/summary.md" >> "$GITHUB_STEP_SUMMARY"
cat "$OUT/summary.md"
if [ "$nfail" != 0 ]; then
  [ "$infra" = 1 ] && exit 2
  exit 1
fi
exit 0
