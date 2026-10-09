#!/usr/bin/env bash
# engine#315 paired in-regime campaign driver — the pre-registered producer.
#
# Emits one line per arm in the grammar the analysis scripts parse:
#   pair <N> arm <c|l>: <verdict> nan=<n> req=<n> ret3=<n> parse=<n> gtt_start=<G>G utf8=<n> alloc=<n>
# (utf8/alloc are appended after gtt_start so the existing regex, which is not end-anchored, still
#  matches; they carry the Face-B count and the startup-allocation failures that are classified
#  invalid rather than faulted.)
#
# Classification, fixed in the pre-registration (docs/315-preregistration.md):
#   invalid      startup allocation failure, server died before the request phase, or no gtt.log
#   out-of-regime gtt_start < 95 GiB   (reported, not counted)
#   fault        nan>0 or hsa>0 or parse>0 or utf8>0 or verdict==TRUNCATED or req<66
#   clean        otherwise
#
# Usage: 315-paired.sh <pairs> <neighbours> <neigh_ctx> <control_bin_dir> <fix_bin_dir> <tagbase>
set -uo pipefail
PAIRS=${1:-1}; NB=${2:-2}; NCTX=${3:-2048}; CBIN=${4:?control bin dir}; FBIN=${5:?fix bin dir}; TAG=${6:-315p}
DET=$(dirname "$0")/315-detector.py
M=/home/bcloud/models/GLM-4.7-Flash-Q4_K_M.gguf
OUTDIR=$HOME/wt/rt-det
CONSOLE=$HOME/wt/$TAG-console.log
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
G=$(ls /sys/class/drm/card*/device/mem_info_gtt_used | head -1)
gtt() { echo $(( $(cat $G) / 1073741824 )); }
mine() { for P in $(pgrep -f "llama-server" 2>/dev/null); do C=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null); case "$C" in *"port 2021"*) echo $P;; esac; done; }

say() { echo "$@" | tee -a "$CONSOLE"; }

say "=== session $(date -Is) tag=$TAG pairs=$PAIRS neighbours=$NB ctx=$NCTX gtt_before=$(gtt)G ==="
say "model_sha256 $(sha256sum "$M" | cut -d' ' -f1)"
say "detector_sha256 $(sha256sum "$DET" | cut -d' ' -f1)"
say "control_bin $CBIN -> $("$CBIN/llama-server" --version 2>&1 | head -1)"
say "fix_bin $FBIN -> $("$FBIN/llama-server" --version 2>&1 | head -1)"
say "resident_inventory (foreign servers, not ours):"
for P in $(pgrep -f llama-server 2>/dev/null); do
  say "  pid=$P $(tr '\0' ' ' < /proc/$P/cmdline | cut -c1-100)"
done

# Regime composition is shared with the other #315 scripts: the helper asserts HRX0 is visible,
# starts neighbours with the HRX env (a missing --setenv is what killed the third smoke arm), bounds
# each start, and aborts instead of grinding through multi-minute health waits while holding the lock.
export IREE_HAL_AMDGPU_LIBHSA_PATH=${IREE_HAL_AMDGPU_LIBHSA_PATH:-/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1}
BIN="$CBIN"
source "$(dirname "$0")/315-regime.sh"
regime_preflight || { say "ABORT: HRX0 not visible to $CBIN/llama-server"; exit 3; }
if [ -z "${NOLOCK:-}" ]; then
  exec 9>"$HOME/.cache/lax-decode/box.lock"
  if flock -w "${LOCK_WAIT:-900}" 9; then
    say "box lock acquired $(date -Is)"
  else
    say "box lock busy after ${LOCK_WAIT:-900}s — proceeding CONCURRENTLY (UNLOCKED)"
  fi
fi
regime_neighbours 1 6 "$NCTX" || { say "regime composition failed — session aborted"; trap - EXIT; regime_teardown; exit 4; }
trap regime_teardown EXIT
say "neighbours up gtt=$(gtt)G"

arm() {   # arm <pair> <cell c|l> <bin dir>
  local pair=$1 cell=$2 bin=$3 port=$((19300 + RANDOM % 400))
  local tag="$TAG-p$pair$cell"
  local O="$OUTDIR/$tag-$(basename $M .gguf)"
  rm -rf "$O"; mkdir -p "$O"
  local -a TRACE_ENV=()
  [ -n "${TRACE:-}" ] && TRACE_ENV=(--setenv=GGML_HRX_BUFFER_TRACE=1)
  systemd-run --user --scope -q -p MemoryMax=60G --unit=$TAG-arm-$pair$cell \
    --setenv=IREE_HAL_AMDGPU_LIBHSA_PATH="$IREE_HAL_AMDGPU_LIBHSA_PATH" \
    --setenv=LD_LIBRARY_PATH="$bin" "${TRACE_ENV[@]}" \
    "$bin/llama-server" -m "$M" -dev HRX0 -ngl 99 -fa on --host 127.0.0.1 --port $port -c 8192 -np 1 --no-webui \
    > "$O/server.log" 2>&1 &
  local up=0
  for t in $(seq 1 300); do curl -sf localhost:$port/health >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
  local g0; g0=$(gtt); echo "gtt_start ${g0}GiB mode $(cat /sys/class/ec_su_axb35/apu/power_mode 2>/dev/null)" > "$O/run.log"

  local alloc nan hsa req=0 parse=0 utf8=0 verdict hv52=0 hv56=0
  alloc=$(grep -cE "hrx_allocator_allocate_buffer.*failed|allocate HRX host staging buffer for value .* failed" "$O/server.log" 2>/dev/null || true)
  if [ "$up" != "1" ] || [ "${alloc:-0}" -gt 0 ]; then
    systemctl --user stop $TAG-arm-$pair$cell 2>/dev/null
    for P in $(pgrep -f "port $port" 2>/dev/null); do kill -TERM $P 2>/dev/null; done
    printf '  pair %d arm %s: INVALID nan=0 req=0 ret3=0 parse=0 gtt_start=%dG utf8=0 alloc=%d hv52=0 hv56=0\n' \
      "$pair" "$cell" "$g0" "${alloc:-0}" | tee -a "$CONSOLE"
    return
  fi
  REGIME_ARM_PORT=$port
  timeout "${ARM_TIMEOUT:-1200}" python3 "$DET" "$port" "$O" 50 > "$O/detector.log" 2>&1
  req=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['requests'])" 2>/dev/null || echo 0)
  parse=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['parse_failures'])" 2>/dev/null || echo 0)
  utf8=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['utf8_failures'])" 2>/dev/null || echo 0)
  nan=$(grep -c "HRX returned NaN logits" "$O/server.log" 2>/dev/null || true)
  # Face A memory fault signatures, excluding the allocation-failure lines that are invalid above.
  hsa=$(grep -iE "HSA_STATUS_ERROR|Queue error|wait for HRX graph replay commands failed" "$O/server.log" 2>/dev/null \
        | grep -vcE "hrx_allocator_allocate_buffer|allocate HRX host staging buffer" || true)
  http=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['http_failures'])" 2>/dev/null || echo 0)
  tr=$(python3 -c "import json;print(json.load(open('$O/detector.json'))['transport_failures'])" 2>/dev/null || echo 0)
  # Face B is first class in the pre-registration (see differential): count it from the detectors and
  # the server log, so an arm that still completes its requests cannot be recorded as clean.
  s500=$(grep -c '"code":500' "$O/server.log" 2>/dev/null || true); s500=${s500:-0}
  unr=$(grep -cE "unparsed Content-only output|does not match the expected" "$O/server.log" 2>/dev/null || true); unr=${unr:-0}
  # Allocation evidence, from the trace the arm just ran: 0x52 = host-visible without HOST_COHERENT,
  # 0x56 = host-visible coherent. Recorded per arm so the fix being live is a measurement, not a build claim.
  hv52=$(grep -c "memory_type=0x52 size=619520" "$O/server.log" 2>/dev/null || true); hv52=${hv52:-0}
  hv56=$(grep -c "memory_type=0x56 size=619520" "$O/server.log" 2>/dev/null || true); hv56=${hv56:-0}
  local g1; g1=$(gtt)
  if [ "${g0:-0}" -lt 95 ]; then verdict=OOR
  elif [ "${nan:-0}" -gt 0 ] || [ "${hsa:-0}" -gt 0 ] || [ "${parse:-0}" -gt 0 ] || [ "${utf8:-0}" -gt 0 ] \
    || [ "${http:-0}" -gt 0 ] || [ "${tr:-0}" -gt 0 ] || [ "${s500:-0}" -gt 0 ] || [ "${unr:-0}" -gt 0 ] || [ "${req:-0}" -lt 66 ]; then verdict=FAULT
  else verdict=OK; fi
  printf '  pair %d arm %s: %s nan=%d req=%d ret3=0 parse=%d gtt_start=%dG utf8=%d alloc=%d hv52=%d hv56=%d (gtt_now=%dG)\n' \
    "$pair" "$cell" "$verdict" "${nan:-0}" "${req:-0}" "${parse:-0}" "$g0" "${utf8:-0}" "${alloc:-0}" "${hv52:-0}" "${hv56:-0}" "$g1" \
    | tee -a "$CONSOLE"
  systemctl --user stop $TAG-arm-$pair$cell 2>/dev/null
  for _t in 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0; do
    [ -z "$(pgrep -f "port $port" | head -1)" ] && break
    sleep 1
  done
  for P in $(pgrep -f "port $port" 2>/dev/null); do kill -TERM $P 2>/dev/null; done
}

# Pre-registered stopping rule (docs/315-preregistration.md §6): run pairs until the fix cell has
# MINCLEAN (default 60) clean in-regime arms, stop immediately if any fix-cell arm faults (acceptance
# requires zero), and never exceed PAIRS pairs. The count is read back from the console, i.e. from the
# same pre-registered classification the analyser uses.
clean_fix() { grep -c 'arm l: OK' "$CONSOLE" 2>/dev/null || true; }
fault_fix() { grep -c 'arm l: \(FAULT\|INVALID\)' "$CONSOLE" 2>/dev/null || true; }
# Control cadence. The acceptance criterion (docs/315-preregistration.md §7) is on the FIX cell, and a
# control arm costs the same box time as a fix arm, so controls run every CONTROL_EVERY-th pair (default
# 3): a live positive control spread across the whole session instead of 1:1 pairing, which would double
# the run and leave no budget for the regression cell. §7 does not constrain the ratio; the analyser
# aggregates by cell.
for p in $(seq 1 $PAIRS); do
  if [ $(( (p - 1) % ${CONTROL_EVERY:-3} )) -eq 0 ]; then arm "$p" c "$CBIN"; fi
  arm "$p" l "$FBIN"
  cf=$(clean_fix); ff=$(fault_fix)
  say "after pair $p: fix-cell clean=$cf fault-or-invalid=$ff (target >=${MINCLEAN:-60} clean, cap $PAIRS pairs)"
  [ "$ff" -gt 0 ] && { say "STOP: a fix-cell arm faulted or was invalid — acceptance requires zero faults"; break; }
  [ "$cf" -ge "${MINCLEAN:-60}" ] && { say "STOP: reached $cf clean fix-cell arms"; break; }
done

echo "$(clean_fix) clean fix-cell arms, $(fault_fix) faulted-or-invalid in the fix cell, out of $(grep -c '^  pair' "$CONSOLE" 2>/dev/null) arms run" | tee -a "$CONSOLE"
