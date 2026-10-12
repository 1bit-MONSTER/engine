#!/usr/bin/env bash
# engine#315 task-8: attach the KFD SMI eviction/preemption watcher to the detector arm,
# so a faulting arm carries the driver's own eviction events on the same timeline.
#
# Why a wrapper and not a driver edit: 315-paired.sh is part of the frozen pre-registration
# and must not change. This script runs the SAME driver, and starts the watcher alongside it,
# keyed on the detector server's port. A clean-arm trace is recorded as uninformative, not
# as evidence - that is the task-8 contract.
#
# Usage: 315-evict-trace.sh <PAIRS> <TAG> [FIX_BIN] [CTRL_BIN]
set -uo pipefail

PAIRS=${1:-4}
TAG=${2:-315evict}
FIX=${3:-$HOME/wt/315fix-build}
CTRL=${4:-$HOME/wt/315instr3-build}

# 315-campaign.sh takes the BUILD DIR and appends /bin itself. Accept either form so a
# caller passing .../bin cannot silently double it (the first launch of this wrapper died
# with 'no .../315fix-build/bin/bin/llama-server' for exactly that reason).
strip_bin() { local d=${1%/}; [ "$(basename "$d")" = "bin" ] && dirname "$d" || echo "$d"; }
FIX=$(strip_bin "$FIX")
CTRL=$(strip_bin "$CTRL")

WATCH=$HOME/amd-oss/tools/315-trace/kfd_smi_watch
OUT=$HOME/wt/$TAG-evict
CONSOLE=$HOME/wt/$TAG-console.log

# Two facts learned by smoke-testing this instrument before trusting it (2026-10-12):
#  1. the watcher MUST run as root, or it sees no events at all - and it fails silently
#     (exit 0, header-only file). Verified with sudo: 75 Q_EVICT / 79 Q_RESTORE during one
#     HRX load, 0 unprivileged. So a header-only file is 'no privilege', not 'no evictions'.
#  2. it writes TSV with a '#recv_boot_ns...' header, not NDJSON. The name in this script
#     reflects that; correlate.py reads the TSV.
HSA=${IREE_HAL_AMDGPU_LIBHSA_PATH:-/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1}
export IREE_HAL_AMDGPU_LIBHSA_PATH="$HSA"

need() { [ -e "$1" ] || { echo "ABORT: missing $1"; exit 2; }; }
need "$WATCH"; need "$FIX/bin/llama-server"; need "$CTRL/bin/llama-server"; need "$HSA"
sudo -n true 2>/dev/null || { echo "ABORT: the watcher needs passwordless sudo; sudo -n failed"; exit 2; }
mkdir -p "$OUT"

echo "=== task-8 eviction/preemption trace ==="
echo "tag      : $TAG"
echo "pairs    : $PAIRS  (arms alternate control/fix inside one composition)"
echo "fix  bin  : $($FIX/bin/llama-server --version 2>&1 | head -1)   ($FIX)"
echo "ctrl bin  : $($CTRL/bin/llama-server --version 2>&1 | head -1)   ($CTRL)"
echo "watcher  : $WATCH (as root)"
echo "hsa      : $HSA"
echo
echo "This run's PURPOSE is the eviction timeline, not an acceptance rate. A clean run is"
echo "recorded as UNINFORMATIVE (with its clean-arm count), never as evidence about the fault."
echo

# The watcher is started LATER, after the box lock and the GPU-quiet check (see below).
# It was originally started here, which meant it ran its 3h TTL while a peer still held HRX0
# and recorded events from a window that was not ours.

# Run the frozen driver unchanged. NOSTOP=1 so the run continues past a fault and gives us
# more than one faulting arm to correlate; MINCLEAN=0 because this is not an acceptance run.
# Run the frozen driver unchanged, under the box lock.
#
# The 315 harnesses share a convention (315-attrib.sh:41-48): take
# ~/.cache/lax-decode/box.lock on fd 9 and pass NOLOCK=1 down, so a second arm runner
# cannot start concurrently. That comment also warns that the driver's fallback is to
# proceed UNLOCKED - so taking the lock here is what makes concurrent runs impossible,
# not merely unlikely.
#
# MINCLEAN is a goalpost, not a floor: 315-paired.sh:147 stops when the clean count
# REACHES it. Passing MINCLEAN=0 therefore means "stop at zero clean arms", which ended
# the first launch of this wrapper after pair 1 (2 arms run, not 12). Set it above anything
# reachable so the PAIRS cap is what ends the run, and let NOSTOP keep a fault from
# aborting a diagnostic run.
UNREACHABLE=$(( PAIRS * 2 + 1 ))

if [ -n "${NOLOCK:-}" ]; then
  echo "inheriting the caller's box lock (NOLOCK=1)"
else
  # Take BOTH lock conventions, because this box has two and they do not overlap:
  #   ~/npu.lock                        - the box-wide convention. Observed holding 12 pids
  #                                       from three unrelated workers (one peer batch and
  #                                       two NPU-goal workers), so this is the one that
  #                                       actually serialises GPU use.
  #   ~/.cache/lax-decode/box.lock      - narrower, used only by the older 315 harnesses.
  #                                       Held by nobody but us, so on its own it excludes
  #                                       nothing (measured 2026-10-12).
  # Taking only box.lock was the bug: the run looked lock-protected while a peer ran HRX0.
  LOCK=$HOME/.cache/lax-decode/box.lock
  mkdir -p "$(dirname "$LOCK")"
  exec 9>"$LOCK"
  if flock -w "${LOCK_WAIT:-900}" 9; then
    echo "box lock held $(date -Is)  ($LOCK)"
  else
    echo "ABORT: box lock unobtainable after ${LOCK_WAIT:-900}s - a peer is running arms."
    echo "Nothing was started. Retry when the box is quiet."
    exit 3
  fi
  NPULOCK=$HOME/npu.lock
  exec 8>"$NPULOCK"
  if flock -w "${NPU_LOCK_WAIT:-900}" 8; then
    echo "npu lock held $(date -Is)  ($NPULOCK) - the box-wide convention"
  else
    echo "ABORT: ~/npu.lock unobtainable after ${NPU_LOCK_WAIT:-900}s."
    echo "That lock is held by other workers on this box, and the campaign composes ~5"
    echo "neighbours in the 95-110 GiB band, which would evict their memory."
    echo "Nothing was started."
    exit 3
  fi
fi

# The lock only excludes harnesses that FOLLOW the convention. Observed 2026-10-12: a peer
# batch (~/wt/loom-ub389) runs HRX jobs without ever taking box.lock, so the lock was free
# while a peer held HRX0. The composition would have evicted that peer's memory. So also
# poll for actual GPU quiescence: any foreign '-dev HRX0' process, other than the two
# Vulkan servers this box keeps resident.
wait_for_gpu_quiet() {
  local deadline=$(( $(date +%s) + ${QUIET_WAIT:-1800} ))
  while :; do
    local foreign
    foreign=$(ps -eo pid,args 2>/dev/null \
      | grep -- "-dev HRX0" | grep -v grep \
      | grep -v "$OUT" | grep -v "315-" || true)
    # Also refuse if a peer's HRX job is mid-flight even without '-dev HRX0' in argv.
    local peer
    peer=$(ps -eo pid,args 2>/dev/null | grep -E "loom-ub389|repro336" | grep -v grep || true)
    if [ -z "$foreign" ] && [ -z "$peer" ]; then
      echo "GPU quiet: no foreign HRX job (waited as needed)"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "ABORT: GPU still busy after ${QUIET_WAIT:-1800}s; a foreign HRX job would be evicted."
      echo "  foreign: ${foreign:-none}"
      echo "  peer:    ${peer:-none}"
      echo "Nothing was started."
      return 3
    fi
    sleep 20
  done
}

wait_for_gpu_quiet || exit $?

# Now that the box is ours, start the watcher. Root, or the trace is silently empty.
# Start it BEFORE the driver composes neighbours so the neighbours' own BOs get owners in
# the stream (the driver's fallback would otherwise miss their registration).
sudo -n nohup "$WATCH" -e all -t 10800 -o "$OUT/kfd_smi.tsv" -q >"$OUT/watcher.err" 2>&1 &
sleep 3
WPID=$(pgrep -f "kfd_smi_watch.*$OUT/kfd_smi.tsv" | head -1)
if [ -z "$WPID" ]; then
  echo "ABORT: the watcher did not start; stderr:"; head -5 "$OUT/watcher.err"; exit 3
fi
echo "watcher pid $WPID -> $OUT/kfd_smi.tsv"
cleanup() { sudo -n kill "$WPID" 2>/dev/null; }
trap cleanup EXIT INT TERM

# NOLOCK=1 is REQUIRED, not optional. 315-paired.sh:47-52 re-opens
# ~/.cache/lax-decode/box.lock on its OWN fd 9 and flocks it, falling back to "proceeding
# CONCURRENTLY (UNLOCKED)" on timeout. flock is per-open-file-description, so a SECOND open
# of a file this process tree already holds a lock on will BLOCK.
#
# Observed: run 6 (315evict6) passed all four of this wrapper's gates, began composing, then
# deadlocked here - the driver's child sat in locks_lock_inode_wait on box.lock while this
# wrapper held box.lock on fd 9 and npu.lock on fd 8. After LOCK_WAIT it would have proceeded
# UNLOCKED, so the failure is not even protective.
#
# The driver documents the convention in its own header: NOLOCK=1 means the caller already
# holds the lock on an inherited fd 9 - exactly our case.
MINCLEAN="$UNREACHABLE" NOSTOP=1 NOLOCK=1 \
  FIX="$FIX" CTRL="$CTRL" TAG="$TAG" PAIRS="$PAIRS" \
  bash "$HOME/wt/315-campaign.sh" 2>&1 | tail -20

cleanup
trap - EXIT INT TERM

echo
echo "=== watcher summary ==="
if [ -s "$OUT/kfd_smi.tsv" ] && [ "$(wc -l < "$OUT/kfd_smi.tsv")" -gt 1 ]; then
  EV=$(grep -c "Q_EVICT" "$OUT/kfd_smi.tsv" 2>/dev/null || echo 0)
  RS=$(grep -c "Q_RESTORE" "$OUT/kfd_smi.tsv" 2>/dev/null || echo 0)
  echo "events: Q_EVICT=$EV Q_RESTORE=$RS  lines=$(wc -l < "$OUT/kfd_smi.tsv")"
  echo "triggers seen:"
  grep "Q_EVICT" "$OUT/kfd_smi.tsv" 2>/dev/null | grep -o "trigger=[A-Z_]*" | sort | uniq -c | sort -rn | head -6
  echo "restores not rescheduled (the interesting case):"
  grep "Q_RESTORE" "$OUT/kfd_smi.tsv" 2>/dev/null | grep -c "rescheduled=no" || true
else
  echo "watcher recorded ONLY A HEADER (or nothing) - check that it ran as root."
  echo "The trace is UNINFORMATIVE, not evidence. stderr:"; head -5 "$OUT/watcher.err" 2>/dev/null
fi

echo
echo "=== arm verdicts ==="
if [ -f "$CONSOLE" ]; then
  grep -E "^  pair" "$CONSOLE" | tail -20
  # grep -c prints 0 AND exits 1 when there are no matches, so `|| echo 0` appends a SECOND
  # 0 and the value becomes "0\n0" - which made `[ "$FA" -eq 0 ]` fail with
  # 'integer expected'. Take the count and normalise in one step.
  IN=$(grep -cE "^  pair.*gtt_start=(9[5-9]|10[0-9]|110)G" "$CONSOLE" 2>/dev/null)
  IN=${IN:-0}
  FA=$(grep -E "^  pair.*gtt_start=(9[5-9]|10[0-9]|110)G" "$CONSOLE" 2>/dev/null | grep -c "FAULT")
  FA=${FA:-0}
  echo "in-regime arms: $IN   faults: $FA"
  if [ "$FA" -eq 0 ]; then
    echo "NO in-regime fault in this run -> the eviction trace has no faulting half to correlate."
    echo "Record it as UNINFORMATIVE (clean-arm count $IN), per the task-8 contract. A clean"
    echo "trace is NOT evidence about the fault's mechanism."
  else
    echo "A fault is present -> correlate each faulting arm against the eviction timeline."
    echo "  # arm-pid.py binds an arm to its server PID via the PORT in that arm's own"
    echo "  # server.log ('listening on http://127.0.0.1:<port>'). Run it WHILE the arm is"
    echo "  # live: a finished arm is refused rather than guessed, because correlate.py's"
    echo "  # own fallback (most-frequent eviction subject) was observed landing on a dead"
    echo "  # pid, and a correlation against the wrong process fails silently."
    echo "  for D in \$OUTDIR/*-p*c-* \$OUTDIR/*-p*l-*; do"
    echo "      PID=\$(python3 \$HOME/wt/arm-pid.py \"\$D\" 2>/dev/null) || continue"
    echo "      python3 \$HOME/amd-oss/tools/315-trace/correlate.py \\"
    echo "        --smi \$OUT/kfd_smi.tsv --server \"\$D/server.log\" --pid \$PID \\"
    echo "        --timeline \$OUT/correlated-\$(basename \$D).tsv"
    echo "  done"
  fi
else
  echo "no console at $CONSOLE"
fi
