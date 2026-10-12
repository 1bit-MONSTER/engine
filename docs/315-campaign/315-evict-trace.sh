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

# Root, or the trace is silently empty. Start BEFORE the driver composes neighbours so the
# neighbours' own BOs get owners in the stream.
sudo -n nohup "$WATCH" -e all -t 10800 -o "$OUT/kfd_smi.tsv" -q >"$OUT/watcher.err" 2>&1 &
sleep 3
WPID=$(pgrep -f "kfd_smi_watch.*$OUT/kfd_smi.tsv" | head -1)
if [ -z "$WPID" ]; then
  echo "ABORT: the watcher did not start; stderr:"; head -5 "$OUT/watcher.err"; exit 3
fi
echo "watcher pid $WPID -> $OUT/kfd_smi.tsv"

cleanup() { sudo -n kill "$WPID" 2>/dev/null; }
trap cleanup EXIT INT TERM

# Run the frozen driver unchanged. NOSTOP=1 so the run continues past a fault and gives us
# more than one faulting arm to correlate; MINCLEAN=0 because this is not an acceptance run.
FIX="$FIX" CTRL="$CTRL" TAG="$TAG" PAIRS="$PAIRS" MINCLEAN=0 NOSTOP=1 \
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
  IN=$(grep -cE "^  pair.*gtt_start=(9[5-9]|10[0-9]|110)G" "$CONSOLE" 2>/dev/null || echo 0)
  FA=$(grep -E "^  pair.*gtt_start=(9[5-9]|10[0-9]|110)G" "$CONSOLE" 2>/dev/null | grep -c "FAULT" || echo 0)
  echo "in-regime arms: $IN   faults: $FA"
  if [ "$FA" -eq 0 ]; then
    echo "NO in-regime fault in this run -> the eviction trace has no faulting half to correlate."
    echo "Record it as UNINFORMATIVE (clean-arm count $IN), per the task-8 contract. A clean"
    echo "trace is NOT evidence about the fault's mechanism."
  else
    echo "A fault is present -> correlate it against the eviction timeline:"
    echo "  # correlate.py needs --server (a server.log) and --smi; --jsonl is optional."
    echo "  # First find the faulting arm's dir under the driver's OUTDIR, then:"
    echo "  python3 $HOME/amd-oss/tools/315-trace/correlate.py \\"
    echo "    --smi $OUT/kfd_smi.tsv --server <arm-dir>/server.log \\"
    echo "    --timeline $OUT/merged-timeline.tsv"
    echo "  # (add --pid <server pid> to filter, --window-ms N to widen the fault neighbourhood)"
  fi
else
  echo "no console at $CONSOLE"
fi
