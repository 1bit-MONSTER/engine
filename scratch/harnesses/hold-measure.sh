#!/usr/bin/env bash
# Measure a gpu-hold script against the two failure shapes named in engine #1639 and #1649, using the owner's own
# reproducer (~/lb/hold-escalation/stubborn.sh, `trap '' TERM INT HUP`) plus a root-exits payload.
#
# Scratch harness, not engine tree. Everything is sandboxed: own queue dir, own token file, own log, fake Tctl, and the
# real ~/GPU-TOKEN / ~/GPU-QUEUE are never touched. CPU only -- no device work, so no gpu-hold ticket is needed to run it.
#
# usage: measure-hold.sh <path to a gpu-hold script>
# exit: 0 = both shapes behaved (token released, no survivor), 1 = at least one shape failed, 2 = usage
set -uo pipefail

hold=${1:?usage: measure-hold.sh <gpu-hold script>}
[ -x "$hold" ] || { echo "not executable: $hold" >&2; exit 2; }
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT

# --- the two payloads -------------------------------------------------------------------------
# A: the owner's stubborn.sh, verbatim semantics: ignores TERM/INT/HUP and loops.
cat > "$work/stubborn.sh" <<'STUBBORN'
#!/usr/bin/env bash
trap '' TERM INT HUP
echo "stubborn: started $$"
echo $$ > "${PIDDIR:?}/stubborn.pid"
while : ; do sleep 1; done
STUBBORN
# B: the #1649 shape -- the payload's ROOT exits by itself, leaving a TERM-ignoring child behind.
cat > "$work/rootexits.sh" <<'ROOTEXITS'
#!/usr/bin/env bash
( trap '' TERM; exec -a hold-child sleep 300 ) &
echo $! > "${PIDDIR:?}/child.pid"
sleep 1
exit 0
ROOTEXITS
chmod +x "$work/stubborn.sh" "$work/rootexits.sh"

: > "$work/tctl"; printf '200\n' > "$work/tctl"     # a fake Tctl reading: no thermal gate
run() {  # <label> <payload> <seconds-of-max-hold>
    local label=$1 payload=$2 mins=$3 dir="$work/$1"
    mkdir -p "$dir/q"
    local t0 token_wait
    PIDDIR="$dir" GPU_QUEUE_DIR="$dir/q" GPU_TOKEN_FILE="$dir/token" GPU_HOLD_LOG="$dir/log" \
    GPU_HOLD_POLL=1 GPU_HOLD_WAIT=5 GPU_HOLD_RELEASE_WAIT=0 GPU_HOLD_KILL_GRACE=2 \
    GPU_HOLD_TCTL_START=999 GPU_HOLD_TCTL_FILE="$work/tctl" GPU_HOLD_THERMAL_FLAG="$dir/none" \
        "$hold" -t "$mins" -l "$label" -- "$payload" > "$dir/out" 2>&1
    local rc=$?
    # how long after the hold returned is the token still held? (it should be gone immediately)
    t0=$(date +%s)
    while [ -e "$dir/token" ] && [ $(( $(date +%s) - t0 )) -lt 40 ]; do sleep 1; done
    token_wait=$(( $(date +%s) - t0 ))
    local survivor="" pid
    for f in "$dir/stubborn.pid" "$dir/child.pid"; do
        [ -f "$f" ] || continue
        pid=$(cat "$f")
        kill -0 "$pid" 2>/dev/null && survivor="$survivor $pid"
    done
    printf '%-12s hold rc=%-3s token gone %2ss after return; survivor:%s\n' \
        "$label" "$rc" "$token_wait" "${survivor:-none}"
    [ -z "$survivor" ] && [ -e "$dir/token" ] && { echo "  FAIL token still held"; return 1; }
    [ -z "$survivor" ] || return 1
}

echo "== $hold =="
fail=0
run stubborn "$work/stubborn.sh" 0.05     || fail=1     # the owner's 3 s max-hold repro
run rootexits "$work/rootexits.sh" 30     || fail=1     # #1649: payload exits by itself
if [ "$fail" = 0 ]; then echo "BOTH SHAPES CLEAN"; else echo "AT LEAST ONE SHAPE FAILED"; fi
exit "$fail"
