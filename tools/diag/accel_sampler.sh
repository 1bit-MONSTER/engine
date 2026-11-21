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
# hw-gate keeps a failing row's server log (engine#2500, #2513) under $HW_GATE_STORE/failed and names it in the
# failure line. The name must be unique per failure: CI sets HW_GATE_OUT=$RUNNER_TEMP/hw-gate, so a name built from
# basename "$OUT" was the constant "hw-gate-<model>.log" and every failing run on every PR overwrote the last one's
# evidence. These arms run the CI shape (same model, same run, same second), and check that a failed copy prints no
# path. A recording fake gpu-hold is first on PATH, so the internal mode is also proven to take no GPU ticket.
# No GPU, no model, runs in CI.
#
# The gate's phase check must match the phase FIELD, not a prefix of it, and only the phases that actually write
# `model <name>` rows may be satisfied by one (review of engine#2523):
#
#   * `npu` is a prefix of `npuserve`, so `grep -qE "^npu"` read an `npuserve` row as proof that `npu` had run;
#   * `placement` writes `placement <name>`, not `model <name>`, so grouping it with kat|kld|needle let any model row
#     satisfy a requested placement phase -- the same silent pass, one name over.
#
# The loop is EXTRACTED FROM THE SCRIPT and run, so this pins the shipped code rather than a copy of it. A copy would
# pass while the real check drifted.
# engine#2542: attribute the amdxdna IO_PAGE_FAULT bursts by sampling WHICH PROCESS HOLDS /dev/accel/accel0 open, and
# correlating each fault with the samples around it.
#
# Read-only by construction: it scans /proc for open-file symlinks and reads journald. It does not open the accel node,
# does not talk to XRT, never touches :18097, and needs no sudo (so no `lsof` loops). Because it takes no device action it
# needs no npu-hold ticket -- but it also must never *cause* a fault, which is why every read is /proc or the journal.
#
# Two loops in one unit:
#   sample   every SAMPLE_S seconds: the PIDs holding the accel node, with cmdline, cgroup scope and the npu-hold ticket
#            (from ~/NPU-TOKEN) as they stood at that moment
#   watch    journalctl -k -f, and for each IO_PAGE_FAULT line, the samples within +-WINDOW_S seconds, written as one
#            attribution record
#
# Inputs (env, all optional): ACCEL_SAMPLER_DIR, SAMPLE_S (2), WINDOW_S (5), MAX_BYTES (50 MB), ACCEL_NODE.
set -u
# engine#3072: `DIR=` looks as deliberate as `ACCEL_SAMPLER_DIR=` and does NOTHING, so a smoke run wrote the LIVE summary
# while the operator believed it went to scratch (measured: a 22 s smoke replaced the fleet's fault summary with a branch's
# numbers, and `check-deployed` stayed ok because the deployed bytes never changed). Refuse it by name.
if [ -n "${DIR:-}" ] && [ -z "${ACCEL_SAMPLER_DIR:-}" ]; then
  echo "accel-sampler: DIR is not a knob -- the sampler reads ACCEL_SAMPLER_DIR. Nothing was written." >&2
  echo "accel-sampler: use ACCEL_SAMPLER_DIR=<scratch dir> for a smoke run (ACCEL_SAMPLER_LIVE_OK=1 for the live one)." >&2
  exit 2
fi
DIR=${ACCEL_SAMPLER_DIR:-$HOME/.cache/accel-sampler}
SAMPLE_S=${SAMPLE_S:-2}
FULL_SCAN_S=${FULL_SCAN_S:-30}   # a full /proc sweep every 2 s cost 62% of a core (measured); holders are long-lived
                                 # servers, so re-checking the known ones each sample is enough between sweeps
WINDOW_S=${WINDOW_S:-5}
MAX_BYTES=${MAX_BYTES:-52428800}   # 50 MB, rotate rather than grow without bound
NODE=${ACCEL_NODE:-/dev/accel/accel0}
mkdir -p "$DIR" || exit 1
SAMP=$DIR/samples.log
ATTR=$DIR/attributed.log
mkdir -p "$DIR"

rotate() {  # $1 = file: hard cap, one previous kept
  local f=$1 sz
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt "$MAX_BYTES" ]; then
    mv -f "$f" "$f.1" 2>/dev/null
    : > "$f"
  fi
}

ticket() {  # the npu-hold ticket that was live, if one is recorded
  for p in "$HOME/NPU-TOKEN" "$HOME/.npu-hold" "$HOME/lb/NPU-TOKEN"; do
    [ -f "$p" ] && { head -c 200 "$p" | tr '\n' ' '; return; }
  done
  echo "-"
}

holders() {  # print "pid|cmd|scope|fds" for every process holding $NODE open
  local pid fd tgt cmd scope n
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    [ -d "/proc/$pid/fd" ] || continue                 # kernel threads: no fd table, skip cheaply
    n=0
    for fd in /proc/"$pid"/fd/*; do
      tgt=$(readlink "$fd" 2>/dev/null) || continue
      [ "$tgt" = "$NODE" ] && n=$((n + 1))
    done
    [ "$n" -eq 0 ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)
    scope=$(sed -n 's/^0:://p' "/proc/$pid/cgroup" 2>/dev/null | head -1)
    [ -z "$scope" ] && scope=$(awk -F/ '/^0::/{print $NF}' "/proc/$pid/cgroup" 2>/dev/null)
    printf '%s|%s|%s|%s\n' "$pid" "${cmd:--}" "${scope:--}" "$n"
  done
}

KNOWN=$DIR/known.pids
sample_once() {  # one line per (moment, holder); "none" when nothing holds the node
  local ts seen=0 now last=0
  ts=$(date +%s)
  now=$ts
  [ -f "$KNOWN" ] && last=$(cat "$KNOWN.last" 2>/dev/null || echo 0)
  if [ $((now - last)) -ge "$FULL_SCAN_S" ]; then
    holders | cut -d'|' -f1 | sort -u > "$KNOWN.new"
    mv -f "$KNOWN.new" "$KNOWN"; echo "$now" > "$KNOWN.last"
  fi
  # only the processes known to hold the node: count their accel fds (cheap) and record them
  while read -r pid; do
    [ -z "$pid" ] && continue
    local n=0 fd
    for fd in /proc/"$pid"/fd/*; do
      [ "$(readlink "$fd" 2>/dev/null)" = "$NODE" ] && n=$((n + 1))
    done
    [ "$n" -eq 0 ] && continue      # it closed it; the next full sweep drops it from KNOWN
    seen=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$pid" "$n" \
      "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)" \
      "$(awk -F/ '/^0::/{print $NF}' "/proc/$pid/cgroup" 2>/dev/null)" \
      "$(ticket)" >> "$SAMP"
  done < "$KNOWN" 2>/dev/null
  if [ "$seen" = 0 ]; then
    printf '%s\t-\t-\t-\t-\t%s\n' "$ts" "$(ticket)" >> "$SAMP"
  fi
  rotate "$SAMP"
}

nearest() {  # $1 fault epoch -> "<delta>s <holder fields>" for the CLOSEST sample, or "" when there are none
  # A fixed +-WINDOW_S is the wrong instrument here: the loop measures 0.02 s per sample but at nice 19 on a loaded box it
  # loses its schedule, and the observed gaps reach 8-13 s (and once 93 s). A window narrow enough to be precise then finds
  # nothing, and its silence reads as "no holder". So report the nearest sample and its distance instead of pretending to a
  # resolution the loop does not have.
  awk -F'\t' -v t="$1" '
    { d = $1 - t; if (d < 0) d = -d
      if (best == "" || d < bestd) { bestd = d; best = $2"\t"$4"\t"$5"\t"$6 } }
    END { if (best != "") printf "%ds\t%s", bestd, best }' "$SAMP" 2>/dev/null
}

correlate() {  # $1 lo $2 hi -> "pid:cmd:scope" for every sample in [lo,hi], deduplicated
  awk -F'\t' -v lo="$1" -v hi="$2" '$1 >= lo && $1 <= hi' "$SAMP" 2>/dev/null | cut -f2,4,5,6 | sort -u | tr '\n' ';'
}

selftest() {  # prove the correlation before an event needs it: the window, the fields, and the miss case
  local d fail=0 out
  d=$(mktemp -d "$DIR/selftest.XXXXXX")   # never /tmp: this lane's rule, and tmpfs pressure has bitten the box
  printf '1000\t111\t1\t/serve A\t/scope/a\tticket1\n1010\t222\t2\t/serve B\t/scope/b\tticket2\n' > "$d/samples.log"
  SAMP=$d/samples.log
  out=$(correlate 1005 1015)                       # +-5 s around a fault at 1010
  case "$out" in *222*) ;; *) echo "FAIL: the holder at the fault instant is missing"; fail=1 ;; esac
  # the ticket is the field the whole issue is about -- a record without it cannot say which ticket was live. This arm
  # is here because a refactor dropped exactly that field and the other arms stayed green (review of engine#2591).
  case "$out" in *ticket2*) ;; *) echo "FAIL: the npu-hold ticket was dropped from the record"; fail=1 ;; esac
  # nearest: with samples at 1000 and 1010, a fault at 1009 must pick 1010 and say "1s"
  n=$(nearest 1009)
  case "$n" in 1s*) ;; *) echo "FAIL: nearest did not pick the closer sample (got '$n')"; fail=1 ;; esac
  case "$out" in *111*) echo "FAIL: a sample outside the window was included"; fail=1 ;; esac
  out=$(correlate 2000 2010)
  [ -z "$out" ] || { echo "FAIL: a window with no samples produced '$out'"; fail=1; }
  # a "none" sample (nobody holding the node) must be reported as such, not as a holder
  printf '1000\t-\t-\t-\t-\t-\n' > "$d/samples.log"
  out=$(correlate 995 1005)
  case "$out" in *'-:-:-'*|*'-'*) ;; *) echo "FAIL: a no-holder sample was not passed through"; fail=1 ;; esac
  # rotation: a 24 h run WILL hit it, so it gets an arm rather than a hand test
  MAX_BYTES=10; printf 'line one is long enough to exceed the cap\n' > "$d/rot.log"
  rotate "$d/rot.log"
  [ -f "$d/rot.log.1" ] || { echo "FAIL: rotation did not keep a previous file"; fail=1; }
  [ "$(stat -c %s "$d/rot.log")" -eq 0 ] || { echo "FAIL: the live file was not truncated"; fail=1; }
  # engine#2778: the summary must be REPLACED, not truncated in place. The observable is the inode: `> "$out"` keeps it
  # (truncate + fill, so a reader inside the window sees a valid header and nothing else), `> "$out.tmp" && mv -f` changes it
  # (rename over the old entry). This arm is deterministic -- no reader race -- and it fails on the truncating line.
  mkdir -p "$d/sum"
  : > "$d/sum/samples.log"
  SAMP=$d/sum/samples.log ATTR=$d/sum/attributed.log DIR=$d/sum
  summary > /dev/null 2>&1
  ino_before=$(stat -c %i "$d/sum/summary.txt" 2>/dev/null || echo none)
  summary > /dev/null 2>&1
  ino_after=$(stat -c %i "$d/sum/summary.txt" 2>/dev/null || echo none)
  [ "$ino_before" != none ] && [ "$ino_after" != none ] || { echo "FAIL: summary() did not write a summary"; fail=1; }
  [ "$ino_before" != "$ino_after" ] || { echo "FAIL: the summary is truncated in place (same inode) -- a reader can see a header with nothing under it"; fail=1; }
  [ -e "$d/sum/summary.txt.tmp" ] && { echo "FAIL: the temp file was left behind"; fail=1; }
  head -1 "$d/sum/summary.txt" | grep -q '^# accel sampler summary ' || { echo "FAIL: the written summary has no header"; fail=1; }
  # engine#2795: every summary names the bytes that wrote it, and a deployed copy can be told apart from the repo's.
  grep -q '^# script sha256 [0-9a-f]\{64\}$' "$d/sum/summary.txt" \
    || { echo "FAIL: the summary does not carry the running script's sha256"; fail=1; }
  # The arm that must be able to fail: install a copy, modify it behind the provenance, and require DRIFTED.
  printf '#!/bin/bash\necho modified\n' > "$d/deployed.sh"
  install_sampler "$0" "$d/deployed.sh" testcommit >/dev/null 2>&1
  # engine#2795 follow-up: no destination must REFUSE and write nothing -- the old default was the live path.
  rm -f "$d/nodefault.sh" "$d/nodefault.sh.provenance"
  install_sampler "$0" >/dev/null 2>&1 && { echo "FAIL: install with no destination must refuse (it used to write to the live path)"; fail=1; }
  [ -e "$d/nodefault.sh" ] && { echo "FAIL: install with no destination wrote something anyway"; fail=1; }
  # ... and a destination that is given gets its commit from the checkout the SOURCE came from, not `unknown`.
  # engine#3069: this arm used to install from `$0` and then assert a commit was found -- which asserts a fact about where
  # the CALLER happens to sit. The deployed copy lives in ~/.cache/accel-sampler, not in a checkout, so `commit=unknown`
  # is the right answer there and `selftest` failed on the one copy lanes are told to run. The fixture now owns its
  # source: a throwaway repo under $d, the script committed into it, installed from there.
  fixture_repo() {  # <dir> -- a repo holding a copy of this script, so provenance has a commit to find
    mkdir -p "$1/tools/diag" && cp "${0}" "$1/tools/diag/accel_sampler.sh" &&
      git -C "$1" init -q && git -C "$1" -c user.email=t@t -c user.name=t add -A &&
      git -C "$1" -c user.email=t@t -c user.name=t commit -qm fixture
  }
  mkdir -p "$d/repo" && fixture_repo "$d/repo" \
    && install_sampler "$d/repo/tools/diag/accel_sampler.sh" "$d/derived.sh" >/dev/null 2>&1
  grep -q '^commit=unknown$' "$(provenance_file "$d/derived.sh")" \
    && { echo "FAIL: install recorded commit=unknown for a source inside its own git checkout"; fail=1; }
  # the mirror arm: a source that is inside NO checkout must record `unknown`, because that is what it can know. Without
  # this half, "install finds a commit" could be satisfied by inventing one.
  mkdir -p "$d/bare" && cp "${0}" "$d/bare/accel_sampler.sh" \
    && install_sampler "$d/bare/accel_sampler.sh" "$d/bare-installed.sh" >/dev/null 2>&1
  grep -q '^commit=unknown$' "$(provenance_file "$d/bare-installed.sh")" \
    || { echo "FAIL: a source outside any checkout must record commit=unknown"; fail=1; }
  # engine#3072: `DIR=` is the near-miss -- it looks like a knob and does nothing, so a smoke run silently used the LIVE
  # directory and replaced the fleet's fault summary with a branch's numbers. It must refuse BY NAME and write nothing.
  mkdir -p "$d/wrongknob"
  knob_out=$(DIR="$d/wrongknob" bash "$0" sample 2>&1); knob_rc=$?
  [ "$knob_rc" = 2 ] || { echo "FAIL: DIR= gave rc $knob_rc, want 2: it is not a knob and the run must refuse (engine#3072)"; fail=1; }
  case "$knob_out" in
    *"DIR is not a knob"*) ;;
    *) echo "FAIL: a refused DIR= run must NAME the variable it wants; said: $knob_out"; fail=1 ;;
  esac
  [ -s "$d/wrongknob/samples.log" ] && { echo "FAIL: a refused DIR= run wrote samples anyway"; fail=1; }
  # ask 1 on #2815: `check-deployed` with no argument must look at the DEPLOYED copy, not at the checkout it was run from.
  # ask 1 on #2815 needs an arm that DISCRIMINATES the two paths: a provenance-less copy is rc 2 either way, so the arm
  # that pins the fix is one where the deployed copy is valid and `$0` is not. Reverting the default to `$0` fails these.
  mkdir -p "$d/live" && install_sampler "$0" "$d/live/accel-sampler.sh" testcommit >/dev/null 2>&1
  ( DIR=$d/live check_deployed >/dev/null 2>&1 ) \
    || { echo "FAIL: check-deployed with no argument did not resolve the DEPLOYED copy (it looked at \$0 instead)"; fail=1; }
  printf '# drift\n' >> "$d/live/accel-sampler.sh"
  ( DIR=$d/live check_deployed >/dev/null 2>&1 ) \
    && { echo "FAIL: check-deployed with no argument passed a drifted deployed copy"; fail=1; }
  rm -f "$(provenance_file "$d/live/accel-sampler.sh")"
  rc_nodef=$( DIR=$d/live check_deployed >/dev/null 2>&1; echo $? )
  [ "$rc_nodef" = 2 ] \
    || { echo "FAIL: check-deployed with no argument must be 'cannot tell' (rc 2) for a provenance-less deployed copy, got rc $rc_nodef"; fail=1; }
  check_deployed "$d/deployed.sh" >/dev/null 2>&1 \
    || { echo "FAIL: a freshly installed copy must check clean"; fail=1; }
  # engine#3087: --against answers "are the deployed bytes CURRENT?" -- which comparing them to their own provenance cannot.
  # The fixture repo owns both sides: a ref holding the deployed bytes is rc 0, one commit later is rc 1 and says STALE,
  # and with no repo named it is rc 2 cannot-tell -- never a pass.
  mkdir -p "$d/arepo/tools/diag" && git -C "$d/arepo" init -q
  cp "$d/deployed.sh" "$d/arepo/tools/diag/accel_sampler.sh"
  git -C "$d/arepo" -c user.email=t@t -c user.name=t add -A
  git -C "$d/arepo" -c user.email=t@t -c user.name=t commit -qm same
  ACCEL_SAMPLER_REPO=$d/arepo check_deployed --against HEAD "$d/deployed.sh" >/dev/null 2>&1 \
    || { echo "FAIL: --against a ref holding the deployed bytes must be rc 0 (engine#3087)"; fail=1; }
  printf 'one line later\n' >> "$d/arepo/tools/diag/accel_sampler.sh"
  git -C "$d/arepo" -c user.email=t@t -c user.name=t commit -qam newer
  against_out=$(ACCEL_SAMPLER_REPO=$d/arepo check_deployed --against HEAD "$d/deployed.sh" 2>&1)
  against_rc=$?
  [ "$against_rc" = 1 ] || { echo "FAIL: --against a newer ref must be rc 1, got $against_rc"; fail=1; }
  case "$against_out" in
    *STALE*) ;;
    *) echo "FAIL: the stale verdict must say STALE, got: $against_out"; fail=1 ;;
  esac
  ( unset ACCEL_SAMPLER_REPO; check_deployed --against HEAD "$d/deployed.sh" >/dev/null 2>&1 )
  [ "$?" = 2 ] || { echo "FAIL: --against with no repo must be rc 2 cannot-tell, not a pass"; fail=1; }
  printf '# drift\n' >> "$d/deployed.sh"
  check_deployed "$d/deployed.sh" >/dev/null 2>&1 \
    && { echo "FAIL: check-deployed passed a copy that had been modified behind its provenance"; fail=1; }
  rm -f "$(provenance_file "$d/deployed.sh")"
  [ "$(check_deployed "$d/deployed.sh" >/dev/null 2>&1; echo $?)" = 2 ] \
    || { echo "FAIL: a deployed copy with no provenance must be 'cannot tell' (rc 2), not a pass"; fail=1; }
  # engine#2478: the per-device split must be in the summary, because a single total is the number that hid 1472 NVMe
  # records being read as an NPU stream. Fixture: one line of each device, and both must appear as their own row.
  mkdir -p "$d/dev"
  printf '%s\t0s\t-\t-\t-\t-\t1.0 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x64e44000 flags=0x0000]\n' 1 > "$d/dev/attributed.log"
  printf '%s\t1s\t-\t-\t-\t-\t2.0 host kernel: amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0003 address=0x7f29 flags=0x0007]\n' 2 >> "$d/dev/attributed.log"
  : > "$d/dev/samples.log"
  SAMP=$d/dev/samples.log ATTR=$d/dev/attributed.log DIR=$d/dev summary > /dev/null 2>&1
  grep -qE '^ +1 nvme$' "$d/dev/summary.txt" || { echo "FAIL: the summary does not count the nvme fault as its own device"; fail=1; }
  grep -qE '^ +1 amdxdna$' "$d/dev/summary.txt" || { echo "FAIL: the summary does not count the amdxdna fault as its own device"; fail=1; }
  # The fault stream must carry every class the gate refuses on, not page faults alone. Fixture: the two lines this box
  # really logged at 18:10:43 AST on 2026-11-20 -- under the old `IO_PAGE_FAULT`-only grep neither matched, so the arm
  # fails on the shipped pattern and passes on the widened one.
  printf '%s\n' \
    '1795195843.4 host kernel: amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_dead_fw_detect: TDR timeout detected' \
    '1795195843.4 host kernel: amdxdna 0000:c6:00.1: [drm] aie2_tdr_dead_fw_recover_all: Array stopped dispatching, power-cycling NPU to recover' \
    > "$d/tdr.txt"
  [ "$(grep -cE "$FAULT_RE" "$d/tdr.txt")" = 2 ] \
    || { echo "FAIL: an NPU TDR / power-cycle line does not reach the fault stream (engine#3067)"; fail=1; }
  printf '%s\n' '1795195844.0 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x64e44000 flags=0x0000]' \
    | grep -qE "$FAULT_RE" \
    || { echo "FAIL: widening the stream dropped the NVMe page-fault class it used to carry"; fail=1; }
  # One line per class the GATE acts on, because the NVMe line above is the sampler's own extra and proves nothing about
  # them: a GPU take refused for a gfxhub fault or a ring timeout is exactly the case this record is asked about.
  printf '%s\n' \
    '1795195850.0 host kernel: amdgpu 0000:0c:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:6)' \
    '1795195851.0 host kernel: amdgpu 0000:0c:00.0: amdgpu: GPU reset begin!' \
    '1795195852.0 host kernel: amdgpu 0000:0c:00.0: ring gfx_0.0.0 timeout' \
    > "$d/gpu.txt"
  [ "$(grep -cE "$FAULT_RE" "$d/gpu.txt")" = 3 ] \
    || { echo "FAIL: a fault class the gate refuses on does not reach the stream (gfxhub page fault / amdgpu reset / ring timeout)"; fail=1; }
  # ... and the superset property STRUCTURALLY, read out of the gate rather than trusted: every alternative in its
  # GPU_FAULT_RE must appear verbatim in FAULT_RE, so a class added to the gate fails this arm until the sampler has it.
  gate_src=$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]:-$0}")")")/scripts/gpu-hold
  if [ -r "$gate_src" ]; then
    gate_re=$(sed -n "s/^GPU_FAULT_RE='\(.*\)'$/\1/p" "$gate_src")
    if [ -n "$gate_re" ]; then
      while IFS= read -r alt; do
        case "$FAULT_RE" in
          *"$alt"*) ;;
          *) echo "FAIL: the gate refuses on '$alt' and FAULT_RE does not carry it -- the superset claim is false"; fail=1 ;;
        esac
      done < <(printf '%s\n' "$gate_re" | tr '|' '\n')
    else
      echo "SKIP: no GPU_FAULT_RE line found in $gate_src, so containment was not checked"
    fi
  else
    echo "SKIP: $gate_src is not readable (a deployed copy has no gate beside it), so containment was not checked"
  fi
  # ... and the device split still names amdxdna for a TDR line, through the same token the summary's awk reads.
  printf '1795195843\t-\t-\t-\t-\t-\t1795195843.4 host kernel: amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_dead_fw_detect: TDR timeout detected\n' > "$d/dev/attributed.log"
  SAMP=$d/dev/samples.log ATTR=$d/dev/attributed.log DIR=$d/dev summary > /dev/null 2>&1
  grep -qE '^ +1 amdxdna$' "$d/dev/summary.txt" \
    || { echo "FAIL: a TDR line is not attributed to amdxdna in the summary"; fail=1; }
  rm -rf "$d"
  [ "$fail" = 0 ] && echo "accel-sampler self-test: ok (nearest+distance, fields+ticket, empty window, no-holder sample, rotation, summary replaced not truncated, faults split by device)"
  return "$fail"
}

# WHICH LINES REACH THE FAULT STREAM. This instrument exists to say whether a fault is inside a window, so its pattern
# must be a SUPERSET of the one the gate refuses takes on: it may record a line the gate ignores, never miss one the gate
# acts on. So the gate's five alternatives (`scripts/gpu-hold` GPU_FAULT_RE) are copied here VERBATIM and first -- a
# `[gfxhub]` page fault, an `amdgpu` reset/hang, a ring timeout, an MES reset/failure, an amdxdna command-timeout,
# power-cycle or fault -- and the sampler's own extras follow. Verbatim matters: the self-test reads that pattern out of
# the gate and asserts each of its alternatives appears in this line, so a class added to the gate fails the arm until the
# sampler carries it, instead of the two lists agreeing by luck (the #3042 shape: several readers, one of them wrong).
# The gate's SKIP list is deliberately NOT shared: it exists to stop one reset counting many times, and a recorder keeps
# the lines. Consequence, stated because it is real: `attributed faults:` counts matching LINES, so an amdgpu reset that
# re-logs ~20 stack frames now counts ~20. Grouping those into events is a change to the summary; importing the skip would
# silently drop the TDR lines this pattern exists to add.
# It used to grep `IO_PAGE_FAULT` alone. Measured 2026-11-20 on strixhalo: an NPU TDR + power-cycle at 18:10:43 AST
# (22:10:43Z) logged four lines (`aie2_tdr_dead_fw_detect: TDR timeout detected`, `aie2_tdr_work_func`, ...`recover_all:
# ... power-cycling NPU`, `aie2_hw_reset: NPU firmware unhealthy: power-cycling NPU`) while this sampler was RUNNING, and
# none of them matched: attributed.log gained nothing and the summary for that window read clean. The device split below
# needed no change -- a TDR line's token after `kernel: ` is still `amdxdna`, which is the token the awk reads.
FAULT_RE='\[gfxhub\] page fault|amdgpu.*(reset|hang)|ring .*timeout|MES.*(reset|failure|failed to respond)|amdxdna.*(command timeout|power.?cycl(e|ing)|fault)|IO_PAGE_FAULT|aie2_tdr|aie2_hw_reset'

watch_faults() {  # correlate each fault with the samples around it
  local line ts lo hi hits
  journalctl -k -f -o short-unix 2>/dev/null | grep --line-buffered -E "$FAULT_RE" |
  while IFS= read -r line; do
    ts=${line%% *}                       # short-unix: "<epoch.frac> host kernel: ..."
    ts=${ts%%.*}
    lo=$((ts - WINDOW_S)); hi=$((ts + WINDOW_S))
    hits=$(nearest "$ts")
    printf '%s\t%s\t%s\n' "$ts" "${hits:--}" "$(printf '%s' "$line" | cut -c1-200)" >> "$ATTR"
    rotate "$ATTR"
  done
}

script_sha256() {  # the bytes that are RUNNING, so every summary names the version that produced it (engine#2795)
  # BASH_SOURCE first, $0 second: a summary produced through a SOURCED copy must still name the real file. I hit exactly
  # that while testing this -- `source fns.sh; summary` wrote `sha256 unavailable` into the live summary, i.e. a provenance
  # line that was there and useless, which is the shape this issue is about.
  if [ -n "${SELF_SHA:-}" ]; then printf '%s' "$SELF_SHA"; return; fi
  command -v sha256sum >/dev/null || { printf 'unavailable'; return; }
  local f=${BASH_SOURCE[0]:-$0}
  SELF_SHA=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  printf '%s' "${SELF_SHA:-unavailable}"
}

provenance_file() { printf '%s.provenance' "$1"; }

summary() {  # what the run observed, so a 24 h expiry is not silence
  local out=$DIR/summary.txt n attr
  if [ -f "$SAMP" ]; then n=$(wc -l < "$SAMP"); else n=0; fi
  if [ -f "$ATTR" ]; then attr=$(wc -l < "$ATTR"); else attr=0; fi
  {
    echo "# accel sampler summary $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# script sha256 $(script_sha256)"
    echo "samples: $n   attributed faults: $attr"
    echo
    # engine#2478 correction: the fault reader greps `IO_PAGE_FAULT` with NO device filter, so this stream carries NVMe
    # faults on the root SSD and amdxdna faults on the NPU. Measured 2026-11-16: 1472 nvme vs 90 amdxdna in 1562 records --
    # and a report that treated the total as the NPU stream was wrong by 16x. A single total is the number that hid it, so
    # the summary splits by device before anything else is read.
    echo "## faults by device (from the recorded lines)"
    if [ -f "$ATTR" ]; then
      awk -F'\t' '{print $NF}' "$ATTR" 2>/dev/null | grep -oE 'kernel: [a-z0-9_]+' | awk '{print $NF}' \
        | sort | uniq -c | sort -rn | sed 's/^/  /'
    fi
    echo
    echo "## holders seen (pid, cmd, scope, ticket, sample count)"
    awk -F'\t' '$2 != "-" {print $2"\t"$4"\t"$5"\t"$6}' "$SAMP" 2>/dev/null | sort | uniq -c | sort -rn | head -20
    echo
    if [ "$attr" -gt 0 ]; then
      echo "## attribution records"
      cut -c1-300 "$ATTR" | tail -20
    else
      echo "## no fault line (page fault, amdgpu/MES reset, amdxdna TDR/power-cycle) was seen in this window"
      echo "The sampler cannot attribute a burst that did not happen. What it does establish: the holders above were"
      echo "live for the whole window, so if a burst had occurred, its record would name them."
    fi
  } > "$out.tmp" && mv -f "$out.tmp" "$out"
  # engine#2778: written to a temp file and renamed, never truncated in place. Since #2613 refreshes this every ~15 min, a
  # 24 h capture has 96 windows per day in which a reader (accel_attribution.py, or a lane grepping the summary at the moment
  # of a fault) opens a file that exists, has a valid header line, and is empty after it -- i.e. "nothing was held" for a
  # window in which something was. rename(2) within one directory makes a reader see the old summary or the new one.
  echo "summary written: $out"
}

# engine#2795: the deployed copy drifted 52 lines from the repo and nothing noticed -- the running instrument was not the
# instrument in the repository, and every summary it wrote carried a timestamp but no version. `install` records what it
# deployed; `check-deployed` is the arm that can fail.
install_sampler() {  # install <repo-root-or-script> <dest> [commit] -- dest is REQUIRED
  # The destination used to default to $DIR/accel-sampler.sh, i.e. the LIVE path the unit runs. Measured cost of that
  # default, twice in one evening: a lane testing a PR branch redeployed the box's instrument with the branch under test,
  # and the summary it wrote then claimed a provenance the instrument did not have. Deployment writes the bytes a unit
  # will execute on its next restart, so it is never done by omission: no destination, no write. (engine#2795 follow-up)
  local src=$1 dest=${2:-} commit=${3:-${GIT_COMMIT:-}}
  if [ -z "$dest" ]; then
    echo "install: a destination is required: install <repo-or-script> <dest> [commit]" >&2
    echo "install: refusing to write, because the old default was the live path ($DIR/accel-sampler.sh) and deploying" >&2
    echo "install: the running instrument must never happen by omission. Nothing was written." >&2
    return 2
  fi
  [ -f "$src" ] || src=$src/tools/diag/accel_sampler.sh
  [ -f "$src" ] || { echo "install: no accel_sampler.sh at $1" >&2; return 2; }
  bash -n "$src" || { echo "install: $src does not parse" >&2; return 2; }
  # No commit given: read it from the checkout the source came from, if it is one. `unknown` only when neither exists,
  # because a provenance line that says `unknown` beside a real deployment is a fact nobody can use later.
  if [ -z "$commit" ]; then
    commit=$(git -C "$(dirname "$src")" rev-parse --short=12 HEAD 2>/dev/null || true)
    [ -n "$commit" ] || commit=unknown
  fi
  mkdir -p "$(dirname "$dest")"
  # copy then rename, so a running sampler never sees a half-written script
  cp "$src" "$dest.tmp" && mv -f "$dest.tmp" "$dest" || return 2
  chmod +x "$dest"
  local sha; sha=$(sha256sum "$dest" | cut -d' ' -f1)
  {
    echo "commit=$commit"
    echo "sha256=$sha"
    echo "source=$src"
    echo "deployed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$(provenance_file "$dest").tmp"
  mv -f "$(provenance_file "$dest").tmp" "$(provenance_file "$dest")"
  echo "install: $dest sha256 $sha commit $commit"
}

check_deployed() {  # check-deployed [dest] -- rc 0 the deployed copy is the bytes its provenance names, 1 drifted, 2 cannot tell
  # No argument means THE DEPLOYED COPY, not "$0": run from a checkout, `$0` is the repo's copy and the question a reader
  # is asking ("are the bytes the unit runs the bytes that were deployed?") gets answered about the wrong file. That was
  # review ask 1 on #2815: the two commands must agree on one path, so both name $DIR/accel-sampler.sh.
  local against="" dest prov sha_actual sha_expected
  if [ "${1:-}" = "--against" ]; then
    against=${2:-}
    [ -n "$against" ] || { echo "check-deployed: cannot tell: --against needs a git ref" >&2; return 2; }
    dest=${3:-$DIR/accel-sampler.sh}
  else
    dest=${1:-$DIR/accel-sampler.sh}
  fi
  prov=$(provenance_file "$dest")
  if [ ! -f "$prov" ]; then
    echo "check-deployed: cannot tell: no provenance beside $dest (run: accel_sampler.sh install <repo> $dest <commit>)" >&2
    return 2
  fi
  sha_expected=$(sed -n 's/^sha256=//p' "$prov")
  sha_actual=$(sha256sum "$dest" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$sha_expected" ] || [ -z "$sha_actual" ]; then
    echo "check-deployed: cannot tell: could not read a sha256 from $prov or $dest" >&2
    return 2
  fi
  if [ "$sha_expected" != "$sha_actual" ]; then
    echo "check-deployed: DRIFTED $dest: $(sed -n 's/^commit=//p' "$prov") deployed sha256 $sha_expected, on disk $sha_actual" >&2
    return 1
  fi
  echo "check-deployed: ok $dest sha256 $sha_actual commit $(sed -n 's/^commit=//p' "$prov")"

  [ -n "$against" ] || return 0
  local repo=${ACCEL_SAMPLER_REPO:-} ref_sha sha_ref behind
  if [ -z "$repo" ]; then
    echo "check-deployed: cannot tell: --against needs ACCEL_SAMPLER_REPO=<engine checkout>; nothing was compared" >&2
    return 2
  fi
  ref_sha=$(git -C "$repo" rev-parse --verify --quiet "$against^{commit}" 2>/dev/null) || {
    echo "check-deployed: cannot tell: '$against' is not a commit in $repo" >&2; return 2; }
  sha_ref=$(git -C "$repo" show "$ref_sha:tools/diag/accel_sampler.sh" 2>/dev/null | sha256sum | cut -d' ' -f1)
  [ -n "$sha_ref" ] || { echo "check-deployed: cannot tell: $repo has no tools/diag/accel_sampler.sh at ${ref_sha:0:9}" >&2; return 2; }
  if [ "$sha_ref" = "$sha_actual" ]; then
    echo "check-deployed: current: the deployed copy is ${ref_sha:0:9} version"
    return 0
  fi
  behind=$(git -C "$repo" rev-list --count "$ref_sha" -- tools/diag/accel_sampler.sh 2>/dev/null)
  echo "check-deployed: STALE $dest is not the ${ref_sha:0:9} version (deployed $sha_actual, that ref $sha_ref, $behind commit(s) touching this file in its history)" >&2
  return 1
}

# engine#3072: a run that resolves to the unit's own directory overwrites what the fleet reads. NOT called by `selftest`,
# which writes only under its own mktemp dir ($DIR/selftest.XXXXXX) and is the instrument's own verification -- gating it
# would stop the tool from checking itself on the box where it matters (engine#3069).
live_dir_guard() {
  # The unit itself runs exactly this file from the live directory with no variables set, so a guard refusing every writer
  # of $DIR would stop the fleet's summary from ever being refreshed. `INVOCATION_ID` is systemd's marker for "started by a
  # unit" and "$0 under $DIR" is that unit's ExecStart; both, and only that pair, pass.
  local is_the_unit=0
  if [ -n "${INVOCATION_ID:-}" ] && [ "$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")" = "$DIR/accel-sampler.sh" ]; then
    is_the_unit=1
  fi
  if [ -z "${ACCEL_SAMPLER_LIVE_OK:-}" ] && [ "$is_the_unit" = 0 ] && [ "$DIR" = "$HOME/.cache/accel-sampler" ] &&
     systemctl --user is-active --quiet accel-sampler.service 2>/dev/null; then
    echo "accel-sampler: DIR resolves to the LIVE directory ($DIR) and accel-sampler.service is active -- refusing to write it." >&2
    echo "accel-sampler: use ACCEL_SAMPLER_DIR=<scratch dir>, or set ACCEL_SAMPLER_LIVE_OK=1 if you mean the live one." >&2
    return 2
  fi
  return 0
}

# the modes that WRITE into $DIR (samples, attribution, or the summary the unit publishes); `selftest` writes only under its
# own mktemp dir and `holders`/`check-deployed` write nothing there.
case "${1:-both}" in
  sample|watch|both|summary) live_dir_guard || exit $? ;;
esac

case "${1:-both}" in
  sample) sample_once ;;
  install) shift; install_sampler "$@" ;;
  check-deployed) shift; check_deployed "$@" ;;
  watch)  watch_faults ;;
  both)
    # the fault reader first, so no fault is missed between start and the first sample
    watch_faults &
    W=$!
    trap 'kill "$W" 2>/dev/null; summary; exit 0' TERM INT
    trap 'kill "$W" 2>/dev/null' EXIT
    LAST=0
    while :; do
      # gap record: a stall is data. Measured need: the loop lost 93 s at the peak of a fault storm, and those minutes became
      # "no holder" attribution records -- silence that looked like a finding. Now the gap is written (and kept) instead.
      NOW=$(date +%s)
      if [ "$LAST" -gt 0 ] && [ $((NOW - LAST)) -gt $((SAMPLE_S * 3)) ]; then
        printf '%s\tGAP\t%s\t-	-	-\n' "$NOW" "$((NOW - LAST))" >> "$SAMP"
      fi
      LAST=$NOW
      sample_once
      # refresh the summary every ~15 min (450 samples at 2 s): a reboot kills the unit and SIGTERM never arrives, so a
      # summary written only on exit can be hours stale exactly when it is needed
      SAMP_N=$(( ${SAMP_N:-0} + 1 ))
      [ $(( SAMP_N % 450 )) -eq 0 ] && summary
      sleep "$SAMPLE_S"
    done
    ;;
  holders) holders ;;
  selftest|--self-test) selftest ;;
  summary) summary ;;
  *) echo "usage: $0 [both|sample|watch|holders|summary|selftest]" >&2; exit 2 ;;
esac
