#!/usr/bin/env bash
# engine#315: build the *instrumented* HRX backend at the engine-pinned revision.
#
# Reproduces the study's own build configuration (from ~/wt/build-3971.sh) so the instrumented
# binary differs from the sibling cell only by the diagnostic patch:
#   -DCMAKE_BUILD_TYPE=Release, TheRock amdclang{,++}, GGML_HIP=OFF, GGML_HRX=ON,
#   -DHRX_SOURCE_DIR=~/wt/pin-fork-hrx/third_party/hrx-system, server on, tests off, -j16.
#
# The diagnostic is inert unless GGML_HRX_WRITEBACK_FINGERPRINT=1, and the build is verified by the
# same style of binary-level assertion the study used: the marker string must be present in
# libggml-hrx.so, otherwise the experiment would silently measure an uninstrumented binary.
#
# Usage: 315-instr-build.sh [patch]
set -euo pipefail
PIN=e44c9d01a4d5110cecaf45472a717f46cd1abbf9
BASE=$HOME/wt/llama-hybrid
SRC=$HOME/wt/llama-315instr
B=${B:-$HOME/wt/315instr-build}
HRX=$HOME/wt/pin-fork-hrx/third_party/hrx-system
PATCH=${1:-$HOME/wt/315-instrumentation.patch}
LOG=$HOME/wt/315instr-build.log

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "preflight"
[ -d "$BASE" ] || { echo "ABORT: fork clone $BASE missing"; exit 1; }
[ -d "$HRX" ]  || { echo "ABORT: hrx-system $HRX missing"; exit 1; }
[ -f "$PATCH" ] || { echo "ABORT: patch $PATCH missing"; exit 1; }
say "patch sha256 $(sha256sum "$PATCH" | cut -d' ' -f1)"

say "fetch + worktree at $PIN"
git -C "$BASE" fetch --quiet origin || true
if git -C "$BASE" cat-file -e "$PIN^{commit}" 2>/dev/null; then
  say "pinned commit present locally"
else
  echo "ABORT: pinned commit $PIN not in $BASE (or not reachable through origin)"; exit 1
fi
if [ ! -d "$SRC/.git" ] && [ ! -f "$SRC/.git" ]; then
  git -C "$BASE" worktree add -f "$SRC" "$PIN"
else
  say "worktree already present at $(git -C "$SRC" rev-parse --short HEAD)"
fi
say "src HEAD $(git -C "$SRC" rev-parse HEAD)"

say "apply the diagnostic patch"
if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  say "patch already applied"
else
  git -C "$SRC" apply --check "$PATCH"
  git -C "$SRC" apply "$PATCH"
  say "patch applied"
fi
git -C "$SRC" diff --stat | tail -3

say "configure"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRX" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "ABORT: cmake failed, see $LOG"; tail -20 "$LOG"; exit 1; }

say "build (-j16)"
cmake --build "$B" -j 16 >> "$LOG" 2>&1 || { echo "ABORT: build failed, see $LOG"; tail -30 "$LOG"; exit 1; }

say "binary identity"
"$B/bin/llama-server" --version

say "binary-level assertion: both diagnostic markers must be present"
LIB=$(ls "$B"/bin/libggml-hrx.so* 2>/dev/null | head -1)
if [ -z "$LIB" ]; then echo "ABORT: libggml-hrx.so not found"; exit 1; fi
M1=$(grep -c "hrx-wb-fp" "$LIB" || true)   # deferred writeback record format
M2=$(grep -c "hrx-dl-fp" "$LIB" || true)   # host download probe record format
say "lib=$LIB  hrx-wb-fp=$M1  hrx-dl-fp=$M2"
[ "$M1" -gt 0 ] || { echo "ABORT: binary lacks the writeback marker"; exit 1; }
[ "$M2" -gt 0 ] || { echo "ABORT: binary lacks the download-probe marker"; exit 1; }

say "INSTRBUILD_DONE bin_dir=$B/bin"
