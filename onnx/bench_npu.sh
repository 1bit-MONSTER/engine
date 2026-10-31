#!/usr/bin/env bash
# bench_npu.sh — one command that turns a model and a kernel directory into an AMD-protocol row on
# the NPU, using our own engine (never someone else's): stage the model as a private-route dir,
# serve it with `1bit serve --device npu --npu-opt dx=...`, wait for health, run
# bench_npu_serve.py, and tear the server down.
#
# WHY A SCRIPT
#   The row in BENCHMARK-PARITY.md was first produced from a hand-built tree: symlinks, a kernel
#   directory found by hand, a server started by hand. None of that is an argument anyone can check.
#   Everything here is an input you can point at.
#
# THE KERNEL DIRECTORY IS AN INPUT, NOT A BUILD STEP
#   The dx kernels are AIE designs compiled with IRON + Peano by the private add-on, one design per
#   layer, ~6 min per shape:
#     addons/npu-dx/scripts/build-kernels.sh <iron-prefix> <out>/<hidden>x<intermediate>x<heads>x<kv>
#   They are not in this repository (LICENSE-BINARY), so this script takes the directory and refuses
#   to invent it. Qwen3-0.6B is 1024x3072x16x8 — the shape must match the model or the route picks
#   nothing and the serve says so.
#
# USAGE
#   ONEBIT_DX_KERNELS=~/wt/ross-npu-product/kern/1024x3072x16x8 onnx/bench_npu.sh
#   onnx/bench_npu.sh --lens 1024,512,256,128 --gen 64 -w 1 -r 3
#   onnx/bench_npu.sh --keep        # leave the server up (and print its port)
#
# ENV
#   ONEBIT_BIN         the engine binary            (default ~/src/engine-laya-npu/build-npu/1bit)
#   ONEBIT_MODEL       a model dir with model.q4nx, config.json, tokenizer.json
#                                                  (default ~/.config/flm/models/Qwen3-0.6B-NPU2)
#   ONEBIT_DX_KERNELS  the dx kernel dir for the model's shape                  (required)
#   PORT               serve port                                              (default 52625)
set -uo pipefail

BIN="${ONEBIT_BIN:-$HOME/src/engine-laya-npu/build-npu/1bit}"
MODEL="${ONEBIT_MODEL:-$HOME/.config/flm/models/Qwen3-0.6B-NPU2}"
KERNELS="${ONEBIT_DX_KERNELS:-}"
PORT="${PORT:-52625}"
KEEP=0
BENCH_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --keep) KEEP=1; shift ;;
        --bin) BIN="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --kernels) KERNELS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --lens|--gen|-w|--warmup|-r|--repeat|--mode) BENCH_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "bench_npu.sh: unknown option $1" >&2; exit 2 ;;
    esac
done

[ -x "$BIN" ] || { echo "bench_npu.sh: no engine binary at $BIN (set ONEBIT_BIN)" >&2; exit 2; }
[ -d "$KERNELS" ] || {
    echo "bench_npu.sh: no kernel directory (set ONEBIT_DX_KERNELS)" >&2
    echo "  build one with addons/npu-dx/scripts/build-kernels.sh <iron-prefix> <out>/<shape> ..." >&2
    exit 2
}
[ -f "$KERNELS/layout.json" ] || {
    echo "bench_npu.sh: $KERNELS has no layout.json — build-kernels.sh writes it last" >&2
    exit 2
}
for f in model.q4nx config.json tokenizer.json; do
    [ -f "$MODEL/$f" ] || { echo "bench_npu.sh: $MODEL has no $f" >&2; exit 2; }
done

# The route serves a model directory with no npu/: with one it would try the fast lane first and
# fail on kernel files that belong to a different design. Symlinks, so nothing is copied.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/q06-dx.XXXXXX")"
trap 'stty sane 2>/dev/null; :' EXIT
for f in config.json model.q4nx tokenizer.json tokenizer_config.json chat_template.jinja; do
    [ -e "$MODEL/$f" ] && ln -s "$MODEL/$f" "$STAGE/$f"
done

LOG="$(mktemp "${TMPDIR:-/tmp}/bench_npu.XXXXXX.log")"
# A stale server on the same port would answer the health check while our own never binds, and the
# bench would silently time someone else's process -- the same class of mistake that once reported
# parity from a lane whose lmhead ELF was missing.
if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "bench_npu.sh: something already answers on port $PORT; refusing to measure it" >&2
    echo "  kill it or pass --port" >&2
    exit 2
fi
"$BIN" serve -m "$STAGE" --device npu --npu-opt "dx=$KERNELS" -p "$PORT" >"$LOG" 2>&1 &
SERVER=$!
cleanup() {
    kill "$SERVER" 2>/dev/null
    wait "$SERVER" 2>/dev/null
    rm -rf "$STAGE"
}
[ "$KEEP" = 1 ] || trap 'cleanup' EXIT

# Loading the kernels is what takes time (620 ms measured), but a wedged XRT load never returns, so
# the wait is bounded and the log is printed when it expires.
up=0
for _ in $(seq 1 120); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then up=1; break; fi
    kill -0 "$SERVER" 2>/dev/null || break
    sleep 1
done
if [ "$up" != 1 ]; then
    echo "bench_npu.sh: the server never became healthy; log:" >&2
    cat "$LOG" >&2
    exit 1
fi

MODEL_ID="$(curl -s "http://127.0.0.1:$PORT/v1/models" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "bench_npu.sh: $(basename "$STAGE") on the NPU as '$MODEL_ID' ($KERNELS), log $LOG"

python3 -u "$(dirname "$0")/bench_npu_serve.py" --host "127.0.0.1:$PORT" --model "$MODEL_ID" \
    "${BENCH_ARGS[@]}"
rc=$?
[ "$KEEP" = 1 ] && echo "bench_npu.sh: server left running on $PORT (pid $SERVER), log $LOG"
exit "$rc"
