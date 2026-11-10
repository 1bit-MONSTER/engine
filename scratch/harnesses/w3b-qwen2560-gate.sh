#!/usr/bin/env bash
# 2048x2560 (Qwen2.5-3B hybrid) device gate: serve the imported container on my kernel set, 3 greedy repeats + edge, tok/s.
export M=$HOME/scratch-packs/qwen25-3b-b8-container
export K=$HOME/ogaload/oga13/kernels-2048-2560-b8
export RUNS="1 2 3 edge"
exec "$HOME/.local/bin/npu-hold" -t 9 -l "w3b-qwen2560" -- "$HOME/ogaload/oga13/w3b-device-inner.sh"
