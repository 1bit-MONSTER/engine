#!/usr/bin/env bash
# 2048x3072 family device gate: the 1.7.1 hybrid container on the rebuilt set (sizes match the swept, device-proven original).
export M=$HOME/scratch-packs/llama32-1b-rai171-b8
export K=$HOME/ogaload/oga13/kernels-2048-3072-b8
export RUNS="1 2 3 edge"
exec "$HOME/.local/bin/npu-hold" -t 9 -l "w3b-llama3072" -- "$HOME/ogaload/oga13/w3b-device-inner.sh"
