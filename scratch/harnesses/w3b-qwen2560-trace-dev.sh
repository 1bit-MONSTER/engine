#!/usr/bin/env bash
export M=$HOME/scratch-packs/qwen25-3b-b8-container
exec "$HOME/.local/bin/npu-hold" -t 7 -l "w3b-qwen2560-trace" -- \
  "$HOME/ogaload/oga13/w3b-layerdump.sh" device "$HOME/ogaload/oga13/kernels-2048-2560-b8" /home/bcloud/.cache/scratch-pi-rlk/ld-qwen-dev
