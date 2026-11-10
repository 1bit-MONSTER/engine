#!/usr/bin/env bash
# Host reference trace for the 2048x2560 family (device-free): NPU_INFER_HOST_GEMM=1 + per-layer [LB] dump.
export M=$HOME/scratch-packs/qwen25-3b-b8-container
exec "$HOME/ogaload/oga13/w3b-layerdump.sh" host "$HOME/ogaload/oga13/kernels-2048-2560-b8" /home/bcloud/.cache/scratch-pi-rlk/ld-qwen-host
