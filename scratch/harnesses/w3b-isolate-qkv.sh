#!/usr/bin/env bash
# Isolate the suspect artifact: llama-b8's device-proven O/GU/D (content-identical to mine, verified by byte-diff)
# plus MY QKV K2048 N6144. If the text is still wrong, the QKV ELF is the fault; if it is right, one of my other
# roles is -- and since those are content-identical to proven files, the answer is then "the container, not the ELFs".
export K=$HOME/ogaload/oga13/kernels-2048-6144-hybrid-qkv
export RUNS=1
exec "$HOME/.local/bin/npu-hold" -t 5 -l "w3b-isolate-qkv" -- "$HOME/ogaload/oga13/w3b-device-inner.sh"
