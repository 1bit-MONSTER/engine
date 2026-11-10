#!/usr/bin/env bash
# Build the engine (1bit) at the merged bump pin c85596a / llama.cpp d8bd0ff6052a, -j2 under thermal-run so the box
# cannot be cooked by it, retrying a thermal abort like the fork builds do.
set -u
T=$(grep -l k10temp /sys/class/hwmon/*/name | head -n 1 | xargs dirname)/temp1_input
cool() { echo $(( $(cat "$T") / 1000 )); }
cd ~/wt/i680-engine
export HIP_DEVICE_LIB_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
for attempt in $(seq 6); do
    for _ in $(seq 120); do
        if [ "$(cool)" -le 78 ] && sleep 15 && [ "$(cool)" -le 80 ]; then break; fi
        echo "waiting for cool: $(cool)C $(date +%T)"; sleep 20
    done
    echo "attempt $attempt at $(cool)C $(date +%T)"
    ~/lb/thermal-run.sh ninja -C build-priv -j2 1bit
    rc=$?
    echo "attempt $attempt rc=$rc $(date +%T)"
    [ "$rc" -eq 0 ] && { echo "ENGINE BUILD OK $(date +%T)"; strings build-priv/1bit | grep -c "refusing to run"; exit 0; }
    sleep 30
done
echo "ENGINE BUILD GAVE UP"; exit 1
