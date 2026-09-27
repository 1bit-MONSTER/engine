#!/usr/bin/env bash
# Print quiet-box evidence: any HRX0/GPU workload holders.
echo "--- quiet check $(date -Is) ---"
echo "KFD holders:      $(fuser /dev/kfd 2>/dev/null || echo none)"
echo "renderD128:       $(fuser /dev/dri/renderD128 2>/dev/null || echo none)"
ps -eo pid,etimes,args | grep -iE "llama-bench|llama-server" | grep -v grep || echo "no llama processes"
/opt/rocm-therock/bin/amd-smi monitor -q 2>/dev/null | sed -n '1,2p'
