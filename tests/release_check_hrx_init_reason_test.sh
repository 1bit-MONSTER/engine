#!/bin/bash
# The hrx-device row repeats the runtime's own init failure (engine#1766).
#
# Three arms on the helper that extracts it, because a wrong extraction is silent in both directions: the row
# would report some *other* HRX failure as the init reason, or invent one where the backend printed none. The
# second is the arm that matters -- without it, "always claims an init failure" would look like an improvement
# over "no HRX0 listed" and would send the reader looking for a symbol that is not missing.
#
# usage: release_check_hrx_init_reason_test.sh <path-to-release-check.sh>
set -u

RC=${1:?usage: release_check_hrx_init_reason_test.sh <path-to-release-check.sh>}
[ -r "$RC" ] || { echo "not readable: $RC"; exit 1; }

FN=$(awk '/^ *hrx_init_reason\(\) \{/{f=1} f{print} f&&/^ *\}$/{exit}' "$RC")
[ -n "$FN" ] || { echo "FAIL no hrx_init_reason() in $RC - the row cannot name an init failure"; exit 1; }
eval "$FN"

fail=0
expect() {  # expect <label> <got> <want>
    if [ "$2" = "$3" ]; then
        echo "ok   $1"
    else
        echo "FAIL $1: got '$2', wanted '$3'"
        fail=1
    fi
}

# The line the backend printed on this box: a ROCr whose libhsa-runtime64 lacks hsa_amd_queue_create.
expect "the real backend line yields its reason, not the file:line prefix" \
    "$(hrx_init_reason 'ggml-hrx.cpp:1587: hrx_gpu_initialize(0) failed: NOT_FOUND: symbol '"'"'hsa_amd_queue_create'"'"' not found in library')" \
    "NOT_FOUND: symbol 'hsa_amd_queue_create' not found in library"

# A healthy runtime that registered no device: no init-failure line, so the row keeps its existing message.
expect "a healthy '(none)' device list yields nothing, so the wrapper message is unchanged" \
    "$(hrx_init_reason 'Available devices: (none)')" \
    ""

# A different HRX failure must not be attributed to init.
expect "a later hrx_gpu_device_count failure is not mistaken for the init reason" \
    "$(hrx_init_reason 'gemm.cpp:912: hrx_gpu_device_count failed: GPU accelerator not initialized')" \
    ""

# Multi-line capture: the reason is taken from the failure line even when it is not the first line.
expect "the reason is found when the failure line is not first" \
    "$(hrx_init_reason 'ggml_hrx: warning
ggml-hrx.cpp:1587: hrx_gpu_initialize(0) failed: UNKNOWN: no device')" \
    "UNKNOWN: no device"

exit "$fail"
