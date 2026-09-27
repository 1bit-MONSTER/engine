# HRX #123 repro environment (engine-pin build, rig branch)
export HRX123_BIN=/home/bcloud/1bit-engine/third_party/llama.cpp/build-hrx-engine/bin
export HRX123_MODEL=/home/bcloud/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
export LD_LIBRARY_PATH="$HRX123_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
