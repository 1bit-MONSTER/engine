#!/bin/bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Hyperloom on ZAYA1-8B HRX decode (strixhalo). Detached; log in ~/hyperloom-zaya/hyperloom.log.
cd ~/hyperloom-zaya || exit 1
. ~/hyperloom-ws/.venv/bin/activate
export HYPERLOOM_BENCHMARK_BACKEND=bypass
export HYPERLOOM_RUN_MODE=baremetal
export USER_DATA_PATH=$HOME/hyperloom-zaya/data
# Use the logged-in claude CLI instead of an API key (local patch: hl_cli_login.py).
export ONEBIT_HYPERLOOM_CLI_LOGIN=1
export RAY_VERSION=2.55.1
export PATH="$HOME/.local/bin:$PATH"   # claude CLI
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
exec python -m hyperloom.inference_optimizer.cli -v optimize \
  --framework custom \
  --framework-path "$HOME/hyperloom-zaya/llama.cpp" \
  --benchmark-scripts-dir "$HOME/hyperloom-zaya/scripts" \
  --gpu-type strixhalo --tp 1 \
  --model "$HOME/hyperloom-zaya/model" \
  --max-hours 12
