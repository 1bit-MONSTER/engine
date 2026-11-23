#!/usr/bin/env bash
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
#
# Hyperloom 1.2.0 custom-workload entrypoint: Qwen3.8-27B UD-Q4_K_XL on the LOOM route (the shipped route since
# engine#3390: card unsloth-Qwen3.8-27B-UD-Q4_K_XL, preset chat, amdgpu://, --pool_backing=fixed). The measurement,
# the gate and the box ticket are inkling-loom's (box_inkling.sh + gauntlet.py + hl_score.py); only the card and the
# lever grid differ, so this directory pins HL_CARD_NAME and carries its own levers.json for Hyperloom's EXPLORE.
set -uo pipefail
export HL_CARD_NAME=${HL_CARD_NAME:-qwen38-27b-3e43d41d}
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../inkling-loom/custom_radeon8060s.sh" "$@"
