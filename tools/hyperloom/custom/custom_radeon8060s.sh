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
# Upstream looks for the entrypoint as custom_<runner-type>.sh inside --benchmark-scripts-dir.
# `--gpu-type radeon8060s` (Strix Halo, gfx1151) resolves to runner_type `radeon8060s`, so this
# name is the one it picks first; the workload itself lives in custom_strixhalo.sh.
set -uo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/custom_strixhalo.sh" "$@"
