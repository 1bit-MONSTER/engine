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
#
# host_venv_fixup.sh VENV -- make a Hyperloom 1.2.0 venv on the orchestration host its own benchmark interpreter.
#
# For `--framework custom`, 1.2.0 (_benchmark_interpreter._resolve_probe_python) uses /opt/venv/bin/python whenever that
# file exists; only a `vllm` on PATH takes precedence, and then the `python` beside it is used. On a host where
# /opt/venv belongs to another account (ryzen: a wrapper into another user's venv), the Ray preflight then fails with
# "Ray install failed ... Permission denied". This puts a refusing `vllm` shim in VENV/bin (keep VENV/bin first on
# PATH) and installs ray[default]>=2.44.1 into VENV (Python 3.14 has no 2.44.1 wheel; preflight accepts newer).
# Touches nothing outside VENV. Idempotent.
set -euo pipefail
V=${1:?usage: host_venv_fixup.sh VENV}
[ -x "$V/bin/python" ] || { echo "no python at $V/bin/python" >&2; exit 2; }
cat > "$V/bin/vllm" <<'SHIM'
#!/bin/sh
# Not vLLM: exists only so Hyperloom 1.2.0 picks the python beside it as the benchmark interpreter (host_venv_fixup.sh).
echo "vllm shim: not vLLM; exists only for Hyperloom interpreter resolution" >&2
exit 127
SHIM
chmod 755 "$V/bin/vllm"
"$V/bin/python" -c 'import ray' 2>/dev/null || "$V/bin/python" -m pip install -q --disable-pip-version-check --no-cache-dir "ray[default]>=2.44.1"
PATH=$V/bin:$PATH "$V/bin/python" - <<'PY'
from hyperloom.orchestrator.actions.executors._benchmark_interpreter import _resolve_probe_python
import ray, sys
bp = _resolve_probe_python("custom")
print("benchmark interpreter:", bp, "| ray", ray.__version__)
sys.exit(0 if bp.startswith(sys.prefix) else 1)
PY
