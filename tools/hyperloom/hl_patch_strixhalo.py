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
# Adds a "strixhalo" (gfx1151, Radeon 8060S, 40 CU RDNA3.5, 256 GB/s LPDDR5X, 128 GB unified) GPU
# type to a local Hyperloom 1.0.0 install, next to each "mi355x" table entry.
import os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/hyperloom-ws/.venv/lib/python3.14/site-packages/hyperloom")
PEAK = '{"bf16": 59.4, "fp16": 59.4, "fp8": 59.4, "int8": 59.4, "fp32": 29.7}'
edits = {
    "common/gpu_identity.py": [('    "mi355x": ("gfx950", 256),\n', '    "mi355x": ("gfx950", 256),\n    "strixhalo": ("gfx1151", 40),\n')],
    "inference_optimizer/gpu_types.py": [('    "gfx950": "mi355x",\n', '    "gfx950": "mi355x",\n    "gfx1151": "strixhalo",\n')],
    "agents/kernel/tools/_bypass_roofline.py": [('    "mi355x": {"hbm_bw_gbps": 8000.0, "peak_tflops": _PEAK_TFLOPS_MI355},\n',
        '    "mi355x": {"hbm_bw_gbps": 8000.0, "peak_tflops": _PEAK_TFLOPS_MI355},\n    "strixhalo": {"hbm_bw_gbps": 256.0, "peak_tflops": {"bf16": 59.4, "bfloat16": 59.4, "f16": 59.4, "fp16": 59.4, "float16": 59.4, "fp8": 59.4, "f8": 59.4, "fp32": 29.7, "f32": 29.7, "float32": 29.7}},\n')],
    "agents/kernel/tools/kernel_optimization.py": [('        "build_flag": "--offload-arch=gfx950",\n    },\n}',
        '        "build_flag": "--offload-arch=gfx950",\n    },\n    "strixhalo": {\n        "name": "Strix Halo (Radeon 8060S)",\n        "arch": "gfx1151",\n        "uarch": "RDNA3.5",\n        "cus": 40,\n        "mem": "LPDDR5X-8000 unified (~256 GB/s peak, ~220 measured), 128 GB",\n        "build_flag": "--offload-arch=gfx1151",\n    },\n}')],
    "agents/robustness/signals/preflight.py": [('    "mi355x": 288.0,\n', '    "mi355x": 288.0,\n    "strixhalo": 128.0,\n')],
    "agents/kernel/tools/diffusion_flops.py": [('    "mi355x": {"bf16": 2516.6, "fp16": 2516.6, "fp8": 5033.2, "mxfp4": 10066.4, "fp32": 157.3},\n',
        '    "mi355x": {"bf16": 2516.6, "fp16": 2516.6, "fp8": 5033.2, "mxfp4": 10066.4, "fp32": 157.3},\n    "strixhalo": ' + PEAK + ',\n')],
    "agents/kernel/tools/tracelens_analysis.py": [('    "mi355x": "gfx950",\n', '    "mi355x": "gfx950",\n    "strixhalo": "gfx1151",\n')],
    "agents/kernel/tools/backends/forge_submit.py": [('    "mi355x": "gfx950",\n', '    "mi355x": "gfx950",\n    "strixhalo": "gfx1151",\n')],
    "orchestrator/phases/framework.py": [('        "mi355x": "gfx950",\n', '        "mi355x": "gfx950",\n        "strixhalo": "gfx1151",\n')],
    "orchestrator/knowledge/recipe_kb_t0.py": [('    "mi355x": "gfx950",\n', '    "mi355x": "gfx950",\n    "strixhalo": "gfx1151",\n')],
    "orchestrator/kernel/roofline_ceiling.py": [
        ('    "mi355x": {\n        "hbm_gb": 288.0,\n        "hbm_bw_gbps": 8000.0,\n        "peak_tflops": _MI355X_PEAK_TFLOPS,\n    },\n',
         '    "mi355x": {\n        "hbm_gb": 288.0,\n        "hbm_bw_gbps": 8000.0,\n        "peak_tflops": _MI355X_PEAK_TFLOPS,\n    },\n    "strixhalo": {\n        "hbm_gb": 128.0,\n        "hbm_bw_gbps": 256.0,\n        "peak_tflops": ' + PEAK + ',\n    },\n'),
        ('    "mi355x": {\n        "hbm_bw_gbps": 8000.0,\n        "hbm_gb": 288.0,\n        "peak_tflops": _MI355X_ACHIEVABLE_TFLOPS,\n    },\n',
         '    "mi355x": {\n        "hbm_bw_gbps": 8000.0,\n        "hbm_gb": 288.0,\n        "peak_tflops": _MI355X_ACHIEVABLE_TFLOPS,\n    },\n    "strixhalo": {\n        "hbm_bw_gbps": 220.0,\n        "hbm_gb": 128.0,\n        "peak_tflops": {"bf16": 55.0, "fp16": 55.0, "fp8": 55.0, "int8": 55.0, "fp32": 25.0},\n    },\n')],
}
for f, reps in edits.items():
    p = os.path.join(root, f)
    s = open(p).read()
    if "strixhalo" in s:
        print("already", f); continue
    for a, b in reps:
        if a not in s:
            print("MISSING in", f, repr(a[:60])); continue
        s = s.replace(a, b, 1)
    open(p, "w").write(s)
    print("patched", f)
kw = os.path.join(root, "agents/framework/keywords.py")
s = open(kw).read()
if '"strixhalo"' not in s:
    s = s.replace('        "mi355x",\n', '        "mi355x",\n        "strixhalo",\n        "gfx1151",\n', 1)
    open(kw, "w").write(s); print("patched keywords")
