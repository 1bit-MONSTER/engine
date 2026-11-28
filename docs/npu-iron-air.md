<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# AMD mlir-air and IRON on 1bit OS

1bit OS has no XRT. Instead it ships a Python module named `pyxrt`, at
`/opt/1bit-package/1bit/python/pyxrt.cpython-314-x86_64-linux-gnu.so`. AMD's
[mlir-aie / IRON](https://github.com/Xilinx/mlir-aie) and [mlir-air](https://github.com/Xilinx/mlir-air) host
code imports it as XRT's binding, so programs built with those tools run on the appliance's NPU (XDNA2,
Strix / Strix Halo / Krackan) as they are.

It runs **full ELFs only**: one self-contained ELF per design, the form `aiecc --get-full-elf` writes. An xclbin
cannot be loaded. Every xclbin call (`pyxrt.xclbin`, `pyxrt.kernel`, `device.register_xclbin`,
`hw_context(device, uuid)`) raises a `RuntimeError` that says so. `pyxrt.implementation` is `"npu_rt"`, which
tells this module apart from XRT's.

## Set up

AMD's tools come from their wheels, in a virtual environment of the appliance's Python (3.14):

```bash
python3 -m venv ~/iron && . ~/iron/bin/activate
pip install 'mlir_air[aie]' numpy ml_dtypes \
  -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-5 \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH=/opt/1bit-package/1bit/python:$SP/mlir_air/python:$SP/mlir_aie/python
export PEANO_INSTALL_DIR=$SP/llvm-aie PATH=$SP/mlir_air/bin:$SP/mlir_aie/bin:$PATH
```

`/opt/1bit-package/1bit/python` must come first on `PYTHONPATH`, ahead of any other `pyxrt`. Check it with
`python -c 'import pyxrt; print(pyxrt.implementation)'`, which prints `npu_rt`. The NPU device node belongs to
root on the appliance, so run NPU programs with `sudo -E`.

mlir-air pins a Peano (`llvm-aie`) version in `utils/build-mlir-air-using-wheels.sh` of its checkout. Install
that version when you build its LLM examples.

## IRON

Ask the JIT for a full ELF:

```python
import numpy as np, aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform

@iron.jit(full_elf=True)
def add_one(inp: In, out: Out, *, n: CompileTime[int] = 1024):
    return transform(lambda x: x + 1, np.ndarray[(n,), np.dtype[np.int32]], tile_size=32)

iron.ensure_current_device()
x = iron.arange(1, 1025, dtype=np.int32, device="npu"); y = iron.zeros_like(x)
add_one(x, y)
assert (y.numpy() == x.numpy() + 1).all()
```

For a design built outside the JIT, use `aiecc --get-full-elf` (with `--get-scratchpad-parameters` when it takes
runtime parameters). Then load the ELF with `pyxrt.hw_context(device, pyxrt.elf(path))` and
`pyxrt.ext.kernel(context, "device:sequence")`. Runtime parameters are written with mlir-aie's
`ParameterScratchpad` into `run.get_ctrl_scratchpad_bo()`, as under XRT.

## mlir-air

Compile with `output_format="elf"` (`XRTBackend(output_format="elf", ...)`). The LLM examples' decodes are
already full ELFs (`DECODE_OUTPUT_FORMAT=elf`, `make compile-decode-elf`). Their per-op prefills are full ELFs
too (`make compile-prefill CTX=<n>`). The examples' **fused** prefill builds only as an xclbin today, because
mlir-air's ELF output takes one function per device and the fused prefill has one per operation. Run those
examples with the per-op prefill (`--fused-prefill ''`).

## What is covered

The covered part of XRT's binding is what mlir-aie's `aie.utils.hostruntime.xrtruntime` / IRON and
mlir-air's backend and examples call on the full-ELF path:
- `device` and `get_info` (`name`, `bdf`);
- `bo`, sub-buffers, `ext.bo`, with `map`, `read`, `write`, `sync`, `address` and `size`;
- `elf` and `module`;
- `hw_context(device, elf)`;
- `ext.kernel`;
- `run` (`set_arg`, `start`, `wait`, `wait2`, `state` and `get_ctrl_scratchpad_bo`);
- `ert_cmd_state`.

Not covered:
- the xclbin flow (it raises, as above);
- user-pointer buffers (`ext.bo(device, pointer, size)`);
- `runlist`;
- XRT's system, profiling and AIE debug interfaces.
