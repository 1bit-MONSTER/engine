#!/usr/bin/env python3
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
"""What the pinned ONNX Runtime does with `ConvInteger`, by type combination (engine#1436).

A very common shape of public "quantized ONNX" (the Optimum / `onnx-community` / `Xenova` exports) is ONNX's *own* integer
ops: `DynamicQuantizeLinear` feeding `ConvInteger` with **uint8 activations and int8 weights**. On the pinned ORT that export
cannot create a session at all — and the interesting part, measured here, is that the pin *does* have a `ConvInteger` kernel:
it is the type combination that is unsupported. `uint8 x uint8` runs; `uint8 x int8` and `int8 x int8` do not.

That matters twice: it is the difference between "our runtime cannot run these models" and "these models were quantised into a
combination this kernel skips", and it gives the workaround (re-quantise to `uint8 x uint8`, or use the repo's fp32 export)
something checked behind it instead of a comment in an issue.

**What is asserted vs what is reported.** The `uint8 x uint8` case is asserted: it is the workaround and we depend on it. The two
unsupported combinations are *reported*: if a future runtime starts supporting them this test says so and passes, because that
would make the workaround unnecessary — a test that goes red when the world improves is a bad test.

usage: tests/onnx_conv_integer_types_test.py <ort_run binary>   (needs python `onnx`, no model, no device)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper

FAIL = []


def check(ok, what, why=""):
    print(("ok   " if ok else "FAIL ") + what + (("  -- " + why) if not ok and why else ""))
    if not ok:
        FAIL.append(what)


def probe(dir_path, name, xt, wt):
    """One single-node ConvInteger graph: [1,1,8,8] x [1,1,3,3] of all-ones weights -> int32 [1,1,6,6]."""
    w = np.ones((1, 1, 3, 3), dtype=np.uint8 if wt == TensorProto.UINT8 else np.int8)
    zp = np.array([0], dtype=np.uint8 if xt == TensorProto.UINT8 else np.int8)
    g = helper.make_graph(
        [helper.make_node("ConvInteger", ["x", "W", "xz", "wz"], ["y"])],
        f"ci_{name}",
        [helper.make_tensor_value_info("x", xt, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [1, 1, 6, 6])],
        [helper.make_tensor("W", wt, [1, 1, 3, 3], w.ravel().tobytes(), raw=True),
         helper.make_tensor("xz", xt, [], zp.tobytes(), raw=True),
         helper.make_tensor("wz", wt, [], zp.tobytes(), raw=True)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 10)])
    # python onnx 1.23 stamps IR 14 and ONNX Runtime 1.23.2 refuses anything above 11.
    m.ir_version = 10
    model = os.path.join(dir_path, f"{name}.onnx")
    onnx.save(m, model)
    # activations of 1 (not 0): with an all-ones kernel and no padding every output is 1*1*9 = 9, so the
    # check is on arithmetic and not on "did it write zeros anywhere".
    data = np.ones((1, 1, 8, 8), dtype=np.uint8 if xt == TensorProto.UINT8 else np.int8)
    tensor = os.path.join(dir_path, f"{name}.bin")
    data.tofile(tensor)
    return model, tensor, ("uint8" if xt == TensorProto.UINT8 else "int8")


def run(runner, model, tensor, dtype, out_dir):
    return subprocess.run([runner, "--model", model, "--input", f"x={tensor}:{dtype}:1x1x8x8",
                           "--dump-all", out_dir], capture_output=True, text=True)


def main(argv):
    runner = argv[1] if len(argv) > 1 else "build/ort_run"
    types = {"uint8": TensorProto.UINT8, "int8": TensorProto.INT8}
    results = {}
    with tempfile.TemporaryDirectory() as td:
        for (xn, wn) in (("uint8", "uint8"), ("uint8", "int8"), ("int8", "int8")):
            name = f"{xn}_{wn}"
            model, tensor, dtype = probe(td, name, types[xn], types[wn])
            out = os.path.join(td, f"out_{name}")
            r = run(runner, model, tensor, dtype, out)
            results[(xn, wn)] = (r.returncode, r.stdout + r.stderr)
            print(f"     x {xn:5s} W {wn:5s} -> rc {r.returncode}"
                  + ("" if r.returncode == 0 else "  " + (r.stdout + r.stderr).strip().splitlines()[-1][:110]))

        # ASSERTED: the combination the workaround uses, and its arithmetic.
        rc, log = results[("uint8", "uint8")]
        check(rc == 0, "uint8 x uint8 ConvInteger runs on the pinned runtime (the workaround)", log[:200])
        y = os.path.join(td, "out_uint8_uint8", "y.bin")
        if rc == 0 and os.path.exists(y):
            got = np.fromfile(y, dtype="<i4").reshape(1, 1, 6, 6)
            # all-ones activations and weights, no padding: every output is 1*1*9 = 9
            check(bool((got == 9).all()), "and its result is the right integer (all-ones kernel -> 9 everywhere)",
                  f"got {got.ravel()[:6].tolist()}")
        else:
            check(False, "and it wrote an output tensor", y)

        # REPORTED, not asserted: if these start working, the workaround is unnecessary and this test should say so.
        for combo in (("uint8", "int8"), ("int8", "int8")):
            rc, log = results[combo]
            if rc == 0:
                print(f"note x {combo[0]} W {combo[1]} NOW RUNS -- the pinned runtime gained this combination, so "
                      f"engine#1436's workaround (re-quantise to uint8 x uint8, or use the fp32 export) is no longer "
                      f"needed for it; update the issue and the ONNX notes")
            else:
                check("ConvInteger" in log,
                      f"x {combo[0]} W {combo[1]} is refused BY NAME as a ConvInteger gap (not a crash or a wrong "
                      f"answer)", log[-200:])

    print(f"onnx_conv_integer_types_test: {len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
