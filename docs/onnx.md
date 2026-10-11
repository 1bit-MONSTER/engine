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

# ONNX Runtime GenAI

`1bit serve --device onnx` serves ONNX Runtime GenAI models: a directory with `genai_config.json`, `model.onnx` and the tokenizer, the format of Lemonade's ONNX models. The engine runs [ryzenai-server](https://github.com/lemonade-sdk/ryzenai-server) (MIT, pinned in `third_party/ryzenai-server`) as a child process, the way it runs llama-server, and forwards the OpenAI API to it. A directory with a `genai_config.json` goes to `--device onnx` on its own:

```sh
cmake -B build -DONEBIT_ONNX=ON          # builds ryzenai-server into build/onnx/
build/1bit serve -m ~/models/oga/Qwen2.5-0.5B-Instruct-cpu-onnx
```

## Where it runs

The execution mode comes from the model's `genai_config.json`:

| mode | runtime | how to get it |
| --- | --- | --- |
| CPU | Microsoft's ONNX Runtime GenAI 0.11.2 + ONNX Runtime 1.23.2 (MIT) | `scripts/build-onnx.sh`: fetched pinned, sha256-checked |
| NPU (Linux and Windows), hybrid NPU + iGPU (Windows only) | AMD's Ryzen AI Software (the RyzenAI execution provider) | its own licence and download; set `RYZENAI_INSTALL_PATH` to it before building ([Linux NPU](#linux-npu)) |

The engine does not ship Ryzen AI Software, and copies none of it into the repository: the build links against the install the user made, and `1bit serve` runs it. With it installed, the same build serves Lemonade's `-npu` ONNX models, and on Windows its `-hybrid` ones. In Ryzen AI 1.8 the hybrid mode and model generation (quantizing a model for the NPU) are Windows-only; Linux runs the NPU-only models AMD publishes.

## Build

`scripts/build-onnx.sh <prefix>` (run by `-DONEBIT_ONNX=ON`):
- **Libraries.** It fetches the two MIT releases, or uses `$RYZENAI_INSTALL_PATH`.
- **Build.** It builds ryzenai-server against them.
- **Output.** It puts the binary and the libraries it loads in `<prefix>`. The binary carries `DT_RPATH $ORIGIN`, so ONNX Runtime GenAI finds `libonnxruntime.so` beside it when it loads it at run time.

`ONEBIT_ONNX_SERVER` or `--device onnx` with another `ryzenai-server` on `PATH` overrides the build's.

With `RYZENAI_INSTALL_PATH`, ryzenai-server's build copies the whole `deployment/lib` beside the binary, and the script then scans those copies for a `PT_GNU_STACK` that asks for an executable stack and clears it with `patchelf --clear-execstack` (a 7.x kernel refuses to load such a library). It stops with a clear message when a copy needs it and `patchelf` is missing. Only the copies change; nothing in the user's install does.

On Windows the engine builds no ONNX server: `1bit.exe` uses the `ryzenai-server.exe` Lemonade installs for its `ryzenai-server` backend (under `$LEMONADE_CACHE_DIR`, else `%USERPROFILE%\.cache\lemonade`, in `bin\ryzenai-server\npu`). Lemonade publishes that server for Windows only, so Linux keeps building it from `third_party/ryzenai-server`.

## The GPU

The engine no longer builds ONNX Runtime with its WebGPU provider. That provider runs on Dawn over
Vulkan, and the engine builds no Vulkan (RFC #213 stage 3, [HRX](hrx.md)). It ran Qwen3-4B at 52.9 tok/s
against 26.8 on the CPU (2026-09-26). ONNX models run on the CPU, or on the NPU with Ryzen AI
Software; GPU work goes through HRX.

## Linux NPU

Ryzen AI Software 1.8.0 for Linux (`ryzen_ai-1.8.0.tgz`, AMD's EULA download) runs the NPU through the same `--device onnx` route: ryzenai-server built against it, and the model's `genai_config.json` naming the `RyzenAI` provider with `hybrid_opt_token_backend: npu`. Checked on Strix Halo (kernel 7.3, the engine's own amdxdna module, NPU firmware 1.1.2.65) on 2026-10-11.

### Installing it without touching the system

AMD's package wants its XRT debs installed and its installer makes a venv. Both can stay out of the system:

1. **XRT into a prefix, not installed.** AMD's `RAI_1.8_Linux_NPU_XRT` debs (`xrt_*-base.deb`, `xrt_*-npu.deb`, `xrt_plugin.*-amdxdna.deb`) are extracted with `dpkg-deb -x <deb> <prefix>` into one directory, say `~/ryzenai-1.8/sysroot`, so XRT 2.25 lives at `<prefix>/opt/xilinx/xrt`. Installing them instead is invasive: the amdxdna plugin's `postinst` builds AMD's module with dkms, `rmmod`s the running `amdxdna` and `modprobe`s its own, and ships NPU firmware into `/usr/lib/firmware/amdnpu`, replacing the box's. The extracted prefix does none of that.
2. **The venv.** `install_ryzen_ai.sh -a yes -p <venv>` from the tgz, with `XILINX_XRT=<prefix>/opt/xilinx/xrt` and its `lib` first on `LD_LIBRARY_PATH`. The installer wants a `python3.12` on `PATH`; with uv's python, `python3.12 -m venv --copies` fails, so remove `--copies` from that line (a distro python3.12 needs no change). The OGA runtime and the RyzenAI provider land in `<venv>/deployment/lib`: `libonnxruntime-genai.so`, `libonnxruntime.so.1.27.0`, `libonnxruntime_providers_ryzenai.so`.
3. **Build the engine against it.** `RYZENAI_INSTALL_PATH=<venv> cmake -B build -DONEBIT_ONNX=ON && cmake --build build --target onebit` (or `scripts/build-onnx.sh` by hand with the same variable).

Three traps on a current kernel, and their fixes:

- **Executable stacks.** Linux 7.x refuses a library whose `PT_GNU_STACK` asks for RWE. The build clears it on the copies beside ryzenai-server (none of 1.8.0's `deployment/lib` asks for one; the scan is the guard). AMD's python quicktest loads two more that do: `<venv>/lib/python3.12/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.so` and `.../site-packages/lib/lnx64.o/libsystemc.so`, which need `patchelf --clear-execstack` by hand inside the venv before `python quicktest.py` passes ("Setting environment for STX/KRK … Test Finished"). That is a change to the user's venv, so the engine never makes it.
- **A stale XRT in the venv.** AMD's `voe` wheel bundles `libxrt_coreutil.so.2.19`, and the venv's `activate` prepends `voe/lib` to `LD_LIBRARY_PATH`, so it shadows XRT 2.25 and `smi::get_option_options` fails. The XRT `lib` must come first: `1bit serve` puts it first in the child's environment, and a shell that sourced `activate` must re-prepend it.
- **`venv --copies`** with uv's python, above (install time only).

### Serving

One knob names the venv, and `1bit serve` builds the child's environment from it when the model declares the RyzenAI provider:

```sh
export ONEBIT_RYZENAI_ROOT=~/ryzenai-1.8/venv              # the venv (deployment/lib inside)
export ONEBIT_XRT_ROOT=~/ryzenai-1.8/sysroot/opt/xilinx/xrt   # else $XILINX_XRT, else /opt/xilinx/xrt
build/1bit serve -m ~/models/oga/Qwen2.5-0.5B-Instruct_rai_1.8.0_npu_4K    # genai_config.json picks onnx
```

ryzenai-server then starts with `XILINX_XRT=<xrt>`, `LD_LIBRARY_PATH=<xrt>/lib:<dir of ryzenai-server>:<venv>/deployment/lib:<what it held>` and `RYZENAI_EP_PATH` set to the provider library (the copy beside the server when the build made one, else the venv's). A model without the RyzenAI provider (a CPU model) and a run without `ONEBIT_RYZENAI_ROOT` get the environment as it was, so the CPU path is unchanged; a model that declares the provider without the knob set gets a note on stderr naming it. `RYZENAI_VERSION` stays unset: ryzenai-server then reads the model's `rai_config.json` under its 1.7.0 key, where AMD's 1.8.0 models list their prompt length (4096 for the `_4K` models).

**The first dispatch after idle.** The NPU's first command after a cold or idle period can hit a one-off firmware timeout (`aie2_set_cmd_timeout` / TDR in dmesg; the device recovers on its own) and the request fails. For an NPU model, `1bit serve` retries a failed non-streamed request once, one second later, and says so on stderr; a streamed request is not retried, since its bytes have gone out. The retry is per request, not per server: idle periods recur.

**Stopping.** Stopping `1bit serve` sends ryzenai-server SIGTERM and waits up to 30 s before SIGKILL (it exits within a second). The long wait is deliberate: a process that holds an NPU hardware context must close it itself. Killing one outright with a live context, after an earlier firmware timeout, took the kernel down on amdxdna 0.17.0 (2026-10-11: `list_del corruption` in `amdxdna_client_cleanup`). Never `kill -9` a process that holds `/dev/accel/accel0`.

**Checking it.** `tests/onnx_npu_smoke.sh build/1bit <model dir>` (also `ctest -R onnx_npu_smoke` in an `-DONEBIT_ONNX=ON` build; it skips without the knob, the NPU or a model): /health, /v1/models, a chat reply, the child holding `/dev/accel/accel0`, the decode rate, and that stopping serve stops the child.

### Measured on Linux

Strix Halo, 2026-10-11, AMD's NPU-only 1.8.0 models (`*_rai_1.8.0_npu_4K`):

| model | AMD's `model_benchmark` (raw) | `1bit serve --device onnx` |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 2,571 tok/s prompt, **72.0** tok/s decode | **~60** tok/s decode over HTTP (TTFT 0.20 s; 128 tokens in 2.06 s wall) |
| Llama-3.2-1B-Instruct | 2,326 tok/s prompt, **50.2** tok/s decode | (first run hit the first-dispatch timeout above; the retry ran clean) |

The child held `/dev/accel/accel0` open throughout generation. Hybrid (NPU + iGPU) and model generation are Windows-only in Ryzen AI 1.8, so Linux serves the NPU-only models AMD publishes (no Qwen3 among them yet).

## Measured

On Strix Halo, 2026-09-25, with `amd/Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx` (CPU, int4) and `1bit serve -m <dir>` (no `--device`):
- **Chat.** "The capital of France is Paris." and a list of three colors, at 125–129 tok/s decode.
- **Streaming.** A reply streamed as SSE.
- **Model list.** `/v1/models` names the model with device `onnx`.
- **Shutdown.** Stopping `1bit serve` stopped the backend.

llama-server's extra routes (`/tokenize`, `/slots`, …) answer 501 on `--device onnx`.

## Next

- **Windows.** `--device onnx` runs Lemonade's `ryzenai-server.exe` ([Windows](windows.md)); the engine no longer builds one. Not yet run against a Lemonade install on Windows.
- **The Ryzen AI NPU on Windows.** Through Lemonade's `ryzenai-server.exe` with Ryzen AI Software installed; not yet run. Linux: [above](#linux-npu).
- **Routing by provider.** `--device auto` sends every `genai_config.json` directory to onnx; the registry has no notion of an OGA directory's provider (NPU vs CPU) yet.
