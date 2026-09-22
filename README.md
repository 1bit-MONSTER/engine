# 1bit engine

A model-agnostic LLM inference backend in C++ for AMD Ryzen AI (Strix Halo):
XDNA 2 NPU, Radeon iGPU (Vulkan and HRX), and a CPU reference path. It serves the
llama-server-compatible OpenAI HTTP API, so [Lemonade](https://github.com/lemonade-sdk/lemonade)
and any OpenAI client can drive it as a drop-in backend.

> **Status: scaffold.** Nothing runs yet. Components land one at a time, each with a
> test against the CPU reference (see [CONTRIBUTING.md](CONTRIBUTING.md)). The
> development history lives in [1bit-MONSTER](https://github.com/1bit-MONSTER/1bit-MONSTER);
> this repository contains only code that has been verified.

## Layout

| Path | What |
|---|---|
| `engine/core/` | GGUF / safetensors / Q4NX loading, tokenizer, arch registry, sampler, KV cache |
| `engine/route/` | Backend selection: a hard capability check, then the Laya scorer |
| `engine/backends/npu/` | XDNA 2 via XRT, full-ELF kernels only (no xclbins) |
| `engine/backends/gpu/` | llama.cpp (AMD-Ecosystem fork) with the Vulkan and HRX20 devices |
| `engine/backends/cpu/` | fp32 reference forward pass; the correctness oracle |
| `engine/server/` | llama-server-compatible HTTP: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health` |
| `tests/` | Golden-logit tests per backend against the CPU reference |
| `lemonade/` | Upstream patch: `BackendDescriptor` + `WrappedServer` for this engine |

## Milestone 1

Qwen3-0.6B end to end on NPU, GPU and CPU through the server, loaded by Lemonade.

## Build

```bash
cmake -B build -DENGINE_NPU=OFF -DENGINE_GPU=OFF
cmake --build build
ctest --test-dir build
```

## License

Apache-2.0. See [LICENSE](LICENSE).
