# Porting map

Where each component comes from in [1bit-MONSTER](https://github.com/1bit-MONSTER/1bit-MONSTER),
and what has to be true before it lands here. Refs are branches or commits in that repo.

| Component | Source | State at source | Gate to land here |
|---|---|---|---|
| CPU reference | `src/gguf_reader.cpp`, `src/tokenizer.cpp`, `tools/qwen36_full_ref.py` | **landed** (qwen3; docs/cpu-reference.md). Tokenizer **landed** (docs/tokenizer.md) | matches HF transformers fp32 logits on Qwen3-0.6B |
| Arch registry | `src/model_registry.cpp` + `Testing/census_*.json` | 569 tokens map 2,030 HF arch strings (mapping only) | data file + separate verified list |
| NPU backend | `engine/npu/src/npu_engine_universal.cpp` (`I8Ctx::init_elf`) | ELF-native, matches its own baseline | golden test vs CPU reference |
| NPU ELF dispatch table | branch `backup/iso-build-elf-native-2026-09-22` (`kElfDesigns`) | ELF and xclbin modes match token for token; 0 xclbin opens | same, in this engine |
| NPU 16-tile layer kernel | branch `bench/fastlane-16tile-corrections-2026-09-22` (435bf36e7) | 24/24 tokens, summed KL 0.000466, 0.340 ms/layer | rebuild from source, reproduce |
| GPU backend | AMD-Ecosystem/llama.cpp fork, `GGML_VULKAN` + `GGML_HRX2`; recipe on `fix/zaya-lmhead-evidence` | Vulkan0 74.8 tok/s, HRX20 18.4 on zaya1-8b; Q4NX is HRX20-only | pinned submodule, linked (no dlopen of copied structs) |
| Router | `src/model_router.cpp` (Q4NX rule) + Laya scorer, branch `backup/laya-and-results-2026-09-22` | Laya not yet checked against its Python reference | Laya matches its Python reference within a stated tolerance |
| Server | `src/server/` | works | llama-server flag and endpoint compatibility test |
| Lemonade recipe | lemonade `src/cpp/include/lemon/backends/*` pattern | n/a | loads Qwen3-0.6B through a local Lemonade build |

Known traps carried over:

- NPU concurrency: one device; each engine instance uses 4 hw contexts; throughput
  peaks at about 4 concurrent instances.
- Q4NX containers have separate formats per family (unsigned q4_1 for Qwen3;
  additive `w = q*scale + min` for the 35B MoE experts). Verify against a reference, never by eye.
- The NPU model containers currently live in `~/.config/flm/models/*-NPU2` on the dev box.
