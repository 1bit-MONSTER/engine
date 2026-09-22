# Contributing

This repository exists to be reviewable. Four rules keep it that way.

1. **Nothing lands without a test.** A backend or kernel change comes with a
   golden-logit test against the CPU reference: same model, same prompt, token
   agreement and per-step KL within a stated tolerance.
2. **Numbers come from committed files.** Every benchmark or accuracy claim in a
   doc names the command, the model hash and the binary it came from, and anyone
   can re-run it from a clean checkout.
3. **Recognized is not the same as verified.** The arch registry reports which HF
   architectures it *maps* and, separately, which ones have *passed* the golden
   test on each backend. Docs quote the second number.
4. **No binaries without source.** NPU kernels are built from source in this
   repository (or a pinned submodule) into full ELFs. No vendored xclbins.

Porting from 1bit-MONSTER: port the smallest piece that can be tested, not whole
files. [docs/PORTING.md](docs/PORTING.md) lists the source of each component.
