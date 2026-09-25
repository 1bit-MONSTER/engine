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
# Contributing

This repository exists to be reviewable. Five rules keep it that way.

1. **Nothing lands without a test that runs it.** Every ported component comes
   with a check that exercises it on real hardware or in CI, and accuracy claims
   state the reference they were measured against.
2. **Numbers come from committed files.** Every benchmark or accuracy claim in a
   doc names the command, the model hash and the binary it came from, and anyone
   can re-run it from a clean checkout.
3. **Recognized is not the same as verified.** The arch registry reports which HF
   architectures it *maps* and, separately, which ones have *run* and been
   checked on each backend. Docs quote the second number.
4. **No binaries without source.** NPU kernels are built from source in this
   repository (or a pinned submodule) into full ELFs. No vendored xclbins.
   A private add-on (docs/npu.md, "Private routes") builds its kernels from
   source in its own repository.
5. **Every file carries the copyright and Apache-2.0 notice.** Run
   `python3 tools/copyright.py --fix` before committing; CI runs `--check`.
   Files that cannot hold a comment (binaries, JSON, test data) are exempt,
   listed in the tool.

This repository is a port of the working code in 1bit-MONSTER, our private development
repository, without its history.
[docs/PORTING.md](docs/PORTING.md) lists the source of each component and the
order it lands in.
