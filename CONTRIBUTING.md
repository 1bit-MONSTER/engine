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

## RFCs: design before code

Changes that shape the engine start as an RFC in
[Discussions](https://github.com/1bit-MONSTER/engine/discussions/categories/rfcs), before any pull
request. This holds for everyone: maintainers, agents and outside contributors alike.

**An RFC is needed for:**
- a new backend, device route or serving mode, or removing one;
- a change to `1bit serve`'s command line or HTTP API that existing users would notice;
- a new third-party dependency or pinned upstream, or a change to how pins move;
- anything touching security: authentication, network exposure, secrets, workflows' permissions.

**A pull request is enough for:** bug fixes, documentation, performance work that does not change
behaviour, and new model architectures that come with the check rule 1 asks for.

**How it goes:**
1. Open a discussion in the **RFCs** category. The form asks for the problem, the design, the
   alternatives, how it will be tested and what it adds to NOTICE.
2. Discuss it there. A maintainer marks it accepted (or declined, with the reason) in the thread.
3. The pull request links the accepted RFC. A pull request that needs an RFC and has none is
   closed with a pointer here.

Security problems are the exception: report them privately ([SECURITY.md](SECURITY.md)), never as
an RFC or issue.

## Automation

- **PR-Agent** reviews every pull request with a local coding model on the Strix Halo box
  (`.github/workflows/pr-agent.yml`); comment `/review`, `/describe`, `/improve` or `/ask` for more.
- **Issue agent** (`tools/issue_agent.py`) takes a first pass on issues opened by people outside the
  project: one label, a request for the bug-report fields when a bug skipped the form, a registry
  lookup for linked Hugging Face models, and a pointer to Q&A for questions. The model only picks
  the label; every comment is a template. Maintainers can run it on any issue from the Actions tab.
- **Pins only move forward.** CI fails a pull request whose submodule pin is behind, or has
  diverged from, the pin on `main` (`tools/check_pins.py`), because that drops commits the engine
  already shipped. A rollback on purpose carries the `pin rollback` label.

This repository is a port of the working code in 1bit-MONSTER, our private development
repository, without its history.
[docs/PORTING.md](docs/PORTING.md) lists the source of each component and the
order it lands in.
