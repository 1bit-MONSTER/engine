---
name: lemonade-rea
description: Reverse-engineer the shipped Lemonade SDK and answer questions about how the shipped product resolves, downloads, launches and serves models. Use for any question about Lemonade's backend pins, its lemond/lemonade split, how a backend process is spawned and driven, its HTTP surfaces, or the asset/ISA naming it downloads from GitHub. Also use as the worked example of the REA-on-a-shipped-artifact method, including its traps.
---
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

# Lemonade SDK reverse engineering

## What exists, and where

A completed, auditor-approved account of the **shipped** artifact (never the source repository):

    deliverable   ~/rea-lemonade-analysis      git HEAD 2a4ed3e, tree clean
      REPORT.md                                696 lines
      tools/verify-lemonade-report.py          317 checks, 0 failed, exits non-zero on failure
      tools/string-refs.py                     string literals a function references
      tools/find-exec-site.py                  which functions call which process primitives
    artifact      ~/Downloads/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
                  sha256 6253558fbdd4b1b6ee058af4c099f2aea5eda9df0f415ceea5e7bfe166d74b12
    extracted     ~/rea-lemonade-11.7.0   (lemond f70a54c4..., lemonade cf8e6727...)
    REA evidence  ~/rea-lemonade-evidence.json (14 MB, 2 retained records)

**Read REPORT.md before re-deriving anything.** It is deliberately self-checking: run the verifier from
`~/rea-lemonade-11.7.0` to confirm the artifact still matches before trusting any of the below.

## Standing facts (verified by content, at the pinned revision)

**The split.** `lemond` is the server; `lemonade` is an HTTP client for it. The serving and backend stack is
*exclusively* the daemon's: `lemon::Server::` (1003 symbol lines), `WebSocketServer` (82), `HttpServer` (6),
`BackendUtils` (88), `LlamaCppServer` (68), `FastFlowLMServer` (51), `make_spec` (39) — all exactly **zero** in
the client. But the registry and utility layers are **shared** (`lemon::utils` 300/194, `LinuxProcessPlatform`
41/41, `ProcessManager` 23/23, `RegistrySearchResult` 28/28). Do not describe the client as lacking them; it has a
process platform because `handle_launch_command` (in `lemonade`) is an interactive **agent launcher** and the
client's only `ProcessManager::start_process` call site.

**Protocol.** Client defaults to `127.0.0.1:13305` (`LEMONADE_HOST`/`LEMONADE_PORT`/`LEMONADE_API_KEY`), plus a
UDP-beacon discovery path. The daemon serves **both** an Ollama-style `/api/*` set (`tags`, `show`, `ps`, `pull`,
`push`, `create`, `copy`, `delete`, `generate`, `chat`, `embed`) and an OpenAI-style `/v1/*` set
(`chat/completions`, `completions`, `embeddings`, `responses`, `images/generations`, `audio/transcriptions`).

**Launch path.** locate (`BackendUtils::get_backend_binary_path`, keyed `stable` / `rocm-`) → `choose_port` →
`LinuxProcessPlatform::spawn_process` (fork + execvp) → `ProcessHandle` → `wait_for_ready` →
`forward_streaming_request` → watchdog reset → reap. Children are named by the client's own `--help-all`:
`llama-server`, `vllm-server`, `flm serve`, `sd-server`, `whisper-server`, `moonshine-server`, each with a
passthrough-args flag. Device selection is a backend-name+ordinal string (e.g. `Vulkan0`).

**Versions and assets.** `resources/backend_versions.json` holds the pins (llamacpp
`{vulkan b10375, rocm-stable b10470, rocm-nightly b1311, cuda b10397, metal b10375, cpu b10375}`,
whispercpp v1.8.4, sd-cpp master-820-de298c2). `lookup_expected_asset_hash` accepts **two** config shapes —
`checksums.github.<owner/repo>.<tag>.<asset> = "sha256:<hex>"`, which is what the shipped config uses, and an
`artifact_hashes` object shape with `algorithm`/`value` that this config does **not** use.
`rocm_asset_families` maps a detected ISA to the published family — **`gfx1201` (this box's RX 9070 XT) →
`gfx120X`**, `gfx1100-1103 → gfx110X`, `gfx1030-1035 → gfx103X`; unlisted ISAs (`gfx908`, `gfx90a`) are verbatim.
That mapping is why `is_concrete_gfx_arch` is only a `gfx` prefix test.

**Relevant to 1bit.cpp:** that repo pins lemonade because **lemonade's `hrx-b*` pin is the actual product HRX
carrier**. Read the pin from lemonade's own configuration, never guessed from a pin table, and note that the
shipped `llama.cpp` numbers above are *release tags* in this config, not commit hashes.

## Traps — these cost real time and produced wrong claims

1. **A filtered view is not a fact.** Reading an absence off `nm -C | grep BackendUtils` produced a false
   headline claim about the client. Check the unfiltered view, and check both binaries.
2. **A checker that validates a chosen subset certifies a subset.** A green 250-check run passed
   `lemond links libm, client does not` while the prose asserted libm was *not* a difference. Make the checker
   parse the report's tables and re-derive them, in both directions.
3. **`strings`/`grep` artifacts that look like source typos.** `resources/backennd_versions.json` looks like a
   typo; it is a string-pool fragment. The disassembly stores 16 bytes at +0 and the next 16 at **+15**, a
   one-byte overlap that rebuilds the correct 31-character path. Check for a correct copy elsewhere first.
4. **`strings -n` is a MINIMUM length.** `-n 8` returns *none* of the six- and seven-character `gfx` names.
5. **Lowest-address symbol wins is wrong.** `.text.cold` clones and generated lambdas sort low, so a naive
   matcher analyses an error path or a lambda destructor. Filter `.cold`/lambda/`clone`, and print every match
   so a wrong pick is visible.
6. **A name-keyed symbol dict collapses weak aliases**, so "the next symbol" skips too far and yields a bogus
   function size. Collect addresses, not name→address.
7. **A decompile that "succeeded" can return the wrong function.** The rea CLI returned exit 0 with 20 KB of
   pseudo-code for `run_with_output`, and it was `Version::parse`. Grep the returned body for a call the real
   function must make (`fork`/`execvp` here); if it is absent, you have someone else's function. Exit status
   proved nothing.
8. **A stale `--snapshot` is worse than none**, because it makes point 7 more likely rather than less. A snapshot
   that is not a real analysis database (here a 22 KB file) yields confident wrong answers instead of errors.
9. **Function "size" has two answers.** Next-symbol-minus-start includes alignment padding; `nm -S` gives the
   symbol's own size. Say which you mean (112 vs 108 for `is_concrete_gfx_arch`).

## REA specifically — the MCP path cancels, the CLI works

`rea_open_binary` works and content-verifies the artifact. But `rea_batch_decompile`,
`rea_procedure_pseudo_code` and `rea_binary_overview` **appear to hang and then time out**, and the retained
evidence shows why:

    code: cancelled   category: cancelled
    message: "Analysis was cancelled. Start it again when ready."

**Your own client timeout cancels the analysis it started.** Retrying through MCP is actively counterproductive.
REA's declared profile is `{cpu_seconds: 30, file_size_bytes: 67108864}` with a ~10 MB MCP response budget, which
is why `inspect_binary_layout` succeeded (as `observed` / `shipped-artifact`, via the target-free `pwntools-elf`
provider — 19,064 symbols, 6,957 relocations, entry `0x1a4850`) but exceeded the stdio budget and had to be read
from an exported bundle.

**The CLI removes those two constraints but NOT the 30-CPU-second one, so it also fails — and one of its failures
is shaped like success.** `rea decompile lemond 0x<addr> --snapshot ~/rea-lemonade-snap --format json` over five
launch-path functions gave four exits 1 (`invalid_request`, `"No procedure exists at the requested address"`) and
**one exit 0 that returned a different function**: the body for `run_with_output` (0x654bf0) contains `Version` x8
and `parse` x4 and `fork`/`execvp`/`waitpid` **zero** times — it is `Version::parse`. The snapshot explains it:
`~/rea-lemonade-snap` is a single **22,467-byte file** against the tens of megabytes a Ghidra database needs for a
binary with 15,286 FUNC symbols. **Check the body, never the exit status**: assert the calls the function must
contain are present. If that check cannot be satisfied, report control flow as undetermined.

    export PATH="$HOME/.nvm/versions/node/v24.21.0/bin:$PATH"
    rea decompile <path> <address> --snapshot <dir> --format json

Two further REA notes: retained evidence is reachable with `rea_get_evidence_bundle` → `export_evidence_bundle`
(a "timed out" call may have left a complete or failed record), and unresolved control flow is a legitimate
result — state it as open rather than guessing.

## Working rules

Content-pin the artifact and never substitute another build. Verify by content, never exit status. Record any
claim that measurement contradicts as a correction in the account, with the command that refuted it. Do not
modify or re-fetch the artifact. No daemon was started and no network request made for this work: the client was
executed only with `--help`/`--help-all` under a scratch `HOME`.

## Known open items (do not present as settled)

Control flow of any function (the decompiler never completed under REA's 30-CPU-second profile); what the
client's agent launcher spawns, with what argv/environment; why `ProcessManager::find_free_port` is needed
client-side; who assembles the child's argv daemon-side; whether the unused `artifact_hashes` shape is forward- or
backward-compatible; and the second string-pool fragment whose reconstruction I could not reproduce.
