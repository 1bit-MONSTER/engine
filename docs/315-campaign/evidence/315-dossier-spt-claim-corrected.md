<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The dossier's `_spt` claim is WRONG — corrected by the REA decompilation

The peer dossier I preserved at `315-hrx-system-dossier-by-peer-agent.md` contains, in its §4.5 and its
architecture diagram, this claim:

    | 0x31d8a0 | 6 B | JMP qword ptr [0x0032a3e0] | GOT-indirect thunk |
    …
    So the plain symbol name resolves through a GOT slot that is patched to the per-thread-stream
    variant.

**That mechanism is wrong**, and it has been independently corrected by a much larger decompilation effort
now on this box (`~/amd-oss/hrx-re/`, 83 MB of decompiled output plus a 55-document private set). Its
`INDEX.md` records the correction as its own item 7, re-derived from source and re-verified later (item 20):

- `0x31d8a0` is **vaddr `0x21d8a0`, inside `.plt`** — a PLT stub, not an HRX-authored thunk;
- **no `hipMalloc_spt` export exists at all** (39 `_spt` symbols are exported; that one is not);
- the runtime check: with `PER_THREAD_DEFAULT_STREAM`, `hipGetProcAddress("hipMalloc")` returns the
  **same** address — so nothing is patched for it;
- the real mechanism is **symbol-name resolution** — `dlsym(symbol + "_spt")` — confirmed live in that
  direction: `hipMemcpy`/`hipMemset`/`hipStreamSynchronize`/`hipEventRecord`/`hipLaunchKernel` all resolve
  **differently**, while `hipMalloc` does not.

Source-verified at `api.c:28329` in that effort, and re-verified rather than trusted.

## Why this is recorded in this tree

The dossier is preserved here as evidence and cited as a pointer, so leaving its wrong claim unannotated
would propagate a refuted mechanism into this issue's record. Nothing in this goal's own reasoning depends
on it — the #315 work never used the `_spt` mechanism — but the file is a citation, and a citation that is
known-wrong must say so.

## Standing of the two sources

The dossier is a peer agent's summary; the REA effort is a decompilation with a corrections ledger (14
self-corrections recorded, including a retracted public bug report). Where they differ on this point the REA
work should be preferred: it states its method, its source line, and its live verification, and it records
having re-derived the correction a second time rather than trusting its own earlier note.
