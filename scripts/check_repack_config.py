#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# check_repack_config.py -- oracle-free repack validation.
#
# Why this exists: MiniCPM4-8B decoded garbage for hours because the repack's
# config.json dropped `embedding_scale`, `residual_scale` and `logit_scale`,
# which the GGUF declares (`minicpm.*`) and llama.cpp applies.  Nothing in the
# pipeline noticed, and every argmax-level check that *did* run was blind to it
# (the first two move the answer, the third does not move an argmax at all).
# This checks the class directly: every scalar the GGUF declares under a
# scale-like name must survive into the repacked config.json.
#
# Usage: python3 check_repack_config.py <model.gguf> <repacked_dir>
# Exit:  0 = every declared scale-like key is present and equal
#        1 = at least one is missing or different
import json
import os
import sys

# Keys whose absence from config.json silently changes the forward's numerics.
# Matched as a suffix on the arch-prefixed GGUF field name (minicpm.logit_scale)
# and as a bare key in config.json.
SCALE_LIKE = ("embedding_scale", "residual_scale", "logit_scale", "scale_emb",
              "scale_depth", "dim_model_base", "final_logit_softcapping",
              "attn_logit_softcapping")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    gguf_path, cfg_dir = sys.argv[1], sys.argv[2]
    try:
        from gguf import GGUFReader
    except ImportError:
        print("needs the `gguf` module on PYTHONPATH")
        return 2

    reader = GGUFReader(gguf_path)
    declared = {}
    for f in reader.fields.values():
        tail = f.name.split(".")[-1]
        if tail in SCALE_LIKE:
            v = f.contents()
            if isinstance(v, (list, tuple)):
                v = v[-1]
            try:
                declared[tail] = float(v)
            except (TypeError, ValueError):
                pass

    cfg_path = os.path.join(cfg_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    print(f"{os.path.basename(gguf_path)}: GGUF declares {len(declared)} scale-like key(s)")
    bad = 0
    for k, want in sorted(declared.items()):
        got = cfg.get(k)
        if got is None:
            print(f"  MISSING  {k:24s} GGUF={want!r}   <-- forward would not apply it")
            bad += 1
        elif abs(float(got) - want) > 1e-9:
            print(f"  DIFFER   {k:24s} GGUF={want!r} config={got!r}")
            bad += 1
        else:
            print(f"  ok       {k:24s} {want!r}")
    if not declared:
        print("  (none declared -- nothing to carry; e.g. MiniCPM5-1B / Qwen2.5-7B)")
    print("PASS" if bad == 0 else f"FAIL ({bad} problem(s))")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
