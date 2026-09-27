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

# Two tiers, deliberately (@agent-b35e4d):
#   * FAIL on the known-critical list -- keys we have proven the forward must apply.
#   * WARN (never fail) on any other *scale-ish* key the GGUF declares and the
#     config drops.  The next arch's factor will have a name nobody listed, so a
#     silent skip would repeat this bug; but some scale-ish keys are legitimately
#     not carried (GLM's `deepseek2.expert_weights_scale` is a routing scale, not
#     a logit scale), so a blanket rule would false-fail correct repacks.
# Keys whose absence from config.json silently changes the forward's numerics.
# Matched as a suffix on the arch-prefixed GGUF field name (minicpm.logit_scale)
# and as a bare key in config.json.
SCALE_LIKE = ("embedding_scale", "residual_scale", "logit_scale", "scale_emb",
              "scale_depth", "dim_model_base", "final_logit_softcapping",
              "attn_logit_softcapping")

# Substrings that mark a key as scale-ish for the WARN tier.
SCALE_ISH_HINTS = ("scale", "softcap", "mup")


def declared_scales(reader):
    """-> (critical, unknown) dicts of GGUF-declared scale-like scalars."""
    critical, unknown = {}, {}
    for f in reader.fields.values():
        tail = f.name.split(".")[-1]
        low = tail.lower()
        if not (tail in SCALE_LIKE or any(h in low for h in SCALE_ISH_HINTS)):
            continue
        v = f.contents()
        if isinstance(v, (list, tuple)):
            v = v[-1]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        (critical if tail in SCALE_LIKE else unknown)[tail] = v
    return critical, unknown


def check(gguf_path, cfg_dir, out=print):
    """-> list of problems (critical only).  Warnings are printed, not returned."""
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)
    critical, unknown = declared_scales(reader)
    cfg_path = os.path.join(cfg_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    bad = []
    for k, want in sorted(critical.items()):
        got = cfg.get(k)
        if got is None:
            out(f"  MISSING  {k:26s} GGUF={want!r}   <-- forward would not apply it")
            bad.append(k)
        elif abs(float(got) - want) > 1e-9:
            out(f"  DIFFER   {k:26s} GGUF={want!r} config={got!r}")
            bad.append(k)
        else:
            out(f"  ok       {k:26s} {want!r}")
    for k, want in sorted(unknown.items()):
        if cfg.get(k) is None:
            out(f"  warn     {k:26s} GGUF={want!r} not carried "
                f"(scale-ish but not on the known-critical list -- check whether the "
                f"forward needs it)")
    if not critical and not unknown:
        out("  (no scale-like keys declared -- nothing to carry)")
    return bad


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    gguf_path, cfg_dir = sys.argv[1], sys.argv[2]
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("needs the `gguf` module on PYTHONPATH")
        return 2

    print(f"{os.path.basename(gguf_path)}: scan of scale-like declared keys")
    bad = check(gguf_path, cfg_dir)
    print("PASS" if not bad else f"FAIL ({len(bad)} critical missing/differing)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
