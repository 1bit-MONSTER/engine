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
"""Write golden next-token logits from HF transformers (fp32, eager attention).

usage: make_golden.py --model <hf_dir_or_id> --prompt "..." --n-gen 16 --out <dir>

Output (read by tests/golden_cpu.cpp):
  tokens.txt  prompt ids followed by the fp32 model's greedy continuation
  logits.f32  float32 [n_tokens][vocab], the logits after each token (teacher-forced)
  meta.json   model id, revision, prompt, library versions, sha256 of logits.f32
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids]), max_new_tokens=args.n_gen, do_sample=False,
            eos_token_id=None, pad_token_id=tok.eos_token_id,
        )[0].tolist()
        # Teacher-forced pass over the whole sequence: logits after every token.
        logits = model(torch.tensor([gen])).logits[0].float().numpy()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tokens.txt"), "w") as f:
        f.write(" ".join(map(str, gen)) + "\n")
    blob = np.ascontiguousarray(logits, dtype=np.float32).tobytes()
    with open(os.path.join(args.out, "logits.f32"), "wb") as f:
        f.write(blob)
    meta = {
        "model": args.model,
        "revision": getattr(model.config, "_commit_hash", None),
        "prompt": args.prompt,
        "n_prompt": len(ids),
        "n_tokens": len(gen),
        "vocab": int(logits.shape[1]),
        "dtype": "float32",
        "attn_implementation": "eager",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "logits_sha256": hashlib.sha256(blob).hexdigest(),
        "continuation": tok.decode(gen[len(ids):]),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
