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
"""mtp_draft_vocab.py - an MTP draft head that drafts from part of the vocabulary.

    tools/mtp_draft_vocab.py --mtp HEAD.gguf --target TARGET.gguf|DIR --ids IDS.txt --out OUT.gguf

An MTP draft head computes logits over the whole vocabulary once per drafted token. For
Qwen3.8-Flash-Next that is the target's 644 MiB Q8_0 output matrix, read three times per decode
step at draft length 3. The draft only proposes tokens; the target checks every one of them, so
the draft can be restricted to the tokens it is likely to propose without changing what the
model outputs. Only the acceptance rate can move.

HEAD.gguf is a "shared" MTP head (Unsloth's mtp-*-shared-*.gguf: no output.weight of its own,
it borrows the target's). The output adds
    blk.<L>.nextn.shared_head_head.weight  the target's output.weight rows for the listed ids
    d2t                                    I64 [K], draft row -> token id
    onebit.draft_vocab = K                 marks the file
and copies everything else unchanged. Our llama.cpp fork's qwen4exp MTP graph runs the K-row
head and puts its logits back into a full-vocabulary row that is -inf elsewhere
(src/models/qwen4exp-draft-vocab.cpp).

IDS.txt has one token id per line; '#' starts a comment. config/draft-vocab/ has the
65,536-id English + code list from dime-online's qwen3.8-Flash-DGX-UltraFast (Apache-2.0, see
NOTICE). Text in other scripts (CJK and so on) uses many ids outside it, so its acceptance drops.
"""
import argparse
import glob
import gzip
import os
import sys

import numpy as np


def load_ids(path, n_vocab):
    opener = gzip.open if path.endswith(".gz") else open
    ids = []
    with opener(path, "rt") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                ids.append(int(line))
            except ValueError:
                sys.exit(f"{path}:{lineno}: not a token id: {line!r}")
    arr = np.unique(np.asarray(ids, dtype=np.int64))
    if arr.size == 0:
        sys.exit(f"{path}: no token ids")
    if arr.size != len(ids):
        print(f"note: {len(ids) - arr.size} duplicate ids dropped", file=sys.stderr)
    if arr[0] < 0 or arr[-1] >= n_vocab:
        sys.exit(f"{path}: ids must be in [0, {n_vocab}), got [{arr[0]}, {arr[-1]}]")
    return arr


def find_output(gguf, target):
    paths = sorted(glob.glob(os.path.join(target, "*.gguf"))) if os.path.isdir(target) else [target]
    for p in paths:
        for t in gguf.GGUFReader(p).tensors:
            if t.name == "output.weight":
                return t
    sys.exit(f"no output.weight in {target}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtp", required=True, help="shared MTP head GGUF (no output.weight of its own)")
    ap.add_argument("--target", required=True, help="target model GGUF, or the directory of its shards")
    ap.add_argument("--ids", required=True, help="token ids, one per line (.gz is fine)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import gguf

    mtp = gguf.GGUFReader(a.mtp)
    arch = mtp.fields["general.architecture"].contents()
    n_nextn = int(mtp.fields[f"{arch}.nextn_predict_layers"].contents()) if f"{arch}.nextn_predict_layers" in mtp.fields else 0
    if n_nextn != 1:
        sys.exit(f"{a.mtp}: expected one MTP layer ({arch}.nextn_predict_layers = 1), got {n_nextn}")
    layer = int(mtp.fields[f"{arch}.block_count"].contents()) - 1
    head_name = f"blk.{layer}.nextn.shared_head_head.weight"
    names = {t.name for t in mtp.tensors}
    if names & {head_name, "d2t", "output.weight"}:
        sys.exit(f"{a.mtp} already has its own output head or d2t; start from the shared head file")

    out_t = find_output(gguf, a.target)
    n_vocab = int(out_t.shape[1])
    if "tokenizer.ggml.tokens" in mtp.fields and len(mtp.fields["tokenizer.ggml.tokens"].contents()) != n_vocab:
        sys.exit("the head's tokenizer and the target's output.weight disagree on the vocabulary size")
    ids = load_ids(a.ids, n_vocab)
    rows = np.asarray(out_t.data).reshape(n_vocab, -1)
    head = np.ascontiguousarray(rows[ids])

    w = gguf.GGUFWriter(a.out, arch=arch)
    for field in mtp.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    w.add_uint32("onebit.draft_vocab", int(ids.size))
    for t in mtp.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.add_tensor_info(head_name, head.shape, head.dtype, head.nbytes, out_t.tensor_type)
    w.add_tensor_info("d2t", ids.shape, ids.dtype, ids.nbytes, gguf.GGMLQuantizationType.I64)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in mtp.tensors:
        w.write_tensor_data(t.data, tensor_endianess=mtp.endianess)
    w.write_tensor_data(head)
    w.write_tensor_data(ids)
    w.close()
    print(f"{a.out}: {out_t.tensor_type.name} head of {ids.size} of {n_vocab} rows "
          f"({head.nbytes / 2**20:.1f} MiB instead of {rows.nbytes / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
