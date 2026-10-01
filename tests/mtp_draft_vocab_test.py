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
"""tests/mtp_draft_vocab_test.py - tools/mtp_draft_vocab.py on a small made-up target and MTP head.

The head rows must be the target's Q8_0 output rows for the listed ids, byte for byte, d2t the
sorted ids, every other tensor and key unchanged; bad inputs must be refused.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "mtp_draft_vocab.py")
ARCH = "qwen4exp"
N_EMBD, N_VOCAB = 64, 320
Q8_0_ROW = N_EMBD // 32 * 34  # bytes per output row
rng = np.random.default_rng(7)
tokens = [f"tok{i}" for i in range(N_VOCAB)]


def write(path, tensors, extra_kv=True):
    w = gguf.GGUFWriter(path, arch=ARCH)
    w.add_block_count(3)
    w.add_uint32(f"{ARCH}.nextn_predict_layers", 1)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    if extra_kv:
        w.add_string("general.name", "draft-vocab-test")
    for name, arr, qtype in tensors:
        w.add_tensor(name, arr, raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as d:
        out_rows = rng.integers(0, 256, size=(N_VOCAB, Q8_0_ROW), dtype=np.uint8)
        target = os.path.join(d, "target.gguf")
        write(target, [("token_embd.weight", rng.standard_normal((N_VOCAB, N_EMBD), dtype=np.float32), None),
                       ("output.weight", out_rows, gguf.GGMLQuantizationType.Q8_0)])
        eh = rng.standard_normal((N_EMBD, 2 * N_EMBD), dtype=np.float32)
        norm = rng.standard_normal(N_EMBD, dtype=np.float32)
        head = os.path.join(d, "head.gguf")
        write(head, [("blk.2.nextn.eh_proj.weight", eh, None), ("blk.2.nextn.enorm.weight", norm, None)])

        ids = sorted(rng.choice(N_VOCAB, size=96, replace=False).tolist())
        ids_path = os.path.join(d, "ids.txt")
        with open(ids_path, "w") as f:
            f.write("# test ids\n" + "\n".join(map(str, ids)) + f"\n{ids[0]}  # a duplicate\n")

        out = os.path.join(d, "out.gguf")
        r = run("--mtp", head, "--target", target, "--ids", ids_path, "--out", out)
        assert r.returncode == 0, r.stderr

        src, res = gguf.GGUFReader(head), gguf.GGUFReader(out)
        t = {x.name: x for x in res.tensors}
        sh = t["blk.2.nextn.shared_head_head.weight"]
        assert sh.tensor_type == gguf.GGMLQuantizationType.Q8_0
        assert [int(v) for v in sh.shape] == [N_EMBD, len(ids)], sh.shape
        assert np.array_equal(np.asarray(sh.data).reshape(len(ids), -1), out_rows[ids]), "head rows differ"
        d2t = t["d2t"]
        assert d2t.tensor_type == gguf.GGMLQuantizationType.I64
        assert np.asarray(d2t.data).tolist() == ids, "d2t differs from the id list"
        for x in src.tensors:
            assert np.array_equal(np.asarray(t[x.name].data), np.asarray(x.data)), x.name
        assert len(t) == len(src.tensors) + 2
        for name, field in src.fields.items():
            if name.startswith("GGUF."):
                continue
            assert res.fields[name].contents() == field.contents(), name
        assert res.fields["onebit.draft_vocab"].contents() == len(ids)

        bad = os.path.join(d, "bad.txt")
        open(bad, "w").write(f"0\n{N_VOCAB}\n")
        assert run("--mtp", head, "--target", target, "--ids", bad, "--out", out).returncode != 0, "out-of-range id accepted"
        open(bad, "w").write("5\nfive\n")
        assert run("--mtp", head, "--target", target, "--ids", bad, "--out", out).returncode != 0, "non-integer id accepted"
        assert run("--mtp", out, "--target", target, "--ids", ids_path, "--out", out + "2").returncode != 0, "head with d2t accepted"
        assert run("--mtp", target, "--target", target, "--ids", ids_path, "--out", out + "3").returncode != 0, "head with output.weight accepted"
    print("mtp_draft_vocab_test: OK")


if __name__ == "__main__":
    main()
