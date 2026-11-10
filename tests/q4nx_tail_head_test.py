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
"""`tools/q4nx_tail_head.py`'s geometry and its envelope decoder, on data the tree already ships.

The tool exists so that engine#1693's measurement can be re-run (it was a scratch script). Two halves of it are
checkable without a pack, and both are checked here:

  * the **geometry** — env, tile bytes, tile count and the int4 ratio — against numbers derived by hand from
    `npu/pack_import.cpp`'s constants, so a drift in the tool's copy of them fails instead of silently measuring
    different bytes;
  * the **envelope decoder** — against `tests/golden/oga_attn_tile.{bin,json}`, a vendor-written tile of exactly the
    head's geometry (K 896, rows 64, 34,816 B). Its bias subvolumes decode to the golden's 64 raw bf16 codes, and its
    code address map is a bijection over the 2,048 code bytes — the two properties that can be pinned without a pack.

And three behaviours engine#1740 showed the tool needed, checked on synthetic packs built here (no download):

  * a head declared at block_size 32, even inside an `If` subgraph under a vendor op name (`MatMulNBitsBf`, the
    shape of `Qwen-2.5_1.5B_Instruct_rai_1.9.0_npu`), is **refused by name** (exit 4), not decoded with the block-128
    tile into NaN scales;
  * an **untied** pack without `--ref` decodes but reports no correlation (exit 5), since its embedding is not its
    head's oracle (`Phi-3-mini`: corr ~0 against the embedding, +0.994 against its upstream `lm_head`);
  * `--ref` supplies that oracle, and the trailing-zero count is reported, not assumed away.

What is deliberately NOT asserted: a correlation on a real pack. No pack ships in the tree, and the measured values are
in the tool's docstring with the packs they came from.

Usage: tests/q4nx_tail_head_test.py   (no arguments)
"""
import json
import os
import sys

import numpy as np

try:
    import onnx  # noqa: F401
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import q4nx_tail_head as T  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "oga_attn_tile")

failures = 0


def check(ok, what):
    global failures
    if not ok:
        failures += 1
        print("FAIL %s" % what)
    else:
        print("ok   %s" % what)


def make_pack(d, tied, head_block, zeros=0, declare=True, embed=False):
    """A pack the tool can open: config, a graph whose head sits in an If subgraph, a .fconst of the right tail size."""
    V, H = 2 * T.K_TILE_ROWS, T.K_KBLOCK
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        cfg = {"vocab_size": V, "hidden_size": H}
        if declare:                     # engine#1747 review: "declared untied" and "did not declare" are different states
            cfg["tie_word_embeddings"] = tied
        json.dump(cfg, f)
    if HAVE_ONNX:
        write_graph(d, V, H, head_block, embed=embed)
    tile = T.attn_tile_bytes(H)
    rng = np.random.default_rng(1740)
    blob = bytearray(rng.integers(0, 256, 1000 + 2 * tile, dtype=np.uint8).tobytes())
    sv = T.subv_bytes(T.K_TILE_ROWS)
    for t in range(2):                     # sane bf16 scales (~0.0078) in every code subvolume: a finite decode
        for j in range(2, tile // sv):
            off = 1000 + t * tile + j * sv + 64 * (T.K_TILE_ROWS // 2) + 64
            blob[off:off + 64] = np.full(32, 0x3C00, "<u2").tobytes()
    blob += bytes(zeros)
    with open(os.path.join(d, "dd_metastate_Llm_Token_x.fconst"), "wb") as f:
        f.write(bytes(blob))
    return V, H


def write_graph(d, V, H, head_block, embed=False):
    """The head inside an If subgraph under the vendor op name -- the shape of the real 1.9 graphs.

    `embed=True` also writes `model.embed_tokens.weight` as external data, which is what the tied (and the undeclared)
    oracle paths read: without it the tool correctly says "no embed_tokens initializer in the graph" and there is
    nothing to correlate against.
    """
    import onnx
    from onnx import helper, TensorProto
    head = helper.make_node("MatMulNBitsBf", ["x", "w"], ["y"], name="/lm_head/MatMul_Q4", domain="com.ryzenai",
                            K=H, N=V, block_size=head_block, bits=4)
    layer = helper.make_node("MatMulNBitsBf", ["x", "w"], ["z"], name="/model/layers.0/attn/qkv_proj/MatMulNBits",
                             domain="com.ryzenai", K=H, N=3 * H, block_size=T.K_KBLOCK, bits=4)
    vi = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, None)
    inits = []
    if embed:
        import numpy as np
        from onnx import numpy_helper
        emb = numpy_helper.from_array(np.full((V, H), 0.5, dtype="<f2"), name="model.embed_tokens.weight")
        emb.ClearField("raw_data")                 # the bytes live in model.onnx.data, as they do in a real pack
        emb.data_location = TensorProto.EXTERNAL
        for k, v in (("location", "model.onnx.data"), ("offset", "0"), ("length", str(V * H * 2))):
            e = emb.external_data.add()
            e.key, e.value = k, v
        with open(os.path.join(d, "model.onnx.data"), "wb") as f:
            f.write(np.full((V, H), 0.5, dtype="<f2").tobytes())
        inits.append(emb)
    branch = helper.make_graph([layer, head], "b", [], [vi("y")])
    cond = helper.make_node("If", ["c"], ["y"], then_branch=branch, else_branch=branch)
    g = helper.make_graph([cond], "g", [helper.make_tensor_value_info("c", TensorProto.BOOL, [])], [vi("y")],
                          initializer=inits)   # the embedding is a top-level initializer, as in a real pack
    onnx.save(helper.make_model(g), os.path.join(d, "model.onnx"))


def synthetic():
    import tempfile
    work = tempfile.mkdtemp(prefix="q4nx-tail-head-")
    try:
        if HAVE_ONNX:
            p32 = os.path.join(work, "block32")
            make_pack(p32, tied=True, head_block=32)
            decl = T.head_decl(p32)
            check(decl is not None and decl["block_size"] == 32 and decl["name"] == "/lm_head/MatMul_Q4",
                  "head_decl finds a vendor-named head inside an If subgraph (block %s)" % (decl or {}).get("block_size"))
            out, status, why = T.measure(p32, 2)
            check(out is None and status == 4 and "block_size 32" in (why or ""),
                  "a block-32 head is refused by name, exit 4 (%s)" % status)
        else:
            # CI installs numpy, not onnx: the graph cannot be written, so the block gate is checked where onnx is
            print("skip the block-32 refusal: onnx is not installed (the other synthetic checks still run)")

        # The undeclared arm reads its embedding through onnx (T.embedding loads model.onnx), so like the block-32 arm
        # it runs only where onnx is installed -- in CI it skips (ci.yml installs numpy and gguf; see #1747).
        if HAVE_ONNX:
            # engine#1747 review: a pack that does not declare tie_word_embeddings is not an untied pack. It must still be
            # correlated against its own embedding, labelled provisional, and NOT exit 5 with "untied".
            p_undecl = os.path.join(work, "undeclared")
            make_pack(p_undecl, tied=True, head_block=T.K_KBLOCK, declare=False, embed=True)
            r_undecl = T.measure(p_undecl, 2)
            check(r_undecl[1] == 0, "an undeclared tie_word_embeddings measures (rc %s, not 5): %s" % (r_undecl[1], r_undecl[0] or r_undecl[2]))
            rep = r_undecl[0] or {}
            check("does not declare tie_word_embeddings" in (rep.get("oracle") or ""),
                  "and it says the oracle is provisional rather than calling the pack untied (%s)" % (rep.get("oracle"),))
            check([t["finite"] for t in rep.get("tiles_measured", [])] == [1.0, 1.0],
                  "and the decode is reported as finite on both tiles")
            # The synthetic embedding is a constant, so the correlation is nan by construction: the tool must not invent a
            # number there. That is the point of the label, not a gap in the arm.
            check(all(t["corr"] != t["corr"] for t in rep.get("tiles_measured", [])),
                  "and a constant oracle yields no correlation rather than a plausible one")
        else:
            print("skip the undeclared-tie arm: onnx is not installed (it reads model.onnx's embedding)")
        # ...whereas a pack that DECLARES untied and gets no --ref still has no oracle, and says so.
        p_untied = os.path.join(work, "declared_untied")
        make_pack(p_untied, tied=False, head_block=T.K_KBLOCK, declare=True)
        r_untied = T.measure(p_untied, 2)
        check(r_untied[1] == 5, "a DECLARED untied pack without --ref still exits 5 (rc %s)" % (r_untied[1],))

        pu = os.path.join(work, "untied")
        V, H = make_pack(pu, tied=False, head_block=T.K_KBLOCK)
        out, status, _ = T.measure(pu, 2)
        check(status == 5 and out["oracle"] is None and all("corr" not in r for r in out["tiles_measured"]),
              "an untied pack without --ref decodes but reports no correlation, exit 5 (%s)" % status)
        check(all(r["finite"] == 1.0 for r in out["tiles_measured"]) and len(out["tiles_measured"]) == 2,
              "and its decode is reported as finite on both tiles")

        # a file that ends in padding (the 1.5B packs' .fconst ends in 110 MB of zeros): the count is reported AND used
        # to place the grid. It used to be reported only, so the read landed inside the padding and came out non-finite --
        # the symptom engine#1693 started from, on a pack where nothing is wrong (engine#1740)
        pz = os.path.join(work, "padded")
        make_pack(pz, tied=False, head_block=T.K_KBLOCK, zeros=4096)
        out, status, _ = T.measure(pz, 2)
        check(out["geometry"]["trailing_zero_bytes"] == 4096,
              "trailing zero bytes are counted (%d)" % out["geometry"]["trailing_zero_bytes"])
        check(min(r["finite"] for r in out["tiles_measured"]) == 1.0,
              "and the grid is placed BEFORE them, so a padded pack decodes finitely instead of reading its own padding "
              "(min finite %.3f)" % min(r["finite"] for r in out["tiles_measured"]))

        # --ref as the oracle: the decode itself, written as f32 rows, must correlate 1.0 with itself
        with open(T.fconst(pu), "rb") as f:
            f.seek(T.head_offset(os.path.getsize(T.fconst(pu)), V // T.K_TILE_ROWS, T.attn_tile_bytes(H)))
            w0, _ = T.decode_tile(f.read(T.attn_tile_bytes(H)), H, T.K_TILE_ROWS)
        ref = os.path.join(work, "ref.f32")
        w0.astype("<f4").tofile(ref)
        out, status, _ = T.measure(pu, 4, ref, "f32")
        got = out["tiles_measured"]
        # with onnx the synthetic graph declares a block-128 head, so the run is fully checked (0); without it the
        # block size is unverified and the tool says so (6) instead of passing an unchecked layout off as 0
        want = 0 if HAVE_ONNX else 6
        check(status == want and len(got) == 1 and abs(got[0]["corr"] - 1.0) < 1e-9,
              "--ref is the oracle, and only the tiles it covers are measured (%d, exit %s)" % (len(got), status))
        g = out["geometry"]
        check(g["ref_rows"] == T.K_TILE_ROWS and g["ref_tiles"] == 1 and g["tiles_wanted"] == 4,
              "and the truncation is the reference's, stated: ref_rows %s, ref_tiles %s of %s wanted"
              % (g["ref_rows"], g["ref_tiles"], g["tiles_wanted"]))

        # engine#1747 review: a block size nobody checked is not block 128. Runs in both environments -- without onnx
        # the reason is "onnx is not importable", with it this pack has no model.onnx to read.
        pc = os.path.join(work, "unchecked")
        make_pack(pc, tied=False, head_block=T.K_KBLOCK)
        if os.path.exists(os.path.join(pc, "model.onnx")):
            os.remove(os.path.join(pc, "model.onnx"))
        out, status, _ = T.measure(pc, 1, ref, "f32")
        why = out["block_unchecked_why"] if out else None
        check(status == 6 and out["block_checked"] is False
              and why == ("no model.onnx" if HAVE_ONNX else "onnx is not importable"),
              "a correlation over an UNCHECKED block size exits 6, not 0, and says why (%s: %s)" % (status, why))
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def main():
    # ── geometry, by hand from npu/pack_import.cpp: kNormOff 64, kTileRows 64, kKBlock 128 ──
    check(T.subv_bytes(64) == 2176, "subv_bytes(64) is 2176 B (64*32 + 64 zeros + 64 scales)")
    check(T.attn_tile_bytes(896) == 34816, "attn_tile_bytes(896) is 34816 B = (2*7 + 2) subvolumes")
    check(T.attn_payload_off(896) == 1856, "attn_payload_off(896) is 1856 B = 64 + 2*896")
    check(T.attn_tile_bytes(896) % 128 == 0 and T.attn_tile_bytes(896) % 64 == 0,
          "a tile is a whole number of 128-byte rows and 64-byte halves")
    # an IDENTITY, not evidence (engine#1740): with ntiles = V/64 the whole-tail int4 ratio and the per-tile ratio are
    # the same number for any bytes at all -- it agreed on a block-32 head -- so it is pinned as one, and the tool no
    # longer prints it as a property of the region
    V, H = 151936, 896
    tiles = V // T.K_TILE_ROWS
    whole = (tiles * T.attn_tile_bytes(H)) / (V * H / 2.0)
    per_tile = T.attn_tile_bytes(H) / (T.K_TILE_ROWS * H / 2.0)
    check(abs(whole - per_tile) < 1e-12, "whole-tail ratio %.6f == per-tile ratio %.6f by construction" % (whole, per_tile))

    # ── the head's offset rule: the last grid, not the region after env (engine#1693's 0.885 was an env-based read)
    check(T.head_offset(291655296, 2374, 34816) == 209002112, "head_offset is fconst_bytes - ntiles*tile")
    check(T.head_offset(291655296, 2374, 34816) + 2374 * 34816 == 291655296, "the head grid ends exactly at EOF")
    check(T.head_offset(291655296, 2374, 34816) != T.attn_payload_off(896),
          "attn_payload_off(H) alone is not the head's offset -- the mistake engine#1693 records")
    check(T.head_offset(291655296, 2374, 34816, 0) == T.head_offset(291655296, 2374, 34816),
          "no padding leaves the offset unchanged (the case engine#1693 measured)")
    check(T.head_offset(291655296, 2374, 34816, 110208000) == T.head_offset(291655296, 2374, 34816) - 110208000,
          "a trailing zero run moves the grid back by exactly its length (engine#1740's padded packs)")
    check(T.head_offset(291655296, 2374, 34816) == 209000256 + T.attn_payload_off(896),
          "it equals the importer's layers*stride + env (209000256 is this pack's 24 layer blocks)")

    # ── the envelope decoder, against the vendor tile ──
    with open(GOLDEN + ".json") as f:
        meta = json.load(f)
    blob = open(GOLDEN + ".bin", "rb").read()
    check(len(blob) == T.attn_tile_bytes(meta["k"]) == meta["tile_bytes"],
          "the golden tile is exactly attn_tile_bytes(%d) = %d B" % (meta["k"], T.attn_tile_bytes(meta["k"])))
    _, bias = T.decode_tile(blob, meta["k"], meta["rows"])
    # the golden's bias_bf16 list is the raw 16-bit codes (48245, 15569, ...), not the decoded floats
    want = [int(round(x)) for x in meta["bias_bf16"]]
    check(bias.tolist() == want, "all %d bias codes match the golden's bias_bf16 list" % len(want))

    rows, k, kbs = meta["rows"], meta["k"], meta["k"] // T.K_KBLOCK
    subv = T.subv_bytes(rows)
    seen = {}
    for h in range(2):
        for kb in range(kbs):
            s = (2 + 2 * kb + h) * subv
            for cp in range((rows // 2) // 2):
                base = (cp % 4) + (cp // 4) * 512
                for kk in range(T.K_KBLOCK):
                    seen[s + base + kk * 4] = seen.get(s + base + kk * 4, 0) + 1
    code_bytes = 64 * (rows // 2) * (2 * kbs)
    check(set(seen.values()) == {1}, "every code address is written exactly once (no collisions)")
    check(len(seen) == code_bytes, "the code map covers all %d code bytes (%d)" % (code_bytes, len(seen)))

    synthetic()

    print("all q4nx_tail_head checks passed" if not failures else "%d check(s) failed" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
