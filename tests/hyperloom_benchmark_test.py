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
"""The long-context leg of the Hyperloom custom workload (engine#679).

#679 is a decode rate at 18.8k context, so the workload needs a way to score *that* number
instead of the short-prompt geometric mean, without changing what the short legs report and
without letting the long leg's text leak into the correctness gate. This checks the three
pieces that can be checked without a GPU: the filler prompt is long enough and deterministic,
the score switches to the long-context rate only when the leg runs, and an unmeasured long
leg scores zero rather than silently falling back to the geomean.

Run directly (`python3 tests/hyperloom_benchmark_test.py`) or through ctest.
"""
import hashlib
import importlib.util
import json
import pathlib
import platform
import sys

BENCH = pathlib.Path(__file__).resolve().parents[1] / "tools/hyperloom/custom/benchmark.py"


def load():
    spec = importlib.util.spec_from_file_location("hl_benchmark", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    bench = load()

    # The filler prompt: measured at ~4.85 characters per token on this box (a 75,200-character prompt
    # came back as 15,492 prompt tokens), so 18,800 tokens needs >= 18,800 * 4.85 characters, and the
    # same request twice has to be byte-identical or candidates are not comparable.
    prompt = bench.long_prompt(18800)
    assert len(prompt) >= 18800 * 4.85, len(prompt)
    assert prompt == bench.long_prompt(18800)
    assert len(bench.long_prompt(512)) < len(prompt)

    # Score: the long leg replaces the geomean when it ran...
    tps = {"code": 28.0, "prose": 16.0, "short": 17.0, "long": 12.67}
    assert bench.score_of(400.0, 12.0, tps, 18800) == 12.67
    # ...and an absent long measurement is a zero, not a quiet fallback to the short legs.
    assert bench.score_of(400.0, 12.0, {"code": 28.0}, 18800) == 0.0
    # Without the leg the metric is unchanged: geomean of pp512, tg128, serve code.
    geomean = bench.score_of(400.0, 12.0, {"code": 28.0}, 0)
    assert abs(geomean - (400.0 * 12.0 * 28.0) ** (1 / 3)) < 1e-9, geomean
    assert bench.geometric_mean(400.0, 12.0, 28.0) == geomean

    # The gate compares the short legs only. A reference that carries a "long" reply — which happens when
    # the reference is established by a long-context run — must not gate anyone on it, not even a short run
    # that never produces that key. (pi-prs review on engine#769, 2026-11-01.)
    assert bench.gated_text_keys({"code": "a", "prose": "b", "short": "c", "long": "d"}) == [
        "code", "prose", "short"]
    assert bench.gated_text_keys({"code": "a"}) == ["code"]
    assert bench.gated_text_keys({}) == []

    # The reviewer's exact scenario, as the gate line itself runs it: a reference carrying "long" and a run
    # that only produced the short legs must compare clean, not fail on a missing key.
    ref_texts = {"code": "a", "prose": "b", "short": "c", "long": "d"}
    texts = {"code": "a", "prose": "b", "short": "c"}
    assert [k for k in bench.gated_text_keys(ref_texts) if texts.get(k) != ref_texts[k]] == []
    # And a short-leg difference is still caught.
    texts["prose"] = "B"
    assert [k for k in bench.gated_text_keys(ref_texts) if texts.get(k) != ref_texts[k]] == ["prose"]

    # Box rule 2026-11-01 12:13: the drafter at long context is the arm that faulted this box, so a
    # candidate that asks for both must be refused (scores zero) and not load a drafter.
    import os

    for longctx, dflash, want in ((18800, "1", True), (18800, "", False), (0, "1", False)):
        os.environ["HL_LONGCTX"], os.environ["HL_DFLASH"] = str(longctx), dflash
        assert bench.longctx_drafter_conflict() is want, (longctx, dflash, want)
    os.environ.pop("HL_LONGCTX"), os.environ.pop("HL_DFLASH")

    # The artifact has to name the binary it measured: this workload is run on more than one build of the
    # same tree, and a two-result report is not decidable by build without this. (Untested, it was absent:
    # the pair-of-record JSONs carried no build field at all.)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        onebit = pathlib.Path(tmp) / "1bit"
        onebit.write_bytes(b"#!/bin/sh\necho not really a build\n")
        ident = bench.build_identity(onebit, pathlib.Path(tmp) / "hrx/llama/bin")
        assert ident["onebit"] == str(onebit), ident
        assert ident["bin_dir"].endswith("hrx/llama/bin"), ident
        assert ident["sha256"] == hashlib.sha256(onebit.read_bytes()).hexdigest(), ident

    # HL_LONGCTX and HL_CTX are independent variables and the 8192 default is smaller than every
    # long-context value in use, so a run that set only HL_LONGCTX asked for an 18.8k prompt on an
    # 8192-token server: rejected, or answered on a shifted context = a different measurement wearing
    # the same label. One definition of the pinned context, and a refusal before anything loads.
    for longctx, ctx, longtok, want in ((18800, "20480", 128, False), (18800, None, 128, True),
                                        (18800, "18800", 128, True), (8000, None, 128, False),
                                        (0, None, 128, False)):
        os.environ["HL_LONGCTX"], os.environ["HL_LONGTOK"] = str(longctx), str(longtok)
        if ctx is None:
            os.environ.pop("HL_CTX", None)
        else:
            os.environ["HL_CTX"] = ctx
        assert bench.longctx_exceeds_serve_ctx() is want, (longctx, ctx, longtok, want)
        assert bench.serve_ctx() == (8192 if ctx is None else int(ctx))
        assert bench.serve_batch_args()[0] == "-c" and bench.serve_batch_args()[1] == str(bench.serve_ctx())
    for k in ("HL_LONGCTX", "HL_LONGTOK", "HL_CTX"):
        os.environ.pop(k, None)

    # The artifact has to name the device stack too: the box reboots between series and the launcher
    # applies no kernel gate, so before this a report could not say which kernel/driver a number came
    # from (the pair-of-record JSONs carried no kernel field at all).
    ident = bench.device_stack()
    assert ident["kernel"] == platform.release(), ident
    with tempfile.TemporaryDirectory() as tmp:
        stub = pathlib.Path(tmp) / "libhsa-runtime64.so.1"
        stub.write_bytes(b"\x7fELF not really a runtime\n")
        os.environ["IREE_HAL_AMDGPU_LIBHSA_PATH"] = str(stub)
        ident = bench.device_stack()
        assert ident["hsa_runtime"] == str(stub), ident
        assert ident["hsa_runtime_sha256"] == hashlib.sha256(stub.read_bytes()).hexdigest(), ident
        os.environ["IREE_HAL_AMDGPU_LIBHSA_PATH"] = str(pathlib.Path(tmp) / "absent.so.1")
        assert bench.device_stack()["hsa_runtime"] is None, bench.device_stack()
    os.environ.pop("IREE_HAL_AMDGPU_LIBHSA_PATH")

    # And the verdict has to name what it was relative to: `HL_STATE` is per-box and the base logits are
    # regenerated when missing, so a KLD number reads PASS against a reference the JSON never named.
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp) / "kld-base.kld"
        texts = pathlib.Path(tmp) / "texts.json"
        base.write_bytes(b"logits " * 1000)
        texts.write_text(json.dumps({"texts": {"code": "a"}}), encoding="utf-8")
        ident = bench.gate_reference({"model": "deadbeef", "levers": {"HL_DFLASH": "1"}}, base, texts)
        assert ident["model"] == "deadbeef" and ident["levers"] == {"HL_DFLASH": "1"}, ident
        assert ident["base_kld_sha256"] == hashlib.sha256(base.read_bytes()).hexdigest(), ident
        assert ident["texts_sha256"] == hashlib.sha256(texts.read_bytes()).hexdigest(), ident
        # A different base of the same name is a different reference, which is the point.
        base.write_bytes(b"logits " * 999)
        assert bench.gate_reference({}, base, texts)["base_kld_sha256"] != ident["base_kld_sha256"]

    # And it has to name what *drafted*: `--dflash` is handed `HL_DFLASH_GGUF` (relative default
    # `dflash.gguf`, so the run's cwd decided it too) and this model has two drafter quants on the box,
    # so `{"HL_DFLASH": "1"}` -- the arm that produced the 50.87 best of record -- is not a
    # configuration. The tuner drops a drafter-quant variant because overwriting that env "would make
    # its number incomparable to the baseline"; the result JSON never recorded it at all.
    with tempfile.TemporaryDirectory() as tmp:
        drafter = pathlib.Path(tmp) / "q4k.gguf"
        drafter.write_bytes(b"GGUF not really a drafter\n")
        os.environ["HL_DFLASH"], os.environ["HL_DFLASH_GGUF"] = "1", str(drafter)
        ident = bench.drafting_identity()
        assert ident["dflash"]["path"] == str(drafter), ident
        assert ident["dflash"]["sha256"] == hashlib.sha256(drafter.read_bytes()).hexdigest(), ident
        assert ident["dflash"]["env"] == "HL_DFLASH_GGUF", ident
        assert bench.dflash_gguf_arg() == str(drafter)          # the value serve is really handed
        # Same lever, other file: a different configuration, and the report says which one.
        other = pathlib.Path(tmp) / "q8_0.gguf"
        other.write_bytes(b"GGUF another drafter\n")
        os.environ["HL_DFLASH_GGUF"] = str(other)
        assert bench.drafting_identity()["dflash"]["path"] == str(other)
        assert bench.drafting_identity()["dflash"]["sha256"] == \
            hashlib.sha256(other.read_bytes()).hexdigest()
        # Unset: the default is resolved and marked as the default, not left unnamed...
        os.environ.pop("HL_DFLASH_GGUF")
        assert bench.dflash_gguf_arg() == "dflash.gguf"
        ident = bench.drafting_identity()
        assert ident["dflash"]["env"] == "default", ident
        assert pathlib.Path(ident["dflash"]["path"]).is_absolute(), ident
        # ...and an absent file is a null hash, not a guess.
        assert ident["dflash"]["sha256"] is None, ident
        # Drafting off is an empty map: no claim about a drafter that never ran.
        os.environ["HL_DFLASH"] = "0"
        assert bench.drafting_identity() == {}, bench.drafting_identity()
        # An external MTP head is the same class of input, and is reported when it is the one in use.
        os.environ.pop("HL_DFLASH")
        os.environ["HL_MTP_HEAD"], os.environ["HL_MTP_MAX"] = str(drafter), "6"
        ident = bench.drafting_identity()
        assert ident["mtp_head"]["sha256"] == hashlib.sha256(drafter.read_bytes()).hexdigest(), ident
        assert ident["mtp_head"]["env"] == "HL_MTP_HEAD", ident
    for k in ("HL_DFLASH", "HL_DFLASH_GGUF", "HL_MTP_HEAD", "HL_MTP_MAX"):
        os.environ.pop(k, None)

    print("hyperloom_benchmark_test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
