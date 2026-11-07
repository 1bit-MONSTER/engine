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

    # The score's unit has to name BOTH dimensions that set its value: the context, and the number of tokens
    # the decode actually produced (its denominator -- not the number requested, because the model can stop at
    # EOS first). Measured on this box 2026-11-07: 13.3986 tok/s over 128 decoded tokens and 14.1784 over 278,
    # both carrying the same unit string, so the two rows read as the same measurement. (engine#1268)
    assert bench.decode_unit(18762) == "tok/s (decode at 18762-token ctx)", bench.decode_unit(18762)
    assert bench.decode_unit(18762, 128) == "tok/s (decode at 18762-token ctx, 128 decoded)", \
        bench.decode_unit(18762, 128)
    # The regression itself: two rows whose denominators differ must not serialise to the same unit.
    assert bench.decode_unit(18762, 128) != bench.decode_unit(18762, 278)
    # An unknown decode count must not fabricate "0 decoded" or a KeyError.
    assert bench.decode_unit(18762, None) == bench.decode_unit(18762)

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

    # Raw logs are kept per measurement. They used to be fixed names inside RESULT_DIR, so a second
    # measurement in the same RESULT_DIR overwrote the first one's serve.log / llama-bench.log — measured
    # on 2026-11-02, the 19:01 arm destroyed the 18:45 baseline's raw logs, and a pair's A and B halves
    # would have collided the same way. This is that reproducer: two measurement names, one RESULT_DIR.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        for fname, payload in (("arm-A", "FIRST"), ("arm-B", "SECOND")):
            bench.run(["sh", "-c", f"echo {payload}"], "llama-bench", 10, bench.log_dir_for(root, fname))
        assert (root / "logs-arm-A/llama-bench.log").read_text().strip() == "FIRST"
        assert (root / "logs-arm-B/llama-bench.log").read_text().strip() == "SECOND"
        # A repeat of the same measurement overwrites only itself, and the result file is untouched by it.
        assert bench.log_dir_for(root, "arm-A") == root / "logs-arm-A"

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

    # Same class, one level up: nothing named the *workload script* that computed the number, so an arm run
    # from another checkout (or from an unmerged branch) produced a JSON indistinguishable from `main`'s.
    # Measured: 2026-11-05 04:06 ran from the #1034-only `hl-boxcond` tree, where the artifact could not
    # have carried `drafting` at all. The hash is checked against an independently computed one, so a field
    # that reads the wrong file (or a prefix) fails here.
    scripts = bench.scripts_identity()
    assert scripts["path"].endswith("tools/hyperloom/custom/benchmark.py"), scripts
    assert scripts["sha256"] == hashlib.sha256(BENCH.read_bytes()).hexdigest(), scripts

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

    # And the conditions it ran in: a score is measured on a shared box, and on 2026-11-01 the same
    # workload at the same configuration produced 44.69 (quiet die, no swap) and 31.68 (85 GB of 122 GB
    # in use, 37 GB of swap, die climbing) — both gate-passing, and the JSONs could not tell them apart.
    # Read from this box rather than stubbed: these are the files the field reads, and every one of them
    # has to come back a number here. Absent is a null, never a zero or a guess, so the fake root check
    # below stands in for a box that exposes nothing.
    box = bench.box_conditions()
    assert box["tctl_c"] is None or 20.0 < box["tctl_c"] < 120.0, box
    assert box["gtt_used_gib"] is not None and box["gtt_total_gib"] is not None, box
    assert 0.0 <= box["gtt_used_gib"] <= box["gtt_total_gib"], box
    assert box["mem_available_gib"] is not None and box["mem_available_gib"] > 0, box
    # The hwmon channel is found by label, not by a hardcoded index: the k10temp Tctl has come back as
    # hwmon2 and as hwmon4 across six reboots in three days. A root with no Tctl label yields null.
    real_root = bench.HWMON_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        bench.HWMON_ROOT = pathlib.Path(tmp)
        assert bench.die_temp_c() is None, bench.die_temp_c()
        (pathlib.Path(tmp) / "hwmon7").mkdir()
        (pathlib.Path(tmp) / "hwmon7/temp1_label").write_text("edge\n", encoding="ascii")
        (pathlib.Path(tmp) / "hwmon7/temp1_input").write_text("99000\n", encoding="ascii")
        assert bench.die_temp_c() is None  # a different sensor is not the die
        (pathlib.Path(tmp) / "hwmon7/temp1_label").write_text("Tctl\n", encoding="ascii")
        assert bench.die_temp_c() == 99.0, bench.die_temp_c()
        bench.HWMON_ROOT = real_root
    # Same class, other direction: a counter the box does not expose is null, not 0.
    assert bench.meminfo_gib(pathlib.Path("/nonexistent/mem_info_gtt_used"), 2 ** 30) is None
    # The arm's own spread, not just its start: the aborted 2026-11-01 arm was claimed at 54 C and hit
    # 93 C inside the run, so the summary keeps both ends and says how many samples it saw.
    samples = [{"tctl_c": 54.0, "gtt_used_gib": 1.0, "gtt_total_gib": 120.0, "mem_available_gib": 9.0},
               {"tctl_c": 93.0, "gtt_used_gib": 1.5, "gtt_total_gib": 120.0, "mem_available_gib": 8.0},
               {"tctl_c": None, "gtt_used_gib": 2.0, "gtt_total_gib": 120.0, "mem_available_gib": 7.0}]
    got = bench.box_summary(samples)
    assert got["tctl_c"] == 54.0 and got["tctl_c_max"] == 93.0 and got["tctl_c_min"] == 54.0, got
    assert got["tctl_samples"] == 2, got  # the unreadable sample counts as nothing, not as 0 C
    assert got["gtt_used_gib"] == 1.0, got  # start GTT, not the last one
    assert bench.box_summary([]) is None

    # And the NPU is a condition of the score, not just of the die: the 2026-11-05 policy entry for this
    # box says the engines coexist safely but that a GPU arm beside a running NPU *matrix* corrupts both
    # lanes' numbers. This lane's arm was refused 27 times on that gate, and a number measured beside a
    # job has to be able to say so -- the owner's resident `npu-serve-keeper` is baseline, not a job.
    real_home = os.environ.pop("HL_REAL_HOME", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            home = pathlib.Path(tmp)
            os.environ["HL_REAL_HOME"] = tmp
            assert bench.npu_conditions() == {"npu_holder": None, "npu_job": False,
                                              "npu_queued": 0}, bench.npu_conditions()
            (home / "NPU-TOKEN").write_text("npu-serve-keeper pid 4675\n", encoding="utf-8")
            assert bench.npu_conditions()["npu_job"] is False   # the resident serve is baseline
            (home / "NPU-QUEUE").mkdir()
            (home / "NPU-QUEUE/.log").write_text("noise\n", encoding="utf-8")
            (home / "NPU-QUEUE/0179392.mosaic-regate9").write_text("pid=1\n", encoding="utf-8")
            (home / "NPU-TOKEN").write_text("gpu-hold mosaic-regate9 pid 1584738 2026-11-05\n",
                                             encoding="utf-8")
            got = bench.npu_conditions()
            assert got["npu_job"] is True and got["npu_queued"] == 1, got  # dotfiles are not tickets
            assert got["npu_holder"].startswith("gpu-hold mosaic-regate9"), got
            # A half-written or unreadable claim file is not a quiet NPU.
            (home / "NPU-TOKEN").write_text("\n", encoding="utf-8")
            assert bench.npu_conditions()["npu_holder"] is None, bench.npu_conditions()
            # And the arm's own summary counts how many of its samples saw a job.
            got = bench.box_summary([{"tctl_c": 60.0, "npu_job": False},
                                     {"tctl_c": 61.0, "npu_job": True},
                                     {"tctl_c": 62.0, "npu_job": True}])
            assert (got["npu_job_samples"], got["npu_samples"]) == (2, 3), got
            assert got["npu_job"] is False, got  # the start snapshot, same shape as gtt_used_gib
    finally:
        os.environ.pop("HL_REAL_HOME", None)
        if real_home is not None:
            os.environ["HL_REAL_HOME"] = real_home

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

    # The artifact has to say whether the graph stayed on HRX, and for a year of runs it could not: the
    # report the parser reads only reaches serve.log when serve lifts llama-server's verbosity, so
    # `placement` was null in 147 of 147 artifacts on this box while no serve.log carried the line.
    # `--placement-report` is that lift; the parser is checked against the engine's own documented
    # line (docs/serve.md, tests/hrx_placement_test.cpp), and a log without it stays None, not 0.
    assert "--placement-report" in bench.serve_batch_args(), bench.serve_batch_args()
    with tempfile.TemporaryDirectory() as tmp:
        clean = pathlib.Path(tmp) / "clean.log"
        clean.write_text("placement_observer: placement: 512 nodes, 0 off HRX (none), 1 graph splits\n",
                         encoding="utf-8")
        assert bench.placement_summary(clean) == {"graphs_reported": 1, "nodes": 512, "off_hrx": 0,
                                                 "worst_graph_off_hrx": 0}, bench.placement_summary(clean)
        worst = pathlib.Path(tmp) / "worst.log"
        worst.write_text(
            "placement_observer: placement: 512 nodes, 0 off HRX (none), 1 graph splits\n"
            "placement_observer: placement: 354 nodes, 20 off HRX (MUL_MAT x 20), 5 graph splits\n",
            encoding="utf-8")
        got = bench.placement_summary(worst)
        assert got["graphs_reported"] == 2 and got["off_hrx"] == 20, got
        assert got["worst_graph_off_hrx"] == 20, got
        silent = pathlib.Path(tmp) / "silent.log"
        silent.write_text("I slot print_timing: id 0 | task 1 | eval time\n", encoding="utf-8")
        assert bench.placement_summary(silent) is None, bench.placement_summary(silent)
    # The artifact has to name the *model file* too, and by the hash its own gate keys on: the references
    # in HL_STATE are per box and a changed file silently drops them, so before this a report could not say
    # which file its score and its KLD verdict belonged to (all 150 result JSONs on this box name the model
    # only by file name, inside a hardcoded `model_id` string).
    with tempfile.TemporaryDirectory() as tmp:
        model = pathlib.Path(tmp) / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 4096)
        ident = bench.model_identity(model)
        assert ident["path"] == str(model), ident
        assert ident["sha256"] == bench.sha256_head(model), ident
        assert ident["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest(), ident
        # A different file under the same name is a different model, which is the point.
        model.write_bytes(b"GGUF" + b"\0" * 4095)
        assert bench.model_identity(model)["sha256"] != ident["sha256"]

    # The artifact has to say whether the decode it scored *drafted*, because on this pin it drafts with
    # the model's own MTP head unless told not to, and the cap is worth ~18 % of 18.8k decode (~25 % of
    # tg128). The lines below are verbatim from the arm that exposed this (2026-11-04 22:39, 27B
    # UD-Q4_K_XL at 20,480 ctx): `drafting` was `{}` while serve.log reported drafting.
    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "serve.log"
        log.write_text(
            "1bit serve: /m/model.gguf carries its own MTP head: drafting with it (--spec-type draft-mtp;"
            " --no-mtp or ONEBIT_NO_MTP=1 turns it off)\n"
            "1bit serve: recipe qwen35-mtp-max-3: --spec-draft-n-max 3\n"
            "0.40.550.563 I slot print_timing: id  0 | task 0 | draft acceptance = 1.00000 (    5 accepted"
            " /     5 generated), mean len =  3.50\n"
            "4.04.014.149 I slot print_timing: id  0 | task 532 | draft acceptance = 0.67460 (   85 accepted"
            " /   126 generated), mean len =  3.02\n",
            encoding="utf-8")
        got = bench.drafting_observed(log)
        assert got["status"] == "drafted", got
        assert got["spec_type"] == "draft-mtp", got
        assert got["recipe"] == "qwen35-mtp-max-3", got
        assert got["recipe_args"] == "--spec-draft-n-max 3", got
        assert got["tasks_reporting"] == 2, got
        assert got["accept_rate_last"] == 0.67460 and got["mean_acc_len_last"] == 3.02, got
        assert (got["accepted_last"], got["generated_last"]) == (85, 126), got
        assert got["mean_acc_len_max"] == 3.50, got
        # Three states used to serialise as the same `None`: a log that says nothing about drafting, a log
        # whose server says drafting is OFF, and a log that could not be read. The middle one is a measured
        # claim and the last is the absence of one, so reading either off `null` is how a lost serve.log came
        # to look like a non-drafting arm (measured 2026-11-05 22:53: the `HL_MTP_MAX=0` arm at 18.8k -- the
        # box's own non-drafting row -- recorded `serve: null`, exactly what an unreadable log produced).
        silent = pathlib.Path(tmp) / "silent.log"
        silent.write_text("I slot print_timing: id 0 | task 1 | eval time\n", encoding="utf-8")
        assert bench.drafting_observed(silent)["status"] == "no-evidence", bench.drafting_observed(silent)
        # Verbatim from that arm's serve.log: the server stating there is no spec implementation at all.
        off = pathlib.Path(tmp) / "off.log"
        off.write_text("0.48.901.174 I spec common_specu: no implementations specified for speculative"
                       " decoding\n0.48.901.156 I srv    load_model: speculative decoding will use"
                       " checkpoints\n", encoding="utf-8")
        not_drafted = bench.drafting_observed(off)
        assert not_drafted["status"] == "not-drafted", not_drafted
        unreadable = bench.drafting_observed(pathlib.Path(tmp) / "missing.log")
        assert unreadable["status"] == "unreadable", unreadable
        # The whole point, and the assertion that would have failed before the fix.
        assert not_drafted != unreadable, (not_drafted, unreadable)
        assert len({bench.drafting_observed(log)["status"], not_drafted["status"],
                    bench.drafting_observed(silent)["status"], unreadable["status"]}) == 4
    # ...and "no lever was set" must not look like "no drafting": unset keeps serve's own MTP default,
    # while HL_MTP_MAX=0 is an explicitly non-drafting measurement. Both used to serialise as {}.
    for k in ("HL_DFLASH", "HL_DFLASH_GGUF", "HL_MTP_HEAD", "HL_MTP_MAX"):
        os.environ.pop(k, None)
    ident = bench.drafting_summary(pathlib.Path("/nonexistent/serve.log"))
    assert ident["requested"] is None and "requested_note" in ident, ident
    assert ident["serve"] == {"status": "unreadable"}, ident
    os.environ["HL_MTP_MAX"] = "0"
    ident = bench.drafting_summary(pathlib.Path("/nonexistent/serve.log"))
    assert ident["requested"] == {"HL_MTP_MAX": "0"}, ident
    assert "requested_note" not in ident, ident
    os.environ.pop("HL_MTP_MAX", None)

    print("hyperloom_benchmark_test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
