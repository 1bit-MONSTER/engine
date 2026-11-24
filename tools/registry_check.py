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
"""Run and check models per backend; record the results in registry/checked.json.

usage: tools/registry_check.py path/to/1bit --models DIR [--npu-models DIR] [--only BACKEND]
       tools/registry_check.py --annotate-decode
       tools/registry_check.py --check-decode
       tools/registry_check.py --self-test

For every row of registry/check_models.tsv (backend, model path under DIR or --npu-models),
runs tests/serve_e2e.sh: the model must load, answer "Paris" to a fixed question under
temperature 0, and stream. The architecture recorded is the one the backend loads: the GGUF
`general.architecture` for hrx and cpu, and config.json's model_type for npu.

Each result replaces the previous one for the same backend and model, so re-running a
subset keeps the rest. Failures are recorded too, and only passes count as checked.

A row can also carry a `decode` object: the CPU-arbiter agreement measurement for a model the
one-prompt serve check cannot clear, kept in registry/decode_agreement.json and copied in here so the
row states what was measured rather than only what it said (engine#3340). `--annotate-decode`
applies or refreshes those without running a model, and `--check-decode` is the row that fails when
a checked row loses one or states one nobody recorded (ctest registry_decode). `--self-test` is that
row's own arm: it drives a scratch tree through the stale row, the annotation and every refusal, so the
check's firing behaviour is pinned by something a machine runs (ctest registry_check_self).
"""
import argparse
import datetime
import json
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "registry/checked.json")
LIST = os.path.join(ROOT, "registry/check_models.tsv")
DECODE = os.path.join(ROOT, "registry/decode_agreement.json")

# A decode record is reviewed data, so its shape is checked, not just its JSON: a commit, a date, a
# link a reader can follow. The fields its numbers may use are the ones tools/logprob_agree.py prints.
COMMIT = re.compile(r"[0-9a-f]{7,40}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
SOURCE = "https://github.com/1bit-MONSTER/engine/"
NUMBERS = {"greedy_agreement": (0.0, 1.0), "token_id_match": (0.0, 1.0), "same_top_token": (0.0, 1.0)}


def gguf_arch(path):
    """general.architecture from a GGUF header (v2/v3)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return ""
        version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            return ""

        def s():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        for _ in range(n_kv):
            key = s()
            (t,) = struct.unpack("<I", f.read(4))
            if t == 8:
                val = s()
                if key == "general.architecture":
                    return val
            elif t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if et == 8:
                    for _ in range(n):
                        s()
                else:
                    f.seek(sizes[et] * n, 1)
            else:
                f.seek(sizes[t], 1)
    return ""


def npu_model_type(path):
    try:
        return json.load(open(os.path.join(path, "config.json"))).get("model_type", "")
    except OSError:
        return ""


class DecodeUnreadable(Exception):
    """registry/decode_agreement.json cannot be read as the record of measurements it claims to be."""


def rel(path):
    return os.path.relpath(path, ROOT)


def load_decode():
    """The parsed registry/decode_agreement.json; raises DecodeUnreadable saying why it is not one."""
    try:
        with open(DECODE) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DecodeUnreadable(f"cannot read {rel(DECODE)}: {exc}") from exc
    if not isinstance(doc, dict):
        raise DecodeUnreadable(f"{rel(DECODE)} is not a JSON object")
    return doc


def decode_records():
    """{(backend, model): decode block} from registry/decode_agreement.json.

    The block is the entry without the two keys that place it: a checked row already names its backend
    and model, so copying them into the block would only make two places to disagree. A tree without the
    file records no decode measurements (an empty set); a file that does not have the shape is not empty.
    """
    if not os.path.exists(DECODE):
        return {}
    entries = load_decode().get("results")
    if not isinstance(entries, list):
        raise DecodeUnreadable(f"{rel(DECODE)} has no 'results' list")
    records = {}
    for entry in entries:
        backend = entry.get("backend") if isinstance(entry, dict) else None
        model = entry.get("model") if isinstance(entry, dict) else None
        if not isinstance(backend, str) or not isinstance(model, str):
            raise DecodeUnreadable(f"{rel(DECODE)} has a result with no backend/model: {entry!r}")
        records[(backend, model)] = {k: v for k, v in entry.items() if k not in ("backend", "model")}
    return records


def sync_decode(data, records):
    """Point every checked row's decode object at what decode_agreement.json records; return the changes.

    Called on every write, so a sweep cannot erase a recorded measurement, and the annotate mode calls it
    with no model run so it can be applied from a tree with no device.
    """
    changed = []
    for row in data["results"]:
        key = (row["backend"], row["model"])
        want = records.get(key)
        if want is None:
            if "decode" in row:
                del row["decode"]
                changed.append((key, "removed"))
        elif row.get("decode") != want:
            row["decode"] = want
            changed.append((key, "set"))
    return changed


def numbers_problems(name, where, nums):
    """A decode block's numbers: the fields the tool prints, in the ranges it prints them in."""
    problems = []
    for field, (low, high) in NUMBERS.items():
        if field in nums:
            value = nums[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
                problems.append(f"{name}: {where}.{field} {value!r} is not a number in [{low}, {high}]")
    if "mean_kld_nats" in nums:
        value = nums["mean_kld_nats"]
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
            problems.append(f"{name}: {where}.mean_kld_nats {value!r} is not null or a number >= 0")
    for field in nums:
        if field not in NUMBERS and field != "mean_kld_nats":
            problems.append(f"{name}: {where}.{field} is not a field this record has")
    return problems


def decode_problems():
    """Every inconsistency between decode_agreement.json and the checked rows, named; [] when they agree."""
    if not os.path.exists(DECODE):
        return [f"{rel(DECODE)} is missing: no checked row can state a decode measurement"]
    try:
        doc = load_decode()
    except DecodeUnreadable as exc:
        return [str(exc)]
    problems = []
    about = doc.get("about")
    if not isinstance(about, str) or not about.strip():
        problems.append(f"{rel(DECODE)} has no non-empty 'about'")
    entries = doc.get("results")
    if not isinstance(entries, list) or not entries:
        return problems + [f"{rel(DECODE)} has no non-empty 'results' list"]
    if not os.path.exists(OUT):
        return problems + [f"{rel(OUT)} is missing: there is no row to state them on"]
    try:
        checked = {(r["backend"], r["model"]): r for r in json.load(open(OUT))["results"]}
        tsv = set()
        for line in open(LIST):
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                tsv.add((parts[0], parts[1]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
        return problems + [f"cannot read the registry: {exc}"]

    records = {}
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(f"a result is not an object: {entry!r}")
            continue
        backend, model = entry.get("backend"), entry.get("model")
        if not isinstance(backend, str) or not isinstance(model, str):
            problems.append(f"a result has no backend/model: {entry!r}")
            continue
        name = f"{backend} {model}"
        if (backend, model) in records:
            problems.append(f"{name}: recorded twice")
        records[(backend, model)] = entry
        for field in ("arbiter", "check", "measured", "engine", "llama_cpp", "source", "why"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{name}: no {field}")
        arms = entry.get("arms")
        if not isinstance(arms, dict) or not arms:
            problems.append(f"{name}: 'arms' is not a non-empty object")
        else:
            # the row states THIS backend's agreement with the arbiter, so that arm has to be in the record
            if backend not in arms:
                problems.append(f"{name}: arms has no {backend} arm, so the row does not state its own agreement")
            for arm, nums in arms.items():
                if not isinstance(nums, dict):
                    problems.append(f"{name}: arm {arm} is not an object")
                else:
                    problems += numbers_problems(name, f"arms.{arm}", nums)
        before = entry.get("before_fix")
        if before is not None:
            if not isinstance(before, dict):
                problems.append(f"{name}: 'before_fix' is not an object")
            else:
                problems += numbers_problems(name, "before_fix", before)
        measured = entry.get("measured")
        if not isinstance(measured, str) or not DATE.fullmatch(measured):
            problems.append(f"{name}: measured {measured!r} is not YYYY-MM-DD")
        for field in ("engine", "llama_cpp"):
            value = entry.get(field)
            if not isinstance(value, str) or not COMMIT.fullmatch(value):
                problems.append(f"{name}: {field} {value!r} is not a commit")
        source = entry.get("source")
        if not isinstance(source, str) or not source.startswith(SOURCE):
            problems.append(f"{name}: source {source!r} is not an engine issue or pull link")
        answers = entry.get("known_answers")
        if answers is not None:
            good = (isinstance(answers, dict) and answers
                    and all(isinstance(k, str) and isinstance(v, str) for k, v in answers.items()))
            if not good:
                problems.append(f"{name}: known_answers is not a non-empty prompt -> answer object")
        if (backend, model) not in tsv:
            problems.append(f"{name}: no registry/check_models.tsv row, so nothing checks it")
        row = checked.get((backend, model))
        if row is None:
            problems.append(f"{name}: no registry/checked.json row to state it on")
        else:
            want = {k: v for k, v in entry.items() if k not in ("backend", "model")}
            if row.get("decode") != want:
                problems.append(f"{name}: the checked row does not state this measurement")
    for backend, model in checked:
        if "decode" in checked[(backend, model)] and (backend, model) not in records:
            problems.append(f"{backend} {model}: the checked row states a decode measurement nothing recorded")
    return problems


def self_test():
    """The decode record's arms (engine#3340), on a scratch registry: no model, no device, no build."""
    import contextlib
    import io
    import tempfile

    fail = 0

    def ok(cond, what):
        nonlocal fail
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            fail += 1

    def run(argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(argv)
        return rc, out.getvalue(), err.getvalue()

    def put(path, value):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f, indent=1)
            f.write("\n")

    global ROOT, OUT, LIST, DECODE
    saved = (ROOT, OUT, LIST, DECODE)
    try:
        with tempfile.TemporaryDirectory(prefix="registry-check-") as tree:
            ROOT = tree
            OUT = os.path.join(tree, "registry", "checked.json")
            LIST = os.path.join(tree, "registry", "check_models.tsv")
            DECODE = os.path.join(tree, "registry", "decode_agreement.json")
            model = "MiniCPM4-8B-Q4_K_M.gguf"
            record = {
                "backend": "hrx", "model": model, "arbiter": "cpu",
                "check": "tools/logprob_agree.py --thinking off --text-only --match-by text",
                "arms": {
                    "hrx": {"greedy_agreement": 0.75, "token_id_match": 0.649,
                            "mean_kld_nats": 0.0266, "same_top_token": 1.0},
                    "loom": {"greedy_agreement": 0.75, "mean_kld_nats": None},
                },
                "before_fix": {"greedy_agreement": 0.25, "token_id_match": 0.252, "mean_kld_nats": 0.305},
                "measured": "2026-11-23", "engine": "1c195c45", "llama_cpp": "edf7f11bc",
                "source": "https://github.com/1bit-MONSTER/engine/issues/3340#issuecomment-7093204931",
                "why": "a one-prompt serve check said Paris on the engine that dropped the last layer's attention term",
            }
            stale = {"backend": "hrx", "model": model, "arch": "minicpm", "passed": True,
                     "said": "Paris", "failed": [], "date": "2026-09-25", "engine": "b9812e0ca35e"}
            os.makedirs(os.path.dirname(LIST), exist_ok=True)
            with open(LIST, "w") as f:
                f.write("# backend\tmodel\nhrx\t%s\n" % model)
            put(DECODE, {"about": "measured with tools/logprob_agree.py", "results": [record]})
            put(OUT, {"results": [stale]})

            # (1) THE STALE ROW, origin/main's shape: a one-prompt serve pass and no decode block. This is the arm
            # that would have caught engine#3340 -- the check must name the row and answer 1, not 0.
            rc, out, err = run(["--check-decode"])
            ok(rc == 1 and "does not state this measurement" in err,
               "--check-decode fails a checked row that carries no decode measurement (rc 1, row named)")

            # (2) --annotate-decode applies the record with no model run.
            rc, out, err = run(["--annotate-decode"])
            row = json.load(open(OUT))["results"][0]
            ok(rc == 0 and row.get("decode", {}).get("before_fix", {}).get("mean_kld_nats") == 0.305,
               "--annotate-decode writes the recorded measurement onto the row")
            # (3) ... and then the check passes; a second annotate is a no-op.
            rc_check, _, err = run(["--check-decode"])
            rc_again, out_again, _ = run(["--annotate-decode"])
            ok(rc_check == 0 and rc_again == 0 and "0 row(s) changed" in out_again,
               "--check-decode passes once the row states it, and a second annotate changes nothing")

            # (4) the record is the only place a measurement may come from: a row stating one nothing recorded fails.
            put(DECODE, {"about": "measured with tools/logprob_agree.py",
                         "results": [dict(record, model="LFM3-1.7B-Q4_K_M.gguf")]})
            rc, _, err = run(["--check-decode"])
            ok(rc == 1 and "states a decode measurement nothing recorded" in err,
               "--check-decode refuses a row that states a measurement no record holds")

            # (5) the record's numbers are range-checked: 75 is a percentage, not a fraction.
            put(DECODE, {"about": "measured with tools/logprob_agree.py",
                         "results": [dict(record, arms={"hrx": {"greedy_agreement": 75}})]})
            rc, _, err = run(["--check-decode"])
            ok(rc == 1 and "not a number in [0.0, 1.0]" in err,
               "--check-decode range-checks a fraction recorded as a percentage")

            # (6) a missing record fails the check, and annotate REFUSES rather than reading it as "record nothing"
            # and stripping the decode block off every row.
            put(DECODE, {"about": "measured with tools/logprob_agree.py", "results": [record]})
            run(["--annotate-decode"])
            os.unlink(DECODE)
            rc_check, _, err = run(["--check-decode"])
            rc_annotate, _, _ = run(["--annotate-decode"])
            kept = "decode" in json.load(open(OUT))["results"][0]
            ok(rc_check == 1 and "is missing" in err and rc_annotate == 2 and kept,
               "a missing record fails --check-decode and annotate refuses instead of stripping the rows")

            # (7) a sweep (which calls sync_decode on every write) keeps a recorded measurement on the row it rewrote.
            put(DECODE, {"about": "measured with tools/logprob_agree.py", "results": [record]})
            swept = {"results": [{"backend": "hrx", "model": model, "arch": "minicpm", "passed": False,
                                  "said": "", "failed": ["FAIL load"], "date": "2026-11-24",
                                  "engine": "deadbeefcafe"}]}
            changed = sync_decode(swept, decode_records())
            ok(changed == [(("hrx", model), "set")] and "decode" in swept["results"][0],
               "a sweep's rewritten row keeps the recorded decode block")
    finally:
        ROOT, OUT, LIST, DECODE = saved
    print(f"registry_check --self-test: {'ok' if not fail else f'{fail} failure(s)'}")
    return 0 if not fail else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bin", nargs="?", help="path to 1bit (required unless a --*-decode mode is given)")
    ap.add_argument("--models", help="model directory (required unless a --*-decode mode is given)")
    ap.add_argument("--npu-models", help="NPU model directories (default: DIR/npu)")
    ap.add_argument("--only", help="run only this backend")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--annotate-decode", action="store_true",
                    help="apply registry/decode_agreement.json to registry/checked.json and exit; runs no model")
    ap.add_argument("--check-decode", action="store_true",
                    help="exit 1 unless every recorded decode measurement is stated on its checked row, "
                         "and every stated one is recorded")
    ap.add_argument("--self-test", action="store_true",
                    help="run the decode block's own arms on a scratch registry; runs no model")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    if a.check_decode:
        problems = decode_problems()
        for problem in problems:
            print(f"registry_check: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"ok: {len(decode_records())} recorded decode measurement(s), every one stated on its row")
        return 0

    if a.annotate_decode:
        if not os.path.exists(OUT):
            print(f"registry_check: {rel(OUT)} does not exist - nothing to annotate", file=sys.stderr)
            return 2
        if not os.path.exists(DECODE):
            print(f"registry_check: {rel(DECODE)} is missing - refusing to read that as 'record nothing' and strip "
                  f"every row's decode block; record the measurement there first", file=sys.stderr)
            return 2
        try:
            records = decode_records()
            with open(OUT) as f:
                data = json.load(f)
        except (DecodeUnreadable, OSError, json.JSONDecodeError) as exc:
            print(f"registry_check: {exc}", file=sys.stderr)
            return 2
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            print(f"registry_check: {rel(OUT)} is not a results object", file=sys.stderr)
            return 2
        changed = sync_decode(data, records)
        with open(OUT, "w") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        for (backend, model), what in changed:
            print(f"{what:8} {backend:6} {model}")
        print(f"{len(changed)} row(s) changed, {len(records)} measurement(s) recorded")
        return 0

    if not a.bin or not a.models:
        ap.error("a 1bit binary and --models are required unless a --annotate-decode/--check-decode/--self-test "
                 "mode is given")

    a.npu_models = a.npu_models or os.path.join(a.models, "npu")
    rows = []
    for line in open(LIST):
        line = line.strip()
        if line and not line.startswith("#"):
            backend, model, *extra = line.split("\t")
            rows.append((backend, model, extra))
    data = json.load(open(OUT)) if os.path.exists(OUT) else {"results": []}
    results = {(r["backend"], r["model"]): r for r in data["results"]}
    try:
        records = decode_records()
    except DecodeUnreadable as exc:
        print(f"registry_check: {exc}; refusing a sweep that would lose the decode blocks", file=sys.stderr)
        return 2
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=12", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    for backend, model, extra in rows:
        if a.only and backend != a.only:
            continue
        # An npu row is either a Q4NX model directory (under --npu-models, the
        # existing fast lane) or a .gguf the GGUF-on-NPU route repacks itself
        # (under --models).  The arch comes from model.json's model_type for the
        # former and the GGUF's general.architecture for the latter.
        npu_gguf = backend == "npu" and model.endswith(".gguf")
        if npu_gguf:
            path = os.path.join(a.models, model)
        else:
            path = os.path.join(a.npu_models if backend == "npu" else a.models, model)
        if not os.path.exists(path):
            print(f"skip {backend} {model}: not found")
            continue
        arch = gguf_arch(path) if (backend != "npu" or npu_gguf) else npu_model_type(path)
        cmd = [os.path.join(ROOT, "tests/serve_e2e.sh"), a.bin, path, backend] + extra
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            out, ok = p.stdout, p.returncode == 0
        except subprocess.TimeoutExpired as e:
            out, ok = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), False
            out += "\nFAIL timeout"
        said = next((l.split("said:", 1)[1].strip() for l in out.splitlines() if "said:" in l), "")
        failed = [l[5:].strip() for l in out.splitlines() if l.startswith("FAIL ")]
        results[(backend, model)] = {
            "backend": backend, "model": model, "arch": arch, "passed": ok,
            "said": said[:120], "failed": failed,
            "date": datetime.date.today().isoformat(), "engine": commit,
        }
        print(f"{'PASS' if ok else 'FAIL'} {backend:6} {arch:10} {model}  said: {said[:60]!r}")
        data["results"] = sorted(results.values(), key=lambda r: (r["arch"], r["backend"], r["model"]))
        # the stored record, not the run: a sweep rewrites the row it ran and must not erase the
        # decode measurement a row already states, nor keep one nothing records any more
        sync_decode(data, records)
        with open(OUT, "w") as f:  # after every run, so an interrupted sweep keeps what finished
            json.dump(data, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
