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
"""One Hyperloom measurement of one lever configuration, on the engine's HRX route.

Hyperloom restarts the entrypoint with a different environment for every candidate and reads
exactly two things back out of ``$RESULT_DIR/$RESULT_FILENAME.json``: ``output_throughput``
(maximised) and ``quality_gate.passed`` (fail-closed correctness). This script produces both.

The measurement, all of it against ONE build of the framework checkout:

* ``llama-bench``: pp512 and tg128, ``-r`` reps, one process (one model load) for both.
* ``1bit serve``: greedy 256-token chats on the three prompts the engine's ``tools/bench.py``
  uses (code, prose, short), median of ``HL_REPS`` requests each.
* The score is the geometric mean of pp512, tg128 and the *code* serve tok/s, so a lever that
  trades one leg for another has to win on balance. Every leg is also reported in
  ``bench_summary`` for the run report.

The gate has two halves, both required (a candidate that fails either is rejected):

* KLD of the prefill path against base logits stored on the first (baseline) measurement,
  mean KLD <= ``HL_GATE_KLD_MAX`` (default 1e-4, the owner's bar). Recorded with
  ``scripts/hw-gate.sh``'s own recipe: ``llama-perplexity``, wiki.test.raw, 512 tokens x 8 chunks.
* The three serve replies must be text-identical to the ones stored at the baseline.

Both references live in ``$HL_STATE`` (default ``~/lb/hl-gate/<model>``), outside the framework
checkout: Hyperloom runs ``git clean -fd`` there when it integrates a patch, which would delete
anything kept inside it.

Nothing here builds anything: the checkout has to be built already (see README.md), because a
build inside the measurement would be charged to whichever candidate happened to run first.
"""
import hashlib
import json
import math
import os
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

#: The three chat prompts, identical to tools/bench.py's, so the numbers line up with the
#: engine's own A/B measurements.
DECODE_PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 timestamp without using datetime, with tests.",
    "prose": "Explain how a B-tree insert works, step by step.",
    "short": "Translate to French: The quick brown fox jumps over the lazy dog, and then it goes home to sleep.",
}

#: The long-context leg (engine#679). When ``HL_LONGCTX`` is set, one more request carries a prompt of
#: about that many tokens and the score becomes its decode rate -- the number #679 is about -- instead
#: of the short-prompt geometric mean. The text is a fixed filler sentence repeated to length, so every
#: candidate sees the same prompt; the real token count comes back from the server and is reported.
LONG_PROMPT_UNIT = ("The archive records a series of routine maintenance events on the northern line. "
                    "Each entry lists a date, a shift, an asset and the action taken. ")
#: Measured on the box: this filler tokenises at ~4.85 characters per token (a 75,200-character prompt
#: came back as 15,492 prompt tokens), so the size below hits the requested count rather than 4 chars/token.
LONG_CHARS_PER_TOKEN = 4.85


def env(name, default=""):
    return str(os.environ.get(name, default) or default).strip()


def env_int(name, default):
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def env_float(name, default):
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default


def log(msg):
    print(f"[1bit-hl] {msg}", flush=True)


def user_home():
    """The box user's home, even when $HOME is a sandbox the caller supplied.

    Hyperloom writes the CLI credential into the ~/.claude of whatever HOME it runs under, so a run
    on a shared box is normally sandboxed; the build, the weights and the reference state still live
    in the real home. $HOME wins when it already holds what we are looking for.
    """
    home = pathlib.Path(os.path.expanduser("~"))
    if os.environ.get("HL_REAL_HOME"):
        return pathlib.Path(env("HL_REAL_HOME")).expanduser()
    try:
        import pwd

        real = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir)
        if (real / "wt").is_dir() or not (home / "wt").is_dir():
            return real
    except (ImportError, KeyError):
        pass
    return home


def model_file():
    """The GGUF this measurement serves: ``$MODEL`` may be the file or the directory holding one."""
    raw = env("HL_MODEL") or env("MODEL")
    if not raw:
        raise SystemExit("neither HL_MODEL nor MODEL names a model")
    path = pathlib.Path(raw).expanduser()
    if path.is_dir():
        cands = sorted(path.glob("*.gguf"), key=lambda p: p.stat().st_size)
        if not cands:
            raise SystemExit(f"no *.gguf under {path}")
        path = cands[-1]
    if not path.is_file():
        raise SystemExit(f"model file not found: {path}")
    return path


def sha256_head(path, limit=64 << 20):
    """Fingerprint the model the way the hardware gate does: header + first tensors, not 17 GB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()


def bench_args():
    """The levers `llama-bench` / `llama-perplexity` take: batch, micro-batch, flash-attn, KV types.

    A quantized KV cache is only usable with flash attention (without `-fa` the bench dies in
    ``ggml_backend_sched_graph_compute_async failed with error -1`` on this fork), so asking for one
    implies `-fa on` unless the caller says otherwise.
    """
    args = []
    for key, flag in (("HL_B", "-b"), ("HL_UB", "-ub"), ("HL_FA", "-fa"),
                      ("HL_CTK", "-ctk"), ("HL_CTV", "-ctv")):
        value = env(key)
        if value:
            args += [flag, value]
    if (env("HL_CTK") or env("HL_CTV")) and not env("HL_FA"):
        args += ["-fa", "on"]
    return args


def serve_batch_args():
    """The batch levers `1bit serve` accepts.

    `1bit serve` is a wrapper with its own argument parser: it takes `-fa` (forwarded to its
    llama-server child) but NOT `-b`/`-ub`/`-ctk`/`-ctv` -- it sets those itself per route, and it
    rejects the flags with `unknown option`. So those levers move the llama-bench legs only; the
    serve leg measures what serve would serve.

    `-c` is not a lever but a measurement condition: without it serve sizes the KV cache for the
    model's OWN context (262144 on Qwen3.8-27B = 16.1 GiB of KV), which only fits when the box
    happens to be idle -- under another lane's GTT the engine's fit check aborts the server and every
    candidate fails for a reason that has nothing to do with the lever under test. `HL_CTX` overrides.
    """
    args = ["-c", str(env_int("HL_CTX", 8192))]
    if env("HL_FA"):
        args += ["-fa", env("HL_FA")]
    return args


def dflash_on():
    """True when the DFlash drafter is requested (`HL_DFLASH`; `0`/`off`/`false` mean off)."""
    value = env("HL_DFLASH")
    return bool(value) and value not in ("0", "off", "false")


def longctx_drafter_conflict():
    """The box rule (2026-11-01 12:13): no long-context measurement may draft.

    The drafter at long context is the arm that faulted the HRX graph on this box (27B at 18.8k, the
    11:13 MODE2 reset), so it is refused until engine#764's admission guardrail is merged. A refused
    candidate scores zero and says why, rather than loading a drafter nobody may measure.
    """
    return env_int("HL_LONGCTX", 0) > 0 and dflash_on()


def serve_lever_args():
    """The levers that only `1bit serve` has: drafting and slots.

    A GGUF that carries its own MTP head drafts with it on HRX **by default** (`1bit serve` says
    so at startup), so `HL_MTP_MAX` is relative to that: unset keeps serve's own default, `0` turns
    drafting off with `--no-mtp`, and N>0 caps the draft length at N. `HL_MTP_HEAD` names an
    external head to draft with instead.
    """
    args = serve_batch_args()
    mtp_max_raw = env("HL_MTP_MAX")
    if dflash_on():
        args += ["--dflash", env("HL_DFLASH_GGUF", "dflash.gguf")]
    elif mtp_max_raw:
        mtp_max = env_int("HL_MTP_MAX", 0)
        if mtp_max <= 0:
            args += ["--no-mtp"]
        else:
            if env("HL_MTP_HEAD"):
                args += ["--mtp", env("HL_MTP_HEAD")]
            args += ["--mtp-max", str(mtp_max)]
    parallel = env_int("HL_PARALLEL", 0)
    if parallel:
        args += ["--parallel", str(parallel)]
    return args


def levers():
    """Echo the configuration into the report, so a KEEP is explicable months later."""
    return {
        k: env(k)
        for k in ("HL_B", "HL_UB", "HL_FA", "HL_CTK", "HL_CTV", "HL_MTP_MAX", "HL_DFLASH",
                  "HL_PARALLEL", "HL_CTX", "HL_LONGCTX", "HL_LONGTOK",
                  "GGML_HRX_Q8_PREFILL_RELAX", "GGML_HRX_GRAPH_PROGRAM_CACHE",
                  "GGML_HRX_TER_Q4_0", "ONEBIT_HRX_DECODE_SPLIT")
        if env(k)
    }


def run(cmd, name, timeout, out_dir):
    """Run one leg, stream it to a log beside the result, return (rc, combined output)."""
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        text = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        text = f"{exc.stdout or ''}{exc.stderr or ''}\ntimed out after {timeout} s"
        rc = 124
    (out_dir / f"{name}.log").write_text(text, encoding="utf-8", errors="replace")
    log(f"{name}: rc={rc} in {time.time() - start:.1f} s -> {out_dir / (name + '.log')}")
    return rc, text


def bench_csv(text):
    """Parse `llama-bench -o csv` into {(n_prompt, n_gen): tok/s}, keyed by header names."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    head = [c.strip() for c in lines[0].split(",")]
    try:
        i_prompt, i_gen, i_ts = head.index("n_prompt"), head.index("n_gen"), head.index("avg_ts")
    except ValueError:
        return {}
    rows = {}
    for line in lines[1:]:
        cols = [c.strip().strip('"') for c in line.split(",")]
        if len(cols) <= max(i_prompt, i_gen, i_ts):
            continue
        try:
            rows[(int(cols[i_prompt]), int(cols[i_gen]))] = float(cols[i_ts])
        except ValueError:
            continue
    return rows


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthy(port, proc, deadline):
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    return False


def ask(port, prompt, max_tokens, timeout):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.load(r)
    if "timings" not in reply:
        raise RuntimeError("backend reported no timings")
    return reply["timings"]["predicted_per_second"], reply["choices"][0]["message"]["content"], reply["timings"]


def long_prompt(n_tokens):
    """A deterministic filler prompt of roughly ``n_tokens`` tokens at this filler's own ratio."""
    return LONG_PROMPT_UNIT * max(1, int(n_tokens * LONG_CHARS_PER_TOKEN / len(LONG_PROMPT_UNIT)) + 1)


def serve_measure(onebit, model, args, reps, tokens, timeout, out_dir, longctx=0, longtok=128):
    """Start `1bit serve` once, run the three prompts (+ the long-context leg), return the numbers."""
    port = free_port()
    cmd = [onebit, "serve", "-m", str(model), "--port", str(port)] + args
    # GGML_HRX_STRICT=0: on this pin the MTP draft graph has one node (the head's PAD) off HRX, and
    # strict placement aborts the server on it -- the engine's own hardware gate runs its server the
    # same way (scripts/hw-gate.sh), counting placement instead of aborting. GGML_HRX_REPORT=2 asks
    # ggml-hrx for the counts; `1bit serve` exposes no `-lv` passthrough, so they only surface when
    # its own verbosity reaches the library's INFO level, and `placement` stays null otherwise.
    run_env = dict(os.environ, GGML_HRX_STRICT="0", GGML_HRX_REPORT="2")
    logtext = (out_dir / "serve.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=run_env, stdout=logtext, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    tps, texts = {}, {}
    long_info = {}
    try:
        if not wait_healthy(port, proc, time.time() + timeout / 2):
            raise RuntimeError(f"`1bit serve` did not become healthy (see {out_dir / 'serve.log'})")
        ask(port, "hi", 4, timeout)  # warm-up: pays the first-token compile, not the measurement
        for name, prompt in DECODE_PROMPTS.items():
            samples = []
            for _ in range(reps):
                value, text, _ = ask(port, prompt, tokens, timeout)
                samples.append(value)
                texts.setdefault(name, text)
            tps[name] = statistics.median(samples)
        if longctx:
            # One request, not a median: the long leg costs a full prefill of the whole history, and a
            # decode rate at 18.8k is what #679 compares. The server reports the prompt it actually saw.
            value, text, timings = ask(port, long_prompt(longctx), longtok, timeout)
            tps["long"] = value
            texts["long"] = text
            long_info.update({"ctx_tokens_requested": longctx, "ctx_tokens_actual": timings.get("prompt_n"),
                              "decode_tokens": timings.get("predicted_n"),
                              "prefill_tok_s": timings.get("prompt_per_second")})
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logtext.close()
    return tps, texts, placement_summary(out_dir / "serve.log"), long_info


PLACEMENT_RE = re.compile(r"placement: (\d+) nodes, (\d+) off HRX")


def placement_summary(path):
    """What the placement report saw, so the report says whether a graph left HRX."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hits = [(int(n), int(off)) for n, off in PLACEMENT_RE.findall(text)]
    if not hits:
        return None
    nodes, off = hits[-1]
    return {"graphs_reported": len(hits), "nodes": nodes, "off_hrx": off,
            "worst_graph_off_hrx": max((o for _, o in hits), default=0)}


KLD_MEAN_RE = re.compile(r"^Mean\s+KLD\s*:\s*([-+0-9.eE]+)", re.M)


def kld(ppl, model, args, base, wiki, timeout, out_dir):
    """Prefill-path KLD against the stored base logits; records the base when it is missing."""
    cmd = [ppl, "-m", str(model), "-f", str(wiki), "-c", "512", "--chunks", "8",
           "--device", "HRX0", "-ngl", "99"] + (args or ["-b", "512", "-ub", "512"])
    recording = not base.is_file()
    cmd += ["--kl-divergence-base", str(base)] + ([] if recording else ["--kl-divergence"])
    rc, text = run(cmd, "kld-record" if recording else "kld", timeout, out_dir)
    if recording:
        return {"recorded": rc == 0 and base.is_file(), "mean": None}
    match = KLD_MEAN_RE.search(text)
    return {"recorded": False, "mean": float(match.group(1)) if match else None}


def main():
    started = time.time()
    # Box rule first: nothing else in this run may even resolve if it asks for the forbidden shape.
    if longctx_drafter_conflict():
        raise SystemExit(
            "refusing HL_DFLASH with HL_LONGCTX>0: the drafter at long context is the arm that faulted the "
            "HRX graph on this box (27B, 18.8k, 2026-11-01 11:13 reset) and the box rule forbids it until "
            "engine#764's refusal guardrail is merged")
    out_dir = pathlib.Path(env("RESULT_DIR") or ".").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = env("RESULT_FILENAME") or "inferencex_result"
    model = model_file()

    # Out-of-tree build (README.md): `1bit` at the build root, the HRX llama.cpp tools built by
    # the llama_hrx ExternalProject under hrx/llama/bin.
    build = pathlib.Path(env("HL_BUILD") or user_home() / "wt/hl-engine-build").expanduser()
    bin_dir = pathlib.Path(env("HL_BIN") or build / "hrx/llama/bin").expanduser()
    onebit = pathlib.Path(env("HL_ONEBIT") or build / "1bit").expanduser()
    bench, ppl = bin_dir / "llama-bench", bin_dir / "llama-perplexity"
    for path in (onebit, bench, ppl):
        if not os.access(path, os.X_OK):
            raise SystemExit(f"missing binary {path}: build the checkout first (README.md)")

    wiki = pathlib.Path(env("HL_WIKI") or user_home() / "models/qwen38/wikitext-2-raw/wiki.test.raw")
    state = pathlib.Path(env("HL_STATE") or user_home() / "lb/hl-gate" / model.stem)
    state.mkdir(parents=True, exist_ok=True)
    ref_path, texts_path, base_kld = state / "ref.json", state / "texts.json", state / "kld-base.kld"
    kld_max = env_float("HL_GATE_KLD_MAX", 1e-4)
    reps = env_int("HL_REPS", 3)
    tokens = env_int("HL_TOKENS", 256)
    # engine#679: an optional long-context decode leg, whose rate becomes the score.
    longctx = env_int("HL_LONGCTX", 0)
    longtok = env_int("HL_LONGTOK", 128)
    deadline = time.time() + env_int("HL_BUDGET_SEC", 780 if not longctx else 1200)

    log(f"model={model} bin={bin_dir} state={state}")
    log(f"levers={levers() or '{}'}")
    result = {
        "framework": "custom",
        "workload_kind": "scriptable",
        "model_id": f"{model.name} (HRX0, gfx1151)",
        "throughput_unit": (f"tok/s (decode at {longctx}-token ctx)" if longctx
                            else "tok/s (geomean of pp512, tg128, serve code)"),
        "output_throughput": 0.0,
        "total_token_throughput": 0.0,
        "request_throughput": 0.0,
        "completed": 0,
        "total_output_tokens": tokens,
        "quality_gate": {"passed": False, "reason": "not run"},
        "bench_summary": {"levers": levers(), "runner_type": env("RUNNER_TYPE")},
    }

    def finish(passed, reason, rc=0):
        result["quality_gate"]["passed"] = bool(passed)
        result["quality_gate"]["reason"] = reason
        result["completed"] = 1 if passed else 0
        result["duration"] = round(time.time() - started, 2)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        log(f"gate={'PASS' if passed else 'FAIL'} ({reason}); score={result['output_throughput']} "
            f"-> {out_dir / (name + '.json')}")
        return rc

    # 1. llama-bench: prefill and decode throughput from one model load.
    rc, text = run([str(bench), "-m", str(model), "-dev", "HRX0"] + bench_args()
                   + ["-p", "512", "-n", "128", "-r", str(reps), "-o", "csv"],
                   "llama-bench", min(600, deadline - time.time()), out_dir)
    rows = bench_csv(text)
    pp512, tg128 = rows.get((512, 0)), rows.get((0, 128))
    if rc != 0 or not pp512 or not tg128:
        return finish(False, f"llama-bench failed (rc={rc}, rows={sorted(rows)})", rc=1)
    result["bench_summary"].update({"pp512_tok_s": round(pp512, 2), "tg128_tok_s": round(tg128, 2)})

    # 2. serve: the decode-side legs and the greedy texts the gate compares.
    try:
        tps, texts, placement, long_info = serve_measure(
            onebit, model, serve_lever_args(), reps, tokens,
            min(1200 if longctx else 600, deadline - time.time()), out_dir,
            longctx=longctx, longtok=longtok)
    except (RuntimeError, OSError) as exc:
        return finish(False, f"serve leg failed: {exc}", rc=1)
    result["bench_summary"]["serve_tok_s"] = {k: round(v, 2) for k, v in tps.items()}
    result["bench_summary"]["placement"] = placement
    if long_info:
        result["bench_summary"]["longctx"] = {k: (round(v, 2) if isinstance(v, float) else v)
                                              for k, v in long_info.items()}
        if long_info.get("ctx_tokens_actual"):
            # Label the unit with the context the server actually saw, not the one we asked for.
            result["throughput_unit"] = f"tok/s (decode at {long_info['ctx_tokens_actual']}-token ctx)"
    result["total_token_throughput"] = round(tps["long"] if longctx else tps["code"], 2)
    result["request_throughput"] = round(result["total_token_throughput"] / tokens, 4)

    # 3. the gate: reference halves are recorded on the first (baseline) measurement only.
    fingerprint = sha256_head(model)
    ref = {}
    if ref_path.is_file():
        try:
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ref = {}
    if ref.get("model") not in (None, fingerprint):
        log("WARNING: stored references are for a different model file; not comparing against them")
        ref = {}
    ref_texts = {}
    if texts_path.is_file():
        try:
            ref_texts = json.loads(texts_path.read_text(encoding="utf-8")).get("texts", {})
        except (OSError, ValueError):
            ref_texts = {}
    establishing = not ref or not ref_texts or not base_kld.is_file()

    kld_stat = kld(ppl, model, bench_args(), base_kld, wiki, min(900, deadline - time.time()), out_dir)
    result["quality_gate"].update({"kld_max_mean": kld_max, "kld_mean": kld_stat["mean"],
                                   "kld_recorded": kld_stat["recorded"]})
    if establishing and kld_stat["recorded"] and kld_stat["mean"] is None:
        # First measurement: this configuration defines the reference. Nothing to compare yet.
        ref_texts = dict(texts)
        texts_path.write_text(json.dumps({"texts": ref_texts}, indent=2) + "\n", encoding="utf-8")
        ref_path.write_text(json.dumps({"model": fingerprint, "levers": levers()}, indent=2) + "\n",
                            encoding="utf-8")
        result["quality_gate"]["reason"] = "reference_established"
        result["quality_gate"]["skipped"] = False
        score = score_of(pp512, tg128, tps, longctx)
        result["output_throughput"] = round(score, 4)
        result["bench_summary"]["score_geomean"] = round(geometric_mean(pp512, tg128, tps["code"]), 4)
        return finish(True, "reference_established")

    differences = [k for k in gated_text_keys(ref_texts) if texts.get(k) != ref_texts[k]]
    identical = not differences
    result["quality_gate"]["texts_identical"] = identical
    result["quality_gate"]["texts_differing"] = differences
    mean_kld = kld_stat["mean"]
    result["quality_gate"]["kld_ok"] = mean_kld is not None and mean_kld <= kld_max

    score = score_of(pp512, tg128, tps, longctx)
    result["output_throughput"] = round(score, 4)
    result["bench_summary"]["score_geomean"] = round(geometric_mean(pp512, tg128, tps["code"]), 4)
    if not result["quality_gate"]["kld_ok"]:
        return finish(False, f"KLD {mean_kld} > {kld_max}")
    if not identical:
        return finish(False, f"greedy text differs on {','.join(differences)}")
    return finish(True, f"KLD {mean_kld:.3e} <= {kld_max}, texts identical")


def geometric_mean(*values):
    vals = [float(v) for v in values if v and v > 0]
    if len(vals) < 3:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def gated_text_keys(ref_texts):
    """The replies the greedy-text gate compares: the short legs only.

    The long reply depends on the context, not on the lever, so it is never part of the gate. Popping it
    here rather than trusting the reference is what makes that true in the case that matters: a run that
    *establishes* the reference with ``HL_LONGCTX`` set records a ``long`` reply (kept as evidence, it is
    useful), and without this every later candidate — including a short-prompt run that never produces
    that key — would be gated on it.
    """
    return [k for k in ref_texts if k != "long"]


def score_of(pp512, tg128, tps, longctx):
    """What the optimizer is asked to maximise: the #679 decode rate when the long leg runs."""
    if longctx:
        return tps.get("long") or 0.0
    return geometric_mean(pp512, tg128, tps["code"])


if __name__ == "__main__":
    sys.exit(main())
