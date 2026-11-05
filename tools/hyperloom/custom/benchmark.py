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
import platform
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
HWMON_ROOT = pathlib.Path("/sys/class/hwmon")
GPU_SYSFS = pathlib.Path("/sys/class/drm/card0/device")


def meminfo_gib(path, unit):
    """A counter from the GPU's sysfs, in GiB, or None when this box does not expose it."""
    try:
        return round(int(path.read_text(encoding="ascii").strip()) / unit, 2)
    except (OSError, ValueError):
        return None


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
    """The arguments every serve leg carries: the batch levers `1bit serve` accepts, and the two
    measurement conditions it has to be started with.

    `1bit serve` is a wrapper with its own argument parser: it takes `-fa` (forwarded to its
    llama-server child) but NOT `-b`/`-ub`/`-ctk`/`-ctv` -- it sets those itself per route, and it
    rejects the flags with `unknown option`. So those levers move the llama-bench legs only; the
    serve leg measures what serve would serve.

    `-c` is not a lever but a measurement condition: without it serve sizes the KV cache for the
    model's OWN context (262144 on Qwen3.8-27B = 16.1 GiB of KV), which only fits when the box
    happens to be idle -- under another lane's GTT the engine's fit check aborts the server and every
    candidate fails for a reason that has nothing to do with the lever under test. `HL_CTX` overrides.

    `--placement-report` is the second one. It is what makes serve pass `-lv 4` to its llama-server,
    and llama-server drops the library's INFO lines below that (`common/log.cpp`), so the report
    `GGML_HRX_REPORT=2` asks ggml-hrx for never reached serve.log: `bench_summary.placement` was null
    in every artifact this workload has ever written, and an unrecorded placement cannot be told
    apart from a graph that left HRX (`docs/serve.md`, engine#390). It reports, it does not check:
    serve's `--strict-device` is the one that refuses, and `GGML_HRX_STRICT=0` already opts out here.
    """
    args = ["-c", str(serve_ctx()), "--placement-report"]
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


def serve_ctx():
    """The context `1bit serve` is pinned to. One definition, so the two readers cannot drift.

    A long-context arm measures the decode rate of a prompt of `HL_LONGCTX` tokens, so an `HL_CTX`
    smaller than prompt+decode does not shorten the leg -- the server either rejects the request or
    shifts the context and answers on the tail, which is a different measurement wearing the same
    label. The refusal in ``longctx_exceeds_serve_ctx`` is what makes that impossible.
    """
    return env_int("HL_CTX", 8192)


def longctx_exceeds_serve_ctx():
    """True when the long leg cannot fit the context serve was pinned to.

    ``HL_LONGCTX`` and ``HL_CTX`` are independent env vars, and the default (8192) is smaller than
    every long-context value this workload was ever run with, so setting only ``HL_LONGCTX`` -- 18800,
    the #679 shape -- asks for a 18.8k-token prompt on an 8192-token server. Refused before anything
    loads, like the drafter refusal, because the alternative is a burned device window.
    """
    longctx = env_int("HL_LONGCTX", 0)
    return bool(longctx) and longctx + env_int("HL_LONGTOK", 128) > serve_ctx()


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
        args += ["--dflash", dflash_gguf_arg()]
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


def dflash_gguf_arg():
    """The drafter path handed to `--dflash`. One definition, so the report cannot name another."""
    return env("HL_DFLASH_GGUF", "dflash.gguf")


def drafting_identity():
    """Name the drafting files the serve leg actually used, as resolved paths with hashes.

    Same class as ``build_identity``/``device_stack``, and the gap the tuner itself points at: it
    *drops* a drafter-quant variant because overwriting `HL_DFLASH_GGUF` "would make its number
    incomparable to the baseline" (`run-26030.log`, 2026-10-30 18:14), and this model has two drafter
    quants on the box (`~/models/qwen38/DFlash2/{q8_0,q4k}.gguf`), yet a result echoed `HL_DFLASH`
    alone -- so `{"HL_DFLASH": "1"}`, the arm that produced the 50.87 best of record, names no
    drafter at all. `--dflash` is handed a *relative* path by default, so the run's cwd decided it
    too: the path here is resolved, and `env` says whether it came from the environment or the
    default rather than leaving the default unnamed.
    """
    ident = {}
    if dflash_on():
        raw = dflash_gguf_arg()
        path = pathlib.Path(raw).expanduser().resolve()
        ident["dflash"] = {"env": "HL_DFLASH_GGUF" if env("HL_DFLASH_GGUF") else "default",
                          "path": str(path),
                          "sha256": sha256_head(path) if path.is_file() else None}
    external_head = env("HL_MTP_HEAD")
    if external_head and not dflash_on():
        path = pathlib.Path(external_head).expanduser().resolve()
        ident["mtp_head"] = {"env": "HL_MTP_HEAD", "path": str(path),
                            "sha256": sha256_head(path) if path.is_file() else None}
    return ident


def build_identity(onebit, bin_dir):
    """Name the binary that produced the measurement, inside the artifact itself.

    A session report outlives the shell it was made in, and this workload is routinely measured on
    more than one build of the same tree (out-of-tree `hl-engine-build` / `-c20`). Without this field
    two results that differ only by build are indistinguishable, so a KEEP can be read against a binary
    it was never measured on. ``1bit`` is far under ``sha256_head``'s 64 MiB head, so this is the whole
    binary's hash, not a prefix.
    """
    onebit, bin_dir = pathlib.Path(onebit), pathlib.Path(bin_dir)
    return {
        "onebit": str(onebit),
        "bin_dir": str(bin_dir),
        "sha256": sha256_head(onebit),
    }


def model_identity(model):
    """Name the model file, with the fingerprint the quality gate itself compares against.

    Same class as ``build_identity`` and ``device_stack``, one level closer to the number: the gate's
    references are per box in ``HL_STATE`` and are keyed by this fingerprint (``ref.json``'s ``model``), so
    a report that names the model only by file name cannot say which file its score and its KLD verdict
    belong to. That case is not hypothetical here: the weights were re-downloaded after a restore while
    the gate references survived, i.e. the same file name on this box can be a different file.
    ``sha256_head`` is the gate's own notion of model identity (header + first tensors, not 17 GB), so a
    result and a reference can be checked against each other by hash rather than by name.
    """
    model = pathlib.Path(model)
    return {"path": str(model), "sha256": sha256_head(model)}


def device_stack():
    """Name the device stack the measurement came from: kernel and the HSA runtime HRX loads.

    Same class as ``build_identity``: this box reboots between measurement series (three times on
    2026-11-03 alone) and the tuner launcher applies no kernel gate, so a report that names no kernel
    cannot say which driver stack a number belongs to -- every long-context number of record so far is
    tied to its kernel by prose only. The runtime is named by path AND whole-file hash; when the env
    does not point at one, the box's ROCm-TheRock path is hashed if it exists and reported as null
    otherwise, rather than guessing.
    """
    libhsa = pathlib.Path(env("IREE_HAL_AMDGPU_LIBHSA_PATH") or
                          "/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/"
                          "libhsa-runtime64.so.1").expanduser()
    return {
        "kernel": platform.release(),
        "hsa_runtime": str(libhsa) if libhsa.is_file() else None,
        "hsa_runtime_sha256": sha256_head(libhsa) if libhsa.is_file() else None,
    }


def die_temp_c():
    """The die temperature, from the hwmon channel labelled Tctl.

    The hwmon *index* is not stable across boots (this box has rebooted six times in three days and the
    k10temp channel has come back as hwmon2 and as hwmon4), so the channel is found by its label instead
    of by a hardcoded path. None when the box exposes no Tctl channel at all.
    """
    for label in sorted(HWMON_ROOT.glob("hwmon*/temp*_label")):
        try:
            if label.read_text(encoding="ascii", errors="replace").strip() != "Tctl":
                continue
            milli = (label.with_name(label.name.replace("_label", "_input"))
                     .read_text(encoding="ascii").strip())
        except OSError:
            continue
        try:
            return round(float(milli) / 1000.0, 2)
        except ValueError:
            return None
    return None


def box_conditions():
    """The conditions the measurement is running in, device-free (sysfs + /proc).

    A score is a measurement *of a box*, and this box is shared. On 2026-11-01 the same workload at the
    same configuration produced 44.69 with a quiet die and no swap, and 31.68 with 85 GB of 122 GB in use,
    37 GB of swap in play and the die climbing; both passed the correctness gate, which is about the
    output and says nothing about whether the two arms are comparable, and the result JSON recorded
    nothing that could tell them apart. Every field is ``None`` where the box does not expose it, so an
    unreadable die cannot be read back as a cool one.
    """
    gtt_used = meminfo_gib(GPU_SYSFS / "mem_info_gtt_used", 2 ** 30)
    gtt_total = meminfo_gib(GPU_SYSFS / "mem_info_gtt_total", 2 ** 30)
    available = None
    try:
        for line in pathlib.Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available = round(int(line.split()[1]) / (1024 * 1024), 2)
                break
    except (OSError, ValueError, IndexError):
        available = None
    return {"tctl_c": die_temp_c(),
            "gtt_used_gib": gtt_used, "gtt_total_gib": gtt_total,
            "mem_available_gib": available}


def box_summary(samples):
    """The start snapshot plus how far the die moved while the arm ran.

    The die at start alone cannot separate a clean arm from a loaded one: the arm that had to be aborted
    on 2026-11-01 was at 54 C when it was claimed and hit 93 C inside the run. Kept in the same shape as
    the start snapshot, so a reader that only wants the conditions reads the top three fields.
    """
    if not samples:
        return None
    summary = dict(samples[0])
    die = [s["tctl_c"] for s in samples if s.get("tctl_c") is not None]
    summary["tctl_c_min"] = min(die) if die else None
    summary["tctl_c_max"] = max(die) if die else None
    summary["tctl_samples"] = len(die)
    return summary


def levers():
    """Echo the configuration into the report, so a KEEP is explicable months later.

    Every name here has to be one the engine actually reads: an echoed knob nothing consumes makes
    a report claim a configuration the measurement never had (`GGML_HRX_TER_Q4_0` used to be listed
    and appears nowhere in the engine or its llama.cpp pin).
    """
    return {
        k: env(k)
        for k in ("HL_B", "HL_UB", "HL_FA", "HL_CTK", "HL_CTV", "HL_MTP_MAX", "HL_DFLASH",
                  "HL_PARALLEL", "HL_CTX", "HL_LONGCTX", "HL_LONGTOK",
                  "GGML_HRX_Q8_PREFILL_RELAX", "GGML_HRX_GRAPH_PROGRAM_CACHE",
                  "ONEBIT_HRX_DECODE_SPLIT")
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


def log_dir_for(out_dir, name):
    """The directory one measurement's raw logs go in.

    They used to share fixed names with the result directory itself, so a second measurement in the same
    RESULT_DIR silently overwrote the first one's serve.log / llama-bench.log / kld.log: measured on
    2026-11-02, the 19:01 arm destroyed the 18:45 baseline's raw logs, and a pair's two halves (A, B) would
    have collided the same way. One directory per RESULT_FILENAME makes the raw evidence survive the run.
    """
    log_dir = out_dir / f"logs-{name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


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


def serve_measure(onebit, model, args, reps, tokens, timeout, log_dir, longctx=0, longtok=128):
    """Start `1bit serve` once, run the three prompts (+ the long-context leg), return the numbers."""
    port = free_port()
    cmd = [onebit, "serve", "-m", str(model), "--port", str(port)] + args
    # GGML_HRX_STRICT=0: on this pin the MTP draft graph has one node (the head's PAD) off HRX, and
    # strict placement aborts the server on it -- the engine's own hardware gate runs its server the
    # same way (scripts/hw-gate.sh), counting placement instead of aborting. GGML_HRX_REPORT=2 asks
    # ggml-hrx for every graph's counts; `serve_batch_args` carries `--placement-report`, which is
    # what lifts llama-server's verbosity enough for those lines to reach the log at all.
    run_env = dict(os.environ, GGML_HRX_STRICT="0", GGML_HRX_REPORT="2")
    logtext = (log_dir / "serve.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=run_env, stdout=logtext, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    tps, texts = {}, {}
    long_info = {}
    try:
        if not wait_healthy(port, proc, time.time() + timeout / 2):
            raise RuntimeError(f"`1bit serve` did not become healthy (see {log_dir / 'serve.log'})")
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
    return tps, texts, placement_summary(log_dir / "serve.log"), long_info


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


def gate_reference(ref, base_kld, texts_path):
    """What a gate verdict is relative to.

    A KLD mean is a distance to *these* base logits, and `HL_STATE` is per-box and regenerated when
    the base is missing, so the same number reads `PASS` against a different reference. The result
    JSON carried the half's own verdict (`kld_mean`, `texts_identical`) and nothing about the
    reference, so a report from one session could not be compared with — or re-checked against —
    another's.
    """
    return {"model": ref.get("model"), "levers": ref.get("levers"),
            "base_kld_sha256": sha256_head(base_kld), "texts_sha256": sha256_head(texts_path)}


def kld(ppl, model, args, base, wiki, timeout, log_dir):
    """Prefill-path KLD against the stored base logits; records the base when it is missing."""
    cmd = [ppl, "-m", str(model), "-f", str(wiki), "-c", "512", "--chunks", "8",
           "--device", "HRX0", "-ngl", "99"] + (args or ["-b", "512", "-ub", "512"])
    recording = not base.is_file()
    cmd += ["--kl-divergence-base", str(base)] + ([] if recording else ["--kl-divergence"])
    rc, text = run(cmd, "kld-record" if recording else "kld", timeout, log_dir)
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
    # Same class, other direction: a long leg that cannot fit the pinned context measures nothing.
    if longctx_exceeds_serve_ctx():
        raise SystemExit(
            f"refusing HL_LONGCTX={env_int('HL_LONGCTX', 0)} with HL_CTX={serve_ctx()}: the long leg asks for "
            f"{env_int('HL_LONGCTX', 0) + env_int('HL_LONGTOK', 128)} tokens (prompt + HL_LONGTOK) and serve is "
            "pinned to a smaller context, so the server would reject the request or answer on a shifted "
            "context; set HL_CTX >= prompt + decode (20480 with HL_LONGCTX=18800)")
    out_dir = pathlib.Path(env("RESULT_DIR") or ".").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = env("RESULT_FILENAME") or "inferencex_result"
    log_dir = log_dir_for(out_dir, name)
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

    model_ident = model_identity(model)
    # Conditions are sampled at start and at each phase boundary: the arm that had to be aborted on
    # 2026-11-01 was claimed at 54 C and reached 93 C inside the run, so the start value alone is not
    # the measurement's conditions. Nothing here touches the device.
    conditions = [box_conditions()]
    log(f"model={model} bin={bin_dir} state={state}")
    log(f"device={device_stack()}")
    log(f"box={conditions[0]}")
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
        "bench_summary": {"levers": levers(), "runner_type": env("RUNNER_TYPE"),
                          "model": model_ident,
                          "build": build_identity(onebit, bin_dir),
                          "device_stack": device_stack(),
                          "drafting": drafting_identity(),
                          "box": box_summary(conditions),
                          "llama_bench_reps": reps},
    }

    def finish(passed, reason, rc=0):
        result["bench_summary"]["box"] = box_summary(conditions)
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
                   "llama-bench", min(600, deadline - time.time()), log_dir)
    rows = bench_csv(text)
    pp512, tg128 = rows.get((512, 0)), rows.get((0, 128))
    if rc != 0 or not pp512 or not tg128:
        return finish(False, f"llama-bench failed (rc={rc}, rows={sorted(rows)})", rc=1)
    result["bench_summary"].update({"pp512_tok_s": round(pp512, 2), "tg128_tok_s": round(tg128, 2)})
    conditions.append(box_conditions())

    # 2. serve: the decode-side legs and the greedy texts the gate compares.
    try:
        tps, texts, placement, long_info = serve_measure(
            onebit, model, serve_lever_args(), reps, tokens,
            min(1200 if longctx else 600, deadline - time.time()), log_dir,
            longctx=longctx, longtok=longtok)
    except (RuntimeError, OSError) as exc:
        return finish(False, f"serve leg failed: {exc}", rc=1)
    result["bench_summary"]["serve_tok_s"] = {k: round(v, 2) for k, v in tps.items()}
    result["bench_summary"]["placement"] = placement
    conditions.append(box_conditions())
    if long_info:
        result["bench_summary"]["longctx"] = {k: (round(v, 2) if isinstance(v, float) else v)
                                              for k, v in long_info.items()}
        if long_info.get("ctx_tokens_actual"):
            # Label the unit with the context the server actually saw, not the one we asked for.
            result["throughput_unit"] = f"tok/s (decode at {long_info['ctx_tokens_actual']}-token ctx)"
    result["total_token_throughput"] = round(tps["long"] if longctx else tps["code"], 2)
    result["request_throughput"] = round(result["total_token_throughput"] / tokens, 4)

    # 3. the gate: reference halves are recorded on the first (baseline) measurement only.
    fingerprint = model_ident["sha256"]
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

    kld_stat = kld(ppl, model, bench_args(), base_kld, wiki, min(900, deadline - time.time()), log_dir)
    conditions.append(box_conditions())
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

    # Only reachable when all three references exist (otherwise the run established them above).
    result["quality_gate"]["reference"] = gate_reference(ref, base_kld, texts_path)
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
