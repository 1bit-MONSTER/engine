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
"""Time our Whisper encoder against the reference number (engine#1783, W8/F8).

The bar is FastFlowLM v1.0.12's public benchmark: **1,889 ms -> 606 ms per 30 s window** on a Ryzen AI MAX+ PRO 395, both
encoder and decoder on the NPU (fastflowlm.com/docs/benchmarks/whisper_results/). This harness measures *our* side on the
execution unit we actually have -- the AMD ONNX pack's `*_encoder.onnx` on an ORT execution provider -- so the two numbers
can be put side by side with their units named, which is the mistake the issue's first draft made (our 7.33 s was CPU, their
0.606 s was NPU).

What it does, per 30 s window of one recording:

  1. decode the audio to 16 kHz mono with ffmpeg (any container ffmpeg reads);
  2. build Whisper's log-mel front end (n_fft 400, hop 160, 80 mels, 3000 frames, centred like the reference);
  3. run the encoder `--iters` times after one warm-up, and report median / min / max;
  4. report the process's peak RSS, because FLM's 3x encoder speedup cost them 1.3 GB of host memory and a number without
     that is not comparable.

The mel is computed once and reused, and its time is reported separately: a comparison that folds the front end into the
encoder number is comparing two different things.

`--self-test` checks the window arithmetic and the mel shape on synthetic input, so the parts that can be wrong without a
model are pinned without a model.
"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time

SAMPLE_RATE = 16000
WINDOW_S = 30.0
N_FFT = 400
HOP = 160
N_MELS = 80
N_FRAMES = 3000          # 30 s at hop 160, Whisper's fixed encoder width


def windows(duration_s, window_s=WINDOW_S):
    """[start, end) per encoder pass: whole windows, plus a final partial one if any audio is left.

    Whisper runs the encoder on every whole 30 s window AND on a trailing partial window (a 6.4 s clip costs one pass), so
    a duration of exactly 30 s is one window and 30.001 s is two. Measured, not assumed: the benchmark page's "6.4 s clip"
    row is the single-partial-window case.
    """
    if duration_s <= 0:
        return []
    out, t = [], 0.0
    while t < duration_s:
        out.append((t, min(t + window_s, duration_s)))
        t += window_s
    return out


def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    """The standard HTK-mel triangular bank Whisper uses (Slaney-normalised areas are close enough for timing, and the
    comparison is stated as shape-real / content-synthetic when no fixture is supplied)."""
    import numpy as np
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_bins)
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        lo, ce, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (fft_freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ce, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def log_mel(pcm, n_frames=N_FRAMES):
    """[1, N_MELS, n_frames] log-mel, Whisper's shape. Centred with reflect padding like the reference front end."""
    import numpy as np
    fb = mel_filterbank()
    pad = N_FFT // 2
    x = np.pad(pcm, (pad, pad), mode="reflect") if pcm.size > pad else np.pad(pcm, (pad, pad), mode="constant")
    want = (n_frames - 1) * HOP + N_FFT
    if x.size < want:
        x = np.pad(x, (0, want - x.size), mode="constant")
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(N_FFT).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) ** 2
    m = spec @ fb.T                                        # [n_frames, n_mels]
    m = np.log10(np.maximum(m, 1e-10))
    m = np.maximum(m, m.max() - 8.0)
    m = (m + 4.0) / 4.0
    return m.T[None, :, :].astype(np.float32)


def decode_audio(path, tmpdir):
    """16 kHz mono f32le, via ffmpeg: any container it reads."""
    out = os.path.join(tmpdir, "audio.f32")
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("ffmpeg could not decode %s: %s" % (path, r.stderr.strip().splitlines()[-1:]))
    import numpy as np
    return np.fromfile(out, dtype=np.float32)


def synthetic_audio(seconds):
    """Deterministic wideband noise: shape-real, content-synthetic (stated in the output when --audio is absent)."""
    import numpy as np
    rng = np.random.default_rng(1783)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.05).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Time a Whisper encoder against FLM v1.0.12's 606 ms/30 s window (engine#1783).")
    ap.add_argument("--encoder", help="*_encoder.onnx from an amd/whisper-*-onnx-npu pack")
    ap.add_argument("--audio", help="any file ffmpeg reads; absent = 30 s of deterministic noise (shape-real)")
    ap.add_argument("--provider", default="CPUExecutionProvider", help="ORT execution provider (default CPU)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=WINDOW_S, help="audio length when --audio is absent")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        import numpy as np
        bad = 0
        def ck(ok, what):
            nonlocal bad
            print(("ok   " if ok else "FAIL ") + what)
            if not ok:
                bad += 1
        ck(windows(0.0) == [], "zero-length audio needs no encoder pass")
        ck(len(windows(6.4)) == 1, "a 6.4 s clip is ONE pass (the benchmark page's smallest row)")
        ck(len(windows(30.0)) == 1, "exactly 30 s is one pass")
        ck(len(windows(30.001)) == 2, "30.001 s is two: the trailing partial window costs a pass")
        ck(len(windows(308.0)) == 11, "the benchmark's 5-minute recording (308 s) is 11 windows, not 10")
        ck(windows(65.0)[-1] == (60.0, 65.0), "the last window is partial, not dropped")
        m = log_mel(synthetic_audio(30.0))
        ck(m.shape == (1, N_MELS, N_FRAMES), "the encoder input is [1, %d, %d] (got %s)" % (N_MELS, N_FRAMES, m.shape))
        # The invariant is the SPAN, not the absolute range: whisper clamps to an 8-decade window and divides by 4, so
        # (max - min) <= 2 by construction while the absolute range depends on the signal's level. I asserted [0, 1] and
        # then [-1.5, 1] here, and both were wrong while the code was right -- twice in a row, which is why the check now
        # states the property the formula actually guarantees.
        span = float(m.max()) - float(m.min())
        ck(bool(np.isfinite(m).all()) and span > 0.1 and span <= 2.0 + 1e-3,
           "the log-mel is finite and spans at most the 8-decade window (min %.3f max %.3f span %.3f)" % (m.min(), m.max(), span))
        print("whisper_encoder_baseline: %s" % ("ok" if not bad else "%d failure(s)" % bad))
        return 1 if bad else 0

    if not a.encoder:
        ap.error("--encoder is required unless --self-test")
    import numpy as np
    import onnxruntime as ort

    tmpdir = os.path.dirname(os.path.abspath(a.encoder))
    if a.audio:
        pcm, content = decode_audio(a.audio, tmpdir), "real (%s)" % os.path.basename(a.audio)
    else:
        pcm, content = synthetic_audio(a.seconds), "synthetic noise, shape-real"
    dur = pcm.size / float(SAMPLE_RATE)

    t0 = time.perf_counter()
    mel = log_mel(pcm)
    mel_s = time.perf_counter() - t0

    sess = ort.InferenceSession(a.encoder, providers=[a.provider])
    name = sess.get_inputs()[0].name
    sess.run(None, {name: mel})                                     # warm-up, not timed
    per_window = []
    for _ in range(max(1, a.iters)):
        t = time.perf_counter()
        sess.run(None, {name: mel})
        per_window.append((time.perf_counter() - t) * 1000.0)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    # The load belongs IN the number: two runs of this harness on the same encoder here gave medians 121.7 ms and 70.3 ms
    # with a 682 ms outlier, because the box was busy. A timing reported without its load is the thing this lane refuses
    # to quote (engine#1423), so the report carries it and the human output prints it next to the figure.
    load = [round(x, 1) for x in os.getloadavg()]
    rep = {
        "encoder": os.path.basename(a.encoder),
        "provider": a.provider,
        "audio": content,
        "audio_seconds": round(dur, 3),
        "windows": len(windows(dur)),
        "mel_ms": round(mel_s * 1000.0, 1),
        "encoder_ms": {"median": round(statistics.median(per_window), 1), "min": round(min(per_window), 1),
                       "max": round(max(per_window), 1), "iters": len(per_window)},
        "peak_rss_mb": round(rss_mb, 1),
        "load_avg": load,
        "reference": {"source": "fastflowlm.com/docs/benchmarks/whisper_results/", "flm_1_0_12_ms": 606,
                      "flm_before_ms": 1889, "unit": "NPU, 30 s window, Ryzen AI MAX+ PRO 395"},
    }
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print("%s on %s, %s audio (%.1f s, %d window(s))" % (rep["encoder"], a.provider, content, dur, rep["windows"]))
        print("  encoder per 30 s window: median %.1f ms (min %.1f, max %.1f, %d iters)"
              % (rep["encoder_ms"]["median"], rep["encoder_ms"]["min"], rep["encoder_ms"]["max"], rep["encoder_ms"]["iters"]))
        print("  log-mel front end:       %.1f ms (excluded from the encoder figure)" % rep["mel_ms"])
        print("  peak RSS:                %.1f MB" % rep["peak_rss_mb"])
        print("  load average:            %s  <- quote the timing WITH this, or not at all" % (load,))
        print("  reference (NPU):         FLM v1.0.12 606 ms/window, was 1889 ms -- the unit is part of the number")
    return 0


if __name__ == "__main__":
    sys.exit(main())
