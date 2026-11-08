#!/usr/bin/env python3
"""Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Run an AMD "Whisper NPU ONNX" pack's plain ONNX encoder+decoder pair on CPU and print a transcript.

usage: run_whisper_onnx.py ENCODER.onnx DECODER.onnx AUDIO.wav [--tokenizer=openai/whisper-tiny]

What the packs contain: `ggml-<size>-encoder-vitisai.rai` (the encoder compiled for the NPU), plus the same
encoder as `<size>_encoder.onnx` (tiny/base) or `encoder_model.onnx` (small/medium/large) and a plain ONNX
decoder. The .rai is inert outside AMD's RyzenAI runtime, which amd/whisper.cpp documents as Windows-only -
so the ONNX pair is the honest Linux path and this script is it.

Three traps, all measured:
  * NO TOKENIZER is shipped. The reference tokenizer is an argument, and the ids must come from it: the
    English-only vocab shifts every style token by one (<|startoftranscript|> 50257 vs 50258,
    <|transcribe|> 50358 vs 50359, <|notimestamps|> 50362 vs 50363). Hard-coded multilingual ids made
    whisper-small.en emit endoftext at step 1.
  * A missing .onnx.data SIDECAR is a silent wrong answer: small.en's graphs are ~5 MB and 3 MB stubs whose
    weights live in <name>.onnx.data. ORT opens the stub happily with zero weights and the decoder answers
    nonsense. Download the sidecar (sweep_family.py does) or the row is fiction.
  * Timing: RTF(compute) is reported separately from RTF(incl load), and the box's load average is printed,
    because a contended box inflated one load to 236 s for a model that loads in 1 s.
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer


def read_wav_pcm16(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise SystemExit(f"{path}: not a RIFF/WAVE file")
    pos, pcm = 12, None
    while pos < len(raw) - 8:
        cid, size = raw[pos:pos + 4], int.from_bytes(raw[pos + 4:pos + 8], "little")
        if cid == b"data":
            pcm = np.frombuffer(raw[pos + 8:pos + 8 + size], dtype="<i2")
            break
        pos += 8 + size + (size & 1)
    if pcm is None:
        raise SystemExit(f"{path}: no data chunk")
    return pcm.astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encoder")
    ap.add_argument("decoder")
    ap.add_argument("wav", help="16 kHz mono s16 PCM")
    ap.add_argument("--tokenizer", default=None,
                    help="reference tokenizer (default: by the encoder's mel-bin count - 128 -> whisper-large-v3, "
                         "80 -> whisper-tiny); use openai/whisper-small.en for the -en packs")
    a = ap.parse_args()

    # The export decides the mel-bin count: whisper large-v3/turbo take 128 bins, tiny/base/small/medium take 80. A
    # feature extractor of the wrong width is rejected by the graph (and, in a driver, reads as "no output").
    so0 = ort.SessionOptions()
    so0.log_severity_level = 3
    enc = ort.InferenceSession(a.encoder, so0, providers=["CPUExecutionProvider"])
    enc_shape = enc.get_inputs()[0].shape          # expected [batch, 80|128, frames]
    bins = enc_shape[1] if len(enc_shape) == 3 else None
    # REFUSE an unfamiliar width instead of falling back: the whole point of trap 4 is that a wrong-width
    # extractor is rejected by the graph and reads as "no output". A symbolic dim (a str), a frame-major export
    # ([batch, frames, n_mels] -> 3000 here) and a 64/96-bin export all used to land silently on the 80-bin path.
    if not isinstance(bins, int) or bins not in (80, 128):
        raise SystemExit(f"whisper encoder declares shape {enc_shape}: cannot tell the mel-bin count "
                         f"(expected [batch, 80|128, frames]); pass --tokenizer to choose the reference explicitly")
    # the extractor follows the tokenizer reference, so the two cannot drift once a third width is added
    ref = a.tokenizer or ("openai/whisper-large-v3" if bins == 128 else "openai/whisper-tiny")
    fe = WhisperFeatureExtractor.from_pretrained(ref)
    tok = WhisperTokenizer.from_pretrained(ref)
    vocab = tok.get_vocab()

    def tid(token, default):
        """The id of a style token IF this tokenizer has it (no assumption about the multilingual ids)."""
        if token in vocab:
            i = tok.convert_tokens_to_ids(token)
            return i if isinstance(i, int) and i >= 0 else default
        return default

    SOT = tid("<|startoftranscript|>", 50258)
    EN = tid("<|en|>", None)
    TR = tid("<|transcribe|>", 50359)
    NT = tid("<|notimestamps|>", 50363)
    EOT = tok.eos_token_id if tok.eos_token_id is not None else 50257

    audio = read_wav_pcm16(a.wav)
    dur = len(audio) / 16000.0
    feat = fe(audio, sampling_rate=16000, return_tensors="np")["input_features"].astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.time()
    dec = ort.InferenceSession(a.decoder, so, providers=["CPUExecutionProvider"])
    t_load = time.time() - t0
    t0 = time.time()
    xa = enc.run(None, {enc.get_inputs()[0].name: feat})[0]
    enc_s = time.time() - t0

    prompt = [t for t in (SOT, EN, TR, NT) if t is not None]
    d_in = dec.get_inputs()
    if len(d_in) < 2:
        raise SystemExit(f"whisper decoder expects {len(d_in)} input(s), not the token+encoder-state pair: "
                         f"{[i.name for i in d_in]}")
    tok_name, xa_name = d_in[0].name, d_in[1].name
    t_shape = d_in[0].shape                     # [batch, 448] in AMD's exports
    if len(t_shape) != 2 or not isinstance(t_shape[1], int):
        raise SystemExit(f"whisper decoder declares {t_shape}: the token window must be a static [batch, N]")
    width = t_shape[1]
    ids = np.zeros((1, width), dtype=np.int64)
    ids[0, :len(prompt)] = prompt
    t0 = time.time()
    out, steps = [], 0
    while len(prompt) < width:
        logits = dec.run(None, {tok_name: ids, xa_name: xa})[0]
        nxt = int(np.argmax(logits[0, len(prompt) - 1]))
        steps += 1
        if nxt == EOT or len(out) >= 200:
            break
        out.append(nxt)
        prompt.append(nxt)
        ids[0, len(prompt) - 1] = nxt
    dec_s = time.time() - t0
    text = tok.decode(out, skip_special_tokens=True).strip()

    try:
        loadavg = os.getloadavg()[0]
    except OSError:
        loadavg = float("nan")
    print(f"audio={dur:.2f}s load={t_load:.2f}s enc={enc_s:.2f}s dec={dec_s:.2f}s steps={steps} "
          f"RTF(compute)={(enc_s + dec_s) / dur:.3f} RTF(incl load)={(t_load + enc_s + dec_s) / dur:.3f} "
          f"load_avg={loadavg:.1f}/{os.cpu_count()}c"
          + ("  [CONTENDED: load/RTF not quotable, use enc+dec]" if loadavg > os.cpu_count() * 0.5 else "")
          + f" tok={a.tokenizer} transcript={json.dumps(text)}")


if __name__ == "__main__":
    main()
