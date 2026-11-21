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
"""A fixed LibriSpeech test-clean subset for Whisper accuracy checks (engine#1783), without the 350 MB download.

Reads only the needed row groups of the Hugging Face parquet conversion of openslr/librispeech_asr (all/test.clean,
LibriSpeech is CC BY 4.0, OpenSLR SLR12) by HTTP range requests, keeps every third utterance of every third row group
(305 utterances, 24 speakers, 37.7 min, 6,227 words with the defaults), drops anything longer than one 30 s window, and
writes 16 kHz mono wav files plus manifest.json (id, speaker, duration, reference text).

    whisper_ls_subset.py <out dir> [--group-step 3] [--row-step 3] [--max-s 29.5]

Needs pyarrow, fsspec with aiohttp, and ffmpeg on PATH. The selection is deterministic: the same arguments give the
same set.
"""
import argparse
import json
import os
import subprocess
import sys

URL = ("https://huggingface.co/datasets/openslr/librispeech_asr/resolve/refs%2Fconvert%2Fparquet/"
       "all/test.clean/0000.parquet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--group-step", type=int, default=3)
    ap.add_argument("--row-step", type=int, default=3)
    ap.add_argument("--max-s", type=float, default=29.5)
    a = ap.parse_args()
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as e:
        sys.exit(f"needs pyarrow and fsspec (with aiohttp): {e}")
    os.makedirs(os.path.join(a.out, "wav"), exist_ok=True)
    keep = []
    fs = fsspec.filesystem("https")
    with fs.open(URL, "rb", block_size=1 << 20, cache_type="none") as f:
        pf = pq.ParquetFile(f)
        for g in range(0, pf.metadata.num_row_groups, a.group_step):
            rows = pf.read_row_group(g, columns=["audio", "text", "id", "speaker_id"]).to_pylist()
            for i, r in enumerate(rows):
                if i % a.row_step:
                    continue
                uid = r["id"]
                wav = os.path.join(a.out, "wav", uid + ".wav")
                p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", "-f", "wav", wav], input=r["audio"]["bytes"], capture_output=True)
                if p.returncode:
                    print("ffmpeg failed:", uid, p.stderr[:200].decode(errors="replace"))
                    continue
                dur = (os.path.getsize(wav) - 44) / 32000.0
                if dur > a.max_s:
                    os.remove(wav)
                    print(f"skip {uid}: {dur:.1f} s is longer than one window")
                    continue
                keep.append({"id": uid, "speaker": r["speaker_id"], "dur": round(dur, 3), "text": r["text"]})
            print(f"row group {g}: {len(keep)} kept so far", flush=True)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(keep, f, indent=0)
    print(f"{len(keep)} utterances, {sum(k['dur'] for k in keep) / 60:.1f} min, {len(set(k['speaker'] for k in keep))} "
          f"speakers, {sum(len(k['text'].split()) for k in keep)} words")


if __name__ == "__main__":
    main()
