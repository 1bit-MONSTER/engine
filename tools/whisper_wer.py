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
"""Word error rate of whisper_transcribe runs on a manifest (tools/whisper_ls_subset.py), and token agreement between
two runs of the same set (engine#1783).

    whisper_wer.py <manifest.json> <arm dir> [<reference arm dir>] [--tokenizer DIR] [--diffs]

An arm dir holds one `<id>.log` per utterance, the tool's stdout (its `ids:` line is read). The ids are detokenised
with the multilingual Whisper tokenizer (tokenizer.json; default: the first whisper-tiny or whisper-small snapshot in the
Hugging Face cache) and both sides go through Whisper's English text normalizer (normalizer.json, transformers'
EnglishTextNormalizer) before a word-level Levenshtein distance; WER is total errors over total reference words.
With a reference arm it also prints the WER difference in points, how many utterances have identical token sequences,
and the token edit distance over the reference's tokens. Needs `tokenizers` and `transformers`.
"""
import argparse
import glob
import json
import os
import sys


def find_tokenizer():
    for name in ("whisper-tiny", "whisper-small", "whisper-base"):
        hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--openai--{name}/snapshots/*/tokenizer.json"))
        if hits:
            return os.path.dirname(hits[0])
    return None


def ids_of(d, uid):
    p = os.path.join(d, uid + ".log")
    if not os.path.exists(p):
        return None
    with open(p, errors="replace") as f:
        for ln in f:
            if ln.startswith("ids:"):
                return [int(x) for x in ln[4:].split() if x.lstrip("-").isdigit()]
    return None


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("arm")
    ap.add_argument("ref", nargs="?")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--diffs", action="store_true", help="print each utterance whose tokens differ from the reference")
    a = ap.parse_args()
    try:
        import tokenizers
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except ImportError as e:
        sys.exit(f"needs tokenizers and transformers: {e}")
    tdir = a.tokenizer or find_tokenizer()
    if tdir is None:
        sys.exit("no Whisper tokenizer found; pass --tokenizer DIR (tokenizer.json + normalizer.json)")
    tok = tokenizers.Tokenizer.from_file(os.path.join(tdir, "tokenizer.json"))
    with open(os.path.join(tdir, "normalizer.json")) as f:
        norm = EnglishTextNormalizer(json.load(f))
    with open(a.manifest) as f:
        man = json.load(f)
    text = lambda ids: tok.decode([i for i in ids if i < 50257])   # below <|endoftext|>: the text tokens

    def score(d, only=None):
        errs = words = 0
        hyp = {}
        for m in man:
            if only is not None and m["id"] not in only:
                continue
            ids = ids_of(d, m["id"])
            if ids is None:
                continue
            hyp[m["id"]] = ids
            r = norm(m["text"]).split()
            errs += edit_distance(r, norm(text(ids)).split())
            words += len(r)
        return errs, words, hyp

    e, n, hyp = score(a.arm)
    print(f"{a.arm}: {len(hyp)} of {len(man)} utterances, {n} reference words, {e} word errors, WER {100.0 * e / max(n, 1):.3f} %")
    if a.ref:
        _, _, ref_all = score(a.ref)
        both = set(hyp) & set(ref_all)   # both arms are scored on the utterances they share
        e, n, hyp = score(a.arm, both)
        e2, n2, ref = score(a.ref, both)
        same = sum(1 for k in both if hyp[k] == ref[k])
        te = sum(edit_distance(ref[k], hyp[k]) for k in both)
        tn = sum(len(ref[k]) for k in both)
        print(f"on the {len(both)} shared utterances ({n} reference words): {a.arm} {e} errors, WER {100.0 * e / max(n, 1):.3f} %; "
              f"{a.ref} {e2} errors, WER {100.0 * e2 / max(n2, 1):.3f} %")
        print(f"difference: {100.0 * (e - e2) / max(n, 1):+.3f} WER points; identical token sequences "
              f"{same} of {len(both)}; token edits {te} of {tn} reference tokens ({100.0 * te / max(tn, 1):.3f} %)")
        if a.diffs:
            for k in sorted(both):
                if hyp[k] != ref[k]:
                    print(f"  {k}\n    ref: {text(ref[k]).strip()}\n    arm: {text(hyp[k]).strip()}")


if __name__ == "__main__":
    main()
