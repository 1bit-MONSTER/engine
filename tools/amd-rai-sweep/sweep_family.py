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

Sweep one family of AMD's public ONNX packs: download only what the row needs, run it, write the row, delete.

usage: sweep_family.py whisper REPO [REPO ...] [--cache PATH] [--rows PATH]
       sweep_family.py image   REPO [REPO ...] --src=512x512.png [--cache PATH] [--rows PATH]

The row goes into the `result:<repo>` entry of the board cache that `tools/amd-rai-board.py` renders
(status / numbers / note), and into an optional rows fragment so a concurrent writer cannot lose it.

Rules this driver enforces, each learned from a real false result:
  * one download at a time, and refuse to start below a free-space floor;
  * fetch `<graph>.onnx.data` whenever the repo has one - a graph whose external data is missing still opens
    and runs with zero-filled weights, which produced a "loads-wrong-output" row for a model that was fine;
  * pick the Whisper reference tokenizer by the pack's own name (English-only vocabularies shift the style ids);
  * no row is written from a run that did not produce output, and nothing is inferred into the cache.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import hf_hub_download, list_repo_files

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
FLOOR_GB = 150.0                      # stop before the disk is a problem for everyone else
IMG_PRE = {"realesrgan": "no255", "sesr": "no255", "psfrgan": "no255", "hrnet": "imagenet"}


def free_gb():
    st = os.statvfs(os.path.expanduser("~"))
    return st.f_bavail * st.f_frsize / 1e9


def merge_rows(rows, cache_path, rows_path):
    if rows_path:
        frag = json.load(open(rows_path)) if os.path.exists(rows_path) else {}
        frag.update(rows)
        json.dump(frag, open(rows_path, "w"), indent=1)
    if not cache_path:
        return True
    for _ in range(3):                # the cache is shared with other sweepers: verify after writing
        d = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        d.update(rows)
        json.dump(d, open(cache_path, "w"))
        back = json.load(open(cache_path))
        if all(k in back for k in rows):
            return True
    return False


def workdir():
    """A temp dir on the REAL disk, not on the system temp.

    /tmp on this fleet is a 62 GB tmpfs that other lanes fill with build artifacts (56 GB used / 5 GB free on
    2026-11-08), and an AMD Whisper pack is up to 7 GB: a download into tempfile.mkdtemp() died with ENOSPC at ~5 GB
    and the driver reported it as a task error rather than "the box has no room where you put it". Cache dirs live
    under $HOME for exactly this reason; honour TMPDIR when it is set to something else.
    """
    base = os.environ.get("AMD_RAI_SWEEP_TMP") or os.path.join(os.path.expanduser("~"), ".cache", "amd-rai-sweep")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="pull-", dir=base)


def fetch_with_data(repo, name, tmp, names):
    """Download an .onnx AND its .onnx.data sidecar if the repo has one (see the module docstring)."""
    path = hf_hub_download(repo, name, local_dir=tmp)
    if f"{name}.data" in names:
        hf_hub_download(repo, f"{name}.data", local_dir=tmp)
    return path


def sweep_whisper(repo, rows, cache_path, rows_path, wav, expect):
    names = list_repo_files(repo)
    graphs = [n for n in names if n.endswith(".onnx") and not n.endswith(".onnx.data")]
    enc = next((f for f in graphs if "encoder" in os.path.basename(f)), None)
    dec = next((f for f in graphs if "decoder" in os.path.basename(f)), None)
    if not enc or not dec:
        rows[f"result:{repo}"] = {"status": "fails", "numbers": "",
                                  "note": f"no ONNX encoder/decoder pair (graphs: {graphs}) - nothing runs outside AMD's .rai path"}
        return merge_rows(rows, cache_path, rows_path)
    tmp = workdir()
    try:
        pe = fetch_with_data(repo, enc, tmp, names)
        pd = fetch_with_data(repo, dec, tmp, names)
        # Only the English-only packs need an explicit tokenizer (their vocabularies shift every style id by one).
        # Everything else must let the runner pick the reference from the encoder's mel-bin count: large-v3/turbo are
        # 128 bins and tiny's tokenizer is a different vocabulary (51865 vs 51866), so forcing it here is wrong even
        # when the transcript happens to come out right.
        args = [PY, f"{HERE}/run_whisper_onnx.py", pe, pd, wav]
        tokenizer = "by mel bins"
        if "-en-" in repo:
            tokenizer = "openai/whisper-small.en"
            args.append(f"--tokenizer={tokenizer}")
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0 or not res.stdout.strip():
            # the runner's own stderr is the only thing that names a shape mismatch, a missing sidecar or a crash
            err = (res.stderr or "").strip().splitlines()
            raise RuntimeError(f"runner failed rc={res.returncode}: " + (err[-1] if err else "no output"))
        out = res.stdout.strip().splitlines()[-1]
        # the known answer is clip-specific: pass --expect for anything but the bundled jfk sample
        ok = "transcript=" in out and expect.lower() in out.lower()
        rows[f"result:{repo}"] = {"status": "OOB" if ok else "loads-wrong-output",
                                  "numbers": out.split(" tok=")[0].strip(),
                                  "note": f"ONNX encoder+decoder on the CPU EP ({tokenizer}); transcript "
                                          f"{'matches' if ok else 'did NOT match'} the known clip"}
        return merge_rows(rows, cache_path, rows_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sweep_image(repo, rows, cache_path, rows_path, src):
    names = list_repo_files(repo)
    graphs = [n for n in names if n.endswith(".onnx")]
    cand = [n for n in graphs if "fp32" in n] or graphs
    if not cand:
        rows[f"result:{repo}"] = {"status": "fails", "numbers": "",
                                  "note": f"no .onnx in the pack (files: {names[:6]}) - nothing runs outside AMD's compiled path"}
        return merge_rows(rows, cache_path, rows_path)
    pick = cand[0]
    pre = next((v for k, v in IMG_PRE.items() if k in repo.lower()), "no255")
    tmp = workdir()
    try:
        path = fetch_with_data(repo, pick, tmp, names)
        out = subprocess.run([PY, f"{HERE}/run_onnx_image.py", path, src, os.path.join(tmp, "out.png"),
                              "--pre", pre], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        rows[f"result:{repo}"] = {"status": "OOB",
                                  "numbers": out.split(" load_avg=")[0].strip(),
                                  "note": f"plain ONNX graph {os.path.basename(pick)} on the CPU EP with the repo's own "
                                          f"preprocessing ({pre}); a VitisAI cache in the pack is inert without that EP"}
        return merge_rows(rows, cache_path, rows_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=("whisper", "image"))
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--cache", help="board cache JSON (result:<repo> entries)")
    ap.add_argument("--rows", help="rows fragment JSON, so a concurrent writer cannot lose the row")
    ap.add_argument("--src", help="image family: a test PNG")
    ap.add_argument("--expect", default="fellow Americans",
                    help="whisper family: a phrase the transcript must contain (default: the jfk clip's)")
    ap.add_argument("--wav", default=os.path.join(tempfile.gettempdir(), "amd-rai-sweep-jfk.wav"),
                    help="whisper family: 16 kHz mono wav with a known transcript "
                         "(defaults to a temp path; the known clip is fetched there if absent)")
    a = ap.parse_args()
    if a.family == "image" and not a.src:
        sys.exit("image family needs --src=<png>")
    if free_gb() < FLOOR_GB:
        sys.exit(f"refusing: {free_gb():.0f} GB free < {FLOOR_GB:.0f} GB floor")
    rows = {}
    for repo in a.repos:
        print(f"== {repo}")
        try:
            if a.family == "whisper":
                if not os.path.exists(a.wav):
                    subprocess.run(["curl", "-sL", "-o", a.wav,
                                    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"],
                                   check=True)
                done = sweep_whisper(repo, rows, a.cache, a.rows, a.wav, a.expect)
            else:
                done = sweep_image(repo, rows, a.cache, a.rows, a.src)
        except Exception as e:                          # noqa: BLE001 - a sweep reports and continues
            print(f"   ERROR {type(e).__name__}: {e}")
            done = False
        print("   row written" if done else "   NOT written")
    print(f"free after: {free_gb():.0f} GB")


if __name__ == "__main__":
    main()
