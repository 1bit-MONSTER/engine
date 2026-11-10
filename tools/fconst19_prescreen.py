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
# SPDX-License-Identifier: Apache-2.0
"""Pre-screen a 1.9 pack + its 1.8 twin for the twin-diff route WITHOUT downloading either pack.

Why: `tools/fconst19_measure.py` aligns the two packs' first block, and that needs the 1.8 twin's Token `.fconst` to be the same
layout minus the 1.9 headers. On amd/Phi-4-mini-instruct that premise failed (1.8 has 0 occurrences of the per-head header
`00 00 7f ff`, the 1.9 has 768, and past the header 751,858 of the first 1,000,000 B differ) -- discovered only after a 9.5 GB
download. The first block of the header is 8.7 MB into the file, so a range request for the first ~12 MB answers the same
question for a few MB of traffic.

What it reports, per pack: the 1.9 side's header count in the fetched window (should be `layers` if the window covers one block),
whether the 1.8 side has any header at all, and the first differing offset between the two windows. It does NOT measure a rule --
it says whether measuring one is worth the download.

Usage:  ./fconst19-prescreen.py amd/Model_rai_1.9.0_npu amd/Model_rai_1.8.0_npu_4K
"""
import json
import os
import re
import sys
import urllib.request

HDR = b"\x00\x00\x7f\xff"
WINDOW = 12 * 1024 * 1024


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = os.path.expanduser("~/.cache/huggingface/token")
    return open(path).read().strip() if os.path.exists(path) else ""


def api(repo, path=""):
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}{path}",
                                 headers={"Authorization": f"Bearer {token()}"})
    return json.load(urllib.request.urlopen(req, timeout=90))


def token_fconst(repo):
    files = [i["path"] for i in api(repo, "/tree/main") if i["type"] == "file"]
    cand = [f for f in files if f.startswith("dd_metastate_Llm_Token_") and f.endswith(".fconst") and "_Prefill_" not in f]
    exact = [f for f in cand if f.startswith("dd_metastate_Llm_Token_Token_")]
    return (exact or cand or [None])[0]


def head_bytes(repo, name, n=WINDOW):
    """First `n` bytes of a file, by range request."""
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}", "Range": f"bytes=0-{n-1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def geometry(repo):
    for fn in ("config.json", "model_config.json"):
        try:
            c = api(repo, f"/resolve/main/{fn}")
        except Exception:
            continue
        c = c.get("text_config") or c
        if all(k in c for k in ("hidden_size", "num_attention_heads", "num_hidden_layers")):
            return int(c["hidden_size"]), int(c["num_attention_heads"]), int(c["num_hidden_layers"])
    return None, None, None


def main():
    """Decide whether a twin-diff is plausible, by CONTENT past the 1.9 header -- not by counting the twin's headers.

    Two corrections from review are baked in here:
      1. a 1.8 Token .fconst never carries the per-head header (the 1.9 block is the 1.8 block PLUS it), so "the twin has no
         markers" is the normal case, not a bad twin. The first version treated it as fatal and rejected every pair,
         including amd/Qwen2.5_3B_Instruct's, which is already measured into the table as {2048, 16, 36, ...} (#1774);
      2. the head count is not read from a config (that returned None and made the fallback mis-align everything) -- it is
         found by asking which header length makes the two files agree past the header.
    """
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    r19, r18 = sys.argv[1], sys.argv[2]
    try:
        f19, f18 = token_fconst(r19), token_fconst(r18)
    except Exception as e:
        print(f"{r19}: cannot list files ({e}) -- check the twin's exact repo name")
        return 1
    if not f19 or not f18:
        print(f"{r19}: missing Token .fconst ({f19}, {f18})")
        return 1
    win, at, b, a = WINDOW, -1, b"", b""
    while win <= 48 * 1024 * 1024:
        try:
            b = head_bytes(r19, f19, win)
            a = head_bytes(r18, f18, win)
        except Exception as e:
            print(f"  fetch failed at {win // 1048576} MB: {e}")
            return 1
        at = b.find(HDR * 2) if at < 0 else at     # any run of at least two markers is a header
        if at >= 0:
            break
        win *= 2
    if at < 0:
        print(f"  the 1.9 window ({win // 1048576} MB) contains no header markers: fetch more before judging")
        return 0
    n = min(1_000_000, len(a) - at, len(b) - at - 512)
    if n <= 0:
        print("  the windows are too small to compare past the first header")
        return 0
    best = None
    for heads in range(1, 129):
        hdr_len = heads * 4
        if at + hdr_len + n > len(b):
            continue
        step = max(1, n // 20000)
        idx = range(0, n, step)
        same = sum(1 for i in idx if a[at + i] == b[at + hdr_len + i])
        agree = same / len(range(0, n, step))
        if best is None or agree > best[0]:
            best = (agree, heads, hdr_len)
    agree, heads, hdr_len = best
    print(f"{r19.split('/')[-1][:44]:46s}")
    print(f"  1.9 {f19[:52]:54s} first header at {at:,} in the first {win // 1048576} MB")
    print(f"  1.8 {f18[:52]:54s} best alignment at heads={heads} ({hdr_len} B header): {100.0 * agree:.2f} % of the "
          f"{n:,} sampled bytes past the header identical")
    if agree > 0.99:
        print("  -> the twin lines up past the header: a twin-diff measurement is plausible. Run "
              "tools/fconst19_measure.py on the full packs.")
    else:
        print("  -> the twin does NOT line up past the header: its payload is a different division or rotation, so the "
              "twin-diff route cannot produce this row. It needs the geometry-agnostic path (engine#1740 W1a) or another "
              "1.8 reference. Do not spend the download.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
