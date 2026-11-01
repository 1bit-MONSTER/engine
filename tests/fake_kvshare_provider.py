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
#
# fake_kvshare_provider.py: the reference provider's shape (llama.cpp tools/kvshare-prefill) without a
# llama_context, so the *plumbing* of the shared-KV handoff can be tested with no model and no dma-buf:
#
#   --tokens <ids file> --model <model> --layout <header the decoder sent> --fds <n[,n...]>
#
# It fails unless it was given a real, readable fd per region -- i.e. unless the engine really received the
# fds over the socket and handed them on through exec -- and it says how many bytes it could pread from each,
# which is what the test asserts on. Print one JSON line, exit 0, exactly like the real provider.
import json, os, sys

args = sys.argv[1:]
if "--tokens" not in args or "--layout" not in args or "--fds" not in args:
    print("fake_kvshare_provider: the shared route needs --tokens/--layout/--fds (got %s)" % (args,),
          file=sys.stderr)
    sys.exit(2)

opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
with open(opt["--tokens"]) as f:
    ids = [int(line) for line in f if line.strip()]
with open(opt["--layout"]) as f:
    header = json.load(f)
fds = [int(x) for x in opt["--fds"].split(",") if x]

regions = header.get("regions") or []
if len(fds) != len(regions):
    print("fake_kvshare_provider: %d fd(s) for %d region(s)" % (len(fds), len(regions)), file=sys.stderr)
    sys.exit(1)

# the provider's whole job here: the fds must be usable descriptors, in region order
readable = []
for fd, region in zip(fds, regions):
    try:
        data = os.pread(fd, min(16, int(region.get("size", 1))), int(region.get("offset", 0)))
    except OSError as e:
        print("fake_kvshare_provider: fd %d is not usable: %s" % (fd, e), file=sys.stderr)
        sys.exit(1)
    if not data:
        print("fake_kvshare_provider: fd %d read 0 bytes" % fd, file=sys.stderr)
        sys.exit(1)
    readable.append(len(data))

params = header.get("params") or {}
# serve reads our stdout for its numbers and does not echo it, and it deletes the ids file and the slot
# directory as soon as we exit -- so the proof of what we were handed goes where the test asks for it
marker = os.environ.get("FAKE_KVSHARE_MARKER")
if marker:
    with open(marker, "w") as f:
        json.dump({"fds": len(fds), "readable": readable, "n_ctx": params.get("n_ctx")}, f)
print(json.dumps({"n_tokens": len(ids), "prefill_ms": 1.0, "fds": len(fds), "readable": readable,
                  "n_ctx": params.get("n_ctx")}), flush=True)
