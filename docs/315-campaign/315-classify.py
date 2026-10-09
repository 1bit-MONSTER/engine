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
import json, re, os, glob
def arm(t):
    O = glob.glob(os.path.expanduser(f"~/wt/rt-det/{t}-*"))
    if not O: return None
    O = O[0]; d = {}
    try: d = json.load(open(f"{O}/detector.json"))
    except Exception: pass
    log = ""
    try: log = open(f"{O}/server.log", errors="replace").read()
    except Exception: pass
    g = lambda k: int(d.get(k) or 0)
    nan = len(re.findall(r"HRX returned NaN logits", log))
    hsa = len(re.findall(r"HSA_STATUS_ERROR|Queue error|wait for HRX graph replay commands failed", log))
    # Face B as pre-registered: invalid-UTF-8 output surfacing as HTTP 500 / parse failures
    five = len(re.findall(r'"code":500', log))
    unr  = len(re.findall(r"unparsed Content-only output|does not match the expected", log))
    noncoh = len(re.findall(r"\[hrx-buf\][^\n]*host_visible=1[^\n]*direct_host_binding=0", log))
    f = dict(nan=nan, hsa=hsa, utf8=g("utf8_failures"), parse=g("parse_failures"),
             http=g("http_failures"), tr=g("transport_failures"), req=g("requests"),
             ok=g("ok"), s500=five, unparsed=unr, hostvis_noncoh=noncoh)
    v = "clean"
    if any(f[k] for k in ("nan","hsa","utf8","parse","http","tr","s500","unparsed")): v = "FAULT"
    elif f["req"] < 66: v = "INCOMPLETE"
    return v, f
print(f"{'arm':<14}{'verdict':<11}{'nan':>4}{'hsa':>4}{'utf8':>5}{'parse':>6}{'http':>5}{'tr':>4}{'500':>4}{'unparsed':>9}{'noncoh':>7}{'req':>5}")
res = {}
for t in [f"315diff2-{c}{i}" for i in range(1,7) for c in "du"]:
    r = arm(t)
    if not r: continue
    v, f = r; res[t] = v
    print(f"{t:<14}{v:<11}{f['nan']:>4}{f['hsa']:>4}{f['utf8']:>5}{f['parse']:>6}{f['http']:>5}{f['tr']:>4}{f['s500']:>4}{f['unparsed']:>9}{f['hostvis_noncoh']:>7}{f['req']:>5}")
print("\npairs:")
for i in range(1,7):
    d, u = res.get(f"315diff2-d{i}"), res.get(f"315diff2-u{i}")
    if d or u: print(f"  PAIR {i}: default={d} unified={u}")
