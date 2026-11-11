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
"""Metrics from `1bit route --eval --per-case` files, the same definitions as laya/eval.cpp: accuracy, mean top p,
ECE over 10 bins of top p, log loss with p floored at 1e-9, decision ms p50."""
import json, math, sys, statistics as st
def m(path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows); ok = sum(r["chosen"] == r["label"] for r in rows)
    top = [max(r["probabilities"].values()) for r in rows]
    ll = -sum(math.log(max(1e-9, r["probabilities"][r["label"]])) for r in rows) / n
    bins = [[] for _ in range(10)]
    for r, p in zip(rows, top):
        bins[min(9, int(p * 10))].append((p, r["chosen"] == r["label"]))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(c for _, c in b) / len(b)) for b in bins if b)
    return {"n": n, "correct": ok, "acc": round(ok / n, 4), "mean_top_p": round(sum(top) / n, 4), "ece": round(ece, 4),
            "log_loss": round(ll, 4), "min_p_true": round(min(r["probabilities"][r["label"]] for r in rows), 4),
            "ms_p50": round(st.median(r["decision_ms"] for r in rows))}
if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, m(p))
