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
"""Fine-tune the routing head over `1bit route --dump-encoder` output, and report before/after accuracy
(engine#1313, item 3).

The other two thirds of item 3 already exist: `--eval` prints accuracy, per-class recall/precision, the confusion
matrix, ECE and log loss over labelled cases, and `--dump-encoder` writes the head's own input. What was missing is
the middle: nothing took the dump and turned it into a head, so "before/after routing accuracy" had no "after".

`before` is the table's decision, read from each case's own `logits` in the dump (argmax) -- so the two arms are
scored on the same cases, the same encoder and the same class order, and the only thing that changes is the head.
`after` is a ridge regression from the pooled encoder state to the class one-hot, which is closed-form, deterministic
and needs no framework: on a few hundred cases with a few thousand input rows the normal equations are exact.

The split is by CASE, never by row: a case's encoder rows are all in one half, or a head could be scored on rows of a
case it was fitted on and the number would be in-sample while looking held out. `--holdout 0` fits and scores on
everything and says so (`in-sample`), because a gate must not read a number without knowing which it is.

Usage:
    tools/laya_head_finetune.py DUMP_DIR [--holdout F] [--ridge L] [--pool mean|last] [--json]
    tools/laya_head_finetune.py --self-test

The dump is what `1bit route --laya-models T --laya-model DIR --dump-encoder cases.jsonl --out DUMP` writes:
    DUMP/h.f32        float32 encoder rows, concatenated
    DUMP/cases.jsonl  one line per case: {"src", "label", "options", "offset", "len", "logits", ...}
    DUMP/dump.json    {"d": encoder width, "rows", "cases", ...}

A line whose `label` is -1 was written without a label; it is skipped, exactly as `--eval` skips it.
"""
import argparse
import json
import math
import os
import struct
import sys


def read_dump(root):
    with open(os.path.join(root, "dump.json")) as f:
        info = json.load(f)
    d = int(info["d"])
    cases = []
    with open(os.path.join(root, "cases.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    raw = open(os.path.join(root, "h.f32"), "rb").read()
    rows = len(raw) // (4 * d) if d else 0
    encoded = struct.unpack("<%df" % (rows * d), raw[: rows * d * 4])
    return d, encoded, cases


def pool_case(encoded, d, off, length, how):
    """One feature vector per case: the mean (or the last) encoder row."""
    if length <= 0:
        return None
    if how == "last":
        start = (off + length - 1) * d
        return list(encoded[start : start + d])
    acc = [0.0] * d
    for r in range(length):
        base = (off + r) * d
        for i in range(d):
            acc[i] += encoded[base + i]
    inv = 1.0 / length
    return [v * inv for v in acc]


def argmax(v):
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


def fit_ridge(xs, ys, k, lam):
    """Closed-form W (d x k) mapping features to one-hot, with a bias column. Solves (X'X + L)^-1 X'Y."""
    n = len(xs)
    d = len(xs[0])
    D = d + 1
    # A = X'X + lam*I (bias not regularised), B = X'Y
    A = [[0.0] * D for _ in range(D)]
    B = [[0.0] * k for _ in range(D)]
    for x, y in zip(xs, ys):
        xb = list(x) + [1.0]
        for i in range(D):
            xi = xb[i]
            if xi == 0.0:
                continue
            row = A[i]
            for j in range(D):
                row[j] += xi * xb[j]
            B[i][y] += xi
    for i in range(d):
        A[i][i] += lam
    # Gaussian elimination with partial pivoting on [A|B]
    M = [A[i] + B[i] for i in range(D)]
    for col in range(D):
        piv = max(range(col, D), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise SystemExit("laya_head_finetune: the design matrix is singular at column %d (raise --ridge)" % col)
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(D):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                Mr, Mc = M[r], M[col]
                for c in range(col, D + k):
                    Mr[c] -= f * Mc[c]
    # After elimination the top D rows are [I | A^-1 B], so the weights live in columns D..D+k-1.
    # Transposed to one row per class (each D long, bias last) so predict() is a plain dot product.
    return [[M[i][D + c] for i in range(D)] for c in range(k)]


def predict(W, x):
    xb = list(x) + [1.0]
    return [sum(w * v for w, v in zip(row, xb)) for row in W]


def evaluate(feats, cases, W, idxs, k):
    """(correct, total, confusion) for one arm; W is None for the table's own decision."""
    conf = [[0] * k for _ in range(k)]
    correct = 0
    for i in idxs:
        want = cases[i]["label"]
        got = argmax(cases[i]["logits"]) if W is None else argmax(predict(W, feats[i]))
        if 0 <= want < k and 0 <= got < k:
            conf[want][got] += 1
            correct += 1 if want == got else 0
    return correct, len(idxs), conf


def run(args):
    d, encoded, cases = read_dump(args.dump)
    usable, feats, labels = [], [], []
    for c in cases:
        if c.get("label", -1) < 0:
            continue
        f = pool_case(encoded, d, int(c["offset"]), int(c["len"]), args.pool)
        if f is None:
            continue
        usable.append(c)
        feats.append(f)
        labels.append(int(c["label"]))
    if not usable:
        raise SystemExit("laya_head_finetune: no labelled cases in the dump")
    k = max(len(usable[0]["options"]), max(labels) + 1)
    for c, y in zip(usable, labels):
        if not (0 <= y < len(c["options"])):
            raise SystemExit("laya_head_finetune: %s: label %d is outside its options" % (c.get("src", "?"), y))

    # Split by CASE and STRATIFIED BY LABEL, deterministically (no RNG: a re-run on one dump reports one number).
    #
    # Not by index. A decision log is written in arrival order, and a demo's arrival order is class-contiguous --
    # a session of reflex steps, then a planning stretch. An index split then holds out a whole CLASS rather than a
    # sample of each, and the "after" number is measured on classes the head never saw while the "before" arm (which
    # needs no training) scores them fine. My own self-test hit exactly this: 20+20+20 cases, holdout 0.5, train =
    # classes 0 and 1 only, so the ceiling for any head was 1/3. Stratifying by label is what makes "held out" mean
    # held out from the same population. With --holdout 0 nothing is held out and the report says in-sample.
    n = len(usable)
    by_label = {}
    for i, y in enumerate(labels):
        by_label.setdefault(y, []).append(i)
    train, test = [], []
    if args.holdout == 0:
        train = list(range(n))
        test = list(range(n))          # in-sample: fit and score on everything, and say so
    for y in (sorted(by_label) if args.holdout > 0 else []):
        idxs = by_label[y]
        hold = int(round(len(idxs) * args.holdout))
        if args.holdout > 0 and hold == 0:
            raise SystemExit("laya_head_finetune: class %d has %d case(s), too few for a held-out split; "
                             "add cases or use --holdout 0 (in-sample)" % (y, len(idxs)))
        test.extend(idxs[len(idxs) - hold:] if hold else [])
        train.extend(idxs[: len(idxs) - hold] if hold else idxs)
    if not train:
        raise SystemExit("laya_head_finetune: the holdout left no training cases (use a smaller --holdout)")

    W = fit_ridge([feats[i] for i in train], [labels[i] for i in train], k, args.ridge)
    b_correct, b_total, b_conf = evaluate(feats, usable, None, test, k)
    a_correct, a_total, a_conf = evaluate(feats, usable, W, test, k)
    b_acc = b_correct / b_total if b_total else 0.0
    a_acc = a_correct / a_total if a_total else 0.0
    out = {
        "cases": n,
        "train": len(train),
        "test": len(test),
        "holdout": args.holdout,
        "arm": "in-sample" if args.holdout == 0 else "held-out",
        "pool": args.pool,
        "classes": [c["options"] for c in usable[:1]][0],
        "before": {"correct": b_correct, "total": b_total, "accuracy": b_acc},
        "after": {"correct": a_correct, "total": a_total, "accuracy": a_acc},
        "delta": a_acc - b_acc,
        "confusion_before": b_conf,
        "confusion_after": a_conf,
    }
    return out


def self_test():
    """A dump can be synthesised exactly: rows in class-conditional clusters with a known boundary. The check is
    that the fine-tune LEARNS it (the table's logits, made deliberately uninformative, cannot), and that the split
    is by case -- a head fitted on a case's rows must not be scored on the same case."""
    import tempfile

    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails += 1

    rng = 12345

    def rnd():
        nonlocal rng
        rng = (1103515245 * rng + 12345) % (1 << 31)
        return rng / float(1 << 31)

    d, k, per_class, rows_per_case = 8, 3, 20, 3
    centres = [[0.0] * d for _ in range(k)]
    for c in range(k):
        for i in range(d):
            centres[c][i] = 4.0 if (i % k) == c else -4.0
    with tempfile.TemporaryDirectory() as tmp:
        h = open(os.path.join(tmp, "h.f32"), "wb")
        meta = open(os.path.join(tmp, "cases.jsonl"), "w")
        off = 0
        made = []
        for c in range(k):
            for _ in range(per_class):
                vals = []
                for _r in range(rows_per_case):
                    for i in range(d):
                        vals.append(centres[c][i] + (rnd() - 0.5))
                h.write(struct.pack("<%df" % len(vals), *vals))
                # Uninformative table logits: identical for every class, so "before" can only be class 0.
                meta.write(json.dumps({"src": "s%d" % len(made), "label": c, "options": ["a", "b", "c"][:k],
                                       "offset": off, "len": rows_per_case, "logits": [0.0] * k}) + "\n")
                off += rows_per_case
                made.append(c)
        h.close()
        meta.close()
        json.dump({"d": d, "rows": off, "cases": len(made)}, open(os.path.join(tmp, "dump.json"), "w"))

        class A:
            dump = tmp
            holdout = 0.5
            ridge = 1e-3
            pool = "mean"
            json = False

        out = run(A)
        check(out["arm"] == "held-out", "the default run is scored held-out, and says so")
        check(out["after"]["accuracy"] > out["before"]["accuracy"],
              "the fine-tuned head beats the table on the same held-out cases (%.2f vs %.2f)"
              % (out["after"]["accuracy"], out["before"]["accuracy"]))
        check(out["after"]["accuracy"] >= 0.9,
              "and it actually learns a separated boundary (%.2f)" % out["after"]["accuracy"])
        check(out["test"] + out["train"] == out["cases"], "every case lands in exactly one side of the split")

        # In-sample is reported as in-sample: same dump, holdout 0.
        A.holdout = 0.0
        ins = run(A)
        check(ins["arm"] == "in-sample", "holdout 0 is labelled in-sample rather than passing as a held-out number")
        check(ins["before"]["total"] == ins["cases"] and ins["test"] == ins["cases"],
              "in-sample scores every case (%d of %d)" % (ins["test"], ins["cases"]))

        # An unlabelled line is skipped, as `--eval` skips it.
        with open(os.path.join(tmp, "cases.jsonl"), "a") as f:
            f.write(json.dumps({"src": "unlabelled", "label": -1, "options": ["a", "b", "c"][:k],
                                "offset": 0, "len": rows_per_case, "logits": [0.0] * k}) + "\n")
        A.holdout = 0.5
        out2 = run(A)
        check(out2["cases"] == out["cases"], "a label -1 line is skipped, not counted as a class")

    if fails:
        print("laya_head_finetune: %d FAILED" % fails)
        return 1
    print("laya_head_finetune: all self-test arms passed")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--holdout", type=float, default=0.5, help="fraction of CASES held out (0 = in-sample)")
    ap.add_argument("--ridge", type=float, default=1e-3, help="ridge strength on the feature weights")
    ap.add_argument("--pool", choices=["mean", "last"], default="mean", help="how a case's encoder rows become one vector")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.dump:
        ap.error("a DUMP_DIR or --self-test is required")
    if not (0.0 <= args.holdout < 1.0):
        ap.error("--holdout must be in [0, 1)")
    out = run(args)
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    print("cases %d (%d train, %d held out)  arm=%s  pool=%s" %
          (out["cases"], out["train"], out["test"], out["arm"], out["pool"]))
    print("  before (the table): %3d/%3d  %.3f" % (out["before"]["correct"], out["before"]["total"], out["before"]["accuracy"]))
    print("  after  (fine-tune): %3d/%3d  %.3f   delta %+0.3f" %
          (out["after"]["correct"], out["after"]["total"], out["after"]["accuracy"], out["delta"]))
    for name, conf in (("before", out["confusion_before"]), ("after", out["confusion_after"])):
        print("  %s confusion (rows true, cols chosen):" % name)
        for r, row in enumerate(conf):
            print("    %d: %s" % (r, " ".join("%3d" % v for v in row)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
