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
"""KL divergence between two `--dump-logits` directories (engine#803 gate, second half).

`forward-serve --dump-logits <dir>` writes one float32, vocabulary-wide array per generated position as
`step_<response>_<n>.bin` -- the response index exists (engine#803) so that a two-prompt run cannot silently
compare only the last prompt. That is the distribution the P4 gate asks for: *logits KLD 0 against the baseline*,
which no correlation or max|d| number can answer, because two distributions can agree on their argmax and
disagree everywhere that matters to sampling.

This is the comparator, not the measurement: it needs no device and no model, so the gate becomes one command
when a ticket is free. Its thresholds are the HRX gate's own shape (`tools/hw_gate_manifest.py`: mean KLD and a
tail percentile), so the two gates cannot drift into different definitions of "the same".

    tools/logits_kld.py BASELINE_DIR CANDIDATE_DIR [--max-mean-kld 0.01] [--max-kld-p99 0.2]
    tools/logits_kld.py --self-test
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

STEP_RE = re.compile(r"^step_(\d+)_(\d+)\.bin$")


class Bad(Exception):
    """A refusal, not a crash: the message names the file and what is wrong with it."""


def load_steps(directory: str) -> dict[tuple[int, int], list[float]]:
    """Every step_<response>_<n>.bin in `directory`, keyed by (response, step), as float32 values."""
    if not os.path.isdir(directory):
        raise Bad(f"{directory}: not a directory")
    import struct

    out: dict[tuple[int, int], list[float]] = {}
    for name in sorted(os.listdir(directory)):
        m = STEP_RE.match(name)
        if not m:
            continue
        path = os.path.join(directory, name)
        raw = open(path, "rb").read()
        if len(raw) % 4 != 0:
            raise Bad(f"{path}: {len(raw)} bytes is not a whole number of float32 values")
        n = len(raw) // 4
        out[(int(m.group(1)), int(m.group(2)))] = list(struct.unpack(f"<{n}f", raw))
    if not out:
        raise Bad(f"{directory}: no step_<response>_<n>.bin files; was --dump-logits given this directory?")
    return out


def compare_keys(base: dict, cand: dict) -> list[tuple[int, int]]:
    """The matched keys, refusing any asymmetry.

    A candidate that holds FEWER steps is not a smaller comparison, it is a different one -- the code comment in
    app/forward_serve.cpp says the file set is meant to say how many responses a run holds, so a missing file is
    the failure this refuses rather than an average over whatever survived.
    """
    missing = sorted(set(base) - set(cand))
    extra = sorted(set(cand) - set(base))
    if missing:
        raise Bad("candidate is missing " + ", ".join(f"step_{r}_{n}.bin" for r, n in missing[:4]) +
                  ("" if len(missing) <= 4 else f" (and {len(missing) - 4} more)"))
    if extra:
        raise Bad("candidate has " + ", ".join(f"step_{r}_{n}.bin" for r, n in extra[:4]) +
                  ("" if len(extra) <= 4 else f" (and {len(extra) - 4} more)") + " that the baseline does not")
    return sorted(base)


def log_softmax(x: list[float]) -> list[float]:
    """log softmax in float64 with the max subtracted, so a large logit cannot overflow the exponent."""
    m = max(x)
    ex = [math.exp(v - m) for v in x]
    s = sum(ex)
    return [v - m - math.log(s) for v in x]


def kl(p_logits: list[float], q_logits: list[float]) -> float:
    """KL(p || q) with p as the reference: the candidate's mass that the baseline would not have put there."""
    lp = log_softmax(p_logits)
    lq = log_softmax(q_logits)
    total = 0.0
    for a, b in zip(lp, lq):
        pr = math.exp(a)
        if pr > 0.0:
            total += pr * (a - b)
    return total


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", nargs="?", help="the reference --dump-logits directory")
    ap.add_argument("candidate", nargs="?", help="the directory under test")
    ap.add_argument("--max-mean-kld", type=float, default=0.01,
                    help="fail above this mean over positions (the HRX gate's own default)")
    ap.add_argument("--max-kld-p99", type=float, default=0.2, help="fail above this 99th percentile")
    ap.add_argument("--self-test", action="store_true", help="run the built-in cases and exit")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    if not a.baseline or not a.candidate:
        print("usage: logits_kld.py BASELINE_DIR CANDIDATE_DIR [--max-mean-kld X] [--max-kld-p99 Y]",
              file=sys.stderr)
        return 2
    try:
        base = load_steps(a.baseline)
        cand = load_steps(a.candidate)
        keys = compare_keys(base, cand)
        for k in keys:
            if len(base[k]) != len(cand[k]):
                raise Bad(f"step_{k[0]}_{k[1]}.bin: baseline is {len(base[k])} wide, candidate {len(cand[k])}")
    except Bad as e:
        print(f"logits_kld: {e}", file=sys.stderr)
        return 2

    klds = [kl(base[k], cand[k]) for k in keys]
    mean = sum(klds) / len(klds)
    ordered = sorted(klds)
    p99 = ordered[min(len(ordered) - 1, int(math.ceil(0.99 * len(ordered))) - 1)]
    print(f"logits_kld: {len(keys)} position(s); mean KLD {mean:.6f}; p99 {p99:.6f}; "
          f"max KLD {ordered[-1]:.6f} (step_{keys[klds.index(ordered[-1])][0]}_"
          f"{keys[klds.index(ordered[-1])][1]})")
    over = []
    if mean > a.max_mean_kld:
        over.append(f"mean {mean:.6f} > {a.max_mean_kld}")
    if p99 > a.max_kld_p99:
        over.append(f"p99 {p99:.6f} > {a.max_kld_p99}")
    if over:
        print("logits_kld: FAIL " + "; ".join(over), file=sys.stderr)
        return 1
    print("logits_kld: PASS")
    return 0


def _write(directory: str, steps: dict[tuple[int, int], list[float]]) -> None:
    import struct

    os.makedirs(directory, exist_ok=True)
    for (r, n), vals in steps.items():
        with open(os.path.join(directory, f"step_{r}_{n}.bin"), "wb") as f:
            f.write(struct.pack(f"<{len(vals)}f", *vals))


def self_test() -> int:
    """The four cases a KLD comparator can get wrong, each asserted rather than described."""
    import tempfile

    failures = 0

    def check(ok: bool, what: str) -> None:
        nonlocal failures
        print(f"{'ok  ' if ok else 'FAIL'}   {what}")
        if not ok:
            failures += 1

    with tempfile.TemporaryDirectory() as d:
        # 1. identical dumps are exactly 0 -- not "small": the gate asks for 0, so this is the assertion that
        #    decides whether the arithmetic has a bias (a summed KL over float32 noise would show up here).
        a = {(1, 0): [0.5, -1.0, 2.0, 0.0], (1, 1): [1.0, 1.0, 1.0, 1.0]}
        _write(os.path.join(d, "a"), a)
        _write(os.path.join(d, "b"), a)
        base, cand = load_steps(os.path.join(d, "a")), load_steps(os.path.join(d, "b"))
        ks = compare_keys(base, cand)
        v = [kl(base[k], cand[k]) for k in ks]
        check(all(x == 0.0 for x in v), "identical dumps give exactly 0.0 KLD (not merely small)")

        # 2. a shifted distribution is positive, and in the expected direction: KL is not symmetric, and the
        #    reference is the baseline, so swapping the arguments must change the number.
        b = {(1, 0): [3.0, -1.0, 2.0, 0.0], (1, 1): [1.0, 1.0, 1.0, 1.0]}
        _write(os.path.join(d, "c"), b)
        cand = load_steps(os.path.join(d, "c"))
        k0 = ks[0]
        check(kl(base[k0], cand[k0]) > 0.0, "a shifted distribution gives a positive KLD")
        check(kl(base[k0], cand[k0]) != kl(cand[k0], base[k0]),
              "KL is directional: swapping baseline and candidate changes the number")

        # 3. a UNIFORM shift is distribution-preserving: every logit +4 moves max|d| by 4 and the KLD stays
        #    exactly 0, because softmax(v + c) == softmax(v). This is the arm that documents what the gate
        #    measures -- the distribution, not the logits -- and it is the mirror of #2101's lesson that a
        #    correlation cannot see a scale: here a scale-free comparison cannot see a uniform shift, and that
        #    is correct rather than a blind spot.
        uni = {k: [x + 4.0 for x in v] for k, v in a.items()}
        _write(os.path.join(d, "u"), uni)
        cand_u = load_steps(os.path.join(d, "u"))
        check(all(kl(base[k], cand_u[k]) == 0.0 for k in ks),
              "a uniform +4 on every logit leaves KLD at exactly 0 (softmax is shift-invariant)")

        # 4. a missing step is refused by name, not averaged over what survived
        os.remove(os.path.join(d, "c", "step_1_1.bin"))
        try:
            compare_keys(base, load_steps(os.path.join(d, "c")))
            check(False, "a missing step is refused")
        except Bad as e:
            check("step_1_1.bin" in str(e), f"a missing step is refused by name ({e})")

        # 5. an extra step on the candidate side is refused too -- a candidate that ran longer did not measure
        #    the same prompt, and comparing only the overlap hides that.
        _write(os.path.join(d, "e"), {**a, (2, 0): [0.0, 0.0, 0.0, 0.0]})
        try:
            compare_keys(base, load_steps(os.path.join(d, "e")))
            check(False, "an extra step is refused")
        except Bad as e:
            check("step_2_0.bin" in str(e), f"an extra step is refused by name ({e})")

    if failures:
        print(f"logits_kld self-test: {failures} FAILED", file=sys.stderr)
        return 1
    print("logits_kld self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
