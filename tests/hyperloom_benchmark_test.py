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
"""The long-context leg of the Hyperloom custom workload (engine#679).

#679 is a decode rate at 18.8k context, so the workload needs a way to score *that* number
instead of the short-prompt geometric mean, without changing what the short legs report and
without letting the long leg's text leak into the correctness gate. This checks the three
pieces that can be checked without a GPU: the filler prompt is long enough and deterministic,
the score switches to the long-context rate only when the leg runs, and an unmeasured long
leg scores zero rather than silently falling back to the geomean.

Run directly (`python3 tests/hyperloom_benchmark_test.py`) or through ctest.
"""
import importlib.util
import pathlib
import sys

BENCH = pathlib.Path(__file__).resolve().parents[1] / "tools/hyperloom/custom/benchmark.py"


def load():
    spec = importlib.util.spec_from_file_location("hl_benchmark", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    bench = load()

    # The filler prompt: measured at ~4.85 characters per token on this box (a 75,200-character prompt
    # came back as 15,492 prompt tokens), so 18,800 tokens needs >= 18,800 * 4.85 characters, and the
    # same request twice has to be byte-identical or candidates are not comparable.
    prompt = bench.long_prompt(18800)
    assert len(prompt) >= 18800 * 4.85, len(prompt)
    assert prompt == bench.long_prompt(18800)
    assert len(bench.long_prompt(512)) < len(prompt)

    # Score: the long leg replaces the geomean when it ran...
    tps = {"code": 28.0, "prose": 16.0, "short": 17.0, "long": 12.67}
    assert bench.score_of(400.0, 12.0, tps, 18800) == 12.67
    # ...and an absent long measurement is a zero, not a quiet fallback to the short legs.
    assert bench.score_of(400.0, 12.0, {"code": 28.0}, 18800) == 0.0
    # Without the leg the metric is unchanged: geomean of pp512, tg128, serve code.
    geomean = bench.score_of(400.0, 12.0, {"code": 28.0}, 0)
    assert abs(geomean - (400.0 * 12.0 * 28.0) ** (1 / 3)) < 1e-9, geomean
    assert bench.geometric_mean(400.0, 12.0, 28.0) == geomean

    # The gate compares the short legs only. A reference that carries a "long" reply — which happens when
    # the reference is established by a long-context run — must not gate anyone on it, not even a short run
    # that never produces that key. (pi-prs review on engine#769, 2026-11-01.)
    assert bench.gated_text_keys({"code": "a", "prose": "b", "short": "c", "long": "d"}) == [
        "code", "prose", "short"]
    assert bench.gated_text_keys({"code": "a"}) == ["code"]
    assert bench.gated_text_keys({}) == []

    # The reviewer's exact scenario, as the gate line itself runs it: a reference carrying "long" and a run
    # that only produced the short legs must compare clean, not fail on a missing key.
    ref_texts = {"code": "a", "prose": "b", "short": "c", "long": "d"}
    texts = {"code": "a", "prose": "b", "short": "c"}
    assert [k for k in bench.gated_text_keys(ref_texts) if texts.get(k) != ref_texts[k]] == []
    # And a short-leg difference is still caught.
    texts["prose"] = "B"
    assert [k for k in bench.gated_text_keys(ref_texts) if texts.get(k) != ref_texts[k]] == ["prose"]

    print("hyperloom_benchmark_test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
