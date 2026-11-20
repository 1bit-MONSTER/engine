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
"""serve_field_census.py -- say what the in-process /v1/completions route does with each request field.

engine#2893 counted key-string mentions in `app/forward_serve.cpp`, read a zero as "the route cannot be reading
this key", and filed `temperature` and `top_k` as silently dropped. Both are read -- one call away, in
`app/npu_sampling.h:options_from_body()` -- and applied per token. A census that looks at one file cannot tell
"read elsewhere" from "ignored", so this one classifies every field across the three places a disposition lives:

  read-in-route    the route's own parse (app/forward_serve.cpp)
  read-in-options  npu::options_from_body() and friends (app/npu_sampling.h)
  refused          a named error: throw std::runtime_error("<key>: ...") on the route
  ignored          none of the above -- the state the issue exists for

Exit codes: 0 every field is read or refused, 1 a field is ignored and is not in UNOWNED, 2 cannot tell (a source
file is missing or unreadable). A field listed in UNOWNED with an issue reference is reported but not failed, so the
tool can land before the last field is fixed.
"""

import argparse
import pathlib
import re
import sys

# The keys the route is asked about: the OpenAI completion keys, llama-server's aliases, and the sampling knobs the
# route's own sampler takes. A key the route documents and this list lacks is the "sixth field" that keeps escaping
# hand-written censuses -- add it here when the route gains it.
FIELDS = (
    "prompt", "stream", "stop", "model", "max_tokens", "max_completion_tokens", "n_predict",
    "temperature", "top_k", "top_p", "seed", "logprobs", "top_logprobs", "json_object", "grammar",
    "cache_prompt", "id_slot",
)

# A field that is genuinely accepted and discarded, with the issue that owns the fix. Reported as a warning so a
# reader sees it and a check stays green only while the debt is named -- and reported the other way too, because an
# allowance whose debt is PAID is a lie that costs nothing to keep until someone needs UNOWNED to mean something.
# `cache_prompt` was the only entry; engine#2889 (merged 11-20 06:22Z) taught the route to read it, so the entry is
# gone and `stale_allowances()` is what stops the next one from outliving its fix.
UNOWNED = {}

# Keys that are deliberately not read as body keys, with what reads them instead. Each one is a decision, not an
# omission, and a reader can check the reason against the source.
VIA_OTHER = {
    "model": "a single-model server answers with its own id (`out[\"model\"] = e.id`); llama-server does the same",
    "json_object": "reached as a VALUE: `response_format.type == \"json_object\"` sets `opts.json_object`, "
                   "which the route consumes when it builds the grammar",
}

ROUTE = pathlib.Path("app/forward_serve.cpp")
OPTIONS = pathlib.Path("app/npu_sampling.h")


def mentions(text, key):
    """Places a key can be read or refused: a body reader, and a named refusal.

    The reader pattern has to allow arguments between the call and the literal -- the route nests them
    (`npu::int_field(body, "n_predict", npu::int_field(body, "max_tokens", 256))`), which is exactly how a
    pattern anchored on `_field("key"` reported `max_tokens` as unread on my first pass.
    """
    k = re.escape(key)
    return {
        "read": bool(re.search(r'\b(?:at|contains|value)\s*\([^)\n]*"\s*' + k + r'\s*"', text))
        or bool(re.search(r'\w*_field\s*\([^;\n]*"\s*' + k + r'\s*"', text))
        or bool(re.search(r'\bbody\s*\[\s*"' + k + r'"\s*\]', text)),
        "refused": bool(re.search(r'runtime_error\(\s*"' + k + r'\s*:', text)),
    }


def census(route_text, options_text, fields=FIELDS):
    out = {}
    for key in fields:
        in_route = mentions(route_text, key)
        in_options = mentions(options_text, key)
        if in_route["refused"]:
            out[key] = "refused"
        elif in_route["read"]:
            out[key] = "read-in-route"
        elif in_options["read"]:
            out[key] = "read-in-options"
        elif key in VIA_OTHER:
            out[key] = "read-elsewhere"
        else:
            out[key] = "ignored"
    return out


def report(result):
    for key in FIELDS:
        state = result[key]
        note = ""
        if state == "read-elsewhere":
            note = f"  ({VIA_OTHER[key]})"
        elif state == "ignored" and key in UNOWNED:
            note = f"  (refused, with a clear message, is what 'fixed' looks like -- {UNOWNED[key]})"
        print(f"{key:22s} {state}{note}")


def stale_allowances(result):
    """UNOWNED keys whose field is no longer `ignored`: the debt was paid and the entry is now a fiction."""
    return [k for k in UNOWNED if result.get(k) != "ignored"]


def check(root):
    route, options = pathlib.Path(root) / ROUTE, pathlib.Path(root) / OPTIONS
    missing = [str(p) for p in (route, options) if not p.is_file()]
    if missing:
        print(f"serve_field_census: cannot tell: {', '.join(missing)} not found", file=sys.stderr)
        return 2
    result = census(route.read_text(errors="replace"), options.read_text(errors="replace"))
    report(result)
    stale = stale_allowances(result)
    for key in stale:
        print(f"serve_field_census: the allowance for {key} is no longer needed -- the field is {result[key]} on this "
              f"tree, so remove it from UNOWNED ({UNOWNED[key]})", file=sys.stderr)
    if stale:
        return 1
    escaped = [k for k, v in result.items() if v == "ignored" and k not in UNOWNED]
    if escaped:
        print(f"serve_field_census: {len(escaped)} field(s) accepted and discarded: {', '.join(escaped)}", file=sys.stderr)
        return 1
    return 0


def self_test():
    """Arms: each disposition, and the mistake the issue made -- a key read only in the options header."""
    bad = 0

    def arm(cond, what):
        nonlocal bad
        if not cond:
            print(f"FAIL {what}")
            bad += 1

    route = 'int n = npu::int_field(body, "max_tokens", 256);\nif (body.contains("stream")) {}\n'
    options = 'if (body.contains("temperature")) { o.sample.temperature = 1.0f; }\n'

    result = census(route, options, ("max_tokens", "stream", "temperature", "top_k"))
    arm(result["max_tokens"] == "read-in-route", "a key the route reads is read-in-route")
    arm(result["temperature"] == "read-in-options",
        "a key the route does not name but the options header reads is NOT ignored -- the engine#2893 mistake")
    arm(result["top_k"] == "ignored", "a key nobody reads is ignored")

    refused = 'if (!chat) throw std::runtime_error("logprobs: served on /v1/chat/completions, only");\n'
    arm(census(refused, "", ("logprobs",))["logprobs"] == "refused", "a named refusal counts as a disposition")
    arm(census('const auto& v = body.at("stop");', "", ("stop",))["stop"] == "read-in-route",
        "body.at() is a read, not a mention")
    arm(census('o.sample.top_k = int(k);', "", ("top_k",))["top_k"] == "ignored",
        "a bare assignment is not a read of the request's key")

    arm(census("", "", ("anything",))["anything"] == "ignored", "an empty route ignores everything")

    # The allowance mechanism (engine#2889): an entry is for a field that is `ignored`, and it must be reported once
    # the field is read -- otherwise UNOWNED can only grow, and the next entry inherits a precedent that never expires.
    saved = dict(UNOWNED)
    try:
        UNOWNED.clear()
        UNOWNED["paid"] = "engine#0000 (a debt that was paid)"
        arm(stale_allowances({"paid": "read-in-route"}) == ["paid"],
            "an allowance whose field is now read is reported stale")
        arm(stale_allowances({"paid": "ignored"}) == [], "...and one still ignored is not")
        arm(stale_allowances({"paid": "refused"}) == ["paid"],
            "a field that became a named refusal also pays the debt")
    finally:
        UNOWNED.clear()
        UNOWNED.update(saved)

    print("ok" if bad == 0 else f"{bad} FAIL")
    return 1 if bad else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    return self_test() if a.self_test else check(a.root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
