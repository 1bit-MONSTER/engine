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
# reader sees it and a check stays green only while the debt is named.
# Field -> the issue that owns it, for a field that is genuinely accepted and discarded. An entry here is a claim
# that the field is still ignored: when the field starts being read, this dict is out of date and the census fails
# until the entry is removed (engine#2889 made `cache_prompt` read on 11-20, which is how the first version of this
# list rotted within the hour it was written).
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
        if in_route["refused"] and in_route["read"]:
            # id_slot: honoured when it names the one slot this route has, refused by name otherwise
            out[key] = "read-and-refused"
        elif in_route["refused"]:
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


def stale_entries(result):
    """Exemptions and read-elsewhere notes that no longer describe the source."""
    # only keys this run actually censused: an exemption for a field outside --fields is not evidence of anything
    stale = [f"{k} is now {result[k]}, so its exemption is dead" for k in UNOWNED
             if k in result and result[k] != "ignored"]
    stale += [f"{k} is now {result[k]}, so 'read-elsewhere' is out of date" for k in VIA_OTHER
              if k in result and result[k] != "read-elsewhere"]
    return stale


def report(result):
    for key in FIELDS:
        state = result[key]
        note = ""
        if state == "read-elsewhere":
            note = f"  ({VIA_OTHER[key]})"
        elif state == "ignored" and key in UNOWNED:
            note = f"  (refused, with a clear message, is what 'fixed' looks like -- {UNOWNED[key]})"
        print(f"{key:22s} {state}{note}")


def check(root):
    route, options = pathlib.Path(root) / ROUTE, pathlib.Path(root) / OPTIONS
    missing = [str(p) for p in (route, options) if not p.is_file()]
    if missing:
        print(f"serve_field_census: cannot tell: {', '.join(missing)} not found", file=sys.stderr)
        return 2
    result = census(route.read_text(errors="replace"), options.read_text(errors="replace"))
    report(result)
    escaped = [k for k, v in result.items() if v == "ignored" and k not in UNOWNED]
    if escaped:
        print(f"serve_field_census: {len(escaped)} field(s) accepted and discarded: {', '.join(escaped)}", file=sys.stderr)
        return 1
    # the other direction: an exemption that is no longer needed is debt of its own, and a list that only grows is how
    # a census stops describing the code it reads.
    stale = stale_entries(result)
    if stale:
        for line in stale:
            print(f"serve_field_census: {line} -- remove the entry", file=sys.stderr)
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
    both = 'if (body.contains("id_slot") && at("id_slot") != 0) throw std::runtime_error("id_slot: one slot");\n'
    arm(census(both, "", ("id_slot",))["id_slot"] == "read-and-refused",
        "a key that is read for one value and refused for another says both")
    arm(census('const auto& v = body.at("stop");', "", ("stop",))["stop"] == "read-in-route",
        "body.at() is a read, not a mention")
    arm(census('o.sample.top_k = int(k);', "", ("top_k",))["top_k"] == "ignored",
        "a bare assignment is not a read of the request's key")

    arm(census("", "", ("anything",))["anything"] == "ignored", "an empty route ignores everything")

    # both directions: an exemption for a field that is still ignored is fine, one for a field that is now read is not
    saved_unowned, saved_via = dict(UNOWNED), dict(VIA_OTHER)
    options_topk = options + 'if (body.contains("top_k")) { o.sample.top_k = 1; }\n'
    try:
        UNOWNED.update({"top_k": "engine#0 (fixture)"})
        arm(stale_entries(census("", "", ("top_k",))) == [],
            "an exemption for a field that is still ignored is not stale")
        arm(stale_entries(census(route, options_topk, ("top_k",))) != [],
            "an exemption for a field the options header now reads IS stale (the engine#2889 case, in miniature)")
        arm(stale_entries(census(route, options_topk, ("temperature",))) == [],
            "a field the run did not census is not evidence that its exemption is stale")
        VIA_OTHER.update({"temperature": "engine#0 (fixture)"})
        arm(stale_entries(census(route, options, ("temperature",))) != [],
            "a read-elsewhere note for a field that is read normally is stale too")
    finally:
        UNOWNED.clear()
        UNOWNED.update(saved_unowned)
        VIA_OTHER.clear()
        VIA_OTHER.update(saved_via)
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
