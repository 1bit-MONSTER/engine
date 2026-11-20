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

"""serve_routes_check.py -- every path a warm-up posts to is a route the server registers.

engine#2853: `warmup_requests()` posted the short and the spec warm-up to `/completion`, and `1bit serve` has never
registered that path -- it registered `/v1/chat/completions` and `/v1/completions`. Both requests answered 404, so
the warm-up `docs/serve.md` promises moved nothing, and nothing in the suite could notice: a warm-up is a POST the
server itself makes, so a wrong path is a 404 nobody reads.

The check is static on purpose: the route table and the warm-up list are two literals in one file, and the failure is
their disagreement. It parses `srv.Post("<path>"` registrations and `{ "path", "<path>" }` warm-up entries out of
`app/serve.cpp` and requires the second set to be a subset of the first.

engine#2884: the path is only half the request. A warm-up is a request the server makes to ITSELF, so a field the
parser does not read is as silent as a 404 -- the short and spec warm-ups set llama.cpp's `n_predict`, the parser read
`max_completion_tokens`/`max_tokens`, and both requests generated 256 tokens while answering 200. So the same check
also requires the token limit a warm-up body sets to be a field the request parser actually reads.

Exit: 0 every warm-up path is registered and its token limit is read, 1 one is not (named), 77 a literal could not be
read.
"""

import pathlib
import re
import sys

SOURCE = pathlib.Path("app/serve.cpp")
# A route is registered with srv.Get or srv.Post, as a plain literal or as a raw-string regex (`R"(/slots/(\\d+))"`).
# The regex form is why `warmup_route_check.py` existed as a second check; both halves live here now, so one file answers
# "is every warm-up path a route this server registers" and no reviewer has to keep two baselines in step.
ROUTE = re.compile(r'srv\.(?:Get|Post)\(\s*(?:R"\((.*?)\)"|"([^"]+)")\s*,')
WARMUP = re.compile(r'\{\s*"path"\s*,\s*"([^"]+)"')
# The token-limit fields the warm-up bodies set, and the ones the request parser reads (forward_serve.cpp).
# Both shapes the builders use: `{ "n_predict", 0 }` in an initializer list and `body["n_predict"] = 2`.
BODY_LIMIT = re.compile(r'"(n_predict|max_tokens|max_completion_tokens)"\s*(?:[:,]|\])')
# Per builder, not the union: the chat warm-up sets `max_tokens`, so a union hides that the short and spec bodies set
# only `n_predict` -- which is exactly how engine#2884 stayed invisible through a green check.
BODY_FN = re.compile(r"static json (warmup_[a-z_]*body)\s*\([^)]*\)\s*\{(.*?)\n\}", re.S)
READ_LIMIT = re.compile(r'int_field\(\s*body\s*,\s*"(n_predict|max_tokens|max_completion_tokens)"')


def routes_and_warmups(text):
    """(registered paths, warm-up paths), each in file order and de-duplicated."""
    def uniq(seq):
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    return uniq([raw if raw else lit for raw, lit in ROUTE.findall(text)]), uniq(WARMUP.findall(text))


def route_covers(routes, path):
    """True when `path` is a registered route: an exact literal, or a full match of a raw-string regex route."""
    for route in routes:
        if route == path:
            return True
        try:
            if re.fullmatch(route, path):
                return True
        except re.error:
            continue
    return False


def check(root=None):
    root = pathlib.Path(root or ".")
    path = root / SOURCE
    try:
        text = path.read_text(errors="replace")
    except OSError:
        print(f"serve_routes: cannot read {path}", file=sys.stderr)
        return 77
    routes, warmups = routes_and_warmups(text)
    requests = len(WARMUP.findall(text))   # the postings, not the distinct paths: two requests can share one
    if not routes or not warmups:
        print(f"serve_routes: {path} has {len(routes)} route(s) and {len(warmups)} warm-up path(s): one of the two "
              f"literals moved, so this check is looking at nothing", file=sys.stderr)
        return 77
    missing = [w for w in warmups if not route_covers(routes, w)]
    for w in missing:
        print(f"serve_routes: warm-up posts to {w}, which {path} never registers "
              f"(routes: {', '.join(routes)})")
    if missing:
        print(f"serve_routes: {len(missing)} of {len(warmups)} distinct warm-up path(s) have no route -- the request "
              f"404s and the warm-up moves nothing (engine#2853)")
        return 1
    print(f"serve_routes: all {requests} warm-up request(s) post to registered routes "
          f"({len(warmups)} distinct path(s), {len(routes)} routes)")

    # engine#2884: the limit a warm-up body asks for has to be a limit the parser reads, or the request runs at the
    # default 256 and the warm-up does the opposite of what it promises.
    try:
        parser = (root / "app/forward_serve.cpp").read_text(errors="replace")
    except OSError:
        print("serve_routes: cannot read app/forward_serve.cpp, so the body half is looking at nothing", file=sys.stderr)
        return 77
    bodies = {name: sorted(set(BODY_LIMIT.findall(src))) for name, src in BODY_FN.findall(text)}
    read = sorted(set(READ_LIMIT.findall(parser)))
    if not bodies or not read:
        print(f"serve_routes: {len(bodies)} body builder(s) and {len(read)} parser read(s): one literal moved, so the "
              f"body half is looking at nothing", file=sys.stderr)
        return 77
    unread = [(name, keys) for name, keys in bodies.items() if keys and not set(keys) & set(read)]
    for name, keys in unread:
        print(f"serve_routes: {name}() sets {', '.join(keys)}, and the parser reads only {', '.join(read)} -- that "
              f"request answers 200 and runs at the default limit (engine#2884)")
    if unread:
        print(f"serve_routes: {len(unread)} warm-up body builder(s) set a token limit the parser never reads")
        return 1
    print(f"serve_routes: every warm-up body's token limit is read by the parser "
          f"({len(bodies)} builder(s), parser reads {', '.join(read)})")
    return 0


def self_test():
    fails = 0

    def arm(name, ok, detail=""):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
        fails += 0 if ok else 1

    good = '''srv.Post("/v1/chat/completions", post);
srv.Post("/v1/completions", post);
srv.Post("/completion", post);
out.push_back(json{ { "path", "/v1/chat/completions" } });
out.push_back(json{ { "path", "/completion" } });'''
    bad = '''srv.Post("/v1/chat/completions", post);
srv.Post("/v1/completions", post);
out.push_back(json{ { "path", "/v1/chat/completions" } });
out.push_back(json{ { "path", "/completion" } });'''
    r, w = routes_and_warmups(good)
    arm("a warm-up whose route exists parses", r == ["/v1/chat/completions", "/v1/completions", "/completion"] and
        w == ["/v1/chat/completions", "/completion"])
    r, w = routes_and_warmups(bad)
    arm("... and the missing route is visible as a subset failure", [x for x in w if x not in r] == ["/completion"])
    arm("no routes parses as nothing", routes_and_warmups("nothing here") == ([], []))
    # The two capabilities `warmup_route_check.py` had and this file did not, kept here with arms so deleting that
    # file cannot quietly lose them (engine#2853's own check, folded in rather than maintained twice).
    regex_route = 'srv.Get("/health", h);\nsrv.Post(R"(/complet[a-z]*)", post);\nout.push_back(json{ { "path", "/completion" } });'
    r, w = routes_and_warmups(regex_route)
    arm("a raw-string route registration parses", r == ["/health", "/complet[a-z]*"] and w == ["/completion"])
    arm("a srv.Get route counts too", routes_and_warmups('srv.Get("/metrics", m);')[0] == ["/metrics"])
    arm("a regex route covers a warm-up path", route_covers(r, "/completion") and not route_covers(r, "/completions/v1"))
    # engine#2884's half: the limit the body asks for must be one the parser reads.
    body = '{"n_predict", 0}, {"temperature", 0}'
    arm("a body limit is found", BODY_LIMIT.findall(body) == ["n_predict"])
    arm("a parser read is found", READ_LIMIT.findall('int_field(body, "n_predict"') == ["n_predict"])
    arm("an unrelated field is neither", BODY_LIMIT.findall('{"prompt", s}') == [] and
        READ_LIMIT.findall('int_field(body, "top_k"') == [])
    print(f"serve_routes: {'ok' if not fails else f'{fails} failed'}")
    return 0 if not fails else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print("usage: serve_routes_check.py [--self-test] [ROOT]")
        print("Fails when a warm-up in app/serve.cpp posts to a path that file never registers.")
        return 0
    return check(argv[1] if len(argv) > 1 else None)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
