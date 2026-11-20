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

Same class, one field type further (engine#2884's `cache_prompt: false`, which the bodies set and nothing read): a
**boolean** field a warm-up body sets must be a boolean the parser reads, through `bool_field`. The two checks are
separate on purpose -- a key can be read for one type and ignored for another, and a union would hide it.

engine#2895: the same shape one step out. `stats.prefix` records WHY a request was refused the prompt cache, and the
only `[prefix] slot=` line sat inside the branch that ACCEPTS the cache -- so a refusal logged nothing at all and read
exactly like "the prefix path was never asked". Every recorded prefix decision must reach the log.

Exit: 0 every warm-up path is registered, its token limit is read, and every recorded prefix decision is logged; 1 one of
those is not (named); 77 a literal could not be read.
"""

import pathlib
import re
import sys

SOURCE = pathlib.Path("app/serve.cpp")
# A route is registered with srv.Get or srv.Post, as a plain literal or as a raw-string regex (`R"(/slots/(\\d+))"`).
# The regex form is why `warmup_route_check.py` existed as a second check; both halves live here now, so one file answers
# "is every warm-up path a route this server registers" and no reviewer has to keep two baselines in step.
ROUTE = re.compile(r'srv\.(?:Get|Post)\(\s*(?:R"\((.*?)\)"|"([^"]+)")\s*,')
# A warm-up path is either a literal or a named constant (`kWarmupPath`), and the chat warm-up uses the constant. Matching
# literals only would verify two of the three warm-ups and report ok -- the same shape as the 404 this check exists for,
# so an identifier is resolved when the file defines it and is `cannot tell` (77) when it does not, never a silent skip.
WARMUP = re.compile(r'\{\s*"path"\s*,\s*([A-Za-z_]\w*|"[^"]*")')
CONST_DEF = re.compile(r'const\s+char\s*\*\s*const\s+([A-Za-z_]\w*)\s*=\s*"([^"]+)"')
# The token-limit fields the warm-up bodies set, and the ones the request parser reads (forward_serve.cpp).
# Both shapes the builders use: `{ "n_predict", 0 }` in an initializer list and `body["n_predict"] = 2`.
BODY_LIMIT = re.compile(r'"(n_predict|max_tokens|max_completion_tokens)"\s*(?:[:,]|\])')
# Per builder, not the union: the chat warm-up sets `max_tokens`, so a union hides that the short and spec bodies set
# only `n_predict` -- which is exactly how engine#2884 stayed invisible through a green check.
BODY_FN = re.compile(r"static json (warmup_[a-z_]*body)\s*\([^)]*\)\s*\{(.*?)\n\}", re.S)
READ_LIMIT = re.compile(r'int_field\(\s*body\s*,\s*"(n_predict|max_tokens|max_completion_tokens)"')
# A recorded prefix decision, and the call that surfaces it (engine#2895).
PREFIX_DECISION = re.compile(r"stats\.prefix\s*=\s*npu::PrefixDecision\{")
PREFIX_LOG = re.compile(r"log_prefix\(|\[prefix\] slot=")
# The boolean half: a field a builder sets to a literal true/false, and the ones the parser reads through bool_field.
BODY_BOOL = re.compile(r'"([a-z_]+)"\s*,\s*(?:true|false)\b')
READ_BOOL = re.compile(r'bool_field\(\s*body\s*,\s*"([a-z_]+)"')


def routes_and_warmups(text):
    """(registered paths, warm-up paths, identifiers that could not be resolved), warm-ups in file order."""
    def uniq(seq):
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    consts = dict(CONST_DEF.findall(text))
    warmups, unresolved = [], []
    for expr in WARMUP.findall(text):
        if expr.startswith('"'):
            warmups.append(expr[1:-1])
        elif expr in consts:
            warmups.append(consts[expr])
        else:
            unresolved.append(expr)
    routes = uniq([raw if raw else lit for raw, lit in ROUTE.findall(text)])
    return routes, uniq(warmups), uniq(unresolved)


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
    routes, warmups, unresolved = routes_and_warmups(text)
    if unresolved:
        print(f"serve_routes: {path} posts a warm-up to {', '.join(unresolved)}, and this file defines no string for it: "
              f"an identifier it cannot resolve is cannot-tell, not a path to skip", file=sys.stderr)
        return 77
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

    # engine#2897: the sentinel's *behaviour* is pinned by tests/npu_token_limit_test.cpp (a row CI runs), which is the
    # right place for it -- this check can only see the shape of an expression. What it can still catch is a call site
    # that stops using the function every other reader goes through.
    if "n_predict" in read:
        if "npu::token_limit(" not in parser:
            print("serve_routes: app/forward_serve.cpp reads `n_predict` and does not call npu::token_limit(...) -- the "
                  "sentinel's arithmetic is tested (tests/npu_token_limit_test.cpp) but this site bypasses it "
                  "(engine#2897)")
            print("serve_routes: 1 call site bypasses the tested token limit")
            return 1
        print("serve_routes: the token limit is taken from the tested npu::token_limit(...)")

    # The boolean half (engine#2884's `cache_prompt`): a `false` the body sets and the parser never reads is the same
    # silence as a 404 -- the request succeeds and does the opposite of what it asked for.
    try:
        readers = [(root / "app/forward_serve.cpp").read_text(errors="replace"),
                   (root / "app/npu_sampling.h").read_text(errors="replace")]
    except OSError:
        print("serve_routes: cannot read the parser files, so the boolean half is looking at nothing", file=sys.stderr)
        return 77
    bools = {name: sorted(set(BODY_BOOL.findall(src))) for name, src in BODY_FN.findall(text)}
    read_bools = sorted({k for r in readers for k in READ_BOOL.findall(r)})
    if not any(bools.values()) or not read_bools:
        print(f"serve_routes: {sum(len(v) for v in bools.values())} boolean body field(s) and "
              f"{len(read_bools)} bool_field read(s): one literal moved, so the boolean half is looking at nothing",
              file=sys.stderr)
        return 77
    unread_bools = [(name, keys) for name, keys in bools.items() if keys and not set(keys) & set(read_bools)]
    for name, keys in unread_bools:
        print(f"serve_routes: {name}() sets {', '.join(keys)}, and the parser reads only "
              f"{', '.join(read_bools)} through bool_field -- that request answers 200 and the field does nothing "
              f"(engine#2884)")
    if unread_bools:
        print(f"serve_routes: {len(unread_bools)} warm-up body builder(s) set a boolean the parser never reads")
        return 1
    print(f"serve_routes: every warm-up body's boolean field is read by the parser "
          f"(parser reads {', '.join(read_bools)})")

    # engine#2895: a decision that is recorded must be printed. Each `stats.prefix = npu::PrefixDecision{` needs the log
    # call near it -- near, because the line that explains a decision belongs beside it.
    parser_lines = readers[0].splitlines()
    silent = [i + 1 for i, ln in enumerate(parser_lines)
              if PREFIX_DECISION.search(ln) and not PREFIX_LOG.search("\n".join(parser_lines[i:i + 12]))]
    if not PREFIX_DECISION.search(readers[0]):
        print("serve_routes: app/forward_serve.cpp records no prefix decision: that literal moved, so the log half is "
              "looking at nothing", file=sys.stderr)
        return 77
    for ln in silent:
        print(f"serve_routes: line {ln} records a prefix decision and nothing in the next 12 lines logs it -- the "
              f"refusal is invisible in the serve log (engine#2895)")
    if silent:
        print(f"serve_routes: {len(silent)} prefix decision(s) recorded and never surfaced")
        return 1
    print(f"serve_routes: every recorded prefix decision is logged "
          f"({len(PREFIX_DECISION.findall(readers[0]))} site(s))")
    return 0


def self_test():
    fails = 0

    def arm(name, ok, detail=""):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
        fails += 0 if ok else 1

    # The chat warm-up names its path through a constant, so the fixtures do too: a fixture that inlines the literal is
    # easier than the artifact, and passes for the better reason while covering less of it.
    good = '''static const char* const kWarmupPath = "/v1/chat/completions";
srv.Post("/v1/chat/completions", post);
srv.Post("/v1/completions", post);
srv.Post("/completion", post);
out.push_back(json{ { "path", kWarmupPath } });
out.push_back(json{ { "path", "/completion" } });'''
    bad = '''static const char* const kWarmupPath = "/v1/chat/completions";
srv.Post("/v1/completions", post);
srv.Post("/completion", post);
out.push_back(json{ { "path", kWarmupPath } });
out.push_back(json{ { "path", "/completion" } });'''
    r, w, u = routes_and_warmups(good)
    arm("a warm-up whose route exists parses, and a named constant is resolved",
        r == ["/v1/chat/completions", "/v1/completions", "/completion"] and
        w == ["/v1/chat/completions", "/completion"] and u == [])
    r, w, u = routes_and_warmups(bad)
    arm("... and the missing route behind the constant is visible as a subset failure",
        [x for x in w if not route_covers(r, x)] == ["/v1/chat/completions"])
    r, w, u = routes_and_warmups('out.push_back(json{ { "path", kUndeclared } });')
    arm("an identifier this file does not define is cannot-tell, not a skip", (r, w, u) == ([], [], ["kUndeclared"]))
    arm("no routes parses as nothing", routes_and_warmups("nothing here") == ([], [], []))
    # The two capabilities `warmup_route_check.py` had and this file did not, kept here with arms so deleting that
    # file cannot quietly lose them (engine#2853's own check, folded in rather than maintained twice).
    regex_route = 'srv.Get("/health", h);\nsrv.Post(R"(/complet[a-z]*)", post);\nout.push_back(json{ { "path", "/completion" } });'
    r, w, u = routes_and_warmups(regex_route)
    arm("a raw-string route registration parses", r == ["/health", "/complet[a-z]*"] and w == ["/completion"])
    arm("a srv.Get route counts too", routes_and_warmups('srv.Get("/metrics", m);')[0] == ["/metrics"])
    arm("a regex route covers a warm-up path", route_covers(r, "/completion") and not route_covers(r, "/completions/v1"))
    # engine#2884's half: the limit the body asks for must be one the parser reads.
    body = '{"n_predict", 0}, {"temperature", 0}'
    arm("a body limit is found", BODY_LIMIT.findall(body) == ["n_predict"])
    arm("a parser read is found", READ_LIMIT.findall('int_field(body, "n_predict"') == ["n_predict"])
    arm("an unrelated field is neither", BODY_LIMIT.findall('{"prompt", s}') == [] and
        READ_LIMIT.findall('int_field(body, "top_k"') == [])
    # engine#2884's boolean half: `cache_prompt: false` was set by both warm-up bodies and read by nothing.
    arm("a body boolean is found", BODY_BOOL.findall('{"temperature", 0}, {"cache_prompt", false}') ==
        ["cache_prompt"])
    arm("a bool_field read is found", READ_BOOL.findall('bool_field(body, "cache_prompt", true)') == ["cache_prompt"])
    arm("a recorded prefix decision is found",
        len(PREFIX_DECISION.findall("stats.prefix = npu::PrefixDecision{-1, 0, x};")) == 1)
    arm("both log shapes are recognised", bool(PREFIX_LOG.search("log_prefix(stats.prefix);")) and
        bool(PREFIX_LOG.search('fprintf(stderr, "[prefix] slot=')))
    arm("an int field is not a boolean read",
        READ_BOOL.findall('int_field(body, "cache_prompt")') == [] and
        BODY_BOOL.findall('{"n_predict", 0}') == [])
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
