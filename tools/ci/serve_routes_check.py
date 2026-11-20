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

Exit: 0 every warm-up path is registered, 1 one is not (named), 77 the file or a literal could not be read.
"""

import pathlib
import re
import sys

SOURCE = pathlib.Path("app/serve.cpp")
ROUTE = re.compile(r'srv\.Post\(\s*"([^"]+)"')
WARMUP = re.compile(r'\{\s*"path"\s*,\s*"([^"]+)"')


def routes_and_warmups(text):
    """(registered paths, warm-up paths), each in file order and de-duplicated."""
    def uniq(seq):
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    return uniq(ROUTE.findall(text)), uniq(WARMUP.findall(text))


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
    missing = [w for w in warmups if w not in routes]
    for w in missing:
        print(f"serve_routes: warm-up posts to {w}, which {path} never registers "
              f"(routes: {', '.join(routes)})")
    if missing:
        print(f"serve_routes: {len(missing)} of {len(warmups)} distinct warm-up path(s) have no route -- the request "
              f"404s and the warm-up moves nothing (engine#2853)")
        return 1
    print(f"serve_routes: all {requests} warm-up request(s) post to registered routes "
          f"({len(warmups)} distinct path(s), {len(routes)} routes)")
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
