#!/usr/bin/env python3
"""Every path the warm-up plan posts to must be a route the server registers.

`app/serve.cpp` builds a list of warm-up requests and posts each one to its own
localhost server. Nothing tied those paths to the routes the same file
registers, so when the short and spec warm-ups were written against
llama-server's "/completion", this server -- which registered
"/v1/completions" only -- answered 404. The only trace was one stderr line per
request that nothing reads, and the docs went on promising the warm-up.

This is a pure-source check: no build, no device, no server. It reads the
string literals in `warmup_requests()` and the paths handed to `srv.Get` /
`srv.Post`, and fails when a warm-up path has no registration. Regex routes are
matched by compiling the registration as a regex, so `R"(/slots/(\\d+))"`
counts for a warm-up path that matches it.

Exit: 0 every warm-up path is registered, 1 a warm-up path is not, 2 cannot tell.

  warmup_route_check.py --root .        # check the tree at .
  warmup_route_check.py --self-test     # prove the checker on fixture trees
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import pathlib

SERVE = "app/serve.cpp"

RE_WARMUP_FN = re.compile(r"warmup_requests\s*\([^)]*\)\s*\{")
RE_PATH_FIELD = re.compile(r'\{\s*"path"\s*,\s*"([^"]+)"')
RE_KWARMUP = re.compile(r'kWarmupPath\s*=\s*"([^"]+)"')
RE_REGISTER = re.compile(r'srv\.(?:Get|Post)\(\s*(?:R"\((.*?)\)"|"([^"]+)")\s*,')


def _block(src: str, start: int) -> str:
    """Return the brace-balanced body of the function whose '{' is at `start`."""
    depth, i = 0, start
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1 : i]
        i += 1
    return src[start + 1 :]


def warmup_paths(src: str) -> list[str]:
    paths: list[str] = []
    for m in RE_KWARMUP.finditer(src):
        paths.append(m.group(1))
    for m in RE_WARMUP_FN.finditer(src):
        paths.extend(RE_PATH_FIELD.findall(_block(src, m.end() - 1)))
    return sorted(set(paths))


def routes(src: str) -> list[tuple[str, bool]]:
    out = []
    for m in RE_REGISTER.finditer(src):
        raw, lit = m.group(1), m.group(2)
        if raw is not None:
            out.append((raw, True))
        else:
            out.append((lit, False))
    return out


def check(root: pathlib.Path) -> int:
    f = root / SERVE
    try:
        src = f.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"cannot tell: {f}: {e}", file=sys.stderr)
        return 2
    wanted = warmup_paths(src)
    have = routes(src)
    if not wanted:
        print(f"cannot tell: no warm-up path found in {f} -- the parser is out of date", file=sys.stderr)
        return 2
    if not have:
        print(f"cannot tell: no route registration found in {f} -- the parser is out of date", file=sys.stderr)
        return 2

    missing = []
    for p in wanted:
        ok = False
        for path, is_regex in have:
            if is_regex:
                try:
                    if re.fullmatch(path, p):
                        ok = True
                        break
                except re.error:
                    continue
            elif path == p:
                ok = True
                break
        if not ok:
            missing.append(p)

    print(f"warm-up paths ({len(wanted)}): {', '.join(wanted)}")
    print(f"registered routes ({len(have)}): {', '.join(r[0] for r in have)}")
    if missing:
        for p in missing:
            print(f"NOT REGISTERED: warm-up posts to {p} and no srv.Get/srv.Post registers it "
                  f"-- that warm-up 404s and warms nothing")
        return 1
    print("ok: every warm-up path is a registered route")
    return 0


# --- self-test -------------------------------------------------------------
GOOD = r'''
static const char* const kWarmupPath = "/v1/chat/completions";
static json warmup_requests(bool speculative) {
    json out = json::array();
    out.push_back(json{ { "path", kWarmupPath }, { "body", 1 } });
    out.push_back(json{ { "path", "/completion" }, { "body", 2 } });
    return out;
}
static void serve_it(httplib::Server& srv) {
    srv.Get("/health", h);
    srv.Post("/v1/chat/completions", post);
    srv.Post("/completion", post);
}
'''

BAD = GOOD.replace('    srv.Post("/completion", post);\n', '')

REGEX_ROUTE = BAD.replace(
    '    srv.Post("/v1/chat/completions", post);\n',
    '    srv.Post("/v1/chat/completions", post);\n    srv.Post(R"(/complet[a-z]*)", post);\n',
)

NO_WARMUP = '''
static void serve_it(httplib::Server& srv) { srv.Get("/health", h); }
'''


def self_test() -> int:
    cases = [("good", GOOD, 0), ("missing route", BAD, 1), ("regex route counts", REGEX_ROUTE, 0),
             ("parser out of date", NO_WARMUP, 2)]
    bad = 0
    for name, src, want in cases:
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / SERVE
            p.parent.mkdir(parents=True)
            p.write_text(src, encoding="utf-8")
            got = check(pathlib.Path(d))
        ok = got == want
        bad += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: exit {got}, want {want}")
    print(f"self-test: {len(cases) - bad}/{len(cases)} arms passed")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    return check(pathlib.Path(a.root).resolve())


if __name__ == "__main__":
    sys.exit(main())
