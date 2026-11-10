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
"""Every process holding the GPU or NPU device belongs to a live ticket.

Why this exists: "GPU only via `gpu-hold`, NPU only via `npu-hold`" is a rule everyone believes and
nothing checks. On 2026-11-08/09 the box produced three orphans of that shape — a 14 h `llama-server`
holding 51 GiB of GTT that knocked the PR-agent LLM off the iGPU, an NPU `forward-serve` 70 minutes
past a 6-minute ticket, and then, within one half hour, **two different lanes** each starting an HRX
server outside the token while a third lane held it (i1536's `i1536-long-next.sh`, i572's
`mtp_decode.py`). Each one is invisible from the outside: the process looks like work, and only the
token knows it is not.

What a clean result means, and what it does not: this is a **ticket** check, not a health check. `0` says
nobody is outside their ticket — it says nothing about whether the box has memory left. On 2026-11-09 a
single ticketed `supreme max bench` had GTT at 102.7 of 120 GiB with `MemAvailable 2.8 GiB`, which is the
state that got both self-hosted runners OOM-killed (engine#1504), and this check correctly printed "every
one inside a ticket". Pair it with `MemAvailable` and the GTT counter; the floor is 20 GiB.

The test is ancestry, not intention: a holder is fine when the pid recorded in `~/GPU-TOKEN` or
`~/NPU-TOKEN` is itself or one of its ancestors. Systemd `.service` units are exempt by name, because
the fleet's own resident services (`1bit-canonical`, `coder-llm`) are supposed to hold a device
handle; a `.scope` is a run, and a run needs a ticket.

    tools/device_holders.py --check          # exit 1 when an unticketed holder is found
    tools/device_holders.py --self-test
"""
import argparse
import os
import re
import sys

DEVICE = re.compile(r"^/dev/(kfd|dri/renderD[0-9]+|accel/accel[0-9]+)$")
TOKEN = re.compile(r"\bpid\s+([0-9]+)\b")


def token_pid(path):
    """The pid a gpu-hold/npu-hold token names, or None when there is no token."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return None
    m = TOKEN.search(text)
    return int(m.group(1)) if m else None


def holders(root="/proc"):
    """{pid: (cmd, [device paths])} for every process with a device fd open."""
    out = {}
    for name in os.listdir(root):
        if not name.isdigit():
            continue
        pid = int(name)
        fds = os.path.join(root, name, "fd")
        try:
            links = os.listdir(fds)
        except OSError:
            continue
        devs = []
        for fd in links:
            try:
                target = os.readlink(os.path.join(fds, fd))
            except OSError:
                continue
            if DEVICE.match(target):
                devs.append(target)
        if not devs:
            continue
        try:
            cmd = open(os.path.join(root, name, "cmdline"), "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = "?"
        out[pid] = (cmd, sorted(set(devs)))
    return out


def ancestry(pid, root="/proc"):
    """[pid, parent, grandparent, …] up to init."""
    chain = []
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        chain.append(pid)
        try:
            stat = open(os.path.join(root, str(pid), "stat"), encoding="utf-8").read()
        except OSError:
            break
        # "pid (comm) state ppid …", comm may contain spaces and parens: split after the last ')'
        pid = int(stat.rsplit(")", 1)[1].split()[1])
    return chain


def cgroup_of(pid, root="/proc"):
    try:
        text = open(os.path.join(root, str(pid), "cgroup"), encoding="utf-8").read().strip()
    except OSError:
        return ""
    return text.rsplit(":", 1)[-1]


def service(cgroup):
    """The INNERMOST systemd .service unit a cgroup path belongs to, or None for a scope.

    The last component matters: a user session's cgroup is
    `user@1000.service/app.slice/1bit-canonical.service`, and matching the first `.service` names the
    session manager instead of the unit that actually holds the device — which is how the first version
    of this check reported both permanent services as unticketed.
    """
    found = re.findall(r"/([A-Za-z0-9@_.\\-]+\.service)", cgroup)
    return found[-1] if found else None


def verdict(held, tokens, chain_of, cgroup_fn, exempt=("1bit-canonical.service", "coder-llm.service")):
    """[(pid, cmd, devs, reason)] for holders that are NOT inside a ticket and NOT a permanent service."""
    bad = []
    tset = {t for t in tokens if t}
    for pid, (cmd, devs) in sorted(held.items()):
        if tset & set(chain_of(pid)):
            continue
        svc = service(cgroup_fn(pid))
        if svc and svc in exempt:
            continue
        why = f"no token names any of {chain_of(pid)[:3]}" if tset else "no ticket is held at all"
        if svc:
            why += f"; cgroup is {svc}, not an exempt service"
        bad.append((pid, cmd, devs, why))
    return bad


def check(root="/proc", home=None):
    home = home or os.path.expanduser("~")
    tokens = [token_pid(os.path.join(home, "GPU-TOKEN")), token_pid(os.path.join(home, "NPU-TOKEN"))]
    held = holders(root)
    bad = verdict(held, tokens, lambda p: ancestry(p, root), lambda p: cgroup_of(p, root))
    label = ", ".join(str(t) for t in tokens if t) or "none"
    if not bad:
        print(f"device_holders: {len(held)} holder(s), every one inside a ticket (token pid {label})")
        return 0
    for pid, cmd, devs, why in bad:
        print(f"device_holders: pid {pid} holds {', '.join(devs)} with no ticket — {why}")
        print(f"                {cmd[:110]}")
    print(f"device_holders: {len(bad)} unticketed holder(s); token pid {label} — this is the shape that cost "
          f"the fleet a 51 GiB GTT orphan (engine#1504) and two contended measurements on 2026-11-09")
    return 1


def self_test():
    fails = 0

    def case(what, held, tokens, chain, cg, want):
        nonlocal fails
        got = len(verdict(held, tokens, chain, cg))
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}  ({got} flagged, wanted {want})")
        if not ok:
            fails += 1

    h = {100: ("llama-server …", ["/dev/kfd"])}
    case("a holder under the token pid is fine", h, [50], lambda p: [100, 50, 1], lambda p: "/run-x.scope", 0)
    case("a holder outside the token is flagged", h, [50], lambda p: [100, 99, 1], lambda p: "/run-x.scope", 1)
    case("no token at all flags the holder", h, [None, None], lambda p: [100, 1], lambda p: "/run-x.scope", 1)
    case("… but not an exempt service",
         {7: ("lemond", ["/dev/kfd"])}, [], lambda p: [7, 1], lambda p: "/app.slice/1bit-canonical.service", 0)
    # The real cgroup of an exempt service starts with the user manager: the FIRST `.service` there is
    # user@1000.service, and taking it reported both permanent services as unticketed on 2026-11-09.
    case("… and the user manager in front does not defeat the exemption",
         {7: ("lemond", ["/dev/kfd"])}, [], lambda p: [7, 1],
         lambda p: "/user.slice/user-1000.slice/user@1000.service/app.slice/1bit-canonical.service", 0)
    case("a NON-exempt service is still flagged",
         {7: ("other", ["/dev/dri/renderD128"])}, [], lambda p: [7, 1], lambda p: "/app.slice/other.service", 1)
    case("the holder being the token pid itself counts", {9: ("x", ["/dev/kfd"])}, [9], lambda p: [9], lambda p: "", 0)
    # /dev/accel/accel0 is the XDNA NPU device, and it is the one the 05:0xZ orphan held. The first
    # version's pattern matched only /dev/kfd and /dev/dri/renderD*, so an NPU-only holder was invisible to
    # the device scan and was caught (when it was caught) only through the token test.
    case("an NPU-only holder (/dev/accel) is seen too",
         {587217: ("1bit forward-serve", ["/dev/accel/accel0"])}, [], lambda p: [587217, 1],
         lambda p: "/app.slice/thermal-run-1.scope", 1)
    print(f"device_holders --self-test: {fails} failure(s)")
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default="/proc")
    a = ap.parse_args(argv[1:])
    if a.self_test:
        return self_test()
    if a.check:
        return check(a.root)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
