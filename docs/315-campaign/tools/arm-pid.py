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
"""engine#315: bind an arm directory to its server PID, exactly.

Why: correlate.py guesses the target PID when not given one, and its fallback is "most
frequent eviction subject" (correlate.py:116-120). On this box the composition starts and
stops many servers, so that guess landed on a dead process (789665) for an arm whose
server was something else. A correlation against the wrong process fails silently, which
is worse than no correlation.

The exact key is the PORT. Each arm's server logs its own bind:

    <ts> I srv llama_server: listening on http://127.0.0.1:19685

and the eviction stream's PROC_START rows give the llama-server pids. A first attempt tried
to infer the PID from timestamps; that was a guess dressed up as a method and was deleted.
The port is authoritative and is already in the arm's own log.

Note the limit, stated plainly: the SMI stream records PROC_START for llama-server without
a port, so the bind is only exact for an arm whose server is STILL RUNNING (we can then
match `pgrep -f "port <p>"`). For a finished arm the pid is gone and this prints nothing
rather than a plausible-looking wrong answer; correlate the finished arm by passing the
subject the stream itself reports, or re-run with the pid captured live.

    ./arm-pid.py <arm_dir> [--smi <smi.tsv>]

stdout: the PID, or nothing. stderr: the reason. exit 0 only when an exact bind is found.
"""
import argparse
import os
import re
import subprocess
import sys

LISTEN = re.compile(r"listening on http://127\.0\.0\.1:(\d+)")


def arm_port(arm_dir):
    log = os.path.join(arm_dir, "server.log")
    if not os.path.exists(log):
        return None, f"no server.log in {arm_dir}"
    with open(log, errors="replace") as fh:
        for line in fh:
            m = LISTEN.search(line)
            if m:
                return int(m.group(1)), None
    return None, "server.log never logged a listening port (arm may have died at load)"


def live_pid(port):
    out = subprocess.run(
        ["pgrep", "-f", f"llama-server.*port {port}"], capture_output=True, text=True
    )
    pids = [int(x) for x in out.stdout.split() if x.isdigit()]
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--smi", default=os.path.expanduser("~/wt/315evict2-evict/kfd_smi.tsv"))
    a = ap.parse_args()

    port, err = arm_port(a.arm_dir)
    if port is None:
        print(err, file=sys.stderr)
        return 1

    pids = live_pid(port)
    if len(pids) == 1:
        print(pids[0])
        return 0
    if len(pids) > 1:
        print(f"ambiguous: {len(pids)} processes match port {port}: {pids}", file=sys.stderr)
        return 1

    # The arm is over. Say so rather than inventing a pid.
    print(
        f"arm on port {port} is no longer running, so its server pid cannot be bound "
        f"exactly. correlate.py without --pid would fall back to the most-frequent "
        f"eviction subject, which is a guess (observed wrong: 789665). For a finished arm, "
        f"read the subject the stream reports in its own rows, or re-run the arm with the "
        f"pid captured while it is live.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
