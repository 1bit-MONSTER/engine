<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Self-hosted CI runners on the Strix Halo box

Two runners serve this repository, both on Strix Halo. Since 2026-11-22 the box is the 1bit OS appliance (Puppy, no
systemd), so the runners are started by `/etc/init.d` scripts instead of the units below; see
[On the 1bit OS appliance](#on-the-1bit-os-appliance-2026-11-27) for what runs there now.

| runner | labels | takes |
|---|---|---|
| `strixhalo-pr-agent` | `self-hosted,Linux,X64,pr-agent,issue-agent` | `pr-agent`, `issue-agent` |
| `strixhalo-bench` | `self-hosted,Linux,X64,strixhalo,bench` | `hrx-build` and the bench/gate jobs |

`strixhalo-engine` (`pr-agent,strixhalo`) was the old dev box's runner, and `ryzen-pr-agent` the stop-gap reviewer on
the desktop from 2026-11-22. The second one is retired: ryzen hard-locked four times between 11-24 and 11-27, three
of them 1-10 minutes after a review on its 9070 XT.

Everything the GitHub-hosted jobs need comes from `ubuntu-*` runners; these two exist because those jobs need
*this box* — its GPU, its NPU and the local LLM at `127.0.0.1:8090`.

## Why this page exists (engine #1504)

On 2026-11-08 both runners were OOM-killed at 08:37 and **stayed dead until 12:23**:

```
$ systemctl show actions.runner.1bit-MONSTER-engine.strixhalo-bench.service -p Result -p NRestarts
Result=oom-kill
NRestarts=0
```

`NRestarts=0` is the whole bug. The units shipped with no restart policy, and `runsvc.sh` exits **0** after the
listener stops ("Runner listener exit with 0 return code, stop the service, no retry needed"), so even
`Restart=on-failure` would never have fired. While they were down, 9 `hrx-build` and 15 `pr-agent` jobs queued,
and every check that needs the box read as *pending* — never green, never red, which looks exactly like "queued
behind a busy runner". It had happened before (2026-11-06, 41 `pr-agent` runs queued behind the same unit).

A check whose runner is gone is the same silent-absence class as [engine #1077](https://github.com/1bit-MONSTER/engine/issues/1077);
a restart policy is what turns a four-hour outage into a thirty-second one.

## The unit, and why each line is there

`scripts/install-ci-runner.sh` writes this (values shown for the defaults):

```ini
[Unit]
Description=GitHub Actions runner strixhalo-engine (dedicated user ghrunner)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=ghrunner
WorkingDirectory=/var/lib/ghrunner/actions-runner-strixhalo-engine
ExecStart=/var/lib/ghrunner/actions-runner-strixhalo-engine/runsvc.sh
Restart=always
RestartSec=30
OOMPolicy=continue
MemoryHigh=43G
MemoryMax=48G
TasksMax=8192
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min

[Install]
WantedBy=multi-user.target
```

- **`Restart=always`, not `on-failure`** — because `runsvc.sh` reports success when it is killed. `always` is the
  only value that brings a runner back from that exit code. `RestartSec=30` keeps a crash-looping runner from
  hammering the box and the GitHub API.
- **`StartLimitIntervalSec=0`** — a runner must never be rate-limited out of existence. The 30-second spacing is
  the rate limit; systemd giving up after five starts in ten seconds (the default) is how "stays dead" happens
  again.
- **`OOMPolicy=continue`** — this is the line the old units lacked. With the default (`stop`), a process in the
  cgroup being OOM-killed stops the *unit* with `Result=oom-kill`. With `continue`, the kernel's victim is the
  one that dies — the job — and the runner keeps running, so the job fails with a reason instead of the runner
  vanishing.
- **`MemoryHigh`/`MemoryMax`** — the cgroup hits its own ceiling before the box's OOM killer has to choose
  between "the largest allocation on the machine" and everything else. The default 48G matches the bench rule
  (`systemd-run --user --scope -p MemoryMax=48G`), so a heavy job is killed by *its* limit, not by the box's.
- **`KillMode=process`** with the 5-minute stop timeout — the behaviour the shipped units already had: stop the
  listener, let the job's own timeout end its children.

## The dedicated user

The jobs used to run as the owner (`User=bcloud`). They now run as `ghrunner`: no shell, home
`/var/lib/ghrunner`, and the runner's `_work` tree under it. Anything a job writes into `$HOME` lands there, not
in the owner's tree, and the account holds no ssh keys and no fleet git remotes of its own.

**It is not read-isolation of the owner's files, though** — `/home/bcloud` is `drwxrwxr-x` on this box, so a job
as `ghrunner` can read whatever in the owner's home is world-readable. The isolation is per-process and
per-account (a job cannot use the owner's keys to push, and its `$HOME` writes are its own); if read-isolation is
wanted too, that is `chmod o-rx /home/bcloud` — the owner's call, because it changes the box for every other lane
as well.

### One queue, two accounts

The workflows reach the box's hold state through `$HOME`: `~/GPU-TOKEN`, `~/GPU-QUEUE` (and `npu-hold`, which
hard-codes `~/NPU-TOKEN` and `~/NPU-QUEUE` around `gpu-hold`). A runner with a different home therefore does not
merely lose `thermal-run.sh` — it takes **its own** GPU token, and the hardware gate then runs GPU work
concurrently with whatever lane holds the real one, which is the one thing the hold rule exists to prevent.

The obvious repair — put the runner's account in the owner's group — is wrong, and the review of this PR measured
why: `/home/bcloud` is `0775`, so group write on the **directory** lets a job rename or replace any entry in it.
A job could rewrite `~/.local/bin/gpu-hold` (which every lane then executes), replace `~/.bashrc`, or move
`~/.ssh` aside and put its own directory in its place. That is an owner-level code path created by a recipe whose
whole point is that jobs no longer run as the owner.

So the hold state lives in its own directory, owned by its own group:

```
/var/lib/1bit-fleet/          0770 fleethold, setgid
  gpu-token                   0660 fleethold     <- what gpu-hold writes, and rm -f's on release
  gpu-queue/                  0770 fleethold, setgid   <- tickets inherit the group
  npu-token, npu-queue/
```

- `ghrunner` joins **`fleethold` and nothing else**. The shared tools (`~/lb`, `~/.local/bin`) are already
  world-readable and executable, so no group is needed to *run* them — only the hold files need the group.
- The unit gets `Environment=GPU_TOKEN_FILE=/var/lib/1bit-fleet/gpu-token GPU_QUEUE_DIR=/var/lib/1bit-fleet/gpu-queue`,
  which `gpu-hold` honours, so a job queues with the fleet.
- **The owner's side has to name the same two paths**, or the split simply reappears in the other direction. The
  installer writes `/etc/profile.d/1bit-fleet.sh` exporting them for the owner's shells; a long-running owner
  service needs them in its own unit or environment.
- **Why a shared *file* inside the owner's home cannot work at all:** `gpu-hold` releases with
  `rm -f "$TOKEN"`, a write to the containing **directory**. `chgrp` on the file is not enough, and a symlink is
  worse — `rm` removes the link and leaves the real token held for ever.
- `npu-hold` wraps `gpu-hold` with `exec env GPU_QUEUE_DIR=~/NPU-QUEUE GPU_TOKEN_FILE=~/NPU-TOKEN …`, so it
  overrides whatever the environment says: **NPU sharing needs that wrapper changed** to honour the two
  variables. Until then the NPU queue is per-account, and that is a named follow-up rather than a claim.
### The tools are reached by a home-relative path

`hrx-build.yml:111,132` and `hw-gate.yml` call `~/lb/thermal-run.sh`, and `hw-gate.sh` resolves `gpu-hold` with
`command -v` — a **home-relative path**, not a permission. So the installer links `~/lb` and `~/.local/bin` from the
runner's home into the account that owns them and puts `.local/bin` first on the unit's `PATH`. The runner reaches
both through `other` (`r-x`), so it can **run** the tools and cannot rewrite them — which is the difference between
this and the owner's group, and the reason the links are back after the review's first round.

`thermal-run.sh` itself is now **in this repo** (`scripts/thermal-run.sh`), because it was in nobody's: the two
workflows call it unconditionally, and the box that ran `hrx_build` on 11-26 had no `~/lb` at all, so every job
died `line 2: /home/bcloud/lb/thermal-run.sh: No such file or directory` (**exit 127**) before configuring
anything — on four branches, so the check could not gate any of them (engine#3517). The runner's
`~/lb/thermal-run.sh` should be a copy of (or a link to) that file. Its contract is the one the workflows already
assume, and the one `scripts/mem-guard.sh --term` depends on: freeze the job at 78 C, thaw at 68 C, kill at 93 C
with rc 137, and on TERM/INT/HUP kill the whole job and exit **143**. `THERMAL_DISABLE=1` runs ungoverned for a box
with no readable sensor, and `THERMAL_HI/LO/KILL/TCTL/NICE/POLL` override the defaults. The shipped copy takes the
process-group form, with a `shortcut:` line naming when to upgrade; the `systemctl --user` scope below is that
upgrade, and a child that calls `setsid()` escapes the process-group form.

`thermal-run.sh` runs its jobs under `systemctl --user`, so the installer also does
`loginctl enable-linger <user>`; without a user manager for that account it falls back to a process-group
`SIGSTOP`, which is the weaker thermal brake the script itself warns about.

`hw-gate.sh` has two more `$HOME` defaults (`:98-99`), and both resolve into an empty home for a runner under
another account:

| variable | default | the installer sets it to |
|---|---|---|
| `HW_GATE_MODELS` | `$HOME/models` | `--owner-user`'s `models/` when it exists — read-only |
| `HW_GATE_LOCK` | `$HOME/npu.lock` | `<hold-state>/npu.lock` — the **fleet's** lock, so a runner cannot take NPU work on a lock of its own (the NPU version of the GPU-token bypass) |

The same reasoning is applied at the other end: `scripts/hw-gate.sh` now prefers `<hold-state>/npu.lock` when that
directory exists, and `scripts/gpu-hold` prefers `<hold-state>/{gpu-token,gpu-queue}`, so **the default is what
moved, not an environment variable**. That matters because the lanes are long-lived tmux/`pi` sessions and
`systemd --user` services, none of them login shells: `/etc/profile.d` reaches none of them, and a caller without
the variables would otherwise fall back to `~/GPU-TOKEN` and silently fork the queue. `ONEBIT_HOLD_DIR` overrides
the directory if it ever lives elsewhere.

One consequence worth writing down, found by `tests/gpu_hold_queue_test.sh`: `gpu-hold` treats the token file
**existing** as "held" — the holder creates it with `noclobber` — so the installer must not pre-create an empty
`gpu-token`. It creates the directory and the queues, and lets the first claim create the token.

- `--no-hold-state` exists for a runner that runs no hardware jobs — the `pr-agent` runner does not (it talks to
  the local LLM over HTTP on `127.0.0.1:8090`), so it needs none of this.
- `--migrate-hold-state` moves an existing `~/GPU-{TOKEN,QUEUE}`, `~/NPU-{TOKEN,QUEUE}` into `/var/lib/1bit-fleet`,
  and **refuses when the queue is not empty**: moving a live queue out from under the lanes is not a step to
  take blind.

What is deliberately **not** shared is `~/.cache`, which is `0700` on the owner's account: the persistent build
trees (`~/.cache/hrx-build/bench`, `~/.cache/hw-gate`) and the `pr-agent-venv-*` environments move, so the first
runs on a fresh install are cold — `hrx-build`'s "rebuild only what changed" starts from zero once.

That persistence is also why the bench tree now **stamps itself** (`bench-stamp.txt`: the engine, llama.cpp and
hrx-system revisions, the host, and the lowering revision — `unknown` when the build did not record one). A tree
built on one runner and measured on another is exactly how engine#2858 got an IQ1_S prefill figure of 11.22 tok/s
standing next to four local builds at 56.5-56.6 on the same box, with nothing in the tree to tell them apart. Before
quoting a number from a persistent tree, ask it:

```
python3 tools/ci/bench_stamp.py check --tree ~/.cache/hrx-build/bench \
    --expect-engine "$(git rev-parse --short=12 HEAD)" --expect-host "$(uname -n)"
```

rc 0 means comparable, rc 1 names the field that moved, and **rc 2 means cannot tell** — no stamp, or a lowering the
build did not record. rc 2 is not a pass: `loom=unknown` is the question #2858 is still open on.

In CI the engine field is the **head sha of the commit being built**, not the checkout's `HEAD`: actions/checkout
gives a PR run a synthetic merge commit, and stamping that (measured on 11-20: `engine 241017c32dbb`) makes every
`check --expect-engine` answer "the field moved" on a tree that is exactly the pin. `hrx-build` passes the head sha
both to `bench_stamp.py write` and to `cmake` as `ONEBIT_BUILD_ENGINE`, and then runs `check` in the same job — because
the first version asked git for one stamp and was handed the sha for the other, and the same job printed
`engine 2767c7914585` in the tree and `engine 0210b23a669c` from the binary (run 44682796555).

The lowering is not a setting anyone passes: it is **read out of the code objects the build emitted**, from any
`.hsaco`'s `.comment` section (`AMD clang version 24.0.0git … e095f1b3becf`), so `loom=unknown` now means "this tree
has no code objects", not "every build". A tree whose objects exist can therefore answer **rc 0, comparable** —
before this, the decisive field was always unknown and the tool could only ever say 1 or 2.

The same facts travel with a shipped binary: `1bit --version` prints `engine … llama.cpp … hrx-system … built …`,
generated at configure time. It deliberately carries **no lowering**: at configure time there are no code objects
yet, so a field there could only say `unknown` and would disagree with the tree stamp about the same build. A submodule that is not checked out prints `unknown`, never the enclosing
checkout's revision — `git -C <missing dir>` walks up to the parent repository, so the obvious way to write that
stamp names the engine three times and looks complete.

What it still needs, and the honest price of each:

| grant | why | price |
|---|---|---|
| `docker` group | the bench jobs run containers (podman/docker images for the distro proofs) | **docker group is root-equivalent** — a job that runs `docker run -v /:/h …` can read the box. It is narrower than the owner's account but not a sandbox |
| `render` (or `video`) group | `/dev/kfd` and `/dev/dri/renderD*` at `0660` for the GPU jobs | device access only |
| `fleethold` (its own group) | write the hold state in `/var/lib/1bit-fleet` | the runner can write the queue and the token files **and nothing else** — not the owner's home, not the tools it runs |

If that trade is not wanted, the next step is rootless podman (or a docker socket proxy that only allows the
images the workflows name) and dropping the `docker` group — that is a change to the workflows, not to this
recipe, and it is not in #1504's scope.

## A fresh install, start to finish

```bash
# 1. the runner, unpacked and verified. Pin the version; the script checks the release's SHA256SUMS and
#    refuses an unverified tarball.
curl -fsSLO https://github.com/actions/runner/releases/download/v2.329.0/actions-runner-linux-x64-2.329.0.tar.gz
mkdir -p /srv/actions-runner-2.329.0 && tar -C /srv/actions-runner-2.329.0 -xzf actions-runner-linux-x64-2.329.0.tar.gz

# 2. a registration token: repo Settings -> Actions -> Runners -> New self-hosted runner, or
#    gh api -X POST repos/1bit-MONSTER/engine/actions/runners/registration-token --jq .token > /root/runner-token
#    It is a one-hour secret: write it to a root-only file and delete it after the install.

# 3. install both runners
sudo scripts/install-ci-runner.sh --name strixhalo-engine --labels pr-agent \
     --token-file /root/runner-token --runner-dir /srv/actions-runner-2.329.0
sudo scripts/install-ci-runner.sh --name strixhalo-bench --labels bench \
     --token-file /root/runner-token --runner-dir /srv/actions-runner-2.329.0
sudo rm -f /root/runner-token

# 4. the owner's own lanes must name the same hold state (the installer writes this file); a long-running
#    owner service needs the two variables in its own unit instead
. /etc/profile.d/1bit-fleet.sh
```

**Order matters, and the installer enforces it.** The default that keeps the queue un-forked lives in
`scripts/gpu-hold`; what *runs* on the box is `~/.local/bin/gpu-hold`, which every lane and the runner reach
through their own home. So the installer puts the repository's copy in place (keeping a dated `.bak-*`, as this
box does for every other tool) before anything else, and `--migrate-hold-state` **refuses** to run against a copy
that still defaults to `~/GPU-TOKEN` — otherwise the migration moves the token and the first lane without
`GPU_TOKEN_FILE` recreates the old path, which is two queues again.

On a box that already has hold state in the owner's home, the first install takes
`--migrate-hold-state --owner-user <name>`, with the queue drained — the installer refuses a non-empty queue.

**NPU is not shared yet, and the installer says so at install time.** `npu-hold` is not in this repository, and it
pins the queue with `exec env GPU_QUEUE_DIR="$HOME/NPU-QUEUE" GPU_TOKEN_FILE="$HOME/NPU-TOKEN" …`
(`~/.local/bin/npu-hold:31`), which resolves to a private NPU queue for any other account whatever `gpu-hold`'s
default is. The unit sets `HW_GATE_LOCK` to the fleet's lock, so the *lock* is shared; the *queue* is not. Until
`npu-hold` honours the two variables, a runner under another account must not take NPU work.

`--dry-run` prints the unit and runs `systemd-analyze verify` on it without changing anything; `--verify` prints
the installed unit's `Result`/`NRestarts`/`OOMPolicy` and exits. The script is idempotent: an existing `ghrunner`
user, an already-configured runner directory and an existing unit are reused or rewritten, not re-created.

## Verifying an install

```bash
# the unit is up and has not been restarting
systemctl show actions.runner.1bit-MONSTER-engine.strixhalo-engine.service \
    -p ActiveState -p Result -p NRestarts -p OOMPolicy -p MemoryMax
#   ActiveState=active  Result=success  NRestarts=0  OOMPolicy=continue  MemoryMax=51539607552

# GitHub agrees it is online and idle
gh api repos/1bit-MONSTER/engine/actions/runners --jq '.runners[] | "\(.name) \(.status) busy=\(.busy)"'
#   strixhalo-bench   online busy=false
#   strixhalo-engine  online busy=false
```

And the OOM behaviour itself, which is what #1504 is about: with `OOMPolicy=continue`, a process killed inside the
unit's cgroup does not stop the unit (`systemd.service(5)`: "the service manager will not take any action"), so
the job fails and the runner keeps listening; with the default `stop` it ends up exactly as it did on 11-08 —
`Result=oom-kill`, `NRestarts=0`.

```bash
# the state to read after a heavy job has been killed, on a freshly installed runner:
systemctl show actions.runner.1bit-MONSTER-engine.strixhalo-bench.service -p Result -p NRestarts
#   Result=success     <- the job died, the runner did not
#   NRestarts=0        <- nothing had to restart it
```

**Not yet demonstrated end to end here.** A throwaway `systemd-run --user` pair with and without
`OOMPolicy=continue` (128M cap, a child allocating 256M) both came back `Result=success` on this box — the child's
allocation failed as an exception rather than as an OOM kill, so that experiment says nothing either way. The
directive is the documented mechanism and it is what the fresh install's single end-to-end run should confirm,
together with the rest of #1504's acceptance.

## The PR-Agent model runs in gpu-hold windows (engine#2494)

The `pr-agent` job talks to a local model, Qwen3-Coder-30B-A3B served by `1bit serve` on HRX at `127.0.0.1:8090` (the
`coder-llm` user service). Until 2026-11-15 that server ran around the clock **outside any gpu-hold ticket**, so "one GPU
job at a time" never held, and its model is the one whose prefill faults the GPU (engine#2494). It now runs only inside a
ticket, in on-demand windows, through `scripts/coder-llm-ticketed` (deployed to `~/.local/bin`):

1. **It waits for demand.** Demand is the file `~/.cache/coder-llm/demand` touched within the last 30 minutes (the
   `pr-agent` workflow's `reviewer` job touches it), or, as the fallback that is on by default (`CODER_LLM_GH_POLL=1`),
   a `pr-agent` **pull_request** run that is queued or in progress (comment runs are not counted: most never review). So
   a review is never starved by a workflow that forgot to ask. A poll that cannot ask (an expired token, a rate limit) is
   logged once per wait as `gh poll failed (<reason>)` and treated as no demand.
2. **It queues for the GPU like any other job**: `gpu-hold -l coder-llm -t 25`. The FIFO queue, the temperature, memory
   and kernel-fault watchdogs, and the release wait that lets the next label in all apply.
3. **Inside the ticket it serves**, and ends the window when no slot in `/slots` has been processing **and** the demand
   file has not been touched for 5 minutes, or when the 25-minute max hold ends it. A server that is stopping gets 30 s
   (`CODER_LLM_STOP_WAIT_S`), and gpu-hold's TERM-to-KILL grace for the window is set 10 s longer. That way the window
   ends its own server before gpu-hold KILLs it. The 30B server needed more than gpu-hold's default 10 s at the first
   live max hold (2026-11-15 10:56Z), which ended rc 137. A KILL after the max hold is still reported as rc 124.
4. **It exits**, and the unit's `Restart=always` starts the next invocation, which waits for demand and queues again at
   the back of the line. After a kernel fault (gpu-hold rc 78) it backs off 10 minutes first. It queues with a long
   `gpu-hold -w` (24 h): a ticket that gave up would be replaced at the back, so a short wait on a busy box starves it.

Anything else that needs the model asks the same way: touch `~/.cache/coder-llm/demand`, keep touching it while it
works, and wait for `:8090` to answer a real completion. Until a window opens, the port refuses.

**The contract this changes: `systemctl --user start coder-llm` now means *queued*, not ":8090 answers".** A window
wrapper that restarts the unit from inside its own gpu-hold ticket and then polls `:8090` is polling a server that is queued
behind that very ticket. The server can answer only after the caller's ticket is released, so such a poll must not count a
silent `:8090` as "the fleet LLM did not come back". The unit's journal says `queued behind <holder>` when it queues. With
the model ticketed, a memory window no longer needs to stop it: it is not resident outside its own windows.
`scripts/window-run.sh` therefore leaves a ticketed unit alone (its window log says `coder-llm ticketed: not stopped`).
Stopping it would only drop its queued ticket. That happened on 2026-11-15 at 10:00Z, when a window stopped the
just-switched unit and its place in the queue was lost.

**How long a measurement ticket must be willing to wait: `-w` > the holder's remaining hold, not `-w 600`.** The ticketed unit holds with
`-t 25` and re-queues with `-w 86400`, so a ticket queued *after* it waits out the remainder of a 25-minute hold before
it becomes head. Measured 2026-11-16: two tickets on this box expired unseen, one at `-w 600` and one at `-w 120`,
while the holder was mid-hold — and the failure line looks identical to a jammed queue (`queue never came round …
(head: …coder-llm, token: gpu-hold coder-llm pid …)`), so the honest reading is *"I did not wait long enough"*, not
*"the queue is broken"*. Queue order is fair: the ticket keeps its place, and the holder's *re-queued* ticket takes a
later id, so waiting out one hold is enough. **Do not stop the unit to shorten the wait** — it is ticketed, stopping it
only drops its queued place (2026-11-15 10:00Z).

**It is not only `coder-llm`.** The same arithmetic applies to every long holder, and the worst one is **`hw-gate`**, whose job timeout is **150 minutes** (`hw-gate.yml`'s `timeout-minutes: 150` — the 5-minute one is `pin_moved`, the hosted diff job that never takes the token) -- measured 2026-11-16 20:30Z, a `-w 300` sweep expired behind `hw-gate_run_43983474842_pid_1295035` which had taken the token ten minutes earlier and was still inside its own timeout. The rule is therefore about the **holder**, not about a particular service: read the token's start time, subtract from the holder's timeout, and wait at least that long — or come back later. `-w 600` is shorter than most holds on this box and should be treated as an expired ticket, not a finding -- and behind `hw-gate` the number to beat is up to **9000 s**, not the 1800 that covers `coder-llm`.

**What waits actually look like, from one night's tickets.** Measured 2026-11-16/17 on this box: five tickets that
reached the device took the token after **686, 734, 910, 1294 and 1840 s**, while three shorter ones **expired** at 120,
300 and 600 s with a holder mid-hold — including one behind an `hw-gate` run, which is unremarkable once the holder's
bound is **150 minutes**. The longest wait logged anywhere in that night's task outputs was **10270 s (2 h 51 m)**, so a
wait of tens of minutes is ordinary and a wait approaching three hours is still not a hang. Size `-w` from the holder's
own timeout plus a margin, and treat an expiry as *"my number was too small"* until the queue has been inspected.

**Read the window's end line before believing a restart failed.** `scripts/window-run.sh` ends every window with one
`end` line in `~/briefs/WINDOWS.log`, and the state it names decides whether anything is wrong. Four tokens, four
meanings — only the last is a fault:

| the line says | what was true when the window ended | what to do |
|---|---|---|
| `restart=queued(WAITING for a GPU ticket when this window ended, not restored)` | `coder-llm` is ticketed and waiting behind other tickets: `systemctl start` means *queued*, so `:8090` is silent **by design** | nothing; it answers when its ticket is granted |
| `restart=deferred(gpu held by <label> (pid N))` | another lane held the GPU token, so the unit could not load: the window started it and did **not** wait on `:8090` | nothing; it comes up when that ticket is released |
| `asleep-no-demand` | the unit is `active` and its journal says it is waiting for demand (ticketed units sleep between requests) | nothing; a request wakes it |
| `restart=FAILED` | `:8090` did not answer a 1-token completion and none of the above held | investigate: it was restarted and did not come back |

A silent `:8090` is therefore **not** evidence of a failed restart on its own — three of these four states are silent on
purpose, and a window that reports them as failures is how lanes learn to ignore failures (#2496, #2547, #2566).
`--restart-rc` encodes the same table, and its self-test asserts each row.

### Other resident models: 1bit-canonical on :8095 (engine#2579)

The same wrapper runs any model that used to stay resident. engine#2579 measured `1bit-canonical.service` (Qwen3-4B
on HRX, `:8095`) costing 3-9 % of every other GPU job while it sat idle. Ticketed, it gets its own gpu-hold label
(`1bit-canonical`), its own demand file (`~/.cache/1bit-canonical/demand`) and its own URL, read from its serve line's
`--port`. The pr-agent poll is coder-llm's alone.

    CODER_LLM_UNIT=1bit-canonical scripts/install-coder-llm-ticketed.sh --diff     # then --install, --status

**A caller asks before it requests**, with `scripts/ticketed-wait UNIT URL [SECONDS]`:

    ticketed-wait 1bit-canonical http://127.0.0.1:8095 900 || { echo "not run: :8095 queued"; exit 75; }

The script touches the demand file every 30 s while it waits. Each poll is a cheap `GET /v1/models` (liveness), then
one 1-token chat completion (readiness): `/v1/models` is a static listing that answers 200 from bind while the
weights are still loading, and `/health` reads 503 both while loading and when idle (okf
`references/health-is-a-progress-probe-not-a-liveness-probe.md`). It returns 0 once the model
answers, at once if the unit is resident or its window is already open, and 75 after the limit. A 75 means the window
is queued behind other tickets, so the run is **not run**, never a fake answer. On a 75 it also **removes the demand
file**, so a caller that has gone does not buy a GPU ticket for nobody. A unit other than coder-llm also takes demand
as stale after 5 minutes (`CODER_LLM_DEMAND_TTL_MIN=5`), so even a crashed caller's touch expires quickly. A caller that works longer than the
unit's idle time (5 min) keeps touching the demand file, or the window closes under it.


- with no argument, or `--diff`, it prints the drop-in change;
- `--status` says whether the wrapper is installed and the running unit executes it (exit 0 only when both are true);
- `--install` backs the drop-in up as `<name>.bak-<YYYYMMDD>`, rewrites it, installs the wrapper, reloads and restarts.

The server command stays byte for byte; the wrapper goes in front of it:

```ini
[Service]
ExecStart=
ExecStart=%h/.local/bin/coder-llm-ticketed -- <the 1bit serve command line>
Environment=CODER_LLM_GH_POLL=1
Restart=always
RestartSec=15
TimeoutStopSec=90
```

`tests/coder_llm_ticketed_test.sh` (ctest `coder_llm_ticketed`) drives the wrapper with the real `scripts/gpu-hold` on a
private queue and a fake `/slots` server. No device is used.

## On the 1bit OS appliance (2026-11-27)

Since 2026-11-22 the box runs the appliance image: Puppy Linux (`PUPMODE=12`, so writes persist on the NVMe), with
**no systemd**, no compiler and no `gpu-hold`. The device lock is `/usr/local/bin/box-hold gpu|npu|both <cmd>`, which
takes **both** the GPU and NPU locks for any ticket. Everything on this page about units, `OOMPolicy`, `fleethold`
and `coder-llm-ticketed` describes the old dev box. On the appliance, the PR reviewer is set up like this:

| piece | where | started by |
|---|---|---|
| runner `strixhalo-pr-agent`, labels `pr-agent,issue-agent` | `/var/lib/ghrunner/actions-runner-pr-agent`, user `ghrunner` (no shell, home `/var/lib/ghrunner`) | `/etc/init.d/52github-runner-pr-agent` ([source](../scripts/appliance/52github-runner-pr-agent)) |
| the window watcher and its server, as `ghrunner` (in `render`/`video` for the GPU, so box-hold serves its tickets in the CI class; it was the owner's account until 2026-11-27) | `/usr/local/bin/coder-llm-window` ([source](../scripts/appliance/coder-llm-window)), root-owned so a job cannot rewrite it | `/etc/init.d/51coder-llm-window` ([source](../scripts/appliance/51coder-llm-window)) |
| the model | `/data/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf` (sha256 `2841aa31…2533`) | — |
| logs | `/data/logs/coder-llm/{window,server}.log`, `<runner>/runner.log` | — |
| the device lock `box-hold` (FIFO since 2026-11-27) | `/usr/local/bin/box-hold` ([source](../scripts/appliance/box-hold)), locks and queue in `/tmp/box-locks` | — |

**Restart policy without systemd.** Each init script starts a detached supervisor loop (`while :; do …; sleep N; done`),
which is `Restart=always` by another name. `run.sh` exits 0 when its listener stops (engine #1504), so a script that
only starts the runner once at boot leaves it dead after the first crash.

**The labels leave out `strixhalo` on purpose.** `hw-gate.yml`'s `hw_gate` job asks for
`[self-hosted, strixhalo, pr-agent]`, which was the old `strixhalo-engine` runner. A pr-agent runner with that label
takes hardware-gate jobs outside `box-hold`. On 2026-11-27 it took a 12-hour-queued one within a second of coming
online. With `pr-agent,issue-agent` it takes only the jobs it is for.

**The window (engine#2494, on box-hold).** The `reviewer` job touches `$HOME/.cache/coder-llm/demand`, which for this
runner is `/var/lib/ghrunner/.cache/coder-llm/demand`. The watcher works like this:

1. When the file is less than 120 s old, the watcher queues `BOX_HOLD_WHO=coder-llm box-hold gpu coder-llm-window window`.
2. Inside the ticket it checks again. If the demand went stale while it was queued (no touch for 7 minutes), or
   `MemAvailable` is under 40 GB, it starts nothing.
3. Otherwise it starts `1bit serve --device hrx --port 8090 --alias qwen3-coder --ctx-size 65536` in its own session.
4. The window closes after 5 minutes without a touch, or at 44 minutes (the box rule is one ticket ≤ 45 min). If the
   demand is still fresh, it re-queues at the back.
5. Closing TERMs the server's whole session, waits 30 s, KILLs what is left, and then KILLs anything still carrying
   `--port 8090`, so the `llama-server` grandchild cannot outlive the ticket.

`tests/coder_llm_window_test.sh` (ctest `coder_llm_window`) checks each of these steps with a fake server and a fake
`box-hold`.

**Measured end to end on 2026-11-27.** The window opened at 10:33:17Z with MemAvailable 121 GB, and the model answered a
one-token completion about 30 s later. While serving, `llama-server` had an RSS of 14.7 GB and MemAvailable was 97 GB.
Two reviews ran in that one window. #3547's `pr_agent` took 67 s and #3537's took 67 s, and both posted their
"PR Reviewer Guide". The window closed at 10:41:36Z, after 499 s, 302 s of them idle. After that no `llama-server` was
left, `:8090` refused connections, and the next lane took the lock in the same second.

**box-hold serves waiters in order (2026-11-27).** Until then `box-hold` was a bare `flock`. Measured on 2026-11-27,
the first coder-llm window queued at 05:54Z and was still waiting when its 3600 s ran out at 06:54Z: five lane tickets
that arrived later took the lock first, and every 10-minute reviewer in that hour failed with "reviewer unavailable".
[`scripts/appliance/box-hold`](../scripts/appliance/box-hold), installed at `/usr/local/bin/box-hold`, keeps the old
behaviour: both locks for every ticket, `BOX_HOLD_WAIT`, `BOX_HOLD_WHO`, the `.who` files, exit 4 on timeout. On top
of that it numbers each waiter and puts a ticket for it in `/tmp/box-locks/queue/<n>.<pid>.<start-time>`. Only the
lowest *live* ticket may take the flock. A ticket whose process is gone, or whose pid now belongs to a different
process (the start time does not match), is skipped. `tests/box_hold_fifo_test.sh` (ctest `box_hold_fifo`) checks this
with stub jobs.

**CI tickets are served next (2026-11-27).** Measured that day, #3550's `hrx_build` gave up after its 90-minute wait
with 32 lane tickets queued, so every PR's device rows would keep timing out. box-hold therefore has a priority class:
- **Who is CI:** a ticket that asks with `BOX_HOLD_PRIORITY=ci` **and** really comes from CI. That means either it runs
  under a GitHub Actions runner (`GITHUB_ACTIONS=true` and a `Runner.Worker` among its ancestors), or it runs as the
  runner account `ghrunner`. The variable alone does not count, and neither does the account alone. A lane that sets
  the variable by hand is queued as a lane, and box-hold reports that it ignored the request, on stderr and in
  `/tmp/box-locks/box-hold.log` (orchestrator, 27 Nov). `hrx-build.yml` and `hw-gate.sh` under CI (`GITHUB_RUN_ID` set)
  ask for it.
- **Order:** CI tickets are served after the current holder and ahead of every queued lane ticket, FIFO among CI
  tickets. Lanes stay FIFO among themselves.
- **Cap:** a CI ticket's command runs under `timeout ${BOX_HOLD_CI_MAX:-2700}`, the 45-minute ticket rule, because it
  jumps the line.
- **Ticket names:** `0-<n>…` for CI and `1-<n>…` for lanes, sorted with `LC_ALL=C`. Tickets of the previous format
  sort between the two, so lanes already queued keep their place ahead of new lanes.

The head of the queue no longer blocks in `flock`; it tries the locks and re-reads the queue between tries. A head
blocked in `flock -w` never saw a CI ticket that arrived behind it. The same change fixed a defect as old as box-hold:
`exec 9>lock 2>/dev/null` on a bare `exec` made the `2>/dev/null` permanent, so **every box-hold job's stderr was
discarded** until 2026-11-27.

The new script was installed by `mv` over the old one, never edited in place, because running tickets read the file
incrementally (a mid-run edit corrupted a run on 11-24). Waiters that started before the swap keep running the old
copy, so they still race on the bare flock until they drain. The reviewer still waits 50 minutes because lanes hold
tickets of up to 45 minutes. The runner takes only `pr-agent`/`issue-agent` jobs, so the long wait blocks nobody.

**Install, start to finish** (as the owner, from ryzen):

```bash
# model: rsync over the LAN, then compare sha256 on both ends
rsync -a --partial ~/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf strixhalo:/data/models/qwen3-coder-30b-a3b/
# account + scripts
ssh strixhalo 'sudo useradd -r -d /var/lib/ghrunner -m -s /bin/false ghrunner && sudo chmod 755 /var/lib/ghrunner'
scp scripts/appliance/{coder-llm-window,51coder-llm-window,52github-runner-pr-agent} strixhalo:/tmp/
ssh strixhalo 'sudo install -m 755 /tmp/coder-llm-window /usr/local/bin/ && sudo install -m 755 /tmp/5[12]* /etc/init.d/'
# runner tree: a copy of the bench runner's release files, owned by ghrunner, then register it (token on stdin, never printed)
gh api -X POST repos/1bit-MONSTER/engine/actions/runners/registration-token --jq .token | ssh strixhalo \
  'sudo -u ghrunner env HOME=/var/lib/ghrunner bash -c "cd /var/lib/ghrunner/actions-runner-pr-agent && t=\$(cat) &&
   ./config.sh --unattended --url https://github.com/1bit-MONSTER/engine --token \"\$t\" --name strixhalo-pr-agent --labels pr-agent,issue-agent"'
ssh strixhalo 'sudo /etc/init.d/51coder-llm-window start; sudo /etc/init.d/52github-runner-pr-agent start'
```

BusyBox `pkill` on this image has no `-u`. A `pgrep -f` pattern that names a command also matches the ssh shell that
runs it (that is how a `kill $(pgrep -f …)` over ssh ends its own session), so the scripts use `kill $(pgrep …)` with
patterns that only their own processes carry.

## Exit codes on the GPU and NPU path: one table

A lane reading an exit code wants one answer: **did my job fail, or did the box not let it run?** The fleet's scheduler
scripts answer it with a few shared codes. Every code but **2** has a single meaning across the scripts. 2 is the exception,
with four causes that its stderr line tells apart (the row says which). The "scripts" column says whether a script
*exits* a code itself, *passes it through* from gpu-hold, or *relabels* it, because that tells you which script to read.
Each script's own header stays the reference for its edge cases.

| rc | meaning | scripts | retry? |
|---|---|---|---|
| 0 | the job (or gate) passed / the window closed on its own | all | - |
| 1 | the job's own failure: a check failed | hw-gate.sh, the payload | no: read the log |
| 2 | one code, four causes; **the stderr line says which** and is part of the contract (each phrase below is what the script prints, so grep your log for it): (a) a **usage error**, `gpu-hold: unknown option` or `gpu-hold: no command given`: the command line is wrong, not the box; (b) **could not run / INCOMPLETE**, `INCOMPLETE (an input was absent: nothing was tested)`: an input was absent (a model, text, a binary, a checkout) or a build or memory floor refused it. hw-gate reports 2 even when some rows failed, because an untested input means the run did not answer its question; (c) the box could not set up the ticket, `gpu-hold: cannot create` or `gpu-hold: cannot write`; (d) a **policy refusal**, `npu-hold: REFUSED:` (a shared queue while npu-serve-keeper reads it) | gpu-hold and hw-gate.sh exit it; npu-hold exits (d) | **read stderr first**. (a) and (d): fix the command or config, because re-queueing repeats the same mistake; (b): after providing the input; (c): after fixing the box |
| 3 | the server under test never came up | hw-gate.sh exits it from its server-start step | read the server log |
| 69 | the job ran, but coder-llm did not answer `:8090` after the window | window-run.sh (computed by `final_rc`, not a literal exit) | see "The PR-Agent model runs in gpu-hold windows" |
| 75 | the queue never came round within `-w`, or a ticketed model did not answer within `ticketed-wait`'s limit | gpu-hold exits it; ticketed-wait exits it; npu-hold and coder-llm-ticketed pass it through | yes, but **at the back** of the queue. That is why coder-llm queues with a 24 h `-w`, and why a lane should size `-w` for the queue instead of treating 75 as a free retry |
| 76 | the memory pre-check refused the run | hw-gate.sh `--mem` exits it | when MemAvailable is back |
| **78** | a GPU/NPU fault line in the kernel log: **during** the hold (the hold was ended) or **at take time** (the ticket was refused, nothing ran). The coder-llm journal says which one | gpu-hold exits it; npu-hold and coder-llm-ticketed pass it through (the coder-llm journal names the cause) | after the fault is understood. Never in a loop |
| **124** | the max hold (`-t`) ended the job: a deadline, and planned. coder-llm-ticketed also maps a KILL after its max hold to 124 | gpu-hold sets it (its own code, after `timeout(1)`'s convention; it also maps a TERM at its deadline from 143); npu-hold passes it through; coder-llm-ticketed passes it through and relabels a post-deadline 137 to it | yes |
| 130 | a stop was requested and the job refused to die; the token was released anyway | gpu-hold sets it; coder-llm-ticketed passes it through | check for a survivor (`~/GPU-QUEUE/.gpu-hold-survivors`) |
| **137** | SIGKILL **before** any deadline: the OOM killer or a `kill -9`. A failure, not a timeout | any payload | no: find the killer |
| 143 | SIGTERM (a cancel) that was not the max hold | any payload | depends on who sent it |

Three rules follow from the table:

- **75, 124 and 78 are not the job's verdict.** A gate or a measurement that ends with one of them has no result. Report
  it as "not run", never as a pass or a fail.
- **78 is never retried in a loop.** coder-llm backs off 10 minutes, and a lane stops and logs the time plus the command
  in its state file (the fault rule).
- **137 is only a timeout when something says so.** gpu-hold maps a TERM at its deadline to 124. A bare 137 is a kill
  from outside the script.

## What a restart policy cannot do

It cannot notice that a runner is *not* coming back (a version mismatch, a revoked token, a full disk under
`/var/lib/ghrunner`). That is [engine #1504](https://github.com/1bit-MONSTER/engine/issues/1504)'s item 3, and it is
`scripts/check-ci-runners.sh`:

```sh
scripts/check-ci-runners.sh --list                                      # what the API says, and the verdict
scripts/check-ci-runners.sh --require strixhalo-engine strixhalo-bench  # the fleet, named
```

The sentence to keep in mind while reading any check list: **every check that needs this box reads as pending,
never red**. That is why the 2026-11-08 outage lasted 3h45m instead of being visible in a minute — a job whose
runner does not exist never runs, so GitHub never fails it, and a pending check looks exactly like a busy queue.
That is the person this row is for: whoever is staring at a green-looking list.

It exits 0 when both named runners are `online`, 1 with a `::error::` line per offender, and **2 when the list
cannot be read at all** — an unchecked runner is exactly the state the row exists to end. It is also 2 when the list
is **truncated** (one page is 100 runners): past that, a runner that exists on page two would be reported missing,
which is a false outage, and a monitor that cannot tell has to say so rather than answer wrongly. `--require` is not decoration: a
runner that has been *deleted* satisfies "every registered runner is online", so the two names have to be asked for
explicitly. A repository with no self-hosted runners registered is not an outage, so a fresh box and a fork both
exit 0.

`scripts/weekly.sh` runs it in the preflight, as a fatal row: a release whose post-release checks queue for ever is
worse than a release that waits an hour, and `WEEKLY_SKIP_RUNNER_CHECK=1` is the deliberate way to publish during an
outage. `tests/check_ci_runners_test.sh` (ctest `check_ci_runners`) covers the arms with a fake `gh`: both online,
one offline, an unrequired dead runner, a required runner that is not registered, nothing registered, an API failure,
and `--list`.

It cannot be a ctest itself — it needs the network and a token, which the GitHub-hosted job does not have. A
*scheduled* red (a cron workflow that fails once an hour while a runner is down) is the complete answer and is not
here: a new workflow changes what the repository's token may do, which is an RFC under CONTRIBUTING.md.

## Two rows, one question: are the runners healthy?

Since #1682 and #1546 both landed there are two checks, and neither alone is enough — each is blind to what the
other sees. Both run in `scripts/weekly.sh`'s preflight. A runner that is **not accepting jobs** stops the release
before it starts. A runner that is up but whose **restart policy is short** is a warning, and the release goes on
(owner decision on engine#1504, 2026-11-11): see [the next section](#a-short-restart-policy-warns-it-does-not-block).

| | `scripts/check-ci-runners.sh` | `scripts/ci-runner-health.sh` |
|---|---|---|
| reads | the GitHub API (`…/actions/runners`) | systemd (`systemctl show`) |
| says | is it **registered** and `online` | is the unit **active**, `Result=`, `NRestarts` |
| notices | a registration that was revoked, expired or never made | `Result=oom-kill`, a restart window, a unit that stopped |
| cannot see | *why* it died — no `oom-kill`, no restart count | a unit that is running fine while its registration is gone |
| speed | after the heartbeat times out | the moment the unit dies |
| exit | 0 online · 1 an offender, named · **2 cannot tell** (unreadable or truncated list) | 0 all accepting, policy right · 1 one is not accepting jobs · 2 no such units on this host · **3 all accepting, a restart policy is short** |

Run either by hand — both are read-only and neither restarts anything:

```sh
scripts/check-ci-runners.sh --require strixhalo-engine strixhalo-bench --list
scripts/ci-runner-health.sh                # every actions.runner.* unit
scripts/ci-runner-health.sh --self-test    # the verdict logic, systemctl stubbed
```

`WEEKLY_SKIP_RUNNER_CHECK=1` releases during an outage, and it skips **both** rows: they fail for the same dead
runner, so a flag that skipped only one of them could not do the job it documents. It also says so in the log --
`preflight: runner checks SKIPPED (WEEKLY_SKIP_RUNNER_CHECK=1)` -- because a release log where the checks quietly did
not run reads exactly like one where they passed. Their exit 2 means "nothing to
inspect on this host" — a fresh box or a fork — which is a note, not a failure; a check that cannot see has to say so
rather than report green.

One case neither row covers: a runner that is `online` in the API *and* active locally, but wedged so it takes no
jobs. That is a hung process, not a registration or unit state; the practical signal for it is a queue that stops
moving while both rows stay green.

## A short restart policy warns, it does not block

`scripts/ci-runner-health.sh` prints two lines per unit: its state (`PASS`/`FAIL`) and its restart policy
(`PASS`/`WARN`). The exit code keeps them apart, because they call for different things:

| exit | meaning | `scripts/weekly.sh` preflight |
|---|---|---|
| 0 | every unit is accepting jobs and would come back after an OOM kill | passes |
| 1 | a unit is not accepting jobs now (dead, or in its restart window) | **blocks**: the checks after the release would queue for ever |
| 2 | no `actions.runner.*` units on this host | a note |
| 3 | every unit is accepting jobs, but a `WARN` line names a policy setting | **warns** and goes on |

A unit that is down outranks a policy line, so 1 wins over 3. The warning in the release log reads
`preflight: WARNING, not blocking: every runner unit is accepting jobs, but a restart policy would leave one down
after the next OOM kill`, with the `WARN` lines above it naming the unit and the setting.

Why a warning is enough: the policy is a box configuration change, which the owner applies (below), and a release
that cannot start does not make the unit any safer. Meanwhile the heavy builds run under
`scripts/mem-guard.sh --term`, which ends the build before the kernel's OOM killer reaches the runner. **A build step
that ends with exit 143 was stopped by that guard**: the log has a `GUARD: MemAvailable N GB < floor GB` line, the
runner is still alive, and the job is re-run once memory frees. The guard covers only the steps that use it. A server
or a bench started without it can still take the unit down, which is why the drop-ins below are the real fix.

`tests/weekly_runner_preflight_test.sh` (ctest `weekly_runner_preflight`) runs the real health script and the real
`preflight_runner_units()` from `scripts/weekly.sh` against a stub `systemctl`: the four exit codes, block on a dead
unit, warn on a short policy.

### The two drop-ins (owner step)

On strixhalo, `strixhalo-bench` and `strixhalo-org` still run the units that predate `scripts/install-ci-runner.sh`:
`Restart=no`, `OOMPolicy=stop`. One OOM kill of any process in the unit stops it for good. `strixhalo-engine` got
the same drop-in on 2026-11-10. This needs `sudo`, so it is the owner's to run. It does not re-register a runner,
change its user, or stop a job in flight.

Both files have the same content, [`scripts/ci-runner-restart.conf`](../scripts/ci-runner-restart.conf):

```ini
# /etc/systemd/system/actions.runner.1bit-MONSTER-engine.strixhalo-bench.service.d/restart.conf
# /etc/systemd/system/actions.runner.1bit-MONSTER.strixhalo-org.service.d/restart.conf
[Unit]
StartLimitIntervalSec=0

[Service]
Restart=always
RestartSec=15
OOMPolicy=continue
```

The three commands, from the root of an engine checkout:

```sh
sudo install -Dm644 scripts/ci-runner-restart.conf /etc/systemd/system/actions.runner.1bit-MONSTER-engine.strixhalo-bench.service.d/restart.conf
sudo install -Dm644 scripts/ci-runner-restart.conf /etc/systemd/system/actions.runner.1bit-MONSTER.strixhalo-org.service.d/restart.conf
sudo systemctl daemon-reload
```

Check it took (read-only, no `sudo`):

```sh
systemctl show actions.runner.1bit-MONSTER-engine.strixhalo-bench.service actions.runner.1bit-MONSTER.strixhalo-org.service -p Id -p Restart -p OOMPolicy
scripts/ci-runner-health.sh; echo "exit $?"
```

Want `Restart=always` and `OOMPolicy=continue` on both. The health script still exits 3 afterwards, with one `WARN`
per unit naming only `MemoryMax=infinity(want a ceiling)`: the drop-in sets no memory ceiling. The ceiling this page
documents (`MemoryHigh=43G`, `MemoryMax=48G`) comes with a full `scripts/install-ci-runner.sh` install, or with
`sudo systemctl set-property <unit> MemoryHigh=43G MemoryMax=48G` per unit. To undo a drop-in, delete the
`restart.conf` and run `sudo systemctl daemon-reload`.

## Landing a change under `.github/workflows/`

A lane's token is the `gh` OAuth token, `gist, read:org, repo` and **no `workflow`** (`gh auth status`, and the
server's own `X-OAuth-Scopes` header agree). GitHub refuses the **HTTPS** push that introduces such a change, and the
refusal names the scope rather than the file:

```
$ git push https://github.com/1bit-MONSTER/engine.git HEAD:refs/heads/cl-1077-transport-probe2
 ! [remote rejected] HEAD -> cl-1077-transport-probe2 (refusing to allow an OAuth App to create or update
   workflow `.github/workflows/ci.yml` without `workflow` scope)
```

The same commit pushed over ssh is accepted — a key authenticates the account, not an OAuth grant:

```
$ git push ssh://git@github.com/1bit-MONSTER/engine.git HEAD:refs/heads/ci/2707-concurrency
 * [new branch]      HEAD -> ci/2707-concurrency
```

Both measured on this box on 2026-11-16 (engine#2707, whose `ci.yml` concurrency group went up that way as #2708;
#2696 landed the `census-watch.yml` excerpt fix the same way an hour earlier). **It grants nothing extra**: whoever
can push a branch can already write every other path in the repository, the change still arrives as a PR with the
same checks, and a *new* workflow is still an RFC under CONTRIBUTING.md. ssh moves the file; it does not move the
decision — so a lane holding a workflow fix that nobody can land should open the PR rather than park the fix on a
scope that only the transport was withholding.

One trap, because it produces the wrong conclusion in the other direction. GitHub scans the commits a push
*introduces*: pushing a ref whose commit is already on the server (an existing branch, or a commit reachable from
one) is accepted by the same token, and reads as "I do have `workflow` scope" when the push only proved the content
was already there. The two measurements above are a commit that exists nowhere else, which is the only form of the
test that means anything.
