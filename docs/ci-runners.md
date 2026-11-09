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

Two runners serve this repository, both on Strix Halo, both installed by
`scripts/install-ci-runner.sh`:

| runner | labels | takes |
|---|---|---|
| `strixhalo-engine` | `self-hosted,Linux,X64,pr-agent,strixhalo` | `pr-agent`, `issue-agent` |
| `strixhalo-bench` | `self-hosted,Linux,X64,strixhalo,bench` | `hrx-build` and the bench/gate jobs |

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
other sees. Both run in `scripts/weekly.sh`'s preflight, so a release refuses to start on a runner it cannot trust.

| | `scripts/check-ci-runners.sh` | `scripts/ci-runner-health.sh` |
|---|---|---|
| reads | the GitHub API (`…/actions/runners`) | systemd (`systemctl show`) |
| says | is it **registered** and `online` | is the unit **active**, `Result=`, `NRestarts` |
| notices | a registration that was revoked, expired or never made | `Result=oom-kill`, a restart window, a unit that stopped |
| cannot see | *why* it died — no `oom-kill`, no restart count | a unit that is running fine while its registration is gone |
| speed | after the heartbeat times out | the moment the unit dies |
| exit | 0 online · 1 an offender, named · **2 cannot tell** (unreadable or truncated list) | 0 all accepting · 1 one is not · 2 no such units on this host |

Run either by hand — both are read-only and neither restarts anything:

```sh
scripts/check-ci-runners.sh --require strixhalo-engine strixhalo-bench --list
scripts/ci-runner-health.sh                # every actions.runner.* unit
scripts/ci-runner-health.sh --self-test    # the verdict logic, systemctl stubbed
```

`WEEKLY_SKIP_RUNNER_CHECK=1` releases during an outage, and it skips **both** rows: they fail for the same dead
runner, so a flag that skipped only one of them could not do the job it documents. Their exit 2 means "nothing to
inspect on this host" — a fresh box or a fork — which is a note, not a failure; a check that cannot see has to say so
rather than report green.

One case neither row covers: a runner that is `online` in the API *and* active locally, but wedged so it takes no
jobs. That is a hung process, not a registration or unit state; the practical signal for it is a queue that stops
moving while both rows stay green.
