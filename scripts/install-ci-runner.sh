#!/usr/bin/env bash
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
#
# install-ci-runner.sh: install one self-hosted GitHub Actions runner as a systemd service, owned by a
# dedicated unprivileged user, with the restart and OOM policy that engine #1504 asks for.
#
#   sudo scripts/install-ci-runner.sh --name strixhalo-engine --labels pr-agent \
#        --token-file /root/runner-token --runner-dir /srv/actions-runner-2.329.0
#
# It runs on the box the jobs run on (Strix Halo), not from CI. It creates the user, the runner directory,
# the unit, and starts it; it is idempotent (an existing user or runner config is reused) and it never
# prints the registration token.
#
# Why each policy is here, in one line each - the detail is in docs/ci-runners.md:
#   Restart=always        runsvc.sh exits 0 after an OOM kill ("no retry needed"), so on-failure never fires
#   OOMPolicy=continue    an OOM kill inside the cgroup kills the JOB; the runner keeps running
#   MemoryHigh/Max        the cgroup hits its own ceiling first, so the box's OOM killer is not choosing
#                         between "the biggest allocation on the machine" and everything else
#   User=ghrunner         the jobs stop running as the owner: no ssh keys, no ~/models, no fleet remotes
#   StartLimitIntervalSec=0  a runner must never be rate-limited out of existence; RestartSec bounds the loop
#
# Options:
#   --name NAME            runner name and unit suffix (default: strixhalo-engine)
#   --labels LIST          extra labels beyond self-hosted,Linux,X64 (default: pr-agent)
#   --token-file FILE      file holding a registration token (repo Settings -> Actions -> Runners -> New)
#   --runner-dir DIR       an already-unpacked runner (config.sh/runsvc.sh in it). Downloaded if omitted,
#                          with --runner-version, and verified against the release's SHA256SUMS
#   --runner-version VER   e.g. 2.329.0 (only used when downloading)
#   --user USER            dedicated user (default: ghrunner)
#   --groups LIST          extra groups for that user (default: docker; use "" for none)
#   --memory-max SIZE      MemoryMax for the unit (default: 48G); MemoryHigh is 90% of it
#   --share-with USER      the existing user whose fleet state the jobs must keep using (default: the owner
#                          account, bcloud). Its ~/lb, ~/.local/bin, ~/GPU-TOKEN, ~/GPU-QUEUE, ~/NPU-TOKEN
#                          and ~/NPU-QUEUE are linked into the runner's home, and the runner's account joins
#                          that user's group so it can write the queue. Without this a job would take ITS OWN
#                          GPU token and run GPU work concurrently with every lane holding the real one
#   --no-share             skip that (only for a box with no fleet to share with)
#   --env-file FILE        optional EnvironmentFile= for the unit
#   --dry-run              print the unit and verify it, change nothing
#   --verify               only report the installed unit's state
set -euo pipefail

name=strixhalo-engine labels=pr-agent user=ghrunner groups=docker memory_max=48G
token_file='' runner_dir='' runner_version='' env_file='' share_with=bcloud no_share=0 dry=0 verify=0
usage() { sed -n '16,48p' "$0"; }

while [ $# -gt 0 ]; do
    case $1 in
        --name) name=$2; shift 2 ;;
        --labels) labels=$2; shift 2 ;;
        --token-file) token_file=$2; shift 2 ;;
        --runner-dir) runner_dir=$2; shift 2 ;;
        --runner-version) runner_version=$2; shift 2 ;;
        --user) user=$2; shift 2 ;;
        --groups) groups=$2; shift 2 ;;
        --memory-max) memory_max=$2; shift 2 ;;
        --env-file) env_file=$2; shift 2 ;;
        --share-with) share_with=$2; shift 2 ;;
        --no-share) no_share=1; shift ;;
        --dry-run) dry=1; shift ;;
        --verify) verify=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "install-ci-runner.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

unit=actions.runner.1bit-MONSTER-engine.$name.service
unit_path=/etc/systemd/system/$unit
dir=/var/lib/$user/actions-runner-$name
memory_high=$(awk -v m="$memory_max" 'BEGIN{ sub(/G$/,"",m); printf "%dG", int(m * 0.9) }')

# <-- the unit this recipe installs
unit_text() {
    cat <<EOF
# engine #1504: installed by scripts/install-ci-runner.sh, documented in docs/ci-runners.md.
[Unit]
Description=GitHub Actions runner $name (dedicated user $user)
After=network-online.target
Wants=network-online.target
# a runner is a daemon: it must never be rate-limited out of existence. RestartSec bounds the loop instead,
# so a crash-looping runner retries every 30 s for ever rather than being given up on silently.
StartLimitIntervalSec=0

[Service]
Type=simple
User=$user
WorkingDirectory=$dir
ExecStart=$dir/runsvc.sh
# hw-gate.sh resolves gpu-hold with `command -v`, and the runner's PATH is what decides which token it takes
Environment=PATH=/var/lib/$user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
$( [ -n "$env_file" ] && echo "EnvironmentFile=$env_file" )
# runsvc.sh exits 0 after an OOM kill ("no retry needed"), so on-failure would leave the runner dead.
Restart=always
RestartSec=30
# ... and an OOM kill inside this cgroup kills the job, not the runner. This is the line whose absence let
# both runners sit dead for 3h45m with Result=oom-kill and NRestarts=0.
OOMPolicy=continue
# The job's ceiling, so the box's OOM killer is not choosing the runner as "the biggest allocation here".
# MemoryHigh throttles first; MemoryMax is the kill. Sized for the fleet's heaviest bench (48G).
MemoryHigh=$memory_high
MemoryMax=$memory_max
TasksMax=8192
# what the shipped units already had: stop the listener, leave its children to finish their own timeout
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min

[Install]
WantedBy=multi-user.target
EOF
}

state() {   # what the unit is doing now, in the terms the issue was filed in
    systemctl show "$unit" -p Result -p NRestarts -p OOMPolicy -p Restart -p MemoryMax -p ActiveState 2>/dev/null
}

if [ "$verify" = 1 ]; then
    state
    exit 0
fi

if [ "$dry" = 1 ]; then
    unit_text
    # verify the DIRECTIVES, not the install: ExecStart does not exist yet, and systemd-analyze treats a missing
    # binary as an error. The copy it verifies points at /bin/true, so an unknown or malformed setting still
    # fails here while the not-yet-installed path does not.
    tmp=$(mktemp --suffix=.service)
    unit_text | sed "s|^ExecStart=.*|ExecStart=/bin/true|" > "$tmp"
    if systemd-analyze verify "$tmp"; then echo "dry run: every directive verifies; nothing was changed"
    else echo "dry run: the unit does not verify" >&2; rm -f "$tmp"; exit 1; fi
    rm -f "$tmp"
    exit 0
fi

[ "$(id -u)" = 0 ] || { echo "install-ci-runner.sh: run as root (it creates a user and a unit)" >&2; exit 2; }
[ -n "$token_file" ] && [ -s "$token_file" ] || { echo "install-ci-runner.sh: --token-file is required and must not be empty" >&2; exit 2; }

# the dedicated user (engine #1504's audit finding): the jobs must not run as the owner
if ! id "$user" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "/var/lib/$user" --shell /usr/sbin/nologin "$user"
    echo "created user $user (no shell, home /var/lib/$user)"
fi
for g in ${groups//,/ }; do
    getent group "$g" >/dev/null 2>&1 || { echo "warning: no group '$g' on this box; $user is not in it (jobs that need it will fail loudly)" >&2; continue; }
    id -nG "$user" | tr ' ' '\n' | grep -qx "$g" || usermod -aG "$g" "$user"
done

# the runner itself
mkdir -p "/var/lib/$user" && chown "$user:" "/var/lib/$user"
if [ ! -x "$dir/config.sh" ]; then
    if [ -n "$runner_dir" ]; then
        [ -x "$runner_dir/config.sh" ] || { echo "install-ci-runner.sh: $runner_dir is not an unpacked runner" >&2; exit 2; }
        mkdir -p "$dir" && cp -a "$runner_dir/." "$dir/"
    else
        [ -n "$runner_version" ] || { echo "install-ci-runner.sh: --runner-dir or --runner-version is required" >&2; exit 2; }
        tarball=actions-runner-linux-x64-$runner_version.tar.gz
        tmpd=$(mktemp -d)
        curl -fsSL -o "$tmpd/$tarball" \
            "https://github.com/actions/runner/releases/download/v$runner_version/$tarball"
        curl -fsSL -o "$tmpd/SHA256SUMS" \
            "https://github.com/actions/runner/releases/download/v$runner_version/SHA256SUMS" || true
        if [ -s "$tmpd/SHA256SUMS" ]; then
            ( cd "$tmpd" && grep " $tarball\$" SHA256SUMS | sha256sum -c - ) || {
                echo "install-ci-runner.sh: $tarball does not match the release's SHA256SUMS" >&2; exit 1; }
        else
            echo "install-ci-runner.sh: no SHA256SUMS on that release; refusing to install an unverified runner" >&2
            exit 1
        fi
        mkdir -p "$dir" && tar -C "$dir" -xzf "$tmpd/$tarball"
        rm -rf "$tmpd"
    fi
    chown -R "$user:" "$dir"
fi

# register, as the runner's own user, with the token read from the file and never echoed
if [ ! -f "$dir/.runner" ]; then
    tok=$(mktemp)
    cp "$token_file" "$tok"; chown "$user:" "$tok"; chmod 0600 "$tok"
    # config.sh takes the token as an argument and there is no stdin form, so it is in the child's argv for the
    # length of the call; the file is 0600 and root-owned, this copy is 0600 and the runner user's, and both the
    # copy and the temp dir go away here. The token itself is a one-hour registration token, never a PAT.
    sudo -u "$user" bash -c "cd '$dir' && ./config.sh --unattended --replace \
        --url https://github.com/1bit-MONSTER/engine \
        --token \"\$(cat '$tok')\" \
        --name '$name' --labels '$labels' --work '_work'" > /dev/null
    rm -f "$tok"
    echo "registered runner $name with labels self-hosted,Linux,X64,$labels"
fi
chown -R "$user:" "$dir"

# --- the fleet's state, at the paths the jobs already assume ---------------------------------------------
# The workflows and scripts reach the box's shared state through $HOME: ~/lb/thermal-run.sh (the thermal
# governor every build runs under), ~/.local/bin/gpu-hold (found with `command -v`), and the queue/token files
# ~/GPU-TOKEN, ~/GPU-QUEUE, ~/NPU-TOKEN, ~/NPU-QUEUE. A runner whose home is empty therefore does not just lose
# the tools - it takes a DIFFERENT GPU token and queues beside nobody, which is the one thing the fleet's hold
# rule exists to prevent (engine #1504 review). So the runner's home carries the same absolute paths, linked
# into the account that already holds them, and the runner joins that account's group to write the queue.
if [ "$no_share" = 0 ]; then
    id "$share_with" >/dev/null 2>&1 || { echo "install-ci-runner.sh: --share-with user '$share_with' does not exist (use --no-share on a box with no fleet)" >&2; exit 2; }
    share_home=$(getent passwd "$share_with" | cut -d: -f6)
    share_group=$(id -gn "$share_with")
    id -nG "$user" | tr ' ' '\n' | grep -qx "$share_group" || usermod -aG "$share_group" "$user"
    # the tools and the hold state, by name, only when they exist: an install on a box that has not set them
    # up yet is not an error, it is a missing feature the docs name
    for rel in lb .local/bin GPU-TOKEN GPU-QUEUE NPU-TOKEN NPU-QUEUE; do
        src="$share_home/$rel" dst="/var/lib/$user/$rel"
        if [ -e "$src" ]; then
            mkdir -p "$(dirname "$dst")"
            [ -L "$dst" ] && [ "$(readlink "$dst")" = "$src" ] && continue
            if [ -e "$dst" ] && [ ! -L "$dst" ]; then
                echo "warning: /var/lib/$user/$rel exists as a real path; leaving it (move it aside to share $src)" >&2
                continue
            fi
            ln -sfn "$src" "$dst"
            echo "shared: ~/$rel -> $src"
        fi
    done
    # the queue and the token files must be writable by the runner's new group; the modes are the owner's, so
    # they are tightened here rather than left to chance (0775 dir / 0664 file: group read+write, no world)
    [ -d "$share_home/GPU-QUEUE" ] && chgrp -R "$share_group" "$share_home/GPU-QUEUE" && chmod -R g+rwX,o-rwx "$share_home/GPU-QUEUE"
    [ -d "$share_home/NPU-QUEUE" ] && chgrp -R "$share_group" "$share_home/NPU-QUEUE" && chmod -R g+rwX,o-rwx "$share_home/NPU-QUEUE"
    for f in GPU-TOKEN NPU-TOKEN; do
        if [ -f "$share_home/$f" ]; then chgrp "$share_group" "$share_home/$f"; chmod g+rw,o-rw "$share_home/$f"; fi
    done
    # thermal-run.sh runs under `systemctl --user`, so the account needs a user manager of its own
    loginctl enable-linger "$user" 2>/dev/null || echo "warning: could not enable-linger $user; thermal-run.sh will use its process-group fallback"
    echo "shared state: $share_with's group ($share_group), queue and token files group-writable"
fi

unit_text > "$unit_path"
chmod 0644 "$unit_path"
systemd-analyze verify "$unit_path"
systemctl daemon-reload
systemctl enable --now "$unit"
sleep 5
echo
echo "installed $unit as user $user, MemoryMax=$memory_max, OOMPolicy=continue, Restart=always"
state
cat <<EOF

Verify it is really up, and can actually take work:
  systemctl show $unit -p Result -p NRestarts     # Result=success, NRestarts unchanged while idle
  gh api repos/1bit-MONSTER/engine/actions/runners --jq '.runners[] | "\\(.name) \\(.status) busy=\\(.busy)"'

The OOM behaviour the issue was filed for is checked by killing the cgroup's biggest child:
  systemd-run --unit=oom-test -p MemoryMax=256M -p OOMPolicy=continue sleep 600 &
  ( the unit reports Result=success and keeps running, instead of oom-kill with NRestarts=0 )
EOF
