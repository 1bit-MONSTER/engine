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
#   --hold-state DIR       where the fleet's GPU/NPU token and queue files live (default /var/lib/1bit-fleet).
#                          The installer creates it group-owned by --hold-group, mode 0770 with the setgid bit so
#                          tickets inherit the group, and puts the runner's account in that group - NOT in the
#                          owner's group. The unit is given GPU_TOKEN_FILE/GPU_QUEUE_DIR so the job queues with
#                          the fleet instead of beside it, and the owner's own environment must name the same
#                          two variables (docs/ci-runners.md, "One queue, two accounts"). A file inside the
#                          owner's home cannot be shared this way: gpu-hold RELEASES with `rm -f "$TOKEN"`,
#                          which needs write on the DIRECTORY, i.e. on the owner's home itself.
#   --hold-group GROUP     the group that owns the hold state (default fleethold)
#   --no-hold-state        leave the runner without hold state, for a runner that runs no hardware jobs (the
#                          pr-agent runner does not: it talks to the local LLM over HTTP). A runner without it
#                          must not run hw-gate/hrx-build, which is why the default is to set it up
#   --migrate-hold-state   move an existing ~/GPU-TOKEN, ~/GPU-QUEUE, ~/NPU-TOKEN, ~/NPU-QUEUE of
#                          --owner-user into --hold-state. Requires a quiet queue: it is the one step that
#                          changes the paths under every running lane
#   --env-file FILE        optional EnvironmentFile= for the unit
#   --dry-run              print the unit and verify it, change nothing
#   --verify               only report the installed unit's state
set -euo pipefail

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two of these ranges had
# drifted past the end of their header and printed code to whoever asked for help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

name=strixhalo-engine labels=pr-agent user=ghrunner groups=docker memory_max=48G
token_file='' runner_dir='' runner_version='' env_file='' hold_state=/var/lib/1bit-fleet hold_group=fleethold owner_user=bcloud no_hold=0 migrate=0 dry=0 verify=0

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
        --hold-state) hold_state=$2; shift 2 ;;
        --hold-group) hold_group=$2; shift 2 ;;
        --owner-user) owner_user=$2; shift 2 ;;
        --no-hold-state) no_hold=1; shift ;;
        --migrate-hold-state) migrate=1; shift ;;
        --dry-run) dry=1; shift ;;
        --verify) verify=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "install-ci-runner.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

unit=actions.runner.1bit-MONSTER-engine.$name.service
unit_path=/etc/systemd/system/$unit
dir=/var/lib/$user/actions-runner-$name
# MemoryHigh is 90% of MemoryMax IN ITS OWN UNIT: awk's numeric coercion reads "512M" as 512 and "2T" as 2,
# so the old one-liner turned 512M into 460G and 2T into 1G. A size with no K/M/G/T suffix is passed through
# unconverted, which makes MemoryHigh equal to MemoryMax: a hard limit with no throttle band above it, which is a
# fine choice for a runner (the job dies at the ceiling) and is what the code does.
share_home=$(getent passwd "$owner_user" 2>/dev/null | cut -d: -f6) || share_home=''
memory_high=$(awk -v m="$memory_max" 'BEGIN{
    if (m !~ /^[0-9.]+[KMGT]$/) { exit }        # no suffix: leave MemoryHigh alone
    v = substr(m, 1, length(m) - 1) + 0; u = substr(m, length(m))
    printf "%.1f%s", v * 0.9, u
}')
[ -n "$memory_high" ] || memory_high=$memory_max

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
Environment=PATH=/var/lib/$user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
$( [ "$no_hold" = 0 ] && printf 'Environment=GPU_TOKEN_FILE=%s/gpu-token GPU_QUEUE_DIR=%s/gpu-queue HW_GATE_LOCK=%s/npu.lock\n' "$hold_state" "$hold_state" "$hold_state" )
$( [ -n "${share_home:-}" ] && [ -d "$share_home/models" ] && printf 'Environment=HW_GATE_MODELS=%s/models\n' "$share_home" )
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

# --- the fleet's hold state, in a directory the fleet group owns -----------------------------------------
# The workflows reach the hold through $HOME (~/GPU-TOKEN, ~/GPU-QUEUE, and npu-hold's hard-coded
# ~/NPU-TOKEN, ~/NPU-QUEUE), and gpu-hold RELEASES a token with `rm -f "$TOKEN"` - a write to the DIRECTORY,
# not to the file. So a hold file that stays inside the owner's home cannot be shared with another account by
# chgrp'ing the file: either the runner is in the owner's group, which is group-write on the owner's whole home
# (/home/bcloud is 0775 here, so a job could replace ~/.bashrc or move ~/.ssh aside - an owner-level escalation
# created by a recipe whose point is that jobs no longer run as the owner), or the file is not shareable at all.
#
# So the hold state moves to a directory the FLEET group owns, and both sides name it:
#   <hold-state>/{gpu-token,gpu-queue,npu-token,npu-queue}   group-owned, 0770, queues setgid so tickets
#                                                            created by either account inherit the group
# The runner's account joins THAT group and nothing else; the shared tools (~/lb, ~/.local/bin) are already
# world-readable and executable, so no group is needed to run them.
# the workflows reach the shared tools by a HOME-RELATIVE path (~/lb/thermal-run.sh, and `command -v gpu-hold`
# for ~/.local/bin), so the runner's home needs those two entries. They are symlinks into the account that owns
# them and the runner reaches them through "other" (r-x), NOT through a group: it can run the tools and cannot
# rewrite them - which is the escalation the first version of this recipe created.
if [ -n "$share_home" ]; then
    for rel in lb .local/bin; do
        src="$share_home/$rel"; dst="/var/lib/$user/$rel"
        [ -d "$src" ] || continue
        mkdir -p "$(dirname "$dst")"
        if [ -e "$dst" ] && [ ! -L "$dst" ]; then
            echo "warning: /var/lib/$user/$rel exists as a real path; leaving it (move it aside to share $src)" >&2
            continue
        fi
        ln -sfn "$src" "$dst"
        echo "tools: ~/$rel -> $src (read-only through \"other\", no group)"
    done
fi

if [ "$no_hold" = 0 ]; then
    getent group "$hold_group" >/dev/null 2>&1 || groupadd --system "$hold_group"
    id -nG "$user" | tr ' ' '\n' | grep -qx "$hold_group" || usermod -aG "$hold_group" "$user"
    install -d -m 0770 -g "$hold_group" "$hold_state" "$hold_state/gpu-queue" "$hold_state/npu-queue"
    chmod g+s "$hold_state/gpu-queue" "$hold_state/npu-queue"
    # NOT the token files: `gpu-hold` treats the token's EXISTENCE as "held" (the holder creates it with
    # noclobber), so pre-creating an empty one makes every claim wait for ever - found by tests/gpu_hold_queue_test.sh.
    # The directory is group-writable so any fleet member can create the token; an existing one is handed over.
    for f in gpu-token npu-token; do
        [ -e "$hold_state/$f" ] || continue
        chgrp "$hold_group" "$hold_state/$f"; chmod 0660 "$hold_state/$f"
    done
    # The default that keeps the queue un-forked lives in scripts/gpu-hold, and what RUNS is the copy on the
    # box: ~/.local/bin/gpu-hold, which every lane and this runner reach through its own home. If that copy is
    # older it still defaults to ~/GPU-TOKEN, so a migration would move the token and the first lane without
    # GPU_TOKEN_FILE would recreate it - two queues again. So the installer puts the repository's copy in place
    # (keeping a dated backup, the way this box keeps every other tool), and the migration refuses to run
    # against a copy that cannot resolve the fleet default.
    tool_dir=$share_home/.local/bin
    if [ -d "$tool_dir" ] && [ -f "$(dirname "$0")/gpu-hold" ] && ! cmp -s "$(dirname "$0")/gpu-hold" "$tool_dir/gpu-hold"; then
        # owned by the account whose tool it is, not by root: this script runs as root, and a root-owned file in
        # ~/.local/bin can be replaced (the directory is theirs) but not edited in place by its owner
        owner_group=$(id -gn "$owner_user")
        install -m 0755 -o "$owner_user" -g "$owner_group" "$(dirname "$0")/gpu-hold" "$tool_dir/gpu-hold.new" &&
            { [ -f "$tool_dir/gpu-hold" ] && cp -a "$tool_dir/gpu-hold" "$tool_dir/gpu-hold.bak-$(date +%Y%m%d-%H%M%S)" &&
              chown "$owner_user:$owner_group" "$tool_dir"/gpu-hold.bak-*; }
        mv -f "$tool_dir/gpu-hold.new" "$tool_dir/gpu-hold"
        echo "installed scripts/gpu-hold over $tool_dir/gpu-hold (previous copy kept as a .bak-*), so every lane's default is the fleet's"
    fi
    if [ "$migrate" = 1 ]; then
        # never migrate against a gpu-hold that would still default to ~/GPU-TOKEN
        if [ -f "$tool_dir/gpu-hold" ] && ! grep -q 'ONEBIT_HOLD_DIR\|1bit-fleet' "$tool_dir/gpu-hold"; then
            echo "install-ci-runner.sh: refusing --migrate-hold-state: $tool_dir/gpu-hold does not resolve the fleet default" >&2
            echo "  (it still defaults to \$HOME/GPU-TOKEN, so the first lane without GPU_TOKEN_FILE would recreate the old path)" >&2
            exit 1
        fi
        owner_home=$(getent passwd "$owner_user" | cut -d: -f6)
        [ -n "$owner_home" ] || { echo "install-ci-runner.sh: --migrate-hold-state needs --owner-user that exists" >&2; exit 2; }
        # a live queue is not a thing to move under the lanes: refuse unless it is empty
        [ -z "$(ls -A "$owner_home/GPU-QUEUE" 2>/dev/null)" ] || {
            echo "install-ci-runner.sh: $owner_home/GPU-QUEUE is not empty: let the queue drain before migrating (or migrate by hand)" >&2; exit 1; }
        for pair in "GPU-TOKEN:gpu-token" "NPU-TOKEN:npu-token"; do
            src="$owner_home/${pair%%:*}"; dst="$hold_state/${pair##*:}"
            if [ -s "$src" ]; then cp -a "$src" "$dst" && chgrp "$hold_group" "$dst" && chmod 0660 "$dst"; fi
            rm -f "$src"   # empty or moved: either way no file stays behind to read as a held token
        done
        for pair in "GPU-QUEUE:gpu-queue" "NPU-QUEUE:npu-queue"; do
            src="$owner_home/${pair%%:*}"; dst="$hold_state/${pair##*:}"
            [ -d "$src" ] && { cp -a "$src/." "$dst/" 2>/dev/null || true; rm -rf "$src"; }
        done
        chgrp -R "$hold_group" "$hold_state"; chmod -R g+rwX,o-rwx "$hold_state"
        echo "migrated the hold state from $owner_home into $hold_state"
    fi
    # the owner's side has to name the same two paths, or the runner queues with the fleet's files and the
    # owner's lanes with their own - which is the bug this whole block exists to prevent
    cat > /etc/profile.d/1bit-fleet.sh <<PROFILE
# engine #1504: one hold queue for the fleet. gpu-hold reads both variables; npu-hold wraps gpu-hold and
# hard-codes \$HOME/NPU-*, so NPU sharing needs that wrapper changed (docs/ci-runners.md).
export GPU_TOKEN_FILE=$hold_state/gpu-token
export GPU_QUEUE_DIR=$hold_state/gpu-queue
PROFILE
    # thermal-run.sh runs its jobs under `systemctl --user`; without a user manager for this account it falls
    # back to a process-group SIGSTOP, which is the weaker mode the file itself warns about
    loginctl enable-linger "$user" 2>/dev/null || echo "warning: could not enable-linger $user; thermal-run.sh will use its process-group fallback"
    if [ -f "$share_home/.local/bin/npu-hold" ] && grep -q 'GPU_TOKEN_FILE="\$HOME/NPU-TOKEN"' "$share_home/.local/bin/npu-hold" 2>/dev/null; then
        echo "warning: npu-hold pins ~/NPU-QUEUE and ~/NPU-TOKEN, so NPU work from $user uses a PRIVATE NPU queue." >&2
        echo "  The unit sets HW_GATE_LOCK to the fleet's lock, so the lock is shared; the queue is not until npu-hold" >&2
        echo "  honours GPU_QUEUE_DIR/GPU_TOKEN_FILE (docs/ci-runners.md). Until then do not let this runner take NPU work." >&2
    fi
    echo "hold state: $hold_state (group $hold_group, 0770, queues setgid); $user joined $hold_group"
    echo "the owner's shells pick up GPU_TOKEN_FILE/GPU_QUEUE_DIR from /etc/profile.d/1bit-fleet.sh;"
    echo "a long-running owner service needs them in its own unit or environment (docs/ci-runners.md)"
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
