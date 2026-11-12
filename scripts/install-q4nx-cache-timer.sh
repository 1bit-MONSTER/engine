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
# install-q4nx-cache-timer.sh: install a systemd --user timer that runs `1bit q4nx-cache evict` on a schedule
# (engine#2263).
#
# The cache's own sweep runs at SERVE time: after writing an entry, `1bit serve` evicts least-recently-used entries
# down to ONEBIT_Q4NX_CACHE_MAX_GIB. A box that never serves - a review, CI or build lane - therefore never sweeps,
# and ~/.cache/1bit/q4nx grows until it pushes the box under the 175 GiB build gate. Measured on this box: 59 GiB in
# 12 h with the evict command merged and nothing invoking it, then 24 GiB after one manual age cut, which regrew.
#
# This installs the invoker. It is OPT-IN and touches nothing outside the calling user's systemd --user directory: no
# root, no /etc, no repo state. Nothing in the repository runs it for you, by design - engine#2263's fix is a box
# configuration change, and this is the one command that makes it.
#
#   scripts/install-q4nx-cache-timer.sh                 # install, hourly, 40 GiB, 24 h age floor
#   scripts/install-q4nx-cache-timer.sh --dry-run       # print the units and verify them, change nothing
#   scripts/install-q4nx-cache-timer.sh --uninstall     # disable and remove the units
#   scripts/install-q4nx-cache-timer.sh --self-test     # the argument and unit generation, no systemd, no cache
#
# Options: --max-gib N (default 40; 0 is refused, it would evict everything eligible), --min-age SECONDS (default
# 86400), --interval HOURLY|DAILY|*:0/30 (a systemd OnCalendar expression, default hourly), --1bit PATH.
#
# Exit: 0 done, 1 refused or failed, 2 usage.
set -uo pipefail

usage() {
    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
}

max_gib=40
min_age=86400
interval=hourly
bin_arg=""
dry=0
uninstall=0

die() { printf 'install-q4nx-cache-timer: %s\n' "$*" >&2; exit 1; }

# The units, generated from the arguments and printed by both the real install and --dry-run, so a dry run shows exactly
# what would be written rather than a description of it.
service_unit() {   # <1bit path>
    cat <<EOF
[Unit]
Description=Evict the 1bit q4nx cache (engine#2263: nothing sweeps a box that never serves)

[Service]
Type=oneshot
ExecStart="$1" q4nx-cache evict --max-gib $max_gib --min-age $min_age
# systemd will not start a second instance of this oneshot unit while one is active, so two sweeps cannot race.
Nice=10
EOF
}

timer_unit() {
    cat <<EOF
[Unit]
Description=Scheduled q4nx cache eviction (engine#2263)

[Timer]
OnCalendar=$interval
Persistent=true

[Install]
WantedBy=timers.target
EOF
}

resolve_1bit() {
    if [ -n "$bin_arg" ]; then printf '%s' "$bin_arg"; return; fi
    if command -v 1bit > /dev/null 2>&1; then command -v 1bit; return; fi
    if [ -x /opt/1bit/1bit ]; then printf '/opt/1bit/1bit'; return; fi
    printf ''
}

while [ $# -gt 0 ]; do
    case "$1" in
        --max-gib) max_gib=${2:-}; shift 2 ;;
        --min-age) min_age=${2:-}; shift 2 ;;
        --interval) interval=${2:-}; shift 2 ;;
        --1bit) bin_arg=${2:-}; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --uninstall) uninstall=1; shift ;;
        --self-test) self=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument '$1' (try --help)" ;;
    esac
done

[ "$uninstall" = 0 ] || [ "${self:-0}" = 0 ] || die "--uninstall and --self-test are different jobs"

if [ "${self:-0}" = 1 ]; then
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    fail=0
    ok() { printf 'ok   %s\n' "$1"; }
    bad() { printf 'FAIL %s\n' "$1"; fail=1; }

    s=$(service_unit /opt/1bit/1bit)
    case "$s" in *'ExecStart="/opt/1bit/1bit" q4nx-cache evict --max-gib 40 --min-age 86400'*)
        ok "the service carries the command and both bounds" ;;
        *) bad "the service ExecStart is wrong: $s" ;;
    esac
    t=$(timer_unit)
    case "$t" in *"OnCalendar=hourly"*) ok "the timer fires hourly by default" ;; *) bad "the timer is not hourly" ;; esac
    case "$t" in *"WantedBy=timers.target"*) ok "and is installed into timers.target" ;; *) bad "no [Install] section" ;; esac
    case "$t" in *"Persistent=true"*) ok "Persistent=true, so a box asleep at the hour still sweeps once it wakes" ;; *) bad "not persistent" ;; esac

    # 0 is the one value that must never be installed: evicting to a 0 GiB bound removes every eligible entry, i.e. the
    # whole cache, which is the opposite of housekeeping.
    max_gib=0
    out=$(bash "$0" --max-gib 0 --dry-run 2>&1); rc=$?
    [ "$rc" != 0 ] && ok "--max-gib 0 is refused (rc $rc)" || bad "--max-gib 0 should be refused, got rc 0"
    case "$out" in *"disables the sweep"*) ok "...and says why: 0 means OFF in the CLI, so the timer would fire and do nothing" ;; *) bad "the refusal does not explain: $out" ;; esac
    max_gib=40

    out=$(bash "$0" --interval "" --dry-run 2>&1); rc=$?
    [ "$rc" != 0 ] && ok "--interval with no value is refused (an empty OnCalendar never fires)" \
        || bad "--interval with no value should be refused"

    # capability probe: a binary that has the command passes, one that predates it is refused (measured on this box: the
    # deployed rc-2026-11-03-c22 says "unknown command 'q4nx-cache'", so a timer installed against it would fail hourly)
    good=$tmp/good-1bit; printf '#!/bin/sh\necho "usage: 1bit q4nx-cache evict [--root DIR] [--max-gib N]" >&2\nexit 2\n' > "$good"; chmod +x "$good"
    out=$(bash "$0" --dry-run --1bit "$good" 2>&1); rc=$?
    [ "$rc" = 0 ] && ok "a binary with the command is accepted ($rc)" || bad "a capable binary was refused: $out"
    old=$tmp/old-1bit; printf '#!/bin/sh\necho "1bit: unknown command '"'"'q4nx-cache'"'"'" >&2\nexit 1\n' > "$old"; chmod +x "$old"
    out=$(bash "$0" --dry-run --1bit "$old" 2>&1); rc=$?
    [ "$rc" != 0 ] && case "$out" in *"predates engine#2251"*) ok "a binary that predates the command is refused, and named" ;; *) bad "refused for the wrong reason: $out" ;; esac \
        || bad "a binary without q4nx-cache should be refused, got rc 0"

    # the dry run writes nothing
    before=$(ls -A "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user" 2>/dev/null | grep -c 'q4nx' || true)
    bash "$0" --dry-run > /dev/null 2>&1
    after=$(ls -A "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user" 2>/dev/null | grep -c 'q4nx' || true)
    [ "$before" = "$after" ] && ok "--dry-run installs nothing ($before q4nx unit(s) before and after)" \
        || bad "--dry-run changed the unit directory ($before -> $after)"

    [ "$fail" = 0 ] && { printf 'install_q4nx_cache_timer self-test: ok\n'; exit 0; }
    printf 'install_q4nx_cache_timer self-test: FAILED\n'; exit 1
fi

case "$max_gib" in
    ''|*[!0-9]*) die "--max-gib needs a whole number of GiB, got '$max_gib'" ;;
esac
if [ "$max_gib" -eq 0 ]; then
    # The CLI treats 0 as "sweep disabled" (app/q4nx_cache.cpp's usage text), not as "evict everything" -- so a timer
    # installed with it would fire hourly and do nothing, which is the failure this whole script exists to remove.
    die "--max-gib 0 disables the sweep in the CLI, so the timer would fire and do nothing; give a positive bound"
fi
case "$min_age" in
    ''|*[!0-9]*) die "--min-age needs a whole number of seconds, got '$min_age'" ;;
esac
# An empty OnCalendar= is a timer that never fires, and systemd accepts the unit anyway -- so refuse it here rather than
# install something that looks scheduled and is not.
[ -n "$interval" ] || die "--interval needs an OnCalendar expression, e.g. hourly or *:0/30"

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
svc=$unit_dir/1bit-q4nx-cache.service
tmr=$unit_dir/1bit-q4nx-cache.timer

if [ "$uninstall" = 1 ]; then
    systemctl --user disable --now 1bit-q4nx-cache.timer 2>/dev/null || true
    rm -f "$svc" "$tmr"
    systemctl --user daemon-reload 2>/dev/null || true
    printf 'removed %s and %s (the cache itself is untouched)\n' "$svc" "$tmr"
    exit 0
fi

bin=$(resolve_1bit)
[ -n "$bin" ] || die "no 1bit binary: not on PATH and not at /opt/1bit/1bit (pass --1bit PATH)"
[ -x "$bin" ] || die "$bin is not executable"

# A timer whose ExecStart names a command the binary does not have fails hourly and silently. The command landed in
# engine#2251, so a box still running an older 1bit has the cache and not the sweep: probe for it, with the subcommand
# that prints usage and returns 2, and refuse rather than schedule a no-op.
probe=$("$bin" q4nx-cache 2>&1 || true)
case "$probe" in
    *"q4nx-cache evict"*) ;;
    *) die "$bin has no q4nx-cache command (it predates engine#2251); install a current 1bit first. It said: $(printf '%s' "$probe" | head -1)" ;;
esac

if [ "$dry" = 1 ]; then
    printf '# would write %s\n%s\n# would write %s\n%s\n' "$svc" "$(service_unit "$bin")" "$tmr" "$(timer_unit)"
    if command -v systemd-analyze > /dev/null 2>&1; then
        tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
        service_unit "$bin" > "$tmp/1bit-q4nx-cache.service"
        timer_unit > "$tmp/1bit-q4nx-cache.timer"
        # The dry run's promise is that the units VERIFY, so say what systemd said. `is not executable` is dropped because a
        # dry run against a not-yet-installed binary is the normal case; every other complaint is passed through and decides
        # the exit code, because a unit systemd refuses would be installed and then never run.
        if verdict=$(systemd-analyze verify "$tmp/1bit-q4nx-cache.service" "$tmp/1bit-q4nx-cache.timer" 2>&1); then
            printf 'verify: systemd accepts both units\n'
        else
                        # systemd adds "is not executable" for any ExecStart it cannot stat -- true of a dry run against a
            # placeholder, and also the exact message a space-riddled path produces. Filtering it out left the
            # operator with a bare "FAILED" for the one failure this check exists to catch, so print it all.
            printf 'verify: FAILED\n%s\n' "$verdict"
            printf 'dry run: nothing was changed, and nothing would be installed until this passes\n'
            exit 1
        fi
        printf 'dry run: nothing was changed\n'
    fi
    exit 0
fi

mkdir -p "$unit_dir"
service_unit "$bin" > "$svc"
timer_unit > "$tmr"
systemctl --user daemon-reload || die "systemctl --user daemon-reload failed (is a user manager running for $USER?)"
systemctl --user enable --now 1bit-q4nx-cache.timer || die "could not enable 1bit-q4nx-cache.timer"
printf 'installed %s\n         %s\n\n' "$svc" "$tmr"
systemctl --user list-timers 1bit-q4nx-cache.timer --no-pager 2>/dev/null || true
printf '\nThe sweep runs %s: `%s q4nx-cache evict --max-gib %s --min-age %s`.\n' \
    "$interval" "$bin" "$max_gib" "$min_age"
