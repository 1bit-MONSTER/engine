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
# update_vm_test.sh <base image.img> <work dir> [--repo DIR]
#
# OS updater U7 part 2 (orchestrator 27 Nov: "approved as you designed it"): the signed in-system update, end to end, in
# QEMU - install -> `1bit-update --latest` -> reboot into the new release -> `--rollback` -> reboot into the old one,
# plus the refusals on the same server. Everything that grants trust is THROWAWAY and lives in <work> only:
#   * a throwaway Ed25519 signing key (its public half is given to the updater with ONEBIT_UPDATE_PUB),
#   * a throwaway CA and a server certificate for 10.0.2.2 (QEMU's address for the host; CURL_CA_BUNDLE),
#   * a fake GitHub release list served over HTTPS from the host (ONEBIT_UPDATE_API).
# Nothing is baked into an image anyone ships: the overrides reach the VM through its own data partition.
# The updater under test is the repo's (os/v3/puppy/embed/usr/local/bin/1bit-update + tools/release_sign.py +
# tools/seal_manifest.py), copied onto the installed disk's data partition: a base image older than the updater
# (v2026.49-rc1) has no --latest of its own.
#
# The "newer" release is the base image with ONE change: etc/1bit/release-identity in its main SFS says TAG_NEW
# (default v2026.49-rc9), and the image's own /1bit/SHA256SUMS line for that SFS is rewritten to match.
#
# Runs ONLY on ryzen's rules (orchestrator): >= 16 GB available (free -g) or it exits 3 before anything; the whole run
# under `flock ~/.cache/ryzen-heavy.lock systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`.
# Needs: qemu-system-x86_64, OVMF, sudo (loop mounts), unsquashfs/mksquashfs, zstd, openssl, ssh, sfdisk, mcopy.
# Exit: 0 every row passed, 1 a row failed, 2 a VM never came up, 3 refused to start (memory / tools / usage).
set -u
prog=update_vm_test
base=${1:-} work=${2:-}
repo=$(cd "$(dirname "$0")/.." && pwd)
[ "${3:-}" = --repo ] && repo=${4:?--repo needs a dir}
[ -f "$base" ] && [ -n "$work" ] || { echo "usage: $prog <base image.img> <work dir> [--repo DIR]" >&2; exit 3; }
TAG_NEW=${TAG_NEW:-v2026.49-rc9}
TAG_FAKE=${TAG_FAKE:-v2026.49-rc10}   # a newer NAME served with TAG_NEW's image: the re-publish attack (#3454)

# ── ryzen's rules: memory floor, one heavy job, a hard cap ────────────────────────────────────────────────────────
if [ -z "${UVT_WRAPPED:-}" ]; then
    avail=$(free -g | awk '/^Mem:/ {print $7}')
    [ "${avail:-0}" -ge 16 ] || { echo "$prog: REFUSED: ${avail:-?} GB available, the rule is >= 16 GB (free -g)" >&2; exit 3; }
    mkdir -p "$HOME/.cache"
    exec flock "$HOME/.cache/ryzen-heavy.lock" systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
        env UVT_WRAPPED=1 TAG_NEW="$TAG_NEW" TAG_FAKE="$TAG_FAKE" bash "$0" "$@"
fi
for t in qemu-system-x86_64 unsquashfs mksquashfs zstd openssl ssh ssh-keygen sfdisk mcopy python3 sudo losetup; do
    command -v "$t" > /dev/null || { echo "$prog: REFUSED: no $t" >&2; exit 3; }
done
OVMF_CODE=${OVMF_CODE:-/usr/share/OVMF/OVMF_CODE_4M.fd}
OVMF_VARS=${OVMF_VARS:-/usr/share/OVMF/OVMF_VARS_4M.fd}
[ -f "$OVMF_CODE" ] && [ -f "$OVMF_VARS" ] || { echo "$prog: REFUSED: no OVMF at $OVMF_CODE / $OVMF_VARS" >&2; exit 3; }
mkdir -p "$work" || exit 3
work=$(cd "$work" && pwd)
R=$work/rows.txt; : > "$R"
row() { printf '%-5s %-22s %s\n' "$1" "$2" "$3" | tee -a "$R"; }
mnt=$work/mnt loop=
cleanup() {
    [ -n "${srv:-}" ] && kill "$srv" 2> /dev/null
    vm_kill
    mountpoint -q "$mnt" 2> /dev/null && sudo umount "$mnt"
    [ -n "$loop" ] && sudo losetup -d "$loop" 2> /dev/null
}
trap cleanup EXIT

# data partition of a GPT image: its byte offset (the 1BIT-DATA partition, as 1bit-update reads it)
data_off() { sfdisk -d "$1" 2> /dev/null | awk '/name="1BIT-DATA"/ { for (i = 1; i <= NF; i++) if ($i ~ /^start=/) {
    v = $i; sub(/start=/, "", v); gsub(/,/, "", v); if (v == "") { v = $(i + 1); gsub(/,/, "", v) }; print (v + 0) * 512; exit } }'; }
mount_data() {   # <image> -> $mnt (rw)
    _o=$(data_off "$1"); [ -n "$_o" ] || { echo "$prog: no 1BIT-DATA in $1" >&2; return 1; }
    loop=$(sudo losetup -f --show -o "$_o" "$1") && mkdir -p "$mnt" && sudo mount "$loop" "$mnt"
}
umount_data() { sudo umount "$mnt" && sudo losetup -d "$loop" && loop=; }

# ── 1. the throwaway trust: signing key, CA + server cert, ssh key ────────────────────────────────────────────────
T=$work/trust; mkdir -p "$T"
python3 -I -c 'import os; print(os.urandom(32).hex())' > "$T/seed"
python3 -I "$repo/tools/release_sign.py" pubkey "$T/seed" > "$T/release-pub" || exit 3
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$T/ca.key" -out "$T/ca.pem" -days 2 -subj "/CN=1bit update test CA" 2> /dev/null
openssl req -newkey rsa:2048 -nodes -keyout "$T/srv.key" -out "$T/srv.csr" -subj "/CN=10.0.2.2" 2> /dev/null
printf 'subjectAltName=IP:10.0.2.2\n' > "$T/san"
openssl x509 -req -in "$T/srv.csr" -CA "$T/ca.pem" -CAkey "$T/ca.key" -CAcreateserial -out "$T/srv.pem" -days 2 \
    -extfile "$T/san" 2> /dev/null || { echo "$prog: cannot make the test certificate" >&2; exit 3; }
rm -f "$T/sshkey" "$T/sshkey.pub"; ssh-keygen -q -t ed25519 -N "" -f "$T/sshkey"
row INFO trust "throwaway signing key $(cut -c1-16 "$T/release-pub")..., throwaway CA, server cert for 10.0.2.2"

# ── 2. the newer release: the base image, its identity -> TAG_NEW ─────────────────────────────────────────────────
W=$work/www; mkdir -p "$W/rel" "$W/tamper"
new=$work/new.img
cp --sparse=always "$base" "$new" && chmod u+w "$new" || exit 3
mount_data "$new" || exit 3
sfs=$(ls "$mnt"/1bit/puppy_*.sfs | head -1)
base_tag=$(unsquashfs -q -cat "$sfs" etc/1bit/release-identity 2> /dev/null | sed -n 's/^payload=lemonade-onebit-//p')
comp=$(unsquashfs -s "$sfs" 2> /dev/null | sed -n 's/^Compression //p' | head -1)
row INFO base "$(basename "$base"): $base_tag (main SFS $(basename "$sfs"), $comp)"
sq=$work/sq; sudo rm -rf "$sq"
sudo unsquashfs -q -d "$sq" "$sfs" > /dev/null || exit 3
sudo sed -i "s/^payload=lemonade-onebit-.*/payload=lemonade-onebit-$TAG_NEW/" "$sq/etc/1bit/release-identity"
sudo mksquashfs "$sq" "$work/new.sfs" -comp "${comp:-xz}" -noappend -quiet -mem 2G > /dev/null || exit 3
sudo cp "$work/new.sfs" "$sfs"
sudo rm -rf "$sq" "$work/new.sfs"
nsum=$(sudo sha256sum "$sfs" | cut -c1-64)
sudo sed -i "s|^[0-9a-f]\{64\}\( \+\*\?\(\./\)\?$(basename "$sfs")\)$|$nsum\1|" "$mnt/1bit/SHA256SUMS"
grep -q "^$nsum " "$mnt/1bit/SHA256SUMS" || { echo "$prog: the new SFS hash did not land in SHA256SUMS" >&2; exit 3; }
umount_data
row INFO newer "$TAG_NEW: identity rewritten, its SHA256SUMS line updated"
zstd -q -T4 -3 -f "$new" -o "$W/rel/1bit-os-$TAG_NEW.img.zst" || exit 3
rm -f "$new"
sign() {   # <dir> <image name>: SHA256SUMS naming it + the throwaway signature
    (cd "$1" && sha256sum "$2" > SHA256SUMS) && python3 -I "$repo/tools/release_sign.py" sign "$1/SHA256SUMS" \
        --key "$T/seed" --key-id update-vm-test > /dev/null
}
sign "$W/rel" "1bit-os-$TAG_NEW.img.zst" || exit 3
# a tampered copy, signed SUMS of the GOOD one: verify-image must refuse it
cp "$W/rel/1bit-os-$TAG_NEW.img.zst" "$W/tamper/" && cp "$W/rel/SHA256SUMS" "$W/rel/SHA256SUMS.sig" "$W/tamper/"
python3 -I -c 'import sys; f = open(sys.argv[1], "r+b"); f.seek(1 << 20); b = f.read(1); f.seek(1 << 20); f.write(bytes([b[0] ^ 0xff]))' \
    "$W/tamper/1bit-os-$TAG_NEW.img.zst"
# the re-publish attack: TAG_NEW's genuine image, signed, under a NEWER name
mkdir -p "$W/fake"; ln -f "$W/rel/1bit-os-$TAG_NEW.img.zst" "$W/fake/1bit-os-$TAG_FAKE.img.zst"
sign "$W/fake" "1bit-os-$TAG_FAKE.img.zst" || exit 3
port=8443
while ss -lnt 2> /dev/null | grep -q ":$port "; do port=$((port + 1)); done
U=https://10.0.2.2:$port
api() {   # <file> <tag> <dir>: a one-release GitHub list
    python3 -I -c 'import json, os, sys
f, tag, d, u = sys.argv[1:5]
name = "1bit-os-%s.img.zst" % tag
asset = lambda n: {"name": n, "size": os.path.getsize(os.path.join(sys.argv[5], d, n)), "browser_download_url": "%s/%s/%s" % (u, d, n)}
json.dump([{"tag_name": tag, "draft": False, "prerelease": True, "published_at": "2026-11-27T00:00:00Z",
            "body": "update_vm_test", "assets": [asset(name), asset("SHA256SUMS"), asset("SHA256SUMS.sig")]}], open(f, "w"))' \
        "$W/$1" "$2" "$3" "$U" "$W"
}
api good.json "$TAG_NEW" rel; api tamper.json "$TAG_NEW" tamper; api fake.json "$TAG_FAKE" fake
python3 -I -c 'import http.server, ssl, functools, sys
d, cert, key, port = sys.argv[1:5]
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", int(port)), h)
c = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); c.load_cert_chain(cert, key)
srv.socket = c.wrap_socket(srv.socket, server_side=True); srv.serve_forever()' "$W" "$T/srv.pem" "$T/srv.key" "$port" \
    > "$work/https.log" 2>&1 &
srv=$!
row INFO server "$U (good / tamper / fake release lists)"

# ── 3. the VMs ─────────────────────────────────────────────────────────────────────────────────────────────────────
disk=$work/installed.img
rm -f "$disk"; truncate -s 48G "$disk"   # sparse; --latest needs room for the download + its decompressed image (run 6: 10 GiB free on a 24G disk, 16 needed)
cp "$OVMF_VARS" "$work/vars.fd"
sshport=2322; while ss -lnt 2> /dev/null | grep -q ":$sshport "; do sshport=$((sshport + 2)); done
vm_kill() { [ -n "${qpid:-}" ] && kill "$qpid" 2> /dev/null; wait "${qpid:-0}" 2> /dev/null; qpid=; }
vm_up() { sleep 3; kill -0 "${qpid:-0}" 2> /dev/null || { row FAIL qemu "qemu exited at start: $(tail -2 "$work/qemu.out" | tr '\n' ' ' | cut -c1-160)"; return 1; }; }
vm() {   # <live|disk>: boot the live stick with the blank disk, or the installed disk alone
    vm_kill; rm -f "$work/ser"
    # the live boot is read over serial (the installer runs there): wait=on makes qemu hold the guest until the reader
    # is attached - with nowait, whatever the guest writes before a client connects is DROPPED, grub's menu included
    # (run 2, 27 Nov: the reader started 3 s late, never saw the menu, and the default entry booted without the serial
    # shell). The installed-disk boots are driven over ssh and nothing reads their serial line: wait=off.
    swait=off; [ "$1" = live ] && swait=on
    if [ "$1" = live ]; then
        cp --sparse=always "$base" "$work/live.img" && chmod u+w "$work/live.img"
        drives=(-drive file="$work/live.img",format=raw,if=none,id=u0 -device qemu-xhci -device usb-storage,drive=u0,bootindex=0
                -drive file="$disk",format=raw,if=none,id=d1 -device nvme,drive=d1,serial=uvt-target)
    else
        # bootindex is a DEVICE property (a -drive with if=virtio cannot take it: qemu refuses to start)
        # NVMe, like the board: the appliance's init finds its data partition on sd*/nvme* disks, and a virtio-blk disk
        # (/dev/vda) left it "still waiting for [UUID=...]" for ever (run 3, 27 Nov: serial transcript of the installed boot)
        drives=(-drive file="$disk",format=raw,if=none,id=d1 -device nvme,drive=d1,serial=uvt-target,bootindex=0)
    fi
    qemu-system-x86_64 -m 4096 -smp 4 -machine q35 -enable-kvm -cpu host -display none -no-reboot \
        -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" -drive if=pflash,format=raw,file="$work/vars.fd" \
        "${drives[@]}" -netdev user,id=n0,hostfwd=tcp:127.0.0.1:"$sshport"-:22 -device virtio-net-pci,netdev=n0 \
        -serial unix:"$work/ser",server=on,wait="$swait" >> "$work/qemu.out" 2>&1 &
    qpid=$!
}
S() { ssh -p "$sshport" -i "$T/sshkey" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -o ConnectTimeout=5 -o LogLevel=ERROR root@127.0.0.1 "$@"; }
wait_ssh() { for _ in $(seq 1 90); do S true 2> /dev/null && return 0; kill -0 "$qpid" 2> /dev/null || return 1; sleep 5; done; return 1; }
down() { S 'sync; poweroff' > /dev/null 2>&1; for _ in $(seq 1 60); do kill -0 "$qpid" 2> /dev/null || { qpid=; return 0; }; sleep 2; done; vm_kill; }
reboot_vm() { S 'sync; reboot' > /dev/null 2>&1; for _ in $(seq 1 60); do kill -0 "$qpid" 2> /dev/null || break; sleep 2; done; vm disk; }
ident() { S "sed -n 's/^payload=lemonade-onebit-//p' /etc/1bit/release-identity; tr ' ' '\n' < /proc/cmdline | sed -n 's/^psubdir=//p'" | tr '\n' ' '; }

# VM 1: the live stick installs onto the blank disk
vm live
sleep 1; kill -0 "${qpid:-0}" 2> /dev/null || { row FAIL qemu "qemu exited at start: $(tail -2 "$work/qemu.out" | tr '\n' ' ' | cut -c1-160)"; exit 2; }
o=$(python3 -I "$repo/tools/update_vm_serial.py" "$work/ser" "$work/live.img" "$work" 1800 \
    "1bit-install /dev/nvme0n1 --yes --no-reboot 2>&1 | tail -4; sync")
vm_kill; rm -f "$work/live.img"
case $o in *"RC=0"*) row PASS install "1bit-install /dev/nvme0n1: $(printf '%s' "$o" | grep -v '^RC=' | tail -1 | cut -c1-80)" ;;
    *) row FAIL install "$(printf '%s' "$o" | tail -3 | tr '\n' ' ' | cut -c1-160)"; exit 1 ;; esac

# the installed disk's data partition: the test key, the updater under test, the throwaway trust
mount_data "$disk" || exit 1
sudo mkdir -p "$mnt/ssh" "$mnt/uvt"
sudo cp "$T/sshkey.pub" "$mnt/ssh/authorized_keys"
sudo cp "$repo/os/v3/puppy/embed/usr/local/bin/1bit-update" "$repo/tools/release_sign.py" "$repo/tools/seal_manifest.py" \
        "$T/release-pub" "$T/ca.pem" "$mnt/uvt/"
umount_data
UPD="env CURL_CA_BUNDLE=/data/uvt/ca.pem ONEBIT_UPDATE_PUB=/data/uvt/release-pub ONEBIT_UPDATE_VERIFIER=/data/uvt/release_sign.py sh /data/uvt/1bit-update"

# VM 2: the installed base; refusals first, then the real update
vm disk; vm_up || exit 2
wait_ssh || { row FAIL boot-installed "no ssh on the installed disk"; exit 2; }
id0=$(ident); row INFO installed "running $id0"
case $id0 in "$base_tag "*) ;; *) row FAIL installed "expected $base_tag"; exit 1 ;; esac
o=$(S "ONEBIT_UPDATE_API=$U/tamper.json $UPD --latest --channel test 2>&1"); rc=$?
[ $rc = 2 ] && printf '%s' "$o" | grep -q "not the released image" && row PASS refuse-tampered "exit 2: $(printf '%s' "$o" | grep -m1 REFUSED | cut -c1-110)" \
    || row FAIL refuse-tampered "rc=$rc: $(printf '%s' "$o" | tail -2 | tr '\n' ' ' | cut -c1-160)"
o=$(S "ONEBIT_UPDATE_API=$U/fake.json $UPD --latest --channel test 2>&1"); rc=$?
[ $rc = 2 ] && printf '%s' "$o" | grep -q "re-published older image\|GitHub's release is named" && row PASS refuse-republished "exit 2: image $TAG_NEW served as $TAG_FAKE" \
    || row FAIL refuse-republished "rc=$rc: $(printf '%s' "$o" | tail -2 | tr '\n' ' ' | cut -c1-160)"
o=$(S "ONEBIT_UPDATE_API=$U/good.json $UPD --check --channel test 2>&1")
printf '%s' "$o" | grep -q "an update is available" && row PASS check "$(printf '%s' "$o" | grep '^test:' | head -1)" \
    || row FAIL check "$(printf '%s' "$o" | tail -2 | tr '\n' ' ' | cut -c1-160)"
o=$(S "ONEBIT_UPDATE_API=$U/good.json $UPD --latest --channel test 2>&1"); rc=$?
[ $rc = 0 ] && printf '%s' "$o" | grep -q "^UPDATED" && row PASS latest "$(printf '%s' "$o" | grep -m1 '^verified:' | cut -c1-90); $(printf '%s' "$o" | grep -m1 '^identity:' | cut -c1-90)" \
    || { row FAIL latest "rc=$rc: $(printf '%s' "$o" | tail -3 | tr '\n' ' ' | cut -c1-200)"; exit 1; }

# VM 3: the reboot runs the new release; then roll back
reboot_vm; wait_ssh || { row FAIL boot-new "no ssh after the update's reboot"; exit 2; }
id1=$(ident)
case $id1 in "$TAG_NEW "*) row PASS boot-new "running $id1" ;; *) row FAIL boot-new "running $id1, expected $TAG_NEW"; exit 1 ;; esac
o=$(S "$UPD --rollback 2>&1"); rc=$?
[ $rc = 0 ] && row PASS rollback "$(printf '%s' "$o" | tail -1 | cut -c1-110)" || { row FAIL rollback "rc=$rc: $(printf '%s' "$o" | tail -2 | tr '\n' ' ')"; exit 1; }

# VM 4: the rollback's reboot runs the base again
reboot_vm; wait_ssh || { row FAIL boot-old "no ssh after the rollback's reboot"; exit 2; }
id2=$(ident)
case $id2 in "$base_tag "*) row PASS boot-old "running $id2" ;; *) row FAIL boot-old "running $id2, expected $base_tag"; exit 1 ;; esac
down

grep -q '^FAIL' "$R" && exit 1
echo "$prog: every row passed (rows: $R)"
exit 0
