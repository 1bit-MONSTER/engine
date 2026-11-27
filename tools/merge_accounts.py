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
"""merge_accounts.py --image DIR [--image DIR]... <live etc dir> [--apply --backup DIR] [--status FILE]
                    [--known-default-sha256 HEX]...

On 1bit OS the save layer holds its own copy of /etc/passwd, group, shadow and gshadow as soon as an account is added
(07onboard-user's useradd, a package's useradd, a person's): an overlay's upper copy wins over the image, so
  (a) the image's account changes - the locked root password above all - never reach a box that has a saved copy,
  (b) a new system account the image adds stays invisible behind the saved files,
  (c) and the files cannot simply be moved out of the save layer either: that would delete every local user.
This merges instead, one file at a time, into what the box should see.

THE IMAGE is a stack of layers: each --image is one layer's etc, highest priority first, in the overlay's mount order
(on 1bit OS lowerdir=/pup_b:/pup_ro2:/pup_f:/pup_z: the base's bdrv SFS sits ABOVE the main one and carries its own
passwd/group/shadow). Each file is the first layer's that has it, exactly as the running system reads it.

  * an IMAGE account or group is one the image's file names; a LOCAL one is only in the saved file. Local ones are
    kept, after the image's, in the saved order; image ones new to the box appear.
  * passwd: the saved line of an image account is KEPT (a shell or home a person set survives); its uid and gid must
    be the image's.
  * shadow: a password a person set is kept - but a saved field that IS a base image's known default (matched by the
    sha256 of the field, so the default itself is in no file of this repository), or an empty one, takes the image's
    field (root: '*', no usable password).
  * group/gshadow: an image group takes the image's line with the members of both; only in the root-equivalent
    groups (disk kmem shadow root sudo wheel, as os/v3/puppy/harden-groups.sh) do image accounts follow the image -
    the image's spot leaves disk, while `usermod -aG video spot` and a local user in wheel are kept.
  * REFUSED (exit 1, nothing written) when a file does not parse; when the result would lose an account or root; when
    a saved account or group has the image's NAME but another uid/gid (a local `lemonade` 1001 must not be re-owned
    by an image's `lemonade` 990); or when an account or group the image ADDS takes a uid/gid a local one already
    has (two names on one id is a privilege crossing). A person decides those; the box keeps its files as they were.
Password fields are never printed, only names and ids (from passwd/group's own reader). Without --apply it only
reports. --apply copies the saved originals to --backup (dir 0700, files 0600; the last 5 backups beside it are
kept) and writes each changed file atomically with its own mode and owner. --status FILE records "ok" or
"refused: <why>" for release-check's account-merge row. --known-default-sha256 adds a fingerprint (the tests' own).
Run by 1bit-update after an install (the new slot's SFS layers, on the save layer's upper/etc) and at every boot by
06onboard-accounts (the running overlay's lower layers), so a box updated by an older 1bit-update is merged on its
first boot into a release that carries this.
"""
import hashlib
import os
import re
import shutil
import sys
import time

# sha256 of root's password field in the base image v2026.47 to v2026.49-rc1 were cut from (the base distribution's
# public default, MD5-crypt). A fingerprint of the HASH STRING: it does not reveal the hash, so it cannot be cracked.
KNOWN_DEFAULT_SHA256 = {
    "c3d7234aee49a2723b63aed242c59fb180d78849cecee971a620fb321c80287f",
}
FILES = (("passwd", 7), ("group", 4), ("shadow", 9), ("gshadow", 4))
PRIVILEGED = {"disk", "kmem", "shadow", "root", "sudo", "wheel"}   # harden-groups.sh's list
KEEP_BACKUPS = 5


def is_known_default(field):
    return hashlib.sha256(field.encode("utf-8", "surrogateescape")).hexdigest() in KNOWN_DEFAULT_SHA256


class Refused(Exception):
    pass


# Output is built ONLY from account names (through safe_name) and ids (through read_ids, which reads passwd and group
# alone): no value that came out of a shadow or gshadow line other than its name column can reach a print.
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}\$?")


def safe_name(name):
    """an account name fit to print, else a placeholder (never the raw text)"""
    return name if _NAME.fullmatch(name) else "<an account whose name is not printable>"


def read_ids(path):
    """{name: (id, gid)} of a passwd file, or {name: (gid, "")} of a group file - its own reader: nothing but the name
    and id columns is ever read into it, and it refuses any other file"""
    out = {}
    if path is None or not os.path.exists(path):
        return out
    kind = os.path.basename(path)
    if kind not in ("passwd", "group"):
        raise ValueError("read_ids reads passwd and group only")
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            cols = line.rstrip("\n").split(":")
            if len(cols) >= 3 and cols[2].isdigit():
                gid = cols[3] if kind == "passwd" and len(cols) > 3 and cols[3].isdigit() else ""
                out[safe_name(cols[0])] = (cols[2], gid)
    return out


def parse(path, nfields):
    """[(name, fields)] in file order; None when the file does not exist."""
    if path is None or not os.path.exists(path):
        return None
    out, seen = [], set()
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(":")
            # shadow lines may stop early (the base has 8-field lines: no trailing reserved field); kept as they are
            lo = 2 if nfields == 9 else nfields
            if not lo <= len(fields) <= nfields or not fields[0]:
                raise Refused("%s line %d: not %d colon-separated fields" % (path, n, nfields))
            if fields[0] in seen:
                raise Refused("%s line %d: %s appears twice" % (path, n, safe_name(fields[0])))
            seen.add(fields[0])
            out.append((fields[0], fields))
    return out


def resolve(layers, kind):
    """the path of the image's file: the first layer (highest priority) that has it, as the overlay reads it"""
    for d in layers:
        p = os.path.join(d, kind)
        if os.path.lexists(p):
            return p
    return None


def check_ids(img_path, live_path, what):
    """refuse a saved account/group that has an image name with another id, and an image addition on a local id"""
    img, live = read_ids(img_path), read_ids(live_path)
    for name, ids in live.items():
        if name in img and img[name] != ids:
            raise Refused("%s %s is %s on this box but %s in the image: a person decides (the image would re-own it)"
                          % (what, name, "/".join(i for i in ids if i), "/".join(i for i in img[name] if i)))
    local_ids = {ids[0]: name for name, ids in live.items() if name not in img}
    for name, ids in img.items():
        if name not in live and ids[0] in local_ids:
            raise Refused("the image adds %s %s with id %s, which the local %s already has: two names on one id "
                          "(a person decides)" % (what, name, ids[0], local_ids[ids[0]]))


def members(s):
    return [m for m in s.split(",") if m]


def merge_members(group, img, saved, image_users):
    """the image's members, then the saved ones; in a root-equivalent group a saved IMAGE account is dropped"""
    out = list(img)
    for m in saved:
        if m in out or (group in PRIVILEGED and m in image_users):
            continue
        out.append(m)
    return ",".join(out)


def merge(kind, image, saved, image_users, notes):
    """the merged list of (name, fields) for one file"""
    sv = dict(saved)
    out = []
    for name, f in image:
        s = sv.get(name)
        if s is None or s == f:
            if s is None:
                notes.append("%s: %s added from the image" % (kind, safe_name(name)))
            out.append((name, f))
            continue
        if kind == "passwd":
            out.append((name, s))   # ids checked equal to the image's: a person's shell/home stay
        elif kind == "shadow":
            pw = s[1]
            if is_known_default(pw):
                notes.append("shadow: %s held the base image's DEFAULT password: it takes the image's field" % safe_name(name))
                out.append((name, f))
            elif pw == "" and f[1] != "":
                notes.append("shadow: %s had NO password: it takes the image's field" % safe_name(name))
                out.append((name, f))
            else:
                out.append((name, s))   # a password a person set (or the image's own), with its own dates
        else:   # group: name:pw:gid:members   gshadow: name:pw:admins:members
            g = list(f)
            if kind == "group":
                g[3] = merge_members(name, members(f[3]), members(s[3]), image_users)
            else:
                g[2] = merge_members(name, members(f[2]), members(s[2]), image_users)
                g[3] = merge_members(name, members(f[3]), members(s[3]), image_users)
            if g != s:
                notes.append("%s: %s takes the image's line with the members of both" % (kind, safe_name(name)))
            out.append((name, g))
    names = {n for n, _ in image}
    for name, s in saved:
        if name not in names:
            out.append((name, s))
    return out


def plan(layers, live_dir):
    """{file: (merged text, live text, notes)} for the files that exist in both and would change"""
    src = {k: resolve(layers, k) for k, _ in FILES}
    if src["passwd"] is None:
        raise Refused("no layer has a passwd: not an image's etc")
    img = {k: parse(src[k], n) for k, n in FILES}
    live = {k: parse(os.path.join(live_dir, k), n) for k, n in FILES}
    check_ids(src["passwd"], os.path.join(live_dir, "passwd"), "account")
    if src["group"]:
        check_ids(src["group"], os.path.join(live_dir, "group"), "group")
    image_users = {n for n, _ in img["passwd"]}
    result = {}
    for kind, _ in FILES:
        if img[kind] is None or live[kind] is None:
            continue
        notes = []
        merged = merge(kind, img[kind], live[kind], image_users, notes)
        got = {n for n, _ in merged}
        lost = ({n for n, _ in live[kind]} | {n for n, _ in img[kind]}) - got
        if lost:
            raise Refused("%s: the merge would lose %s" % (kind, ", ".join(safe_name(n) for n in sorted(lost))))
        if kind in ("passwd", "shadow") and "root" not in got:
            raise Refused("%s: no root after the merge" % kind)
        text = "".join(":".join(f) + "\n" for _, f in merged)
        with open(os.path.join(live_dir, kind), encoding="utf-8", errors="surrogateescape") as fh:
            old = fh.read()
        if text != old:
            result[kind] = (text, old, notes)
    return result


def write_atomic(path, text):
    st = os.stat(path)
    tmp = "%s.merge.%d" % (path, os.getpid())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chown(tmp, st.st_uid, st.st_gid)
        os.chmod(tmp, st.st_mode & 0o7777)
        os.rename(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def prune_backups(backup):
    """keep the newest KEEP_BACKUPS directories beside this backup (they hold old shadow copies)"""
    parent = os.path.dirname(os.path.abspath(backup))
    dirs = [os.path.join(parent, d) for d in os.listdir(parent)]
    dirs = sorted((d for d in dirs if os.path.isdir(d) and not os.path.islink(d)), key=os.path.getmtime, reverse=True)
    for d in dirs[KEEP_BACKUPS:]:
        shutil.rmtree(d, ignore_errors=True)


def write_status(path, line):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), line))
    except OSError:
        pass


def main(argv):
    args, layers, apply, backup, status, i = [], [], False, None, None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--apply":
            apply = True
        elif a in ("--backup", "--status", "--image", "--known-default-sha256") and i + 1 < len(argv):
            i += 1
            if a == "--backup":
                backup = argv[i]
            elif a == "--status":
                status = argv[i]
            elif a == "--image":
                layers.append(argv[i])
            else:
                KNOWN_DEFAULT_SHA256.add(argv[i].lower())
        elif a.startswith("--"):
            print("merge_accounts: unknown option %s" % a, file=sys.stderr)
            return 2
        else:
            args.append(a)
        i += 1
    if len(args) == 2 and not layers:          # the one-layer form: <image etc> <live etc>
        layers, args = [args[0]], [args[1]]
    if len(args) != 1 or not layers or (apply and not backup):
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    live_dir = args[0]
    try:
        changes = plan(layers, live_dir)
    except (Refused, OSError, UnicodeError, ValueError) as e:
        print("merge_accounts: REFUSED, nothing written: %s" % e)
        write_status(status, "refused: %s" % e)
        return 1
    if not changes:
        print("merge_accounts: the account files already agree with the image (local accounts kept)")
        write_status(status, "ok: already agree")
        return 0
    for kind in changes:
        for n in changes[kind][2]:
            print("   " + n)
    if not apply:
        print("merge_accounts: would rewrite %s (dry run)" % ", ".join(changes))
        return 0
    os.makedirs(backup, mode=0o700, exist_ok=True)
    os.chmod(backup, 0o700)
    for kind, (_, old, _) in changes.items():
        fd = os.open(os.path.join(backup, kind), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(old)
    for kind, (text, _, _) in changes.items():
        write_atomic(os.path.join(live_dir, kind), text)
    prune_backups(backup)
    print("merge_accounts: rewrote %s; the previous copies are in %s" % (", ".join(changes), backup))
    write_status(status, "ok: rewrote %s" % ", ".join(changes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
