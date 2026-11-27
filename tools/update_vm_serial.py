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
#
"""update_vm_serial.py <serial socket> <image> <work dir> <timeout s> <command>

OS updater U7 part 2 (tools/update_vm_test.sh): drive a booting 1bit OS VM over its serial line - pick the image's
"serial root shell" grub entry (counted from the image's own grub.cfg, as os/v3/puppy/vm-check.sh does), wait for the
root shell, run ONE command and print its output followed by `RC=<exit code>`. The same serial technique as
vm-check.sh, cut down to the one thing the update test needs from a live boot: running the installer.
Exit 0 with the command's output on stdout (its rc is the RC= line), 2 when the VM never offered a shell.
"""
import os
import re
import shutil
import socket
import subprocess
import sys
import time

sock_path, img, work, timeout, cmd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
log = open(os.path.join(work, "serial.log"), "ab", buffering=0)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
for _ in range(120):
    try:
        s.connect(sock_path)
        break
    except OSError:
        time.sleep(0.5)
else:
    print("FAIL qemu never opened the serial socket", flush=True)
    sys.exit(2)
buf = b""


def pump(stop):
    global buf
    s.settimeout(0.5)
    try:
        d = s.recv(65536)
    except socket.timeout:
        return True
    except OSError:
        return False
    if not d:
        return False
    log.write(d)
    buf += d
    return True


def read_until(token, secs):
    global buf
    stop = time.time() + secs
    while time.time() < stop:
        i = buf.find(token)
        if i >= 0:
            out, buf = buf[:i + len(token)], buf[i + len(token):]
            return out.decode("utf-8", "replace")
        if not pump(stop):
            return None
    return None


def menu_downs():
    """Entries before the serial root shell one, from the image's own grub.cfg (vm-check.sh's menu_downs)."""
    if not shutil.which("mcopy"):
        return 2
    try:
        out = os.path.join(work, "grub.cfg.from-image")
        subprocess.run(["mcopy", "-n", "-o", "-i", img + "@@1048576", "::/EFI/BOOT/grub.cfg", out],
                       check=True, capture_output=True)
        for i, line in enumerate(l for l in open(out) if l.startswith("menuentry ")):
            if "serial root shell" in line:
                return i
    except Exception:
        pass
    return 2


if read_until(b"serial root shell", 180) is not None:
    s.sendall(b"\x1b[B" * menu_downs() + b"\r")
stop = time.time() + 300
nudge = 0.0
while time.time() < stop:
    if b"21serial-console: root shell on ttyS0" in buf or re.search(rb"root@[^\r\n]{0,80}[$#] ", buf):
        break
    if time.time() - nudge > 10:
        s.sendall(b"\n")
        nudge = time.time()
    if not pump(stop):
        break
else:
    print("FAIL no serial root shell", flush=True)
    sys.exit(2)
buf = b""
for i in range(30):                                   # the shell must be READING, not just started
    tok = "UVT-READY-%d" % i
    s.sendall(("echo %s\n" % tok).encode())
    if read_until(tok.encode(), 12) is not None:
        break
else:
    print("FAIL the serial shell never answered", flush=True)
    sys.exit(2)
buf = b""
begin, end = "UVT-BEGIN-%d" % os.getpid(), "UVT-END-%d" % os.getpid()
s.sendall(("echo %s; ( %s ) 2>&1; echo RC=$?; echo %s\n" % (begin, cmd, end)).encode())
read_until(("%s\r\n" % begin).encode(), 30)
out = read_until(end.encode(), timeout)
if out is None:
    print("FAIL the command did not finish within %d s" % timeout, flush=True)
    sys.exit(2)
print(out[:-len(end)].replace("\r", ""), end="", flush=True)
sys.exit(0)
