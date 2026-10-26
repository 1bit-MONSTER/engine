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
# hwprobe.py: on the build machine (which is the appliance's own board), list every bound device
# with its kernel module and say which of those modules the image under <build>/os/root does not
# carry. Run after an image build: `python3 os/mini/hwprobe.py` (IMAGE_ROOT overrides the root).
# Written after the first hardware boots (26 Oct 2026) found the image without a USB keyboard driver.
import os, glob, subprocess, re, collections
kver = os.uname().release
# 1. every bound device on the Ubuntu box -> driver -> kernel module (or built-in)
rows = []
for bus in ("pci","usb","platform","acpi","i2c","serio","hid","spi","mmc","sdio","thunderbolt","typec","usb-serial","nvme","scsi","input","pnp"):
    for dev in sorted(glob.glob(f"/sys/bus/{bus}/devices/*")):
        drv = os.path.join(dev, "driver")
        if not os.path.islink(drv): continue
        drvname = os.path.basename(os.readlink(drv))
        modlink = os.path.join(drv, "module")
        mod = os.path.basename(os.readlink(modlink)) if os.path.islink(modlink) else "(built-in)"
        desc = ""
        if bus == "pci":
            try:
                out = subprocess.run(["lspci","-s",os.path.basename(dev)], capture_output=True, text=True).stdout.strip()
                desc = out.split(" ",1)[1][:70] if " " in out else out
            except Exception: pass
        elif bus == "usb":
            for f in ("product","manufacturer"):
                p = os.path.join(dev,f)
                if os.path.exists(p): desc += open(p).read().strip()[:30]+" "
        elif bus in ("platform","acpi","i2c","serio","hid","pnp"):
            desc = os.path.basename(dev)[:40]
        rows.append((bus, os.path.basename(dev), drvname, mod, desc.strip()))
# 2. modules the image carries (c8 build root) and the image's own kernel modules dir
img_root = os.environ.get("IMAGE_ROOT", os.path.expanduser("~/wt/release-rc/work/os/root"))
have = set()
for ko in glob.glob(f"{img_root}/lib/modules/{kver}/**/*.ko*", recursive=True):
    have.add(os.path.basename(ko).split(".ko")[0].replace("-","_"))
loaded = {l.split()[0] for l in open("/proc/modules")}
needed = collections.OrderedDict()
for bus,dev,drv,mod,desc in rows:
    if mod == "(built-in)": continue
    m = mod.replace("-","_")
    needed.setdefault(m, []).append((bus,dev,drv,desc))
missing = {m:v for m,v in needed.items() if m not in have}
print(f"image modules found in build root: {len(have)}; distinct driver modules bound on this box: {len(needed)}; MISSING from image: {len(missing)}")
print("\n== MISSING (module: devices)")
for m,v in missing.items():
    print(f"  {m:24s} " + "; ".join(f"{b}:{d} [{drv}] {desc}" for b,d,drv,desc in v[:3])[:150])
print("\n== present in image (module -> count of devices)")
for m,v in needed.items():
    if m in have: print(f"  {m:24s} {len(v)}")
print("\n== loaded-but-unbound modules on this box (helpers: hwmon, crypto, input handlers)")
extra = sorted(x for x in loaded if x not in needed and x not in have and not x.startswith(("nf_","nft","x_tables","ip_tables","iptable","bridge","br_","veth","overlay","xt_","ip6","zram","binfmt","snd","soundcore","nls_","cuse","fuse","vhost","tun","tap","kvm","ccp","aes","crypto","cryptd","dm_","raid","md_","loop","squashfs","qrtr","cfg80211","mac80211","rfkill","bluetooth","bt","ecdh","libarc4","mt76","mt792","drm","ttm","video","wmi","i2c_","intel","sunrpc","nfs","lockd","grace","autofs","parport","ppdev","lp","msr","cmac","ecc","input_leds","joydev","mac_hid","sch_","ipmi","efi_pstore","pstore","x86_pkg","edac","ucsi","typec","roles","nvme_fabrics","nvme_tcp","xor","asix","ax88","cdc_","r8152","usbnet","mii","apple","hid_","uinput","uhid","udp_tunnel","ip6_udp","geneve","vxlan","macvlan","ipvlan","libcrc","crc"))) 
print("  " + " ".join(extra)[:600])
