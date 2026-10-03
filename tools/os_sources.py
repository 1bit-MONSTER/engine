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
"""What a 1bit OS image is made of, for the "Source code" section of the release notes.

os/mini/mkimage.sh runs this on the image root before packing the initramfs. For every
regular file in the image it finds the Debian package that owns it (dpkg -S on the builder,
or the .debs mkimage unpacked into its tools directory), that package's source package and
version, and a license class read from /usr/share/doc/<package>/copyright: "gpl", "lgpl",
"firmware", "kernel", "permissive" or "unknown". It also records the kernel (the
third_party/linux pin when the kernel is ours), BusyBox, the firmware packages, the EFI stub
the kernel is wrapped in, kernel modules no package owns, and files no package owns.

Writes sources.json; tools/weekly_changes.py --os-sources renders it. Needs dpkg, no network.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess

SCHEMA = 1
MERGED = ("bin", "sbin", "lib", "lib32", "lib64", "libx32")   # /X and /usr/X are one directory
# our own files, never looked up
SKIP = ("/usr/share/licenses/1bit-os/",)
COPYLEFT = re.compile(r"\b(A|L)?GPL|\bLGPL|\bGPL")
LESSER = re.compile(r"\bLGPL|\bLesser\b|\bLibrary General Public", re.I)
# Where the copyright file's main license is not the license of the parts on the image (the
# libraries of a package that also ships programs, a free-form file that names several
# licenses), say so here: source package (regex) -> (name, license, class or None to keep the
# copyright file's, upstream source link or None). Reviewed against the copyright files.
KNOWN = {
    r"busybox": ("BusyBox", "GPL-2.0", "gpl", "https://busybox.net/downloads/"),
    r"glibc": ("GNU C Library", "LGPL-2.1+", "lgpl", None),
    r"gcc-\d+": ("GCC runtime", "GPL-3.0 with the GCC Runtime Library Exception", "gpl", None),
    r"elfutils": ("elfutils", "LGPL-3.0+ / GPL-2.0+", "lgpl", None),
    r"gmp": ("GMP", "LGPL-3.0+ / GPL-2.0+", "lgpl", None),
    r"util-linux": ("util-linux libraries", "LGPL-2.1+ / BSD-3-Clause", "lgpl", None),
    r"systemd": ("systemd libraries", "LGPL-2.1+", "lgpl", None),
    r"libidn2|libunistring|nettle": (None, "LGPL-3.0+ or GPL-2.0+", "lgpl", None),
    r"gnutls\d*": ("GnuTLS", "LGPL-2.1+", "lgpl", None),
    r"libtasn1-\d+": ("libtasn1", "LGPL-2.1+", "lgpl", None),
    r"libxcrypt": ("libxcrypt", "LGPL-2.1+", "lgpl", None),
    r"readline": ("GNU Readline", "GPL-3.0+", "gpl", None),
    r"krb5": ("MIT Kerberos", "MIT", "permissive", None),
    r"mesa": ("Mesa", "MIT", None, None),
    r"llvm-toolchain-\d+": ("LLVM", "Apache-2.0 with LLVM exceptions", None, None),
    r"vulkan-loader": ("Vulkan loader", "Apache-2.0", None, None),
    r"linux-firmware.*": ("linux-firmware", "redistributable binary-only (see each WHENCE entry)", "firmware",
                          "https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git"),
}


def run(*cmd: str, env=None, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=check)


def aliases(path: str) -> list[str]:
    """the path, its merged-/usr twin, and where it resolves on the builder"""
    parts = path.lstrip("/").split("/")
    out = [path]
    if parts[0] == "usr" and len(parts) > 2 and parts[1] in MERGED:
        out.append("/" + "/".join(parts[1:]))
    elif parts[0] in MERGED:
        out.append("/usr" + path)
    if os.path.lexists(path):
        real = os.path.realpath(path)
        if real not in out:
            out.append(real)
    return out


def image_files(root: str) -> list[str]:
    out = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(d, f)
            if os.path.islink(full) or not os.path.isfile(full):
                continue   # BusyBox's applet links and the like: the file they name is counted
            p = "/" + os.path.relpath(full, root)
            if not p.startswith(SKIP):
                out.append(p)
    return out


def dpkg_owners(paths: list[str]) -> dict[str, str]:
    """path -> owning binary package (with :arch when dpkg gives one), for the paths dpkg knows"""
    out = {}
    for i in range(0, len(paths), 500):
        r = run("dpkg", "-S", *paths[i:i + 500])
        for line in r.stdout.splitlines():
            if line.startswith("diversion ") or ": " not in line:
                continue
            pkgs, path = line.split(": ", 1)
            if "*" in path or path in out:
                continue
            out[path] = pkgs.split(", ")[0]
    return out


def dpkg_info(pkgs: list[str]) -> dict[str, dict]:
    out = {}
    if not pkgs:
        return out
    r = run("dpkg-query", "-W", "-f=${binary:Package}\t${Package}\t${Version}\t${source:Package}\t${source:Version}\n", *pkgs)
    for line in r.stdout.splitlines():
        full, name, ver, src, srcver = line.split("\t")
        info = {"package": name, "version": ver, "source": src or name, "source_version": srcver or ver,
                "origin": "dpkg", "copyright": f"/usr/share/doc/{name}/copyright"}
        out[full] = out[name] = info
    return out


def deb_info(debs_dir: str | None, tools_dir: str | None) -> tuple[dict[str, dict], dict[str, str]]:
    """packages fetched with apt-get download (not in the dpkg database): name -> info, and
    path inside the package -> name"""
    infos, files = {}, {}
    if not debs_dir or not os.path.isdir(debs_dir):
        return infos, files
    for f in sorted(os.listdir(debs_dir)):
        if not f.endswith(".deb"):
            continue
        deb = os.path.join(debs_dir, f)
        r = run("dpkg-deb", "-f", deb, "Package", "Version", "Source")
        fields = dict(re.findall(r"^(\w+): (.*)$", r.stdout, re.M))
        name, ver = fields.get("Package"), fields.get("Version")
        if not name or not ver:
            continue
        src, srcver = name, ver
        if fields.get("Source"):   # "name" or "name (version)"
            m = re.match(r"(\S+)(?: \((.+)\))?", fields["Source"])
            src, srcver = m.group(1), m.group(2) or ver
        cr = os.path.join(tools_dir, "usr/share/doc", name, "copyright") if tools_dir else None
        infos[name] = {"package": name, "version": ver, "source": src, "source_version": srcver,
                       "origin": "deb", "deb": f, "copyright": cr}
        for line in run("dpkg-deb", "-c", deb).stdout.splitlines():
            p = line.split(None, 5)[-1].split(" -> ")[0]
            if p.startswith("./") and not p.endswith("/"):
                files[p[1:]] = name
    return infos, files


def license_terms(text: str) -> list[str]:
    """the license of a debian/copyright: in the machine-readable format, the Files: * stanza
    (the package's own license); else every stanza but debian/*. Old free-form files give []."""
    if not text.startswith("Format:"):
        return []
    main, rest = [], []
    for stanza in re.split(r"\n\s*\n", text):
        files = re.search(r"^Files:\s*(.*(?:\n[ \t].*)*)", stanza, re.M)
        lic = re.search(r"^License:[ \t]*(\S.*)$", stanza, re.M)
        if not files or not lic:
            continue
        globs = files.group(1).split()
        term = lic.group(1).strip()
        if globs == ["*"]:
            main.append(term)
        elif not all(g.startswith("debian/") for g in globs):
            rest.append(term)
    return list(dict.fromkeys(main or rest))


def classify(text: str | None) -> tuple[str, list[str]]:
    """gpl / lgpl / permissive / unknown, and the license terms it read. A choice that offers a
    non-copyleft license ("BSD-3-clause or GPL-2") counts as that license."""
    if text is None:
        return "unknown", []
    terms = license_terms(text)
    if not terms:   # free-form copyright file
        if LESSER.search(text):
            return "lgpl", []
        if re.search(r"General Public License|\bGPL\b", text):
            return "gpl", []
        return "permissive", []
    cls = "permissive"
    for term in terms:
        alts = [a.strip() for a in re.split(r"\bor\b", term)]
        if any(not COPYLEFT.search(a) for a in alts):
            continue
        if any(a.upper().startswith("LGPL") for a in alts):
            cls = "gpl" if cls == "gpl" else "lgpl"
        else:
            cls = "gpl"
    return cls, terms


def read(path: str | None) -> str | None:
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except (OSError, TypeError):
        return None


def git(repo: str, *args: str) -> str | None:
    r = run("git", "-C", repo, *args)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def gitlink(repo: str, path: str) -> str | None:
    line = (git(repo, "ls-tree", "HEAD", path) or "").split()
    return line[2] if len(line) >= 3 and line[1] == "commit" else None


def kernel_version(kver: str) -> str:
    """7.3.0-rc4-1bit -> 7.3-rc4, 7.0.0-34-generic -> 7.0"""
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?(-rc\d+)?", kver)
    if not m:
        return kver
    v = f"{m.group(1)}.{m.group(2)}"
    if m.group(3) and m.group(3) != "0":
        v += "." + m.group(3)
    return v + (m.group(4) or "")


def modinfo(path: str, field: str) -> str | None:
    r = run("modinfo", "-F", field, path)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None


def os_release() -> dict:
    out = {}
    for line in (read("/etc/os-release") or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v.strip('"')
    return {"name": out.get("NAME"), "version_id": out.get("VERSION_ID"), "pretty": out.get("PRETTY_NAME")}


def collect(root: str, kver: str, repo: str | None, build: str | None, tools: str | None,
            debs: str | None, stub: str | None) -> dict:
    files = image_files(root)
    deb_infos, deb_files = deb_info(debs, tools)
    owner_of: dict[str, tuple[str, str]] = {}   # image path -> (package key, origin)
    pending = []
    for p in files:
        hit = next((deb_files[a] for a in aliases(p) if a in deb_files), None)
        if hit and tools and os.path.exists(os.path.join(tools, p.lstrip("/"))):
            owner_of[p] = (hit, "deb")   # mkimage copied it out of the unpacked .deb
        else:
            pending.append(p)
    candidates = sorted({a for p in pending for a in aliases(p)})
    owners = dpkg_owners(candidates)
    for p in pending:
        hit = next((owners[a] for a in aliases(p) if a in owners), None)
        if hit:
            owner_of[p] = (hit, "dpkg")
    vmlinuz = f"/boot/vmlinuz-{kver}"
    kernel_pkg = dpkg_owners([vmlinuz]).get(vmlinuz)
    infos = dpkg_info(sorted({o for o, origin in owner_of.values() if origin == "dpkg"}
                             | ({kernel_pkg} if kernel_pkg else set())))

    # binary packages, grouped by source package
    sources: dict[tuple[str, str], dict] = {}
    for p in files:
        if p not in owner_of:
            continue
        key, origin = owner_of[p]
        info = deb_infos[key] if origin == "deb" else infos.get(key)
        if not info:
            continue
        s = sources.setdefault((info["source"], info["source_version"]),
                               {"source": info["source"], "source_version": info["source_version"],
                                "binaries": {}, "files": 0})
        s["files"] += 1
        b = s["binaries"].setdefault(info["package"], {"package": info["package"], "version": info["version"],
                                                       "origin": info["origin"], "copyright": info["copyright"]})
        if info.get("deb"):
            b["deb"] = info["deb"]
    packages = []
    for s in sources.values():
        classes, terms = [], []
        for b in s["binaries"].values():
            c, t = classify(read(b["copyright"]))
            classes.append(c)
            terms += t
        cls = next((c for c in ("gpl", "lgpl", "unknown") if c in classes), "permissive")
        if s["source"].startswith("linux-firmware") or all(b.startswith("linux-firmware") for b in s["binaries"]):
            cls = "firmware"
        elif re.match(r"linux(-signed|-meta|-upstream)?$", s["source"]) or \
                all(re.match(r"linux-(image|modules)", b) for b in s["binaries"]):
            cls = "kernel"
        terms = list(dict.fromkeys(terms))
        row = {"source": s["source"], "source_version": s["source_version"], "class": cls,
               "class_from_copyright": cls, "name": s["source"], "license": ", ".join(terms) or None}
        for pat, (name, lic, kcls, link) in KNOWN.items():
            if re.fullmatch(pat, s["source"]):
                row.update({k: v for k, v in (("name", name), ("license", lic), ("class", kcls),
                                              ("upstream", link)) if v})
                break
        packages.append({**row, "licenses": terms,
                         "binaries": sorted(({k: v for k, v in b.items() if k != "copyright"}
                                             for b in s["binaries"].values()), key=lambda b: b["package"]),
                         "files": s["files"]})
    packages.sort(key=lambda s: s["source"])

    unowned, oot = [], []
    for p in files:
        if p in owner_of:
            continue
        if p.startswith(f"/lib/modules/{kver}/") and p.endswith((".ko", ".ko.zst", ".ko.xz", ".ko.gz")):
            host = next((a for a in aliases(p) if os.path.exists(a)), os.path.join(root, p.lstrip("/")))
            oot.append({"path": p, "name": os.path.basename(p).split(".ko")[0],
                        "license": modinfo(host, "license"), "version": modinfo(host, "version"),
                        "srcversion": modinfo(host, "srcversion")})
            continue
        if build and p.startswith(build.rstrip("/") + "/"):
            comp = "engine"
        elif p.startswith("/opt/xilinx/xrt/"):
            comp = "xrt"
        elif p.startswith("/lib/firmware/"):
            comp = "firmware"
        elif p.startswith("/lib/modules/"):
            comp = "kernel"
        else:
            comp = "other"
        unowned.append({"path": p, "component": comp})

    busybox = None
    if "/bin/busybox" in files:
        bb = {"path": "/bin/busybox"}
        r = run(os.path.join(root, "bin/busybox"), "--help")
        m = re.search(r"BusyBox v(\S+)", r.stdout + r.stderr)
        bb["upstream_version"] = m.group(1) if m else None
        if "/bin/busybox" in owner_of:
            key, origin = owner_of["/bin/busybox"]
            info = deb_infos[key] if origin == "deb" else infos.get(key, {})
            bb.update({k: info.get(k) for k in ("package", "version", "source", "source_version")})
        busybox = bb

    pins = {}
    if repo:
        for name in ("linux", "xdna-driver"):
            sha = gitlink(repo, f"third_party/{name}")
            if sha:
                pins[name] = sha
    kernel = {"release": kver, "version": kernel_version(kver), "image": vmlinuz,
              "from_pin": kver.endswith("-1bit"), "modules": sum(p.startswith(f"/lib/modules/{kver}/") for p in files)}
    if kernel["from_pin"]:
        kernel.update(pin=pins.get("linux"), build_script="scripts/build-kernel.sh",
                      config=["config/kernel/strixhalo.config", "config/kernel/strixhalo.fragment"])
        # the running kernel is the one built from the pin only if nobody bumped the pin since:
        # compare the checked-out tree's version (when there is one) with the release
        mk = read(os.path.join(repo, "third_party/linux/Makefile")) if repo else None
        if mk:
            v = dict(re.findall(r"^(VERSION|PATCHLEVEL|SUBLEVEL|EXTRAVERSION) = ?(.*)$", mk, re.M))
            tree = f"{v.get('VERSION')}.{v.get('PATCHLEVEL')}.{v.get('SUBLEVEL') or 0}{v.get('EXTRAVERSION', '')}"
            kernel["pin_tree_version"] = tree
            if not kver.startswith(tree):
                print(f"   warning: the image kernel {kver} is not third_party/linux ({tree}): rebuild or fix the notes")
    if kernel_pkg and kernel_pkg in infos:
        i = infos[kernel_pkg]
        kernel["package"] = {k: i[k] for k in ("package", "version", "source", "source_version")}

    fw = [s for s in packages if s["class"] == "firmware"]
    firmware = {"files": sum(p.startswith("/lib/firmware/") for p in files),
                "packages": [{"source": s["source"], "source_version": s["source_version"],
                              "binaries": [b["package"] for b in s["binaries"]]} for s in fw],
                "unpackaged": [u["path"] for u in unowned if u["component"] == "firmware"]}

    efi_stub = None
    if stub and os.path.exists(stub):
        rel = "/" + os.path.relpath(stub, tools) if tools and stub.startswith(tools.rstrip("/") + "/") else stub
        key = deb_files.get(rel)
        info = deb_infos.get(key) if key else None
        if info is None:
            o = dpkg_owners([stub]).get(stub)
            info = dpkg_info([o]).get(o) if o else None
        efi_stub = {"path": rel}
        if info:
            cls, terms = classify(read(info["copyright"]))
            efi_stub.update({k: info[k] for k in ("package", "version", "source", "source_version")},
                            **{"class": cls, "licenses": terms, "license": ", ".join(terms) or None})

    return {"schema": SCHEMA, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "builder": os_release(), "engine_commit": git(repo, "rev-parse", "HEAD") if repo else None,
            "pins": pins, "kernel": kernel, "busybox": busybox, "firmware": firmware, "efi_stub": efi_stub,
            "out_of_tree_modules": oot, "packages": packages, "unowned": unowned,
            "counts": {"files": len(files), "packaged": len(owner_of), "unowned": len(unowned) + len(oot)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="the image root")
    ap.add_argument("--out", required=True, help="sources.json to write")
    ap.add_argument("--kernel", default=os.uname().release, help="the kernel release on the image")
    ap.add_argument("--repo", help="the engine checkout (for the third_party pins)")
    ap.add_argument("--build", help="the engine build directory (its files are the engine's own)")
    ap.add_argument("--tools", help="where mkimage unpacked the .debs it downloads")
    ap.add_argument("--debs", help="where those .debs are (default: the parent of --tools)")
    ap.add_argument("--stub", help="the EFI stub the kernel is wrapped in")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    tools = os.path.abspath(a.tools) if a.tools else None
    debs = a.debs or (os.path.dirname(tools) if tools else None)
    data = collect(root, a.kernel, a.repo, a.build, tools, debs, a.stub)
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    by = {}
    for s in data["packages"]:
        by[s["class"]] = by.get(s["class"], 0) + 1
    c = data["counts"]
    odd = sorted({os.path.dirname(u["path"]) for u in data["unowned"]
                  if u["component"] == "other" and not u["path"].startswith(("/etc/", "/init"))})
    for d in [d for d in odd if not any(d.startswith(o + "/") for o in odd)]:
        print(f"   warning: no package owns the files in {d} (add them to the notes by hand)")
    print(f"   {c['files']} files: {c['packaged']} from {len(data['packages'])} source packages "
          f"({', '.join(f'{v} {k}' for k, v in sorted(by.items()))}), {c['unowned']} from no package")


if __name__ == "__main__":
    main()
