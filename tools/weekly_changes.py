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
"""What moved between two engine commits, for the weekly release (docs/releases.md).

For every pinned upstream (third_party/*) whose pin moved: the commits between the two pins,
with their authors and PR numbers, from GitHub's compare API, and the upstream releases
published in between. Plus the engine's own commits, and the bump PRs the Sunday run held
back. Writes changes.json (what the release announcement is written from) and the release
notes in Markdown. Needs gh, logged in.
"""
import argparse
import base64
import datetime as dt
import json
import re
import subprocess
import urllib.request

# path -> (name people know it by, repo whose releases count, repo the "(#N)" in commit titles
# refers to); the compare runs on the pinned repo, which for our forks holds upstream's commits
UPSTREAMS = {
    "lemonade": ("Lemonade", "lemonade-sdk/lemonade", "lemonade-sdk/lemonade"),
    "llama.cpp-vulkan": ("llama.cpp (Vulkan, upstream release)", None, "ggml-org/llama.cpp"),
    "llama.cpp": ("llama.cpp for HRX (AMD's tested pair)", None, "ggml-org/llama.cpp"),
    "hrx-system": ("HRX", None, None),
    "llama.cpp-rocmfpx": ("ROCmFPX (lean)", None, None),
    "xdna-driver": ("XDNA driver + XRT", "amd/xdna-driver", None),
    "zinc": ("ZINC", "zolotukhin/zinc", None),
    "laya": ("Laya router", None, None),
    "tokenizers": ("Hugging Face tokenizers", "huggingface/tokenizers", None),
    "ryzenai-server": ("ryzenai-server (ONNX)", "lemonade-sdk/ryzenai-server", None),
    "ds4": ("DwarfStar", "antirez/ds4", None),
    "comfyui.cpp": ("ComfyUI.cpp", None, None),
    "linux": ("Linux kernel", None, None),
}
# every pin, for the Updates list at the end of the weekly post: (what it is to the engine, the
# repo whose latest release is quoted, where its changes are read)
PINS = {  # path -> (name, what it is to the engine, repo whose latest release is quoted, changes)
    "lemonade": ("Lemonade", "the server the engine runs inside", "lemonade-sdk/lemonade", "https://github.com/lemonade-sdk/lemonade/releases"),
    "llama.cpp-vulkan": ("llama.cpp", "ggml-org; Vulkan and lean builds", "ggml-org/llama.cpp", "https://github.com/ggml-org/llama.cpp/releases"),
    "llama.cpp": ("HRX llama.cpp", "our patches on AMD's ggml-hrx", None, "https://github.com/1bit-MONSTER/llama.cpp/commits/1bit/hrx-vulkan-patched"),
    "hrx-system": ("hrx-system", "ROCm", "ROCm/hrx-system", "https://github.com/ROCm/hrx-system/commits/main"),
    "xdna-driver": ("xdna-driver", "AMD NPU driver + XRT", "amd/xdna-driver", "https://github.com/amd/xdna-driver/commits/main"),
    "llama.cpp-rocmfpx": ("ROCmFPX", "ROCmFP4 / ROCmI4 lean quants", None, "https://github.com/charlie12345/ROCmFPX/commits"),
    "zinc": ("ZINC", "NVIDIA and Apple GPUs", "zolotukhin/zinc", "https://github.com/zolotukhin/zinc/commits"),
    "laya": ("Laya", "router scorer", "NandhaKishorM/laya", "https://github.com/NandhaKishorM/laya/releases"),
    "tokenizers": ("tokenizers", "Hugging Face", "huggingface/tokenizers", "https://github.com/huggingface/tokenizers/releases"),
    "ryzenai-server": ("ryzenai-server", "ONNX Runtime GenAI", "lemonade-sdk/ryzenai-server", "https://github.com/lemonade-sdk/ryzenai-server/releases"),
    "ds4": ("DwarfStar", "antirez/ds4", "antirez/ds4", "https://github.com/antirez/ds4/commits"),
    "comfyui.cpp": ("ComfyUI.cpp", "ComfyUI in C++", None, "https://github.com/1bit-MONSTER/comfyui.cpp/commits"),
    "linux": ("Linux", "amdxdna driver source", None, "https://github.com/torvalds/linux/commits"),
}
KEEP = 80  # commits per upstream kept in changes.json
PR = re.compile(r"\(#(\d+)\)\s*$")


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def gh(path: str):
    return json.loads(run("gh", "api", "-H", "Accept: application/vnd.github+json", path))


def submodules(src: str, rev: str) -> dict[str, str]:
    """third_party/<name> -> owner/repo, from .gitmodules at rev"""
    text = run("git", "-C", src, "show", f"{rev}:.gitmodules")
    out, path = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("path ="):
            path = line.split("=", 1)[1].strip()
        elif line.startswith("url =") and path:
            url = line.split("=", 1)[1].strip().removesuffix(".git")
            out[path] = "/".join(url.rstrip("/").split("/")[-2:])
    return out


def pin(src: str, rev: str, path: str) -> str | None:
    line = run("git", "-C", src, "ls-tree", rev, path).split()
    return line[2] if len(line) >= 3 else None


def commit_date(src: str, rev: str) -> dt.datetime:
    return dt.datetime.fromisoformat(run("git", "-C", src, "show", "-s", "--format=%cI", rev).strip())


def linux_version(sha: str) -> str:
    text = base64.b64decode(gh(f"repos/torvalds/linux/contents/Makefile?ref={sha}")["content"]).decode()
    v = dict(re.findall(r"^(VERSION|PATCHLEVEL|SUBLEVEL|EXTRAVERSION) = ?(.*)$", text, re.M))
    s = f"{v.get('VERSION')}.{v.get('PATCHLEVEL')}"
    if v.get("SUBLEVEL", "0") not in ("", "0"):
        s += "." + v["SUBLEVEL"]
    return s + v.get("EXTRAVERSION", "")


def compare(repo: str, old: str, new: str) -> dict:
    c = gh(f"repos/{repo}/compare/{old}...{new}?per_page=250")
    commits = []
    for x in c.get("commits", []):
        title = x["commit"]["message"].splitlines()[0]
        if title.startswith("Merge ") or title.startswith("Sync upstream"):
            continue
        m = PR.search(title)
        commits.append({
            "sha": x["sha"][:12],
            "title": PR.sub("", title).strip(),
            "author": (x.get("author") or {}).get("login") or x["commit"]["author"]["name"],
            "pr": int(m.group(1)) if m else None,
        })
    return {"total": c.get("total_commits", len(commits)), "commits": commits[-KEEP:][::-1],
            "url": f"https://github.com/{repo}/compare/{old[:12]}...{new[:12]}"}


def releases(repo: str, after: dt.datetime, until: dt.datetime) -> list[dict]:
    out = []
    for r in gh(f"repos/{repo}/releases?per_page=20"):
        if r.get("draft") or r.get("prerelease") or not r.get("published_at"):
            continue
        t = dt.datetime.fromisoformat(r["published_at"].replace("Z", "+00:00"))
        if after < t <= until:
            out.append({"tag": r["tag_name"], "name": r.get("name") or r["tag_name"], "url": r["html_url"],
                        "published": r["published_at"], "body": (r.get("body") or "")[:8000]})
    return out


def pins(src: str, rev: str, mods: dict[str, str]) -> list[dict]:
    """every pinned upstream at rev: commit, its date, the upstream's latest release"""
    out = []
    for path, repo in mods.items():
        key = path.removeprefix("third_party/")
        sha = pin(src, rev, path)
        if not sha:
            continue
        name, what, rel_repo, changes = PINS.get(key, (key, "", None, f"https://github.com/{repo}/commits"))
        p = {"path": path, "name": name, "what": what, "sha": sha[:7], "changes": changes}
        try:
            p["date"] = gh(f"repos/{repo}/commits/{sha}")["commit"]["committer"]["date"][:10]
        except subprocess.CalledProcessError:
            p["date"] = None
        if rel_repo:
            try:
                r = gh(f"repos/{rel_repo}/releases/latest")
                p["release"] = {"tag": r["tag_name"], "date": r["published_at"][:10], "url": r["html_url"]}
            except subprocess.CalledProcessError:
                pass
        if key == "linux":
            p["version"] = linux_version(sha)
        out.append(p)
    out.sort(key=lambda p: list(PINS).index(p["path"].removeprefix("third_party/"))
             if p["path"].removeprefix("third_party/") in PINS else 99)
    return out


def show(src: str, rev: str, path: str):
    try:
        return json.loads(run("git", "-C", src, "show", f"{rev}:{path}"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def lemonade_models(sha: str | None) -> dict:
    """Lemonade's model catalog (server_models.json) at a pin of the fork, else of upstream"""
    if not sha:
        return {}
    for repo in ("1bit-MONSTER/lemonade", "lemonade-sdk/lemonade"):
        try:
            c = gh(f"repos/{repo}/contents/src/cpp/resources/server_models.json?ref={sha}")
            return json.loads(base64.b64decode(c["content"]))
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
            continue
    return {}


def models(src: str, old: str, new: str, after: dt.datetime) -> dict:
    """what became runnable this week: architectures the registry gained (or that gained a
    backend), models added to Lemonade's catalog, and models published under our HF org"""
    out = {"architectures": [], "lemonade": [], "huggingface": []}
    a, b = show(src, old, "registry/architectures.json"), show(src, new, "registry/architectures.json")
    if a and b:
        before, now = a.get("architectures", {}), b.get("architectures", {})
        for arch, v in now.items():
            gained = sorted(set(v.get("backends", [])) - set(before.get(arch, {}).get("backends", [])))
            if gained:
                out["architectures"].append({"architecture": arch, "new": arch not in before, "backends": gained})
    la = lemonade_models(pin(src, old, "third_party/lemonade"))
    lb = lemonade_models(pin(src, new, "third_party/lemonade"))
    if la and lb:
        out["lemonade"] = [{"name": k, "recipe": v.get("recipe"), "checkpoint": v.get("checkpoint"),
                            "labels": v.get("labels", [])} for k, v in lb.items() if k not in la]
    try:
        q = "https://huggingface.co/api/models?author=1bit-MONSTER&sort=createdAt&direction=-1&limit=50"
        req = urllib.request.Request(q, headers={"User-Agent": "1bit-weekly (https://1bit.gg)"})
        for m in json.load(urllib.request.urlopen(req, timeout=30)):
            if m.get("createdAt") and dt.datetime.fromisoformat(m["createdAt"].replace("Z", "+00:00")) > after:
                out["huggingface"].append({"id": m["id"], "url": f"https://huggingface.co/{m['id']}",
                                           "created": m["createdAt"][:10]})
    except OSError:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", default="HEAD")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--previous", default="")
    ap.add_argument("--bumps", help="bumps.json from scripts/weekly.sh")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()

    old_date, new_date = commit_date(a.src, a.old), dt.datetime.now(dt.timezone.utc)
    mods = submodules(a.src, a.new)
    ups = []
    for path, repo in mods.items():
        key = path.removeprefix("third_party/")
        before, after = pin(a.src, a.old, path), pin(a.src, a.new, path)
        if not after or before == after:
            continue
        name, rel_repo, pr_repo = UPSTREAMS.get(key, (repo, None, None))
        u = {"path": path, "name": name, "repo": repo, "pr_repo": pr_repo or repo, "from": before, "to": after}
        if key == "linux":
            u["versions"] = [linux_version(before) if before else None, linux_version(after)]
            u.update(total=None, commits=[], url=f"https://github.com/{repo}/compare/{before[:12]}...{after[:12]}"
                     if before else f"https://github.com/{repo}/commit/{after}")
        elif before:
            # an old pin can predate a fork (e.g. Lemonade before 1bit-MONSTER/lemonade): ask upstream
            for r in dict.fromkeys(x for x in (repo, pr_repo, rel_repo) if x):
                try:
                    u.update(compare(r, before, after))
                    break
                except subprocess.CalledProcessError:
                    continue
            else:
                u.update(total=None, commits=[], url=f"https://github.com/{repo}/commit/{after}")
        else:
            u.update(total=None, commits=[], url=f"https://github.com/{repo}/commit/{after}")
        if rel_repo:
            u["releases"] = releases(rel_repo, old_date, new_date)
        ups.append(u)
    # Lemonade first (the engine ships inside it), then the busiest
    ups.sort(key=lambda u: (u["path"] != "third_party/lemonade", -(u.get("total") or 0)))

    engine = []
    for line in run("git", "-C", a.src, "log", "--first-parent", "--format=%h%x09%s%x09%an",
                    f"{a.old}..{a.new}").splitlines():
        sha, title, author = line.split("\t", 2)
        m = PR.search(title)
        engine.append({"sha": sha, "title": PR.sub("", title).strip(), "author": author,
                       "pr": int(m.group(1)) if m else None, "bump": title.startswith("Bump ")})
    bumps = json.load(open(a.bumps)) if a.bumps else []

    data = {"tag": a.tag, "previous": a.previous or None, "date": new_date.date().isoformat(),
            "engine_from": a.old, "engine_to": run("git", "-C", a.src, "rev-parse", a.new).strip(),
            "upstreams": ups, "engine": engine, "bumps": bumps, "pins": pins(a.src, a.new, mods),
            "models": models(a.src, a.old, a.new, old_date)}
    json.dump(data, open(a.json, "w"), indent=1)

    md = [f"Weekly build of 1bit engine at the pins below"
          + (f", since {a.previous}." if a.previous else "."), ""]
    held = [b for b in bumps if b["result"] == "held"]
    if held:
        md += ["## Held back", ""] + [f"- #{b['number']} {b['title']}: {b['why']}" for b in held] + [""]
    md += ["## What moved upstream", ""]
    if not ups:
        md += ["No pin moved this week.", ""]
    for u in ups:
        span = f"`{u['from'][:12]}` → `{u['to'][:12]}`" if u["from"] else f"new: `{u['to'][:12]}`"
        count = f", {u['total']} commits" if u.get("total") else ""
        md.append(f"### {u['name']} ({u['repo']})")
        md.append(f"{span}{count} ([{'compare' if u['from'] else 'commit'}]({u['url']}))")
        if u.get("versions"):
            v0, v1 = u["versions"]
            md.append(f"Linux {v0} → {v1}" if v0 else f"Linux {v1}")
        for r in u.get("releases", []):
            md.append(f"- Release [{r['name']}]({r['url']})")
        for c in u["commits"][:25]:
            pr = f" ([#{c['pr']}](https://github.com/{u['pr_repo']}/pull/{c['pr']}))" if c["pr"] else ""
            md.append(f"- {c['title']}{pr} @{c['author']}")
        if len(u["commits"]) > 25:
            md.append(f"- … and {(u.get('total') or len(u['commits'])) - 25} more")
        md.append("")
    mo = data["models"]
    if any(mo.values()):
        md += ["## New models", ""]
        md += [f"- [{m['id']}]({m['url']}) on Hugging Face" for m in mo["huggingface"]]
        md += [f"- {m['name']} in Lemonade's catalog ({m['recipe']}, `{m['checkpoint']}`)" for m in mo["lemonade"]]
        md += [f"- {x['architecture']}: {'new, ' if x['new'] else ''}runs on {', '.join(x['backends'])}"
               for x in mo["architectures"][:40]]
        if len(mo["architectures"]) > 40:
            md.append(f"- … and {len(mo['architectures']) - 40} more architectures")
        md.append("")
    own = [e for e in engine if not e["bump"]]
    if own:
        md += ["## Engine", ""] + [f"- {e['title']}" + (f" (#{e['pr']})" if e["pr"] else "") for e in own] + [""]
    md += ["## Packages", "",
           "| file | what |", "|---|---|",
           f"| `1bit-{a.tag}-linux-x86_64.tar.zst` | 1bit and its backends (Vulkan, HRX, lean, ZINC, ONNX, DwarfStar, NPU with XRT) |",
           f"| `lemonade-onebit-{a.tag}-linux-x86_64.tar.zst` | Lemonade (lemond + CLI) with the onebit recipe |",
           f"| `1bit-{a.tag}-windows-x64.zip` | 1bit.exe with Vulkan and ONNX backends |",
           f"| `1bit-os-{a.tag}.img.zst`, `.efi` | 1bit OS: boot the engine from a USB stick |",
           "| `SHA256SUMS`, `changes.json` | checksums; this list as data |", "",
           "Licenses: the engine is Apache-2.0 (LICENSE, NOTICE in each package); each backend keeps its own "
           "license. 1bit OS also carries GPL programs (the Linux kernel, BusyBox) as built on the release "
           "machine; their complete source is available on request for three years from this release: open "
           "an issue at https://github.com/1bit-MONSTER/engine/issues.", ""]
    open(a.md, "w").write("\n".join(md))
    print(f"{len(ups)} upstreams moved, {len(engine)} engine commits, {len(held)} bumps held")


if __name__ == "__main__":
    main()
