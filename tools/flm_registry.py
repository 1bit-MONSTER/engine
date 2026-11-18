#!/usr/bin/env python3
"""registry/flm-models.json: every model FastFlowLM ships, mapped to the GGUF our engine would run.

FastFlowLM's manifest (/opt/fastflowlm/share/flm/model_list.json on strixhalo) is the list to match:
if FLM ships it, we claim it. FLM ships Q4NX/NPU2 directories for its own NPU runtime; our engine
loads GGUF, so each tag is resolved to an upstream GGUF repo and probed for its architecture, then
joined with registry/architectures.json to say which backends accept it.

usage: tools/flm_registry.py [--manifest PATH] [--out registry/flm-models.json] [--refresh]

Nothing here downloads a model: the architecture comes from an HTTP range request for the first
2 MiB of the file's header.
"""
import argparse, json, os, struct, sys, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED = os.path.join(HERE, "flm_upstream.json")     # hand-checked repos, wins over search

# Families whose artifacts our runtime cannot serve as GGUF text models, with the reason. Recorded
# rather than silently dropped, so the registry states the whole FLM catalog.
NOT_TEXT = {
    "whisper-v3": "speech recognition; not a GGUF text model (ggml whisper uses its own loader)",
    "embed-gemma": "embedding model; served via --embed, not as a chat model",
}

# Our own quantization plan per family: what we would ship instead of the vendor file.
PLAN = {
    "lfm2":     "Q4_0 (native type, our HRX Q4_0 path) + Hadamard rotation on the matmuls",
    "lfm2.5-tk": "Q4_0 + Hadamard rotation; keep the thinking template",
    "qwen3":    "UD-style Q4_K_XL from F16 with imatrix",
    "qwen3-it": "UD-style Q4_K_XL from F16 with imatrix",
    "qwen3-tk": "UD-style Q4_K_XL from F16 with imatrix",
    "qwen2":    "Q4_K_M from F16",
    "qwen3.5":  "UD-style Q4_K_XL from F16 with imatrix",
    "qwen3.6-moe": "UD-style Q4_K_XL, experts on the K-quant path",
    "qwen3.8-mtp": "already ours: UD-Q4_K_XL + MTP head (leaderboard row)",
    "llama3":   "Q4_K_M from F16",
    "gemma3":   "Q4_K_M from F16",
    "gemma4e":  "Q4_K_M from F16",
    "gemma4-12b": "Q4_K_M from F16",
    "gpt-oss":  "MXFP4 kept as-is (native to the file), or Q4_K_M from F16",
    "phi4":     "Q4_K_M from F16",
    "hunyuan":  "Q4_K_M from F16",
    "nanbeige": "Q4_K_M from F16",
    "deepseek-r1": "UD-style Q4_K_XL from F16 with imatrix",
    "deepseek-r1-0528": "UD-style Q4_K_XL from F16 with imatrix",
    "qwen2vl":  "Q4_K_M + mmproj",
    "qwen3vl":  "Q4_K_M + mmproj",
    "qwen3vl-flash": "Q4_K_M + mmproj",
}


def http_json(url):
    r = urllib.request.Request(url, headers={"User-Agent": "flm-registry"})
    return json.load(urllib.request.urlopen(r, timeout=60))


def hf_gguf_files(repo):
    try:
        d = http_json(f"https://huggingface.co/api/models/{repo}")
    except Exception:
        return []
    return sorted(s["rfilename"] for s in d.get("siblings", []) if s["rfilename"].lower().endswith(".gguf"))


def hf_search(name, limit=25):
    q = urllib.parse.quote(name)
    try:
        hits = http_json(f"https://huggingface.co/api/models?search={q}&limit={limit}&full=false")
    except Exception:
        return []
    return [h["modelId"] for h in hits]


SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_header(repo, fname, n=1 << 21):
    """general.architecture and the size keys, from the first 2 MiB of the file."""
    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(fname)}"
    r = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}", "User-Agent": "flm-registry"})
    buf = urllib.request.urlopen(r, timeout=90).read()
    i = buf.index(b"GGUF")
    _, _, n_kv = struct.unpack_from("<IQQ", buf, i + 4)
    p = i + 24
    out = {}

    def s():
        nonlocal p
        (ln,) = struct.unpack_from("<Q", buf, p)
        p += 8
        v = buf[p:p + ln].decode("utf-8", "replace")
        p += ln
        return v

    for _ in range(n_kv):
        k = s()
        (t,) = struct.unpack_from("<I", buf, p)
        p += 4
        if k.startswith("tokenizer."):
            break                      # the vocabulary is what makes this window too small
        if t == 8:
            out[k] = s()
        elif t == 9:
            (et,) = struct.unpack_from("<I", buf, p)
            p += 4
            (ln,) = struct.unpack_from("<Q", buf, p)
            p += 8
            if et in SZ:
                p += SZ[et] * ln
            else:
                break
        elif t == 7:
            out[k] = bool(buf[p])
            p += 1
        elif t in SZ:
            out[k] = struct.unpack_from(FMT[t], buf, p)[0]
            p += SZ[t]
        else:
            break
    return out


# Who publishes each family. A search for "Llama-3.2-1B" returns community remixes (uncensored, heretic,
# abliterated) before Meta's own repo, and a remix is the wrong source for a quant we ship ourselves.
OFFICIAL = {
    "lfm2": ["LiquidAI"], "lfm2.5-tk": ["LiquidAI"],
    "qwen2": ["Qwen"], "qwen3": ["Qwen"], "qwen3-it": ["Qwen"], "qwen3-tk": ["Qwen"],
    "qwen3.5": ["Qwen"], "qwen3.6-moe": ["Qwen"], "qwen3.8-mtp": ["Qwen"],
    "qwen2vl": ["Qwen"], "qwen3vl": ["Qwen"], "qwen3vl-flash": ["Qwen"],
    "llama3": ["meta-llama", "unsloth"],
    "gemma3": ["google", "unsloth"], "gemma3-text": ["google", "unsloth"],
    "gemma4e": ["google", "unsloth"], "gemma4e-flash": ["google", "unsloth"],
    "gemma4-12b": ["google", "unsloth"],
    "gpt-oss": ["openai"],
    "phi4": ["microsoft", "unsloth"],
    "hunyuan": ["tencent"],
    "nanbeige": ["Nanbeige"],
    "deepseek-r1": ["deepseek-ai", "unsloth"],
    "deepseek-r1-0528": ["deepseek-ai", "Qwen", "unsloth"],
}


def official_candidates(family, name):
    """Repos from the family's publisher whose name matches the model FLM names, best first."""
    base = name.replace("-NPU2", "").replace("-NPU", "")
    out = []
    for org in OFFICIAL.get(family, []):
        for repo in hf_search(base, limit=40):
            if not repo.lower().startswith(org.lower() + "/"):
                continue
            leaf = repo.split("/")[1].lower()
            key = base.lower().replace("-", "")
            score = 0 if leaf.startswith(key) else (1 if key in leaf.replace("-", "") else 2)
            out.append((score, len(leaf), repo))
    return [r for _, _, r in sorted(out)]


def pick_file(files, want=("Q4_0", "Q4_K_M", "Q8_0", "F16")):
    for w in want:
        c = [f for f in files if w in f and "/" not in f]
        if c:
            return sorted(c, key=len)[0]          # shortest name = least suffix clutter
    return files[0] if files else None


def resolve(family, name, seed):
    """"(repo, gguf file or None, note). Publisher's own repo wins; a GGUF-less official repo means we
    convert it ourselves (which is the point, not a problem)."""
    for repo in ([seed] if seed else []) + official_candidates(family, name):
        files = hf_gguf_files(repo)
        if files:
            return repo, pick_file(files), None
        if repo.split("/")[0].lower() in [o.lower() for o in OFFICIAL.get(family, [])]:
            return repo, None, "official repo has no GGUF: we convert the F16 ourselves"
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/bcloud/flm-model-list.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "registry", "flm-models.json"))
    args = ap.parse_args()

    man = json.load(open(args.manifest))["models"]
    seeds = json.load(open(SEED)) if os.path.exists(SEED) else {}
    reg = json.load(open(os.path.join(ROOT, "registry", "architectures.json")))
    arches = {}                                     # gguf arch -> backends that build a model for it
    for hf_cls, m in reg["architectures"].items():
        arches.setdefault(m["gguf"], set()).update(m["backends"])
    arches = {k: sorted(v) for k, v in arches.items()}

    rows = {}
    for fam, vers in sorted(man.items()):
        for ver, spec in vers.items():
            tag = f"{fam}:{ver}"
            fam_l = (spec.get("details") or {}).get("family") or fam
            row = {
                "flm_repo": spec["url"],
                "flm_name": spec["name"],
                "family": fam_l,
                "params": spec["details"].get("parameter_size"),
                "vlm": bool(spec.get("vlm")),
                "asr": bool(spec.get("asr")),
                "our_plan": None,
                "upstream": None,
                "file": None,
                "gguf_arch": None,
                "backends": None,
                "verdict": None,
            }
            if fam in NOT_TEXT:
                row["verdict"] = "not a text GGUF: " + NOT_TEXT[fam]
                rows[tag] = row
                continue
            row["our_plan"] = PLAN.get(fam_l) or PLAN.get(fam) or "Q4_K_M from F16"
            repo, f, note = resolve(fam_l, spec["name"], seeds.get(tag))
            row["upstream"], row["file"] = repo, f
            if note:
                row["verdict"] = note
            if not repo:
                row["verdict"] = "no upstream repo found under the publisher's org"
                rows[tag] = row
                continue
            if not f:
                rows[tag] = row
                print(f"{tag:22s} {'(no gguf)':12s} {'-':8s} {repo}")
                continue
            try:
                h = gguf_header(repo, f)
            except Exception as e:
                row["verdict"] = f"header probe failed: {type(e).__name__}"
                rows[tag] = row
                continue
            arch = h.get("general.architecture")
            row["gguf_arch"] = arch
            backs = arches.get(arch) or []
            row["backends"] = backs
            row["verdict"] = ("our runtime builds %s for it" % "/".join(backs) if backs
                              else "ARCH GAP: no model class for %s" % arch)
            if row["vlm"]:
                row["verdict"] += "; needs --mmproj"
            rows[tag] = row
            print(f"{tag:22s} {str(arch):12s} {','.join(backs) or '-':8s} {repo}/{f}")

    out = {
        "about": "Every model FastFlowLM ships, mapped to the GGUF our engine would run. Generated by "
                 "tools/flm_registry.py from FLM's own manifest; architectures read from the files' headers.",
        "source_manifest": args.manifest,
        "flm_version": "1.0.7",
        "count": len(rows),
        "models": dict(sorted(rows.items())),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, sort_keys=False)
    ok = [t for t, r in rows.items() if r["backends"] and "hrx" in r["backends"]]
    conv = [t for t, r in rows.items() if r.get("verdict", "").startswith("official repo has no GGUF")]
    gap = [t for t, r in rows.items() if r["verdict"] and r["verdict"].startswith("ARCH GAP")]
    print(f"\n{len(rows)} FLM tags: {len(ok)} our runtime builds a model class for, {len(conv)} need a "
          f"GGUF conversion from the official weights, {len(gap)} arch gaps, "
          f"{len(rows) - len(ok) - len(gap) - len(conv)} not text/GGUF")
    if gap:
        print("arch gaps:", ", ".join(f"{t}({rows[t]['gguf_arch']})" for t in gap))


if __name__ == "__main__":
    main()
