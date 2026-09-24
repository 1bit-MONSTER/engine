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
# site.py <out-dir>
#
# Builds the documentation site (GitHub Pages, .github/workflows/pages.yml) from
# README.md and docs/*.md: README becomes index.html, docs/<name>.md becomes
# <name>.html. Every doc is published; the ones NAV does not place land under
# "More", so a new doc never goes missing. Links to other docs become page links;
# links to other files in the repository go to them on GitHub.
#
# Needs python-markdown (pip install markdown).
import html
import pathlib
import re
import shutil
import sys

import markdown

REPO = "https://github.com/1bit-MONSTER/engine"
ROOT = pathlib.Path(__file__).resolve().parent.parent

# sidebar groups: (title, [(doc name, label)]); "index" is README.md
NAV = [
    ("Start", [("index", "Overview"), ("serve", "1bit serve"), ("lemonade", "Lemonade")]),
    ("Devices", [("npu", "NPU"), ("hrx", "HRX + Vulkan"), ("vulkan", "Vulkan (upstream)"),
                 ("zinc", "ZINC"), ("apple", "Apple Silicon")]),
    ("Components", [("laya", "Laya router"), ("tokenizers", "Tokenizers"), ("kernel", "Linux kernel")]),
    ("Project", [("PORTING", "Porting map")]),
]


def sources():
    docs = {"index": ROOT / "README.md"}
    for p in sorted((ROOT / "docs").glob("*.md")):
        docs[p.stem] = p
    return docs


def nav_groups(docs):
    placed = {name for _, items in NAV for name, _ in items}
    groups = [(t, [(n, l) for n, l in items if n in docs]) for t, items in NAV]
    extra = [(n, title_of(docs[n].read_text())) for n in docs if n not in placed]
    if extra:
        groups.append(("More", extra))
    return [(t, items) for t, items in groups if items]


def strip_notice(text):
    # the copyright notice at the top of every doc is an HTML comment
    return re.sub(r"\A\s*<!--.*?-->\s*", "", text, count=1, flags=re.S)


def title_of(text):
    m = re.search(r"^# (.+)$", strip_notice(text), flags=re.M)
    return re.sub(r"[`*]", "", m.group(1)).strip() if m else "1bit engine"


def rewrite_links(text, src, docs, labels):
    """Doc links -> page links; other repository paths -> GitHub; anything absolute unchanged."""
    here = src.parent

    def fix(m):
        label, target = m.group(1), m.group(2)
        # a label that is just the file name reads better as the page's name
        path_label = re.fullmatch(r"(?:docs/)?([A-Za-z0-9_-]+)\.md", label.strip("`"))
        if re.match(r"^[a-z]+:|^#|^/", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        frag = "#" + frag if frag else ""
        resolved = (here / path).resolve()
        try:
            rel = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return m.group(0)
        if rel == "README.md":
            return f"[{'Overview' if path_label else label}](index.html{frag})"
        if rel.startswith("docs/") and rel.endswith(".md") and resolved.stem in docs:
            if path_label:
                label = labels.get(resolved.stem) or title_of(docs[resolved.stem].read_text())
            return f"[{label}]({resolved.stem}.html{frag})"
        kind = "tree" if resolved.is_dir() else "blob"
        return f"[{label}]({REPO}/{kind}/main/{rel}{frag})"

    return re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", fix, text)


def nest_lists(text):
    """GitHub nests a list under 2 spaces, python-markdown needs 4: double list-marker indents outside code."""
    out, fenced = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        m = None if fenced else re.match(r"^( +)([-*+]|\d+\.) ", line)
        out.append(" " * (2 * len(m.group(1))) + line[len(m.group(1)):] if m else line)
    return "\n".join(out)


def render(text):
    text = nest_lists(text)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"permalink": "#", "permalink_class": "anchor"}})
    body = md.convert(text)
    # wide tables scroll on their own instead of the page
    body = body.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")
    return body


def sidebar(groups, current):
    out = []
    for title, items in groups:
        out.append(f'<p class="group">{html.escape(title)}</p><ul>')
        for name, label in items:
            cls = ' class="current" aria-current="page"' if name == current else ""
            out.append(f'<li><a href="{name}.html"{cls}>{html.escape(label)}</a></li>')
        out.append("</ul>")
    return "\n".join(out)


def pager(order, name, labels):
    i = order.index(name)
    prev = f'<a class="prev" href="{order[i - 1]}.html"><span>Previous</span>{html.escape(labels[order[i - 1]])}</a>' if i > 0 else "<span></span>"
    nxt = f'<a class="next" href="{order[i + 1]}.html"><span>Next</span>{html.escape(labels[order[i + 1]])}</a>' if i + 1 < len(order) else "<span></span>"
    return f'<nav class="pager">{prev}{nxt}</nav>'


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    docs = sources()
    groups = nav_groups(docs)
    order = [n for _, items in groups for n, _ in items]
    labels = {n: l for _, items in groups for n, l in items}
    template = strip_notice((ROOT / "site" / "template.html").read_text())
    for name, src in docs.items():
        text = strip_notice(src.read_text())
        if name == "index":
            # the README's link to this site is redundant on the site itself
            text = re.sub(r"^\*\*Documentation:\*\*.*\n+", "", text, count=1, flags=re.M)
        body = render(rewrite_links(text, src, docs, labels))
        source = src.relative_to(ROOT).as_posix()
        page = (template
                .replace("{{title}}", html.escape(title_of(text)) + ("" if name == "index" else " · 1bit engine"))
                .replace("{{home}}", "home" if name == "index" else "doc")
                .replace("{{sidebar}}", sidebar(groups, name))
                .replace("{{content}}", body)
                .replace("{{pager}}", pager(order, name, labels))
                .replace("{{source}}", f"{REPO}/blob/main/{source}")
                .replace("{{repo}}", REPO))
        (out / f"{name}.html").write_text(page)
    shutil.copy(ROOT / "site" / "style.css", out / "style.css")
    (out / ".nojekyll").write_text("")
    print(f"{len(docs)} pages -> {out}")


if __name__ == "__main__":
    main()
