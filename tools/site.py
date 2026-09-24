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
# "More", so a new doc never goes missing. Posts in blog/YYYY-MM-DD-<slug>.md become
# blog-<slug>.html, listed newest first on blog.html and in the Atom feed feed.xml. Links to other docs become page links;
# links to other files in the repository go to them on GitHub.
#
# Visitor counts: with GOATCOUNTER set to a GoatCounter site code (pages.yml passes the
# repository variable of that name), every page loads GoatCounter's counter, which sets
# no cookies. Unset, as in a local preview, the pages carry no analytics at all.
#
# Needs python-markdown (pip install markdown).
import datetime
import html
import os
import pathlib
import re
import shutil
import sys

import markdown

REPO = "https://github.com/1bit-MONSTER/engine"
SITE = "https://1bit.monster/"
# the old 1bit.MONSTER site, kept on GitHub Pages under its repository's own address
OLD_SITE = "https://1bit-monster.github.io/1bit-MONSTER/"
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


def posts():
    """[(date, slug, path)], newest first."""
    out = []
    for p in (ROOT / "blog").glob("*.md"):
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-([a-z0-9-]+)", p.stem)
        if not m:
            sys.exit(f"blog post names are YYYY-MM-DD-<slug>.md: {p.name}")
        out.append((m.group(1), m.group(2), p))
    return sorted(out, reverse=True)


def summary_of(text):
    """The first paragraph after the title, as plain text."""
    for para in re.split(r"\n\s*\n", strip_notice(text)):
        para = para.strip()
        if para and not para.startswith(("#", ">", "|", "-", "```")):
            para = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", para)
            return re.sub(r"[`*_]", "", " ".join(para.split()))
    return ""


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


def analytics():
    code = os.environ.get("GOATCOUNTER", "").strip()
    if not code:
        return ""
    if not re.fullmatch(r"[a-z0-9-]+", code):
        sys.exit(f"GOATCOUNTER must be a GoatCounter site code (letters, digits, dashes), not {code!r}")
    return f'<script data-goatcounter="https://{code}.goatcounter.com/count" async src="https://gc.zgo.at/count.js"></script>'


def not_found(template):
    """404.html: GitHub serves it at any depth, so it pins its links to the site root.
    Links into the old 1bit.MONSTER site (1bit-*.html) point at that site's archive."""
    body = ('<h1>Page not found</h1>\n'
            '<p>1bit.MONSTER is now <a href="index.html">1bit engine</a>. '
            'Pages from the old site are kept in the <a id="old" href="' + OLD_SITE + '">1bit.MONSTER archive</a>.</p>\n'
            '<script>\n'
            '  // an old 1bit.MONSTER address: point straight at its archived copy\n'
            '  var m = location.pathname.match(/\\/(1bit-[a-z0-9-]+\\.html)$/);\n'
            '  if (m && m[1] !== "1bit-jarvis.html") document.getElementById("old").href = "' + OLD_SITE + '" + m[1];\n'
            '</script>')
    return (template
            .replace("<head>", f'<head>\n<base href="{SITE}">', 1)
            .replace("{{title}}", "Page not found · 1bit engine")
            .replace("{{home}}", "doc")
            .replace("{{sidebar}}", "")
            .replace("{{content}}", body)
            .replace("{{pager}}", "")
            .replace("{{source}}", REPO)
            .replace("{{repo}}", REPO))


def page(template, **fields):
    for key, value in fields.items():
        template = template.replace("{{" + key + "}}", value)
    return template.replace("{{repo}}", REPO)


def blog(template, out, docs, labels):
    items = posts()
    if not items:
        return
    names = [f"blog-{slug}" for _, slug, _ in items]
    side = ('<p class="group">Blog</p><ul>' + "".join(
        f'<li><a href="{n}.html"{{cur}}>{html.escape(title_of(p.read_text()))}</a></li>'.replace("{cur}", "{{cur_" + n + "}}")
        for n, (_, _, p) in zip(names, items)) + "</ul>")

    def sidebar_for(current):
        text = side
        for n in names:
            text = text.replace("{{cur_" + n + "}}", ' class="current" aria-current="page"' if n == current else "")
        return text

    rows = []
    for i, (date, slug, src) in enumerate(items):
        text = strip_notice(src.read_text())
        title = title_of(text)
        # the post's own date line under its title
        body = render(rewrite_links(text, src, docs, labels))
        body = re.sub(r"(</h1>)", rf'\1\n<p class="date"><time datetime="{date}">{date}</time></p>', body, count=1)
        newer = f'<a class="prev" href="{names[i - 1]}.html"><span>Newer</span>{html.escape(title_of(items[i - 1][2].read_text()))}</a>' if i > 0 else "<span></span>"
        older = f'<a class="next" href="{names[i + 1]}.html"><span>Older</span>{html.escape(title_of(items[i + 1][2].read_text()))}</a>' if i + 1 < len(items) else "<span></span>"
        (out / f"{names[i]}.html").write_text(page(template, title=html.escape(title) + " · 1bit engine", home="doc post",
            sidebar=sidebar_for(names[i]), content=body, pager=f'<nav class="pager">{newer}{older}</nav>',
            source=f"{REPO}/blob/main/{src.relative_to(ROOT).as_posix()}"))
        rows.append((date, names[i], title, summary_of(text), body))

    listing = '<h1>Blog</h1>\n<p class="date"><a href="feed.xml">Atom feed</a></p>\n' + "\n".join(
        f'<section class="post"><p class="date"><time datetime="{d}">{d}</time></p>'
        f'<h2><a href="{n}.html">{html.escape(t)}</a></h2><p>{html.escape(s)}</p></section>' for d, n, t, s, _ in rows)
    (out / "blog.html").write_text(page(template, title="Blog · 1bit engine", home="doc", sidebar=sidebar_for(""),
        content=listing, pager="", source=f"{REPO}/tree/main/blog"))

    def stamp(d):
        return d + "T00:00:00Z"
    entries = "".join(
        f"<entry><title>{html.escape(t)}</title><link href=\"{SITE}{n}.html\"/><id>{SITE}{n}.html</id>"
        f"<updated>{stamp(d)}</updated><summary>{html.escape(s)}</summary>"
        f"<content type=\"html\">{html.escape(b)}</content></entry>" for d, n, t, s, b in rows)
    (out / "feed.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">'
        f'<title>1bit engine</title><link href="{SITE}"/><link rel="self" href="{SITE}feed.xml"/><id>{SITE}</id>'
        f"<updated>{stamp(rows[0][0])}</updated><author><name>bong-water-water-bong</name></author>{entries}</feed>\n")


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    docs = sources()
    groups = nav_groups(docs)
    order = [n for _, items in groups for n, _ in items]
    labels = {n: l for _, items in groups for n, l in items}
    template = strip_notice((ROOT / "site" / "template.html").read_text()).replace("{{analytics}}", analytics())
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
    (out / "404.html").write_text(not_found(template))
    blog(template, out, docs, labels)
    shutil.copy(ROOT / "site" / "style.css", out / "style.css")
    (out / ".nojekyll").write_text("")
    print(f"{len(docs)} pages -> {out}")


if __name__ == "__main__":
    main()
