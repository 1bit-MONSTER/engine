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
#
# site.py <out-dir>
#
# Builds the 1bit engine site (GitHub Pages, .github/workflows/pages.yml) in the
# 1bit.MONSTER site template (site/template.html, site/style.css, site/theme.js):
#
#   index.html     the home page: hero, the latest post, a card for every page
#   overview.html  README.md, "Meet the engine"
#   docs.html      the docs index; docs/<name>.md becomes <name>.html, with the docs sidebar
#                  (every doc is published; the ones NAV does not place land under "More")
#   blog.html      posts in blog/YYYY-MM-DD-<slug>.md, newest first; each becomes
#                  blog-<slug>.html, and feed.xml carries them as an Atom feed
#   404.html       sends old 1bit.MONSTER addresses to the archived old site
#
# A post may start with "tags: a, b" lines before its "# Title"; its first paragraph is its
# lead. Links to other docs become page links; links to other files in the repository go
# to them on GitHub.
#
# Visitor counts: with GOATCOUNTER set to a GoatCounter site code (pages.yml passes the
# repository variable of that name), every page loads GoatCounter's counter, which sets
# no cookies. Unset, as in a local preview, the pages carry no analytics at all.
#
# Docs chat: with CONTEXT7_LIBRARY set to the engine's Context7 library id (pages.yml passes
# the repository variable of that name), every page carries Context7's chat widget, which
# answers from the docs Context7 indexed (see context7.json). Context7 only serves it once the
# library is claimed and 1bit.gg is on the widget's allowed domains; unset, no widget.
#
# Search engines and link previews: every page carries its canonical 1bit.gg address, Open
# Graph and Twitter card tags with site/assets/og-card.png, and JSON-LD (the home page describes
# the project, each post is a BlogPosting). robots.txt and sitemap.xml list every page; a page's
# lastmod is its source file's last commit (pages.yml checks out the full history for that).
#
# Needs python-markdown (pip install markdown).
import datetime
import html
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import markdown

REPO = "https://github.com/1bit-MONSTER/engine"
WIKI = REPO + "/wiki"
SITE = "https://1bit.gg/"
# the old 1bit.MONSTER site, kept on GitHub Pages under its repository's own address
OLD_SITE = "https://1bit-monster.github.io/1bit-MONSTER/"
ROOT = pathlib.Path(__file__).resolve().parent.parent

TAGLINE = ("One engine behind an OpenAI-compatible API, running inside Lemonade: the XDNA 2 NPU, "
           "HRX and Vulkan on the Radeon iGPU, ZINC for NVIDIA and Apple GPUs, MLX on Apple Silicon.")

# top bar: (label, page name or absolute URL)
TOP = [("Engine", "overview"), ("Docs", "docs"), ("Blog", "blog"), ("Benchmarks", WIKI)]

# docs sidebar groups: (title, [(doc name, label)])
NAV = [
    ("Start", [("serve", "1bit serve"), ("lemonade", "Lemonade")]),
    ("Devices", [("npu", "NPU"), ("hrx", "HRX + Vulkan"), ("vulkan", "Vulkan (upstream)"), ("lean", "Lean (ROCmFP4, ROCmI4)"),
                 ("zinc", "ZINC"), ("apple", "Apple Silicon")]),
    ("Components", [("laya", "Laya router"), ("comfyui", "ComfyUI.cpp"), ("tokenizers", "Tokenizers"), ("kernel", "Linux kernel")]),
    ("Project", [("PORTING", "Porting map")]),
]


def docs_sources():
    return {p.stem: p for p in sorted((ROOT / "docs").glob("*.md"))}


def posts():
    """[(date, slug, path)], newest first."""
    out = []
    for p in (ROOT / "blog").glob("*.md"):
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-([a-z0-9-]+)", p.stem)
        if not m:
            sys.exit(f"blog post names are YYYY-MM-DD-<slug>.md: {p.name}")
        out.append((m.group(1), m.group(2), p))
    return sorted(out, reverse=True)


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
            return f"[{'Meet the engine' if path_label else label}](overview.html{frag})"
        post = re.fullmatch(r"blog/\d{4}-\d{2}-\d{2}-([a-z0-9-]+)\.md", rel)
        if post:
            return f"[{label}](blog-{post.group(1)}.html{frag})"
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


def analytics():
    out = []
    code = os.environ.get("GOATCOUNTER", "").strip()
    if code:
        if not re.fullmatch(r"[a-z0-9-]+", code):
            sys.exit(f"GOATCOUNTER must be a GoatCounter site code (letters, digits, dashes), not {code!r}")
        out.append(f'<script data-goatcounter="https://{code}.goatcounter.com/count" async src="https://gc.zgo.at/count.js"></script>')
    # Docs7 (Context7's docs analytics): the site id from the Docs7 dashboard
    site = os.environ.get("DOCS7_SITE", "").strip()
    if site:
        if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", site):
            sys.exit(f"DOCS7_SITE must be the Docs7 site id (a UUID), not {site!r}")
        out.append(f'<script defer src="https://context7.com/docs7-analytics.js" data-site="{site}"></script>')
    return "\n".join(out)


def chat_widget():
    library = os.environ.get("CONTEXT7_LIBRARY", "").strip()
    if not library:
        return ""
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", library):
        sys.exit(f"CONTEXT7_LIBRARY must be a Context7 library id like /owner/repo, not {library!r}")
    return (f'<script src="https://context7.com/widget.js" data-library="{library}" '
            'data-color="#1779e1" data-position="bottom-right" '
            'data-placeholder="Ask about the 1bit engine docs..." '
            'data-welcome-message="Ask anything about the 1bit engine: install, serving, NPU, HRX, Vulkan, quantization." '
            'async></script>')


OG_IMAGE = SITE + "assets/og-card.png"


def last_commit_date(path):
    """The source file's last commit date (YYYY-MM-DD); today when git cannot say."""
    try:
        d = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%cs", "--", str(path)],
                           capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        d = ""
    return d or datetime.date.today().isoformat()


def page_url(name):
    return SITE if name == "index" else f"{SITE}{name}.html"


def seo_head(name, title, description, og_type, jsonld, noindex):
    esc = lambda s: html.escape(s, quote=True)
    if noindex:
        return '<meta name="robots" content="noindex">'
    url = page_url(name)
    tags = [f'<link rel="canonical" href="{url}">',
            f'<meta property="og:url" content="{url}">',
            f'<meta property="og:type" content="{og_type}">',
            f'<meta property="og:image" content="{OG_IMAGE}">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            '<meta property="og:image:alt" content="1bit engine: one engine behind an OpenAI-compatible API">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{esc(title)}">',
            f'<meta name="twitter:description" content="{esc(description)}">',
            f'<meta name="twitter:image" content="{OG_IMAGE}">']
    if jsonld:
        # "</" cannot appear inside a script element
        tags.append('<script type="application/ld+json">' + json.dumps(jsonld, ensure_ascii=False).replace("</", "<\\/") + "</script>")
    return "\n".join(tags)


def plain(markdown_text):
    """One paragraph of markdown as plain text."""
    text = re.sub(r"\s*\(\[[^\]]*\.md\]\([^)]*\)\)", "", markdown_text)  # "(docs/x.md)" asides
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[`*_]", "", " ".join(text.split()))


def split_post(text):
    """-> (tags, title, lead paragraph, the rest) of a post."""
    text = strip_notice(text)
    tags = []
    while True:
        m = re.match(r"\s*tags:\s*(.+)\n", text)
        if not m:
            break
        tags += [t.strip() for t in m.group(1).split(",") if t.strip()]
        text = text[m.end():]
    m = re.match(r"\s*# (.+)\n", text)
    if not m:
        sys.exit("a post starts with its '# Title'")
    rest = text[m.end():].strip()
    lead, _, body = rest.partition("\n\n")
    return tags, re.sub(r"[`*]", "", m.group(1)).strip(), lead.strip(), body


def first_paragraph(text):
    for para in re.split(r"\n\s*\n", strip_notice(text)):
        para = para.strip()
        if para and not para.startswith(("#", ">", "|", "-", "```", "<")):
            return plain(para)
    return ""


def clip(text, n=150):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def nav_groups(docs):
    placed = {name for _, items in NAV for name, _ in items}
    groups = [(t, [(n, l) for n, l in items if n in docs]) for t, items in NAV]
    extra = [(n, title_of(docs[n].read_text())) for n in docs if n not in placed]
    if extra:
        groups.append(("More", extra))
    return [(t, items) for t, items in groups if items]


class Site:
    def __init__(self, out):
        self.out = out
        self.template = strip_notice((ROOT / "site" / "template.html").read_text())
        self.docs = docs_sources()
        self.groups = nav_groups(self.docs)
        self.order = [n for _, items in self.groups for n, _ in items]
        self.labels = {n: l for _, items in self.groups for n, l in items}
        self.posts = posts()
        self.pages = []  # (url, lastmod) for sitemap.xml
        # rewrite_links resolves README.md to the overview page and docs to their pages
        self.link_docs = dict(self.docs)

    def top_nav(self, current):
        out = []
        for label, target in TOP:
            href = target if target.startswith("http") else f"{target}.html"
            cur = ' aria-current="page"' if target == current else ""
            out.append(f'      <a href="{href}"{cur}>{html.escape(label)}</a>')
        return "\n".join(out)

    def write(self, name, title, main, kind, section, description=TAGLINE, extra_head="",
              og_type="website", jsonld=None, lastmod=None, noindex=False):
        if not noindex:
            self.pages.append((page_url(name), lastmod or datetime.date.today().isoformat()))
        page = (self.template
                .replace("{{seo}}", seo_head(name, title, description, og_type, jsonld, noindex))
                .replace("{{title}}", html.escape(title))
                .replace("{{description}}", html.escape(description, quote=True))
                .replace("{{kind}}", kind)
                .replace("{{nav}}", self.top_nav(section))
                .replace("{{main}}", main)
                .replace("{{analytics}}", analytics())
                .replace("{{chat}}", chat_widget())
                .replace("{{repo}}", REPO))
        if extra_head:
            page = page.replace("<head>", "<head>\n" + extra_head, 1)
        (self.out / f"{name}.html").write_text(page)

    def md(self, text, src):
        return render(rewrite_links(text, src, self.link_docs, self.labels))

    # ── docs ──────────────────────────────────────────────────────────
    def sidebar(self, current):
        out = ['<details class="docs-side" open><summary>Contents</summary><nav aria-label="Documentation">']
        for title, items in [("Overview", [("docs", "All docs"), ("overview", "Meet the engine")])] + self.groups:
            out.append(f'<p class="group">{html.escape(title)}</p><ul>')
            for name, label in items:
                cur = ' aria-current="page"' if name == current else ""
                out.append(f'<li><a href="{name}.html"{cur}>{html.escape(label)}</a></li>')
            out.append("</ul>")
        out.append("</nav></details>")
        return "\n".join(out)

    def pager(self, name):
        if name not in self.order:
            return ""
        i = self.order.index(name)
        prev = (f'<a class="prev" href="{self.order[i - 1]}.html"><span>Previous</span>{html.escape(self.labels[self.order[i - 1]])}</a>'
                if i > 0 else "<span></span>")
        nxt = (f'<a class="next" href="{self.order[i + 1]}.html"><span>Next</span>{html.escape(self.labels[self.order[i + 1]])}</a>'
               if i + 1 < len(self.order) else "<span></span>")
        return f'<nav class="pager">{prev}{nxt}</nav>'

    def doc_page(self, name, src, section="docs"):
        text = strip_notice(src.read_text())
        if name == "overview":
            # the README's link to this site is redundant on the site itself
            text = re.sub(r"^\*\*Documentation:\*\*.*\n+", "", text, count=1, flags=re.M)
        main = (f'<div class="container docs">\n{self.sidebar(name)}\n<article class="docs-page prose">\n'
                f'{self.md(text, src)}\n{self.pager(name)}\n'
                f'<p class="source meta"><a href="{REPO}/blob/main/{src.relative_to(ROOT).as_posix()}">View this page\'s source on GitHub ↗</a></p>\n'
                "</article>\n</div>")
        title = title_of(text)
        self.write(name, f"{title} · 1bit engine", main, "doc", section, first_paragraph(text) or TAGLINE,
                   og_type="article", lastmod=last_commit_date(src))

    def docs_index(self):
        rows = []
        for title, items in self.groups:
            rows.append(f"<h2>{html.escape(title)}</h2>\n<div class=\"table\"><table><thead><tr><th>Page</th><th>What it covers</th></tr></thead><tbody>")
            for name, label in items:
                desc = clip(first_paragraph(self.docs[name].read_text()), 180)
                rows.append(f'<tr><td><a href="{name}.html">{html.escape(label)}</a></td><td>{html.escape(desc)}</td></tr>')
            rows.append("</tbody></table></div>")
        main = (f'<div class="container docs">\n{self.sidebar("docs")}\n<article class="docs-page prose">\n'
                "<h1>Documentation</h1>\n"
                f'<p class="lead">How to run the engine, what each device does, and how the pieces fit. '
                f'Measured numbers are on the <a href="{WIKI}">wiki</a>.</p>\n' + "\n".join(rows) +
                "\n</article>\n</div>")
        self.write("docs", "Documentation · 1bit engine", main, "doc", "docs")

    # ── blog ──────────────────────────────────────────────────────────
    def blog(self):
        if not self.posts:
            return []
        rows = []
        for i, (date, slug, src) in enumerate(self.posts):
            tags, title, lead, body = split_post(src.read_text())
            name = f"blog-{slug}"
            tag_html = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
            newer = self.posts[i - 1] if i > 0 else None
            older = self.posts[i + 1] if i + 1 < len(self.posts) else None

            def link(p, cls, word):
                if not p:
                    return "<span></span>"
                return f'<a class="{cls}" href="blog-{p[1]}.html"><span>{word}</span>{html.escape(split_post(p[2].read_text())[1])}</a>'
            main = (f'<article class="art"><div class="container">\n'
                    f'<a class="back" href="blog.html">← all posts</a>\n'
                    f'<div class="art-meta"><time datetime="{date}">{date}</time>{tag_html}</div>\n'
                    f"<h1>{html.escape(title)}</h1>\n"
                    f'<p class="lead">{self.md(lead, src)[3:-4]}</p>\n'
                    f'<div class="prose">\n{self.md(body, src)}\n</div>\n'
                    f'<nav class="pager">{link(newer, "prev", "Newer")}{link(older, "next", "Older")}</nav>\n'
                    "</div></article>")
            self.write(name, f"{title} · 1bit engine", main, "post", "blog", plain(lead), og_type="article",
                       lastmod=max(date, last_commit_date(src)),
                       jsonld={"@context": "https://schema.org", "@type": "BlogPosting", "headline": title,
                               "description": plain(lead), "datePublished": date,
                               "dateModified": max(date, last_commit_date(src)), "image": OG_IMAGE,
                               "url": page_url(name), "mainEntityOfPage": page_url(name),
                               "author": {"@type": "Person", "name": "bong-water-water-bong"},
                               "publisher": {"@type": "Organization", "name": "1bit engine", "url": SITE},
                               "keywords": ", ".join(tags)})
            rows.append((date, name, title, plain(lead), tags, self.md(lead + "\n\n" + body, src)))

        log = "\n".join(
            f'<a class="log-row" href="{n}.html"><span class="log-date">{d}</span><div class="log-body">'
            f"<h3>{html.escape(t)}</h3><p>{html.escape(clip(s, 320))}</p></div></a>" for d, n, t, s, _, _ in rows)
        main = ('<section class="section"><div class="container hero-center hero-narrow">\n'
                '<p class="eyebrow">Blog · field notes</p>\n'
                "<h1>How the engine actually gets built.</h1>\n"
                '<p class="lead">Notes on the work: what we port, what we measure, what we ship.</p>\n'
                "</div></section>\n"
                '<section class="section"><div class="container">\n'
                '<div class="row-between log-head"><div><p class="eyebrow">Serial · 2026</p><h2>Notes, in order.</h2></div>'
                f'<div class="links"><a class="btn btn-ghost" href="feed.xml">Atom feed&nbsp;→</a>'
                f'<a class="btn btn-ghost" href="{OLD_SITE}1bit-blog.html">1bit.MONSTER archive&nbsp;→</a></div></div>\n'
                f"{log}\n</div></section>\n{self.archive()}")
        self.write("blog", "Blog · 1bit engine", main, "blog", "blog", "Notes on building the 1bit engine.")

        def stamp(d):
            return d + "T00:00:00Z"
        entries = "".join(
            f"<entry><title>{html.escape(t)}</title><link href=\"{SITE}{n}.html\"/><id>{SITE}{n}.html</id>"
            f"<updated>{stamp(d)}</updated><summary>{html.escape(s)}</summary>"
            + "".join(f'<category term="{html.escape(x, quote=True)}"/>' for x in tags) +
            f"<content type=\"html\">{html.escape(b)}</content></entry>" for d, n, t, s, tags, b in rows)
        (self.out / "feed.xml").write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">'
            f'<title>1bit engine</title><link href="{SITE}"/><link rel="self" href="{SITE}feed.xml"/><id>{SITE}</id>'
            f"<updated>{stamp(rows[0][0])}</updated><author><name>bong-water-water-bong</name></author>{entries}</feed>\n")
        return rows

    def archive(self):
        """blog/archive.tsv: date, title, page of the old 1bit.MONSTER blog worth keeping."""
        src = ROOT / "blog" / "archive.tsv"
        if not src.exists():
            return ""
        rows = [line.split("\t") for line in src.read_text().splitlines() if line.strip()]
        log = "\n".join(
            f'<a class="log-row" href="{OLD_SITE}{f}"><span class="log-date">{d}</span><div class="log-body">'
            f"<h3>{html.escape(t)}&nbsp;↗</h3></div></a>" for d, t, f in rows)
        return ('<section class="section"><div class="container">\n'
                '<div class="log-head"><p class="eyebrow">Archive · 1bit.MONSTER</p><h2>From the 1bit.MONSTER years.</h2>\n'
                '<p class="lead archive-note">The posts from before the clean repository, as they were written. '
                "They describe the old monorepo; the engine's current numbers are on the wiki.</p></div>\n"
                f"{log}\n</div></section>")

    # ── home ──────────────────────────────────────────────────────────
    def home(self, rows):
        latest = ""
        if rows:
            d, n, t, s, _, _ = rows[0]
            latest = (f'<a class="latest-link" href="{n}.html"><span class="latest-badge"><span class="dot"></span>latest · {d}</span>'
                      f'<span class="latest-title"><strong>{html.escape(t)}.</strong> {html.escape(clip(s, 260))}</span>'
                      '<span class="latest-more">read the post <span class="ar">→</span></span></a>')

        def card(href, title, desc, path):
            return (f'<a class="site-card" href="{href}"><span class="site-title">{html.escape(title)} <span class="ar">→</span></span>'
                    f'<p class="site-desc">{html.escape(desc)}</p><span class="site-path">{html.escape(path)}</span></a>')
        cards = [card("overview.html", "Engine", "What the engine is, what runs where, and who it is built on.", "engine"),
                 card("docs.html", "Docs", "Every device, every component, and how to run them.", "docs"),
                 card("blog.html", "Blog", "The build log: what moved over, what got faster, what broke.", "blog"),
                 card(WIKI, "Benchmarks", "Measured, not projected. Every number with how it was taken.", "wiki")]
        cards += [card(f"{n}.html", l, clip(first_paragraph(self.docs[n].read_text()), 120), n)
                  for _, items in self.groups for n, l in items]
        mark = ('<svg class="mark" viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect class="relay r1" x="2" y="2" width="9" height="9" rx="2"/>'
                '<rect class="relay r2" x="13" y="2" width="9" height="9" rx="2"/><rect class="relay r3" x="2" y="13" width="9" height="9" rx="2"/>'
                '<rect class="relay r4" x="13" y="13" width="9" height="9" rx="2"/></svg>')
        main = ('<section class="section hero-center"><div class="container">\n'
                f"<h1>{mark}1bit engine</h1>\n"
                f'<p class="lead">{html.escape(TAGLINE)}</p>\n'
                '<div class="cta-row"><a class="btn btn-primary" href="serve.html">Get started&nbsp;→</a>'
                '<a class="btn btn-ghost" href="overview.html">Meet the engine&nbsp;→</a></div>\n'
                f"{latest}\n</div></section>\n"
                '<section class="section"><div class="container">\n'
                '<span class="site-badge"><span class="dot"></span>site index</span>\n'
                f'<div class="site-grid">{"".join(cards)}</div>\n</div></section>')
        self.write("index", "1bit engine", main, "home", "", jsonld={"@context": "https://schema.org", "@graph": [
            {"@type": "WebSite", "name": "1bit engine", "url": SITE, "description": TAGLINE},
            {"@type": "SoftwareSourceCode", "name": "1bit engine", "description": TAGLINE, "url": SITE,
             "codeRepository": REPO, "license": "https://www.apache.org/licenses/LICENSE-2.0",
             "programmingLanguage": "C++", "runtimePlatform": "Linux",
             "keywords": "LLM inference, AMD Ryzen AI, Strix Halo, XDNA 2 NPU, Vulkan, ROCm, Lemonade, OpenAI-compatible API, GGUF"}]})

    # ── 404 ───────────────────────────────────────────────────────────
    def not_found(self):
        """GitHub serves 404.html at any depth, so it pins its links to the site root. An old
        1bit.MONSTER address (1bit-*.html) links to its copy on the archived old site."""
        main = ('<section class="section hero-center"><div class="container hero-narrow">\n'
                '<p class="eyebrow">404</p><h1>Page not found</h1>\n'
                '<p class="lead">1bit.MONSTER is now 1bit engine. Pages from the old site are kept in the '
                f'<a id="old" href="{OLD_SITE}"><u>1bit.MONSTER archive</u></a>.</p>\n'
                '<div class="cta-row"><a class="btn btn-primary" href="index.html">Home&nbsp;→</a>'
                '<a class="btn btn-ghost" href="docs.html">Docs&nbsp;→</a></div>\n'
                "<script>\n"
                "  // an old 1bit.MONSTER address: point straight at its archived copy\n"
                r'  var m = location.pathname.match(/\/(1bit-[a-z0-9-]+\.html)$/);' "\n"
                f'  if (m && m[1] !== "1bit-jarvis.html") document.getElementById("old").href = "{OLD_SITE}" + m[1];\n'
                "</script>\n</div></section>")
        self.write("404", "Page not found · 1bit engine", main, "doc", "", extra_head=f'<base href="{SITE}">', noindex=True)

    def build(self):
        readme = ROOT / "README.md"
        for name, src in self.docs.items():
            self.doc_page(name, src)
        self.doc_page("overview", readme, section="overview")
        self.docs_index()
        rows = self.blog()
        self.home(rows)
        self.not_found()
        for f in ("style.css", "theme.js"):
            shutil.copy(ROOT / "site" / f, self.out / f)
        shutil.copytree(ROOT / "site" / "assets", self.out / "assets")
        (self.out / ".nojekyll").write_text("")
        # IndexNow (Bing, Yandex, Seznam, Naver): pages.yml sets INDEXNOW_KEY (public by design), the
        # site serves it at /<key>.txt, and the deploy job submits the sitemap's URLs.
        key = os.environ.get("INDEXNOW_KEY", "").strip()
        if key:
            if not re.fullmatch(r"[0-9a-f]{8,128}", key):
                sys.exit(f"INDEXNOW_KEY must be 8-128 hex digits, not {key!r}")
            (self.out / f"{key}.txt").write_text(key)
        (self.out / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {SITE}sitemap.xml\n")
        urls = "".join(f"<url><loc>{u}</loc><lastmod>{d}</lastmod></url>" for u, d in sorted(self.pages))
        (self.out / "sitemap.xml").write_text('<?xml version="1.0" encoding="utf-8"?>\n'
                                              f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n')
        return len(list(self.out.glob("*.html")))


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"{Site(out).build()} pages -> {out}")


if __name__ == "__main__":
    main()
