// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! What the screen looks like. Top to bottom, as the owner asked on cycle 20 ("can we place the menu above the
//! scrolling logs and below the technical information"), in this order and nothing between:
//!
//!   header | CONNECT | the status facts | MENU (nine numbered rows) | LOGS (scrolling) | one footer line
//!
//! The status facts are the only part that changes shape with the screen:
//!
//!   * 240x67 (the appliance's tty1 on a monitor) - six panels in a 3x2 grid (LEMONADE / GPU / NPU, then
//!     MODELS / FANS+TEMPS / DATA).
//!   * 80x25 (a serial/VGA default console, and what the QEMU rows read) - the same six facts, and their short
//!     forms, on ONE line, because nine menu rows plus that line is what leaves the log pane its three rows.
//!
//! The MENU is always on screen at both sizes: the c15 acceptance row looks for "Show the API key" and
//! "Reboot / power off" on tty1, and the owner's complaint was a menu you had to scroll the log off to read.
//!
//! Colours are the 16-colour VT palette by name only (no RGB: a bare Linux console has no truecolour) and no
//! glyph outside what every console font has - titles are ASCII rules, so a VGA text mode with a plain code page
//! still draws the screen. Nothing is clipped: every string goes through `wrap_words` before it reaches a widget,
//! because the c16 board showed "firmware prese".

use crate::facts::{human_bytes, human_uptime, wrap_hanging, wrap_words, Facts, Mode, SecureBoot};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use ratatui::Terminal;

pub const ACTIONS: [&str; 9] = [
    "Show the API key",
    "Quick chat test (GPU and NPU)",
    "Network (DHCP, static, hostname)",
    "Models (list, pull, delete)",
    "Install 1bit OS to a disk",
    "Start the desktop (JWM) for this session",
    "Shell (bash, returns here on exit)",
    "Logs (Lemonade, kernel)",
    "Reboot / power off",
];

#[derive(Clone, Debug)]
pub enum Modal {
    /// a message; any key closes it
    Info { title: String, body: Vec<String> },
    /// y/n; the payload says which action `y` runs
    Confirm {
        title: String,
        question: String,
        yes: Yes,
    },
    /// power: r reboot, p power off, anything else cancels
    Power,
    /// a handed-off action is running in the foreground
    Busy { title: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Yes {
    ShowKey,
    // Cycle 20: blank is the shipped mode, so item 1 in the TUI also has to be able to TURN a key on. It hands the
    // terminal to the dialog menu's item (1bit-console-menu --action 1), which mints the key, writes lemond.conf and
    // restarts Lemonade - one implementation, not a second copy in Rust.
    MintKey,
    Desktop,
    Reboot,
    Poweroff,
}

pub struct App {
    pub facts: Facts,
    pub log_scroll: usize,
    pub full_logs: bool,
    pub modal: Option<Modal>,
    /// advanced by the event loop ~4x a second while the box is still coming up
    pub spin: usize,
    /// started from an ssh session: `q` is then the way out, which the footer has to say (tty1's does not quit)
    pub ssh: bool,
    /// what just happened, or why the console did not exit, and when it was said (it clears itself after a while)
    pub message: Option<(String, std::time::Instant)>,
}

impl App {
    pub fn new(facts: Facts) -> App {
        App {
            facts,
            log_scroll: 0,
            full_logs: false,
            modal: None,
            spin: 0,
            ssh: false,
            message: None,
        }
    }

    /// Say something under the footer for a few seconds.
    pub fn say(&mut self, what: impl Into<String>) {
        self.message = Some((what.into(), std::time::Instant::now()));
    }
}

fn styled(text: impl Into<String>, color: Option<Color>, bold: bool) -> Line<'static> {
    let mut l = Line::from(Span::raw(text.into()));
    let mut s = Style::default();
    if let Some(c) = color {
        s = s.fg(c);
    }
    if bold {
        s = s.add_modifier(Modifier::BOLD);
    }
    l.style = s;
    l
}

/// `" TITLE ----..."` filled to exactly `width` columns.
fn rule(title: &str, width: usize) -> Line<'static> {
    let label = format!(" {title} ");
    let taken = label.chars().count();
    let bar = width.saturating_sub(taken);
    Line::from(vec![
        Span::styled(
            label,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("-".repeat(bar), Style::default().fg(Color::DarkGray)),
    ])
}

fn header(f: &Facts, width: usize) -> Line<'static> {
    let ip = f.ip().unwrap_or("no address yet");
    let sb = match f.secure_boot {
        SecureBoot::Enabled => ("Secure Boot enabled", Some(Color::Green)),
        SecureBoot::Disabled => ("Secure Boot disabled", Some(Color::Yellow)),
        SecureBoot::Unknown => ("Secure Boot unknown", Some(Color::DarkGray)),
    };
    // Which system is running comes first, beside the edition's own name: after an install the screen must not still
    // look like a stick (#711 - "live usb is loaded up" while the facts came from the NVMe). The word is init.sh's
    // (`MODE: LIVE` / `MODE: INSTALLED`), so the console and the boot banner cannot disagree.
    let mut spans = vec![Span::styled(
        "1bit OS".to_string(),
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )];
    if let Some(w) = f.mode.word() {
        let colour = match f.mode {
            Mode::Installed => Color::Cyan,
            _ => Color::Green,
        };
        spans.push(Span::styled(
            format!(" {w}"),
            Style::default().fg(colour).add_modifier(Modifier::BOLD),
        ));
    }
    spans.extend([
        Span::styled(
            format!(" {} ", f.release),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{}  ", f.hostname),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(format!("{ip}  "), Style::default().fg(Color::Gray)),
        Span::styled(
            format!("up {}  ", human_uptime(f.uptime_s)),
            Style::default().fg(Color::Gray),
        ),
    ]);
    // The two facts that may give way on a narrow screen, in the order they go: the temperature first, then Secure
    // Boot. The mode, the release, the host and the address never do, and the blocks that may not be trimmed at all
    // (the nine actions, the footer) are not this function's. Dropping them instead of letting ratatui clip the line
    // is what keeps a 60-column ssh window from reading as `Secure Boot enab`.
    let mut tail: Vec<Span<'static>> = vec![Span::styled(
        sb.0,
        Style::default().fg(sb.1.unwrap_or(Color::Gray)),
    )];
    if let Some(t) = f.temp_c {
        tail.push(Span::styled(
            format!("  SoC {t}C"),
            Style::default().fg(Color::Gray),
        ));
    }
    let cells =
        |s: &[Span<'static>]| -> usize { s.iter().map(|x| x.content.chars().count()).sum() };
    while !tail.is_empty() && cells(&spans) + cells(&tail) > width {
        tail.pop();
    }
    spans.extend(tail);
    Line::from(spans)
}

/// The CONNECT block: the owner's cycle-18 item, and the reason this screen exists at all ("when you first
/// boot the headless there is no data to display on how to connect, just the menu"). It is composed from the
/// facts on every draw, so it cannot go stale, and while the box is still coming up it says which half is
/// missing instead of showing a URL that does not answer yet.
pub fn connect_block(f: &Facts, spin: &str, width: usize) -> Vec<Line<'static>> {
    /// The wording is the dialog menu's own CONNECT block (cycle 18, pi-os3): vm-check's `U2 menu connect` row
    /// greps these exact strings off tty1 (`web UI http://`, `api  http://`, `ssh  root@`, `Lemonade Mobile`), so
    /// the two surfaces say the same thing and that row keeps passing whichever of them is on the console.
    const IND: &str = "  ";
    let mut out = Vec::new();
    let state = if f.ips.is_empty() {
        Some(format!("waiting for network {spin}"))
    } else if !f.lemond_listening {
        Some(format!("Lemonade starting {spin}"))
    } else {
        None
    };
    let title = match &state {
        Some(s) => format!("CONNECT - {s}"),
        None => "CONNECT".to_string(),
    };
    out.push(rule(&title, width));

    // Before DHCP there is no address, and a URL with a `-` for a host is not a URL anybody can use: the first
    // seconds of a boot used to read as broken output. There is exactly one honest thing to say then (owner
    // decision 2026-11-04 20:19), so the block says it and nothing else - the address rows come back with the lease.
    // `vm-check.sh`'s `U2 menu connect` row needs four dotted digits and is red in this state either way; it reads
    // the booted box, where these rows are all present again.
    let Some(host) = f.ip() else {
        out.push(Line::from(vec![
            Span::styled(IND, Style::default().fg(Color::Cyan)),
            Span::styled(
                "web UI: waiting for network".to_string(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        out.push(Line::from(Span::styled(
            format!(
                "{IND}no address yet - press 3 (Network); DHCP is automatic once a cable is in"
            ),
            Style::default().fg(Color::Yellow),
        )));
        return out;
    };
    let keyed = f.key.is_some();
    // Cycle 20 (owner: "blank key"): the appliance ships with NO API key, so the block says what that means for
    // whoever is reading it - anyone on this network can use Lemonade until item 1 mints one. Keyed mode keeps the
    // wording the gate greps for, so one renderer serves both modes and neither reads as the other.
    let mut rows = vec![
        format!("web UI http://{host}:13305"),
        // the name a phone (and Lemonade Mobile) can use instead of the address: pi-os3 makes the box announce
        // it via avahi, so when the responder is not running the row says so instead of promising a dead name
        if f.mdns {
            "name http://1bit-os.local:13305   (mDNS)".to_string()
        } else {
            "name http://1bit-os.local:13305   (mDNS - avahi not running)".to_string()
        },
        if keyed {
            format!("api  http://{host}:13305/api/v1   (API key: press 1)")
        } else {
            format!("api  http://{host}:13305/api/v1")
        },
        format!("ssh  root@{host}   (your key in /data/ssh/authorized_keys)"),
        if keyed {
            format!("app  Lemonade Mobile: server http://{host}:13305 + that key")
        } else {
            format!("app  Lemonade Mobile: server http://{host}:13305 (any key)")
        },
    ];
    if !keyed {
        rows.push("key  none - anyone here can use it; press 1 to require one".to_string());
    }
    for value in rows {
        let indent = IND.len();
        let style = Style::default().fg(Color::Gray);
        let avail = width.saturating_sub(indent).max(8);
        // a row that fits is emitted VERBATIM: its two spaces are part of what the gate greps for (`api  http://`),
        // and wrap_words normalises runs of spaces by design
        let pieces = if value.chars().count() <= avail {
            vec![value.clone()]
        } else {
            wrap_words(&value, avail)
        };
        for (i, w) in pieces.into_iter().enumerate() {
            if i == 0 {
                out.push(Line::from(vec![
                    Span::styled(IND, Style::default().fg(Color::Cyan)),
                    Span::styled(
                        w,
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
            } else {
                out.push(Line::from(vec![
                    Span::raw(" ".repeat(indent)),
                    Span::styled(w, style),
                ]));
            }
        }
    }

    if f.ips.is_empty() {
        out.push(Line::from(Span::styled(
            format!(
                "{IND}no address yet - press 3 (Network); DHCP is automatic once a cable is in"
            ),
            Style::default().fg(Color::Yellow),
        )));
    } else if !f.lemond_listening {
        out.push(Line::from(Span::styled(
            format!("{IND}Lemonade is not answering on 13305 yet - 8 shows its log"),
            Style::default().fg(Color::Yellow),
        )));
    }
    out
}

/// Each panel's content, one logical line per fact. The layout below decides how many columns and how wide;
/// this function never knows the screen size, so the two layouts cannot disagree about what is true.
pub fn panels(f: &Facts) -> Vec<(&'static str, Vec<String>)> {
    // Before the first probe answers there is nothing truthful to put in these panels: the empty defaults
    // would read as "DOWN", "absent", "not mounted" about a box that is still coming up. One panel that
    // says which question is being asked, and the collector's frame replaces it within a second.
    if f.idle {
        return vec![(
            "STARTING",
            vec!["asking Lemonade, the GPU, the NPU and the model list".to_string()],
        )];
    }
    let mut p = Vec::new();

    let mut lemonade = Vec::new();
    match f.lemond_pid {
        Some(pid) if f.lemond_listening => lemonade.push(format!("up (pid {pid})")),
        Some(pid) if f.lemond_listed => {
            lemonade.push(format!("pid {pid}, 13305 listed but nothing answers"))
        }
        Some(pid) => lemonade.push(format!("pid {pid}, not listening on 13305")),
        None => lemonade.push("DOWN (no lemond process)".to_string()),
    }
    // same rule as the CONNECT block: no address yet is neither the address "-" nor a URL (see `connect_block`)
    lemonade.push(match f.ip() {
        Some(ip) => format!("http://{ip}:13305"),
        None => "web UI: waiting for network".to_string(),
    });
    // Owner decision 2026-11-04 20:19: the console says which engine is serving, in Lemonade's own words - no
    // marketing map. `engine: llamacpp (gpu)` is what the box reports, not what we would like it to say.
    if let Some(e) = &f.engine {
        lemonade.push(format!("engine: {e}"));
    }
    lemonade.push(if f.key.is_some() {
        "API key: set, press 1".to_string()
    } else {
        "API key: none (blank mode), press 1".to_string()
    });
    p.push(("LEMONADE", lemonade));

    let g = &f.gpu;
    let mut gpu = Vec::new();
    match &g.device {
        Some(d) => {
            gpu.push(d.clone());
            if let Some(mib) = g.free_mib {
                gpu.push(format!("{mib} MiB free"));
            }
        }
        None if !g.asked => gpu.push("asking the runtime...".to_string()),
        None => gpu.push(g.note.clone()),
    }
    if g.vram_total > 0 {
        gpu.push(format!(
            "VRAM {} of {} used, GTT {} of {}",
            human_bytes(g.vram_used),
            human_bytes(g.vram_total),
            human_bytes(g.gtt_used),
            human_bytes(g.gtt_total)
        ));
    }
    gpu.push(match g.temp_c {
        Some(t) => format!("{t}C"),
        None => "no sensor".to_string(),
    });
    p.push(("GPU", gpu));

    let n = &f.npu;
    p.push((
        "NPU",
        vec![
            format!("amdxdna {}", if n.module { "loaded" } else { "absent" }),
            format!("firmware {}", if n.firmware { "present" } else { "absent" }),
            match &n.device {
                Some(d) if n.busy => format!("{d} busy"),
                Some(d) => format!("{d} idle"),
                None => "no /dev/accel node".to_string(),
            },
        ],
    ));

    let mut models = Vec::new();
    if f.models.downloaded.is_empty() {
        models.push("none downloaded (4 pulls one)".to_string());
    } else {
        models.push(format!("{} downloaded", f.models.downloaded.len()));
        for m in f.models.downloaded.iter().take(3) {
            models.push(m.clone());
        }
    }
    models.push(match f.models.loaded.len() {
        0 => "loaded: none".to_string(),
        _ => format!("loaded: {}", f.models.loaded.join(", ")),
    });
    p.push(("MODELS", models));

    let fans = if f.fans.is_empty() {
        "not readable".to_string()
    } else {
        f.fans.clone()
    };
    p.push((
        "FANS / TEMPS",
        vec![
            fans,
            match f.temp_c {
                Some(t) => format!("temps max {t}C"),
                None => "no sensor".to_string(),
            },
        ],
    ));

    let mut data = Vec::new();
    match f.data {
        Some((total, free)) => data.push(format!(
            "{} total, {} free",
            human_bytes(total),
            human_bytes(free)
        )),
        None => data.push("not mounted".to_string()),
    }
    data.push(match f.stats.tps {
        Some(t) => format!(
            "last request {t:.1} tok/s{}",
            match f.stats.ttft_s {
                Some(ttft) => format!(", first token {:.0} ms", ttft * 1000.0),
                None => String::new(),
            }
        ),
        None => "last request: none since boot".to_string(),
    });
    if let Some(r) = f.stats.requests {
        data.push(match f.stats.active {
            Some(n) if n > 0 => format!("{r} requests served, {n} streaming now"),
            _ => format!("{r} requests served"),
        });
    }
    p.push(("DATA", data));
    p
}

/// The nine actions, numbered, one row each (the owner's MENU block). Never abbreviated: the labels are what the
/// gates and the owner's own fingers know.
///
/// Item 1 is the SAME string in both key modes on purpose. pi-prs on #935: the appliance ships blank (cycle-20 item
/// 7), and `vm-check.sh:417` greps `Show the API key` off tty1, so a mode-dependent label turns `U2 menu` (and
/// pi-release's c15 menugate) red on every shipped image. The blank-key offer belongs where the item-7 work puts it
/// - the CONNECT block and item 1's own dialog, on `os/v3-c20` - not in the label the gates know.
pub fn menu_lines() -> Vec<String> {
    ACTIONS
        .iter()
        .enumerate()
        .map(|(i, label)| format!("{} {label}", i + 1))
        .collect()
}

/// The same rows as drawn: wrapped (never clipped) with a continuation indent, should the screen be narrower
/// than the longest label. The indent is spent out of the wrap, not added after it: a row indented to the pane
/// width is a row the widget clips (`wrap_hanging`).
pub fn menu_rows(width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for row in menu_lines() {
        for (i, part) in wrap_hanging(&row, width, width.saturating_sub(2))
            .into_iter()
            .enumerate()
        {
            let text = if i == 0 { part } else { format!("  {part}") };
            out.push(styled(text, Some(Color::Cyan), i == 0));
        }
    }
    out
}

/// The newest `height` visual lines of the log, wrapped to the width so a long record is continued rather than
/// cut ("KRN ... amdgpu 0000:c7:00.0: HRX runtime registered HRX0" is wider than 80 columns). Scrolling is in
/// LOGICAL records - one keypress moves one record - and both layouts share this, so the full-screen view and
/// the pane below the panels cannot show different text.
fn log_lines(a: &App, height: usize, width: usize) -> Vec<Line<'static>> {
    let total = a.facts.logs.len();
    if total == 0 {
        // idle: nothing has been read yet, which is not the same statement as "there is nothing in it"
        let empty = if a.facts.idle {
            " (reading the Lemonade log and the kernel ring now)"
        } else {
            " (no Lemonade log and no kernel GPU/NPU lines yet)"
        };
        return vec![styled(empty, Some(Color::DarkGray), false)];
    }
    let end = total.saturating_sub(a.log_scroll.min(total.saturating_sub(1)));
    // collected BACKWARDS and stopped as soon as the pane is full: drawing the log then costs what is on screen
    // instead of the wrapped length of all 400 records, which is pi-prs' second focus area (a record with no
    // spaces costs its own bytes and nothing more)
    let mut blocks: Vec<Vec<(Option<Color>, String)>> = Vec::new();
    let mut lines = 0usize;
    for rec in a.facts.logs[..end].iter().rev() {
        let color = if rec.starts_with("KRN") {
            Some(Color::Magenta)
        } else if rec.contains("error") || rec.contains("ERROR") || rec.contains("fail") {
            Some(Color::Red)
        } else {
            Some(Color::Gray)
        };
        let block: Vec<(Option<Color>, String)> = wrap_hanging(rec, width, width.saturating_sub(2))
            .into_iter()
            .enumerate()
            .map(|(i, w)| (color, if i == 0 { w } else { format!("  {w}") }))
            .collect();
        lines += block.len();
        blocks.push(block);
        if lines >= height {
            break;
        }
    }
    let mut visual: Vec<(Option<Color>, String)> = Vec::new();
    for block in blocks.into_iter().rev() {
        visual.extend(block);
    }
    let start = visual.len().saturating_sub(height);
    visual[start..]
        .iter()
        .map(|(c, t)| styled(t.clone(), *c, false))
        .collect()
}

fn log_title(a: &App, width: usize, full: bool) -> Line<'static> {
    let total = a.facts.logs.len();
    let pos = if a.log_scroll == 0 {
        "newest".to_string()
    } else {
        format!("{} lines up", a.log_scroll.min(total))
    };
    if full {
        return rule(
            &format!("LOGS - full screen, 8 or q back to the dashboard ({total} records)"),
            width,
        );
    }
    rule(&format!("LOGS {pos}"), width)
}

fn panel_grid(f: &Facts, width: usize) -> Vec<Line<'static>> {
    let panels = panels(f);
    let cell = width.saturating_sub(4) / 3;
    let mut out = Vec::new();
    for chunk in panels.chunks(3) {
        let mut cells: Vec<(String, Vec<String>)> = Vec::new();
        for (title, values) in chunk {
            // each logical fact keeps its own row: joining them first would let a wrapped value push the next
            // fact onto the wrong line (measured at 240x67: "GTT 512M" then "of 61G 43C")
            let mut lines = Vec::new();
            for v in values {
                lines.extend(wrap_words(v, cell));
            }
            cells.push((title.to_string(), lines));
        }
        let h = cells.iter().map(|(_, l)| l.len()).max().unwrap_or(1);
        let mut heads: Vec<Span> = Vec::new();
        for (i, (title, _)) in cells.iter().enumerate() {
            let label = format!(" {title} ");
            let taken = label.chars().count();
            let bar = cell.saturating_sub(taken);
            heads.push(Span::styled(
                label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
            if i + 1 < cells.len() {
                heads.push(Span::styled(
                    "-".repeat(bar + 2),
                    Style::default().fg(Color::DarkGray),
                ));
            } else {
                heads.push(Span::styled(
                    "-".repeat(bar),
                    Style::default().fg(Color::DarkGray),
                ));
            }
        }
        out.push(Line::from(heads));
        for row in 0..h {
            let mut spans: Vec<Span> = Vec::new();
            for (i, (_, lines)) in cells.iter().enumerate() {
                let text = lines.get(row).cloned().unwrap_or_default();
                let pad = cell.saturating_sub(text.chars().count());
                spans.push(Span::styled(text, Style::default().fg(Color::Gray)));
                if i + 1 < cells.len() {
                    spans.push(Span::raw(" ".repeat(pad + 2)));
                }
            }
            out.push(Line::from(spans));
        }
    }
    out
}

/// The six panels as ONE line for a text console, because 80x25 has to hold header + CONNECT + this + nine menu
/// rows + three log rows, and the nine rows are the owner's. Each field is the shortest TRUE form of the same
/// fact the panel shows at 240x67, never a truncated field: `wrap_words` breaks the line if the screen is narrower
/// than the whole thing.
fn compact_status(f: &Facts, width: usize) -> Vec<Line<'static>> {
    if f.idle {
        return wrap_words("STARTING - the probes have not answered yet", width)
            .into_iter()
            .map(|l| styled(l, Some(Color::DarkGray), false))
            .collect();
    }
    let mut fields: Vec<String> = Vec::new();

    let mut lemonade = match (f.lemond_pid, f.lemond_listening) {
        (Some(pid), true) => format!("LEMONADE up {pid}"),
        (Some(pid), false) => format!("LEMONADE pid {pid} no answer"),
        (None, _) => "LEMONADE DOWN".to_string(),
    };
    // the same sentence the wide layout's LEMONADE panel carries, on the field it belongs to: a FIELD of its own
    // would push the compact block to a third row at 80x25 and cost the log pane one (measured)
    if let Some(e) = &f.engine {
        lemonade.push_str(&format!(" engine: {e}"));
    }
    fields.push(lemonade);

    let g = &f.gpu;
    fields.push(match (&g.device, g.temp_c) {
        (Some(d), Some(t)) => format!("GPU {} {t}C", short_device(d)),
        (Some(d), None) => format!("GPU {}", short_device(d)),
        (None, _) => "GPU no HRX device".to_string(),
    });

    let n = &f.npu;
    fields.push(
        match (&n.device, n.module) {
            (Some(_), true) if n.busy => "NPU busy",
            (Some(_), true) => "NPU idle",
            (_, false) => "NPU amdxdna absent",
            _ => "NPU no accel node",
        }
        .to_string(),
    );

    fields.push(match f.models.downloaded.len() {
        0 => "MODELS none".to_string(),
        n => format!("MODELS {n}"),
    });
    fields.push(fans_word(f));
    fields.push(match f.data {
        Some((_, free)) => format!("DATA {} free", human_bytes(free)),
        None => "DATA not mounted".to_string(),
    });
    if let Some(t) = f.stats.tps {
        fields.push(format!("{t:.0} tok/s"));
    }

    // the hanging indent belongs to the block, so `pack_fields` puts it on and this only colours the rows; two
    // indents (one here, one there) put a wrapped field four cells in and two cells past the pane
    let text = pack_fields(&fields, width, width.saturating_sub(2));
    text.into_iter()
        .enumerate()
        .map(|(i, l)| {
            if i == 0 {
                styled(l, Some(Color::Gray), false)
            } else {
                styled(l, Some(Color::DarkGray), false)
            }
        })
        .collect()
}

/// Pack `A | B | C` into as few rows as the width allows WITHOUT splitting a field: at 80 columns the six facts
/// come out as two rows, and `DATA` never ends up alone at the end of the first one. A field wider than the whole
/// screen (a long device name on a 40-column serial line) is wrapped by itself rather than cut. `first` is the
/// opening row and `rest` the rows after it (`width - 2`: this function owns the two-cell hanging indent).
fn pack_fields(fields: &[String], first: usize, rest: usize) -> Vec<String> {
    let first = first.max(1);
    let rest = rest.max(1);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for field in fields {
        let width = if out.is_empty() { first } else { rest };
        if cur.is_empty() {
            cur = field.clone();
        } else if cur.chars().count() + 3 + field.chars().count() <= width {
            cur.push_str(" | ");
            cur.push_str(field);
        } else {
            out.push(std::mem::take(&mut cur));
            cur = field.clone();
        }
        // `cur` may start a row that is indented, so ask again after a push rather than reusing the old width
        let width = if out.is_empty() { first } else { rest };
        if cur.chars().count() > width {
            let mut parts = wrap_hanging(&cur, width, rest).into_iter();
            if let Some(first) = parts.next() {
                out.push(first);
            }
            out.extend(parts);
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.into_iter()
        .enumerate()
        .map(|(i, r)| if i == 0 { r } else { format!("  {r}") })
        .collect()
}

/// `HRX0: AMD Radeon 8060S Graphics` -> `HRX0`: the colons separate the runtime's node from the marketing name.
fn short_device(d: &str) -> &str {
    d.split(':').next().unwrap_or(d).trim()
}

/// `fans: controlled (fan1 0 rpm fan2 0 rpm)` -> `FANS controlled`.
// shortcut: the probe's "fans: <state> (fanN rpm)" shape; a different shape says "?" instead of a guess -
// upgrade if the ec_su_axb35 probe's wording changes.
fn fans_word(f: &Facts) -> String {
    if f.fans.is_empty() {
        return "FANS not readable".to_string();
    }
    let state = f.fans.split_once(':').map(|(_, r)| r).unwrap_or(&f.fans);
    match state.split('(').next().unwrap_or("").trim() {
        "" => "FANS ?".to_string(),
        word => format!("FANS {word}"),
    }
}

fn modal_box(m: &Modal) -> (String, Vec<String>, String) {
    match m {
        Modal::Info { title, body } => (
            title.clone(),
            body.clone(),
            " any key returns to the dashboard ".into(),
        ),
        Modal::Confirm {
            title,
            question,
            yes,
        } => {
            let hint = match yes {
                Yes::ShowKey => "y shows the key on this screen, n cancels",
                Yes::MintKey => "y mints one and restarts Lemonade, n cancels",
                Yes::Desktop => "y starts the desktop now, n cancels",
                Yes::Reboot => "y reboots the box, n cancels",
                Yes::Poweroff => "y powers the box off, n cancels",
            };
            (
                title.clone(),
                question.lines().map(str::to_string).collect(),
                format!(" {hint} "),
            )
        }
        Modal::Power => (
            "Reboot / power off".into(),
            vec!["what should the box do?".into()],
            " r reboots, p powers off, any other key cancels ".into(),
        ),
        Modal::Busy { title } => (
            title.clone(),
            vec!["running in the foreground...".into()],
            String::new(),
        ),
    }
}

/// Where one URL sits on the finished screen, in cells: `text` is both what is shown and what the terminal is
/// asked to open. Only whole URLs are collected - a URL split by wrapping is left plain, because half a link is
/// worse than none - and `main` writes them as OSC 8 hyperlinks when the terminal is not the VGA console.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    pub x: u16,
    pub y: u16,
    pub text: String,
}

/// The footer's legend as separate words, so a short frame can give up its tail instead of giving up an action.
/// `1-9 actions` leads (or, in the full-log view, closes) the line: it is the string the gates read.
fn keys_segments(a: &App) -> Vec<&'static str> {
    if a.full_logs {
        vec![
            "q back",
            "arrows scroll",
            "f newest",
            "r refresh",
            "1-9 actions",
        ]
    } else if a.ssh {
        vec![
            "1-9 actions",
            "arrows scroll",
            "f newest",
            "r refresh",
            "8 logs",
            "q quits",
        ]
    } else {
        // the tty1 promise: this console never drops to a bare prompt
        vec![
            "1-9 actions",
            "arrows scroll",
            "f newest",
            "r refresh",
            "8 logs",
            "q stays",
        ]
    }
}

/// The legend line(s), then - while something is being said - the message under them.
fn footer_lines(segments: &[&str], message: Option<&str>, width: usize) -> Vec<Line<'static>> {
    let mut rows = vec![segments.join("   ")];
    if let Some(m) = message {
        rows.push(m.to_string());
    }
    rows.iter()
        .flat_map(|s| wrap_words(s, width))
        .map(|s| styled(s, Some(Color::DarkGray), false))
        .collect()
}

pub fn draw(f: &mut Frame, a: &App, out_links: &mut Vec<Link>) {
    let area = f.area();
    let w = area.width as usize;
    let h = area.height as usize;
    let wide = w >= 100;

    // header, CONNECT, then the status facts - the "technical information" the owner wants the menu below
    let mut top: Vec<Line<'static>> = Vec::new();
    let mut menu: Vec<Line<'static>> = Vec::new();
    let mut facts: Vec<Line<'static>> = Vec::new();
    let mut qr: Option<(u16, Vec<Line<'static>>)> = None;

    // The nine actions are ROWS and the footer is WORDS, so when the frame is short the words give way first.
    // Before this, in a 40-column window whose pane is 12 rows, the frame was exactly the nine actions and two
    // footer rows; one keypress added a two-row message and ratatui clipped the END of the MENU - `9 Reboot /
    // power off`, a `vm-check.sh` / `menugate` string - for the message's 12 s, with the footer still promising
    // `1-9 actions` (issue #1015). So: the legend's tail words, then the legend line, and the nine rows are
    // never charged for any of it.
    let actions = if a.full_logs {
        Vec::new()
    } else {
        menu_rows(w)
    };
    let full = keys_segments(a);
    let mut segments = full.clone();
    let mut message = a.message.as_ref().map(|(m, _)| m.clone());
    let budget = h.saturating_sub(actions.len());
    let mut footer = footer_lines(&segments, message.as_deref(), w);
    // the legend's tail gives way first, then the legend itself, and the message is the last thing dropped: it is
    // the feedback for the key that was just pressed, and `1-9 actions` is only read by the gates at 80x25 and
    // 240x67, where none of this trims.
    while footer.len() > budget {
        if segments.len() > 1 {
            segments.pop();
        } else if !segments.is_empty() {
            segments.clear();
        } else if message.take().is_some() {
            segments = full.clone();
        } else {
            break;
        }
        footer = footer_lines(&segments, message.as_deref(), w);
    }

    if !a.full_logs {
        top.push(header(&a.facts, w));
        top.extend(connect_block(&a.facts, crate::facts::token(a.spin), w));
        facts = if wide {
            panel_grid(&a.facts, w)
        } else {
            compact_status(&a.facts, w)
        };
        // The MENU sits between the facts and the log, always whole, one numbered row per action. The rule above
        // them is decoration, so it is the first thing to give way when the frame is short: before this, at 80x10,
        // 56x11, 40x12 and 30x16 (a tmux pane, an ssh window on a phone) the rule took the row that
        // `9 Reboot / power off` needed and ratatui clipped the END of the menu while the footer still promised
        // "1-9 actions". Swept over 13 widths x 19 heights: 72 sizes gained that row, none lost the
        // frame-is-the-terminal invariant (`the_menu_rule_gives_way_before_an_action_does`).
        if 1 + actions.len() + footer.len() <= h {
            menu.push(rule("MENU", w));
        }
        menu.extend(actions);
        // The QR is drawn to the RIGHT of the CONNECT rows, so the block has to be as tall as the code. It is only
        // drawn when the WHOLE frame still fits (a log title and one log row are the floor): the first version
        // charged it 12 rows, counted by hand, which at 104x29 let it take the panel grid's rows and paint its
        // quiet zone over the MENU rule (measured; `the_qr_never_costs_another_block_a_row` sweeps the sizes).
        if let Some((qw, rows)) = qr_panel(&a.facts, &top[1..], w) {
            let padded = top.len().max(2 + rows.len());
            if padded + facts.len() + menu.len() + 2 + footer.len() <= h {
                top.resize(padded, Line::from(""));
                qr = Some((qw, rows));
            }
        }
    }

    // Nothing is drawn half a block. The nine actions and the footer are what this screen is for - `Reboot /
    // power off` is a gate string - so they are never trimmed, and everything above them gives way in a fixed
    // order: the status facts first (they are "technical information"), then the CONNECT block's own tail rows.
    // Before this, `log_h` was a remainder with no floor, and a 15-row window clipped the MENU itself: 40x15 drew
    // actions 1-5 and a footer that still promised "1-9 actions", and a log title with nothing under it.
    // The nine actions and the footer come first; `avail` is what they leave for everything above them.
    let fixed = menu.len() + footer.len();
    let avail = h.saturating_sub(fixed);
    if top.len() + facts.len() + 2 > avail {
        facts.clear();
    }
    let mut keep = top.len().min(avail.saturating_sub(facts.len()));
    // A log pane is a title and at least one row; one row alone is the tail of a wrapped record ("  tokens,
    // 330.75 tok/s)"), which reads as broken output. When only one row is left it goes back to the block above,
    // so the footer stays on the last row.
    if avail.saturating_sub(keep + facts.len()) == 1 && keep < top.len() {
        keep += 1;
    }
    top.truncate(keep);

    let log_h = avail.saturating_sub(top.len() + facts.len());
    let mut logs: Vec<Line> = Vec::new();
    if log_h >= 2 {
        logs.push(log_title(a, w, a.full_logs));
        logs.extend(log_lines(a, log_h - 1, w));
    }
    top.extend(facts);

    let mut constraints: Vec<Constraint> = Vec::new();
    if !top.is_empty() {
        constraints.push(Constraint::Length(top.len() as u16));
    }
    if !menu.is_empty() {
        constraints.push(Constraint::Length(menu.len() as u16));
    }
    if log_h > 0 {
        constraints.push(Constraint::Length(log_h as u16));
    }
    constraints.push(Constraint::Length(footer.len() as u16));
    let rows = Layout::vertical(constraints).split(area);

    let mut row = 0;
    if !top.is_empty() {
        render_lines(f, rows[row], &top);
        row += 1;
    }
    if !menu.is_empty() {
        render_lines(f, rows[row], &menu);
        row += 1;
    }
    if log_h > 0 {
        render_lines(f, rows[row], &logs);
        row += 1;
    }
    render_lines(f, rows[row], &footer);

    // the QR goes on top of the blank columns the block left free (it was measured to fit before it is drawn),
    // and before the modal, so a dialog is never covered by a code
    if let Some((qw, lines)) = &qr {
        let rect = Rect {
            x: area.x + area.width.saturating_sub(*qw + 1),
            y: area.y + 2,
            width: (*qw).min(area.width),
            height: (lines.len() as u16).min(area.height.saturating_sub(1)),
        };
        f.render_widget(Paragraph::new(lines.clone()), rect);
    }

    out_links.clear();
    for (i, line) in top.iter().enumerate() {
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        for (x, url) in urls_in(&text, w) {
            out_links.push(Link {
                x: x as u16,
                y: i as u16,
                text: url,
            });
        }
    }

    if let Some(m) = &a.modal {
        let (title, text, hint) = modal_box(m);
        let lines: Vec<String> = std::iter::once(title.clone())
            .chain(text.iter().cloned())
            .chain(std::iter::once(hint.clone()))
            .collect();
        // the box is the longest line plus its own padding: sizing it to the longest line and then wrapping
        // inside it made every hint wrap for no reason (measured: "any other key / cancels")
        let width = (lines.iter().map(|l| l.chars().count()).max().unwrap_or(10) + 4)
            .min(w.saturating_sub(4))
            .max(27);
        let wrap = |s: &str| -> Vec<Line<'static>> {
            wrap_words(s, width.saturating_sub(4))
                .into_iter()
                .map(|l| styled(l, Some(Color::White), false))
                .collect()
        };
        let mut title_rows = wrap(&title);
        let mut text_rows: Vec<Line> = text.iter().flat_map(|l| wrap(l)).collect();
        let hint_rows = wrap(&hint);
        // The hint is the only row that says how to answer the question, so it is never the row that goes: a frame
        // too short for the whole box loses the explanation first, whole rows at a time, and then the title. Before
        // this the box was clamped to the frame and ratatui clipped the END - at 40x11 the reboot confirm drew
        // `Reboot the appliance now?` with `y reboots the box, n cancels` off the screen while `y` still rebooted.
        let room = (area.height as usize).saturating_sub(2);
        while title_rows.len() + text_rows.len() + hint_rows.len() > room && !text_rows.is_empty() {
            text_rows.pop();
        }
        while title_rows.len() + hint_rows.len() > room && !title_rows.is_empty() {
            title_rows.pop();
        }
        let body: Vec<Line> = title_rows
            .into_iter()
            .chain(text_rows)
            .chain(hint_rows)
            .collect();
        let bh = (body.len() as u16 + 2).min(area.height);
        let bw = (width as u16 + 4).min(area.width);
        let rect = centred(area, bw, bh);
        f.render_widget(ratatui::widgets::Clear, rect);
        let block = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .style(Style::default().fg(Color::White));
        let inner = block.inner(rect);
        f.render_widget(block, rect);
        f.render_widget(Paragraph::new(body), inner);
    }
}

/// The web UI URL as a scannable block, drawn on paper (black on white: a scanner wants dark-on-light) to the
/// right of the CONNECT rows. `None` when the screen cannot hold both - the QR is 33 columns wide for a
/// 27-character URL and 80x25 has no 33 free columns beside the block, so there the URL stays plain text.
fn qr_panel(f: &Facts, connect: &[Line<'static>], w: usize) -> Option<(u16, Vec<Line<'static>>)> {
    let url = format!("http://{}:13305", f.ip()?);
    let rows = crate::qr::code(&url)?;
    let qw = rows[0].chars().count();
    // the block's first line is the CONNECT rule, which is as wide as the screen: the code only has to clear
    // the rows that carry text
    let widest = connect
        .iter()
        .skip(1)
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.chars().count())
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0);
    // one column of air between the block and the code. The rows are the CALLER's question: it knows the panel
    // grid, the nine menu rows, the log floor and the footer it is about to draw, and only it can count them.
    if w < widest + qw + 2 {
        return None;
    }
    let paper = Style::default().fg(Color::Black).bg(Color::White);
    Some((
        qw as u16,
        rows.into_iter()
            .map(|r| Line::from(Span::styled(r, paper)))
            .collect(),
    ))
}

/// Every whole URL on one line of text, with the column it starts at - the CONNECT block's hyperlink cells.
///
/// A URL that ends exactly on the pane's last column is NOT returned: `wrap_words` hard-breaks a token that is
/// wider than the pane, so such a token may be the first half of one, and a link whose text is half a URL opens
/// the wrong thing (or nothing) when clicked. On an 80- or 240-column screen no CONNECT row reaches that edge,
/// so nothing real is lost; on a 30-column serial line it costs a link instead of opening a broken one.
fn urls_in(text: &str, width: usize) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = text[from..].find("http://") {
        let start = from + i;
        let tail = &text[start..];
        let end = tail
            .find(|c: char| c.is_whitespace() || c == ')' || c == ',')
            .unwrap_or(tail.len());
        let url = &tail[..end];
        let x = text[..start].chars().count();
        if url.len() > "http://".len() && x + url.chars().count() < width {
            out.push((x, url.to_string()));
        }
        from = start + end.max(1);
    }
    out
}

fn centred(area: Rect, w: u16, h: u16) -> Rect {
    Rect {
        x: area.x + area.width.saturating_sub(w) / 2,
        y: area.y + area.height.saturating_sub(h) / 2,
        width: w.min(area.width),
        height: h.min(area.height),
    }
}

fn render_lines(f: &mut Frame, area: Rect, lines: &[Line<'static>]) {
    f.render_widget(Paragraph::new(lines.to_vec()), area);
}

// ── the same facts as plain text: `--status`, and the regression snapshots ───────────────────────────────

pub fn status_text(f: &Facts) -> Vec<String> {
    let mut out = Vec::new();
    // `--status` is a transcript, not a frame: give the header room so every fact it carries is in the text a gate
    // greps (`1bit-console-tui --status` prints what the screen says, and the mode is one of those facts now)
    let h = header(f, 400);
    out.push(
        h.spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>(),
    );
    for l in connect_block(f, "-", 78) {
        out.push(
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>(),
        );
    }
    for (title, values) in panels(f) {
        out.push(format!("{title}  {}", values.join("  ")));
    }
    for l in menu_lines() {
        out.push(l);
    }
    out
}

fn buffer_text(buf: &Buffer) -> String {
    let mut s = String::new();
    for y in 0..buf.area.height {
        let mut line = String::new();
        for x in 0..buf.area.width {
            line.push_str(buf[(x, y)].symbol());
        }
        s.push_str(line.trim_end());
        s.push('\n');
    }
    s
}

/// Render one frame off-screen - what the snapshot tests and `--snapshot` both use.
pub fn render_snapshot(a: &App, width: u16, height: u16) -> String {
    render_snapshot_with_links(a, width, height).0
}

/// The frame and the hyperlink cells it carries, for the tests that check one against the other.
pub fn render_snapshot_with_links(a: &App, width: u16, height: u16) -> (String, Vec<Link>) {
    let backend = TestBackend::new(width, height);
    let mut t = Terminal::new(backend).expect("test terminal");
    let mut links = Vec::new();
    t.draw(|f| draw(f, a, &mut links)).expect("draw");
    (buffer_text(t.backend().buffer()), links)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{Gpu, Models};

    use crate::facts::c16_board;

    fn app(f: Facts) -> App {
        App::new(f)
    }

    /// The dialog menu's own item table: the `N "Label"` rows inside the `--menu` call of the main screen.
    ///
    /// That table is what `--action N` runs, so this screen's row N has to read the same words: keys 2-7 hand off by
    /// NUMBER, and a label that drifts turns "5 Install 1bit OS to a disk" into a row that runs something else.
    /// The anchor is the main dialog's own title - the script has three other `--menu` calls (network, models,
    /// reboot) whose rows are a different thing, and `1bit OS console` appears exactly once in it.
    fn menu_script_items(text: &str) -> Vec<String> {
        let mut in_main_menu = false;
        let mut items = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.contains("1bit OS console") {
                in_main_menu = true;
                continue;
            }
            if !in_main_menu {
                continue;
            }
            let Some(number) = line.chars().next().filter(|c| ('1'..='9').contains(c)) else {
                // the rows are contiguous: the first non-row line ends the table
                if !items.is_empty() {
                    break;
                }
                continue;
            };
            let Some(rest) = line[number.len_utf8()..].trim_start().strip_prefix('"') else {
                break;
            };
            let Some(end) = rest.find('"') else { break };
            items.push(rest[..end].to_string());
        }
        items
    }

    #[test]
    fn every_menu_row_is_the_menu_scripts_own_item() {
        // The nine labels exist twice - `ACTIONS` here and the dialog's --menu table in
        // embed/usr/local/bin/1bit-console-menu - and nothing held them together: #935 and #1163 both reworded a
        // label, and the gate (`vm-check.sh` `U2 menu`) reads the OTHER copy off tty1. A drift is not cosmetic:
        // the hand-off runs `--action N`, so row N would say one thing and do another.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../embed/usr/local/bin/1bit-console-menu");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (this test reads the menu the image ships)",
                path.display()
            )
        });
        let items = menu_script_items(&text);
        assert_eq!(
            items,
            ACTIONS.to_vec(),
            "row-for-row parity with {} is broken: {items:?}",
            path.display()
        );
        assert_eq!(items.len(), 9, "the dialog menu is nine items: {items:?}");
    }

    #[test]
    fn connect_block_names_how_to_reach_the_box() {
        // the owner's cycle-18 complaint, as a test: the first screen must say how to connect
        let f = app(c16_board()).facts;
        let text: Vec<String> = connect_block(&f, "-", 78)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        let joined = text.join("\n");
        for want in [
            "CONNECT",
            "http://192.168.50.110:13305",
            "http://192.168.50.110:13305/api/v1",
            "API key",
            "press 1",
            "ssh",
            "root@192.168.50.110",
            "Lemonade Mobile",
        ] {
            assert!(
                joined.contains(want),
                "CONNECT block does not mention {want:?}:\n{joined}"
            );
        }
    }

    #[test]
    fn before_dhcp_the_connect_block_waits_instead_of_inventing_an_address() {
        // Measured on main before this: a box with no lease yet (the first screen of every boot) printed
        // `web UI http://-:13305` / `api  http://-:13305/api/v1` / `ssh  root@-` - four URLs that cannot be used and
        // a hostname of `-`. Owner decision 2026-11-04 20:19: say `web UI: waiting for network` until there is an
        // address, so the block says one true thing instead of five false ones.
        let mut f = c16_board();
        f.ips.clear();
        f.lemond_listening = false;
        for (w, h) in [(240u16, 67u16), (80, 25)] {
            let text = render_snapshot(&app(f.clone()), w, h);
            assert!(
                text.contains("web UI: waiting for network"),
                "{w}x{h}: the waiting-for-network screen does not say so:\n{text}"
            );
            for bad in ["http://-:", "root@-", "<address>", "web UI http://"] {
                assert!(
                    !text.contains(bad),
                    "{w}x{h}: {bad:?} is back on the no-address screen:\n{text}"
                );
            }
            assert!(
                text.contains("no address yet - press 3"),
                "{w}x{h}: the row that says what to do is gone:\n{text}"
            );
        }
        // ...and the moment there is an address, the whole block is back exactly as the gate row reads it
        let text = render_snapshot(&app(c16_board()), 80, 25);
        for want in [
            "CONNECT",
            "web UI http://192.168.50.110:13305",
            "api  http://192.168.50.110:13305/api/v1",
            "ssh  root@192.168.50.110",
            "Lemonade Mobile",
        ] {
            assert!(
                text.contains(want),
                "the booted block lost {want:?}:\n{text}"
            );
        }
        assert!(!text.contains("waiting for network"), "{text}");
    }

    fn span_text(line: &Line<'static>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn a_log_continuation_fits_the_pane_it_is_drawn_on() {
        // The defect this holds: `log_lines` wrapped a record to the pane width and THEN indented every
        // continuation row by two, so those rows were drawn two cells wider than the pane and the widget clipped
        // their end. Measured with the shipped binary at 80x25: a record of 4-char tokens drew `… a36 a37 a3` and
        // then `a39 a40` - `a38` gone, `a3` a word cut in the middle. A record with one unbroken token is the
        // case that fills a row exactly, so it is the one that fails when the indent is not in the budget.
        let mut f = c16_board();
        f.logs = vec![
            format!("LEM {} tail", "x".repeat(200)),
            "LEM 2026-11-03 12:07:41 lemond: POST /api/v1/chat/completions 200".to_string(),
        ];
        for w in 20..=240usize {
            for (i, line) in log_lines(&app(f.clone()), 12, w).iter().enumerate() {
                let text = span_text(line);
                assert!(
                    text.chars().count() <= w,
                    "width {w}: drawn log row {i} is {} cells: {text:?}",
                    text.chars().count()
                );
            }
        }
    }

    #[test]
    fn a_packed_status_row_fits_the_pane_it_is_drawn_on() {
        // Same class one block up: `pack_fields` packed to the width and `compact_status` indents the rows after the
        // first by two. A field wider than the screen is hard-broken into exactly-width pieces, which is the case
        // that overflowed. `w < 100` is the compact layout.
        let mut f = c16_board();
        f.engine = Some("e".repeat(120));
        for w in 20..99usize {
            for (i, line) in compact_status(&f, w).iter().enumerate() {
                let text = span_text(line);
                assert!(
                    text.chars().count() <= w,
                    "width {w}: drawn status row {i} is {} cells: {text:?}",
                    text.chars().count()
                );
            }
        }
    }

    #[test]
    fn the_console_says_which_engine_is_serving() {
        // Owner decision 2026-11-04 20:19 item 1: the console reports what Lemonade says serves the loaded model,
        // in Lemonade's own words - no marketing map. The wide layout gets its own row, the compact layout carries
        // it on the LEMONADE field (a field of its own costs the 80x25 log pane a row, measured).
        let text240 = render_snapshot(&app(c16_board()), 240, 67);
        assert!(
            text240.contains("engine: llamacpp (gpu)"),
            "the LEMONADE panel does not name the engine:\n{text240}"
        );
        let text80 = render_snapshot(&app(c16_board()), 80, 25);
        assert!(
            text80.contains("engine: llamacpp (gpu)"),
            "the compact status does not name the engine:\n{text80}"
        );
        // the 80x25 log pane still pays for nothing: it keeps its three rows with the engine on the field
        assert_eq!(
            text80.lines().count(),
            25,
            "the frame is not the terminal:\n{text80}"
        );
        // nothing loaded (a box that has not served anything yet) -> no row, and no empty label either
        let mut f = c16_board();
        f.engine = None;
        f.models.loaded.clear();
        for (w, h) in [(240u16, 67u16), (80, 25)] {
            let text = render_snapshot(&app(f.clone()), w, h);
            assert!(
                !text.contains("engine:"),
                "{w}x{h}: an engine row with nothing loaded:\n{text}"
            );
        }
        // and `--status` carries the same line, which is what a booted-image gate row can read without a terminal
        let status = status_text(&c16_board()).join("\n");
        assert!(
            status.contains("engine: llamacpp (gpu)"),
            "--status does not name the engine:\n{status}"
        );
    }

    #[test]
    fn the_c18_gate_row_finds_its_strings_on_this_screen() {
        // vm-check.sh's `U2 menu connect` row (pi-os3, cycle 18) greps these fixed strings off tty1: if the TUI
        // is what tty1 gets, that row must keep passing, so the block uses the dialog menu's own wording.
        for (w, h) in [(240u16, 67u16), (80, 25)] {
            let text = render_snapshot(&app(c16_board()), w, h);
            for want in [
                "CONNECT",
                "web UI http://",
                "api  http://",
                "ssh  root@",
                "Lemonade Mobile",
            ] {
                assert!(
                    text.contains(want),
                    "{w}x{h}: the U2 menu connect row would go red - {want:?} is missing"
                );
            }
            assert!(
                text.contains("web UI http://192.168.50.110:13305"),
                "{w}x{h}: the row's address regex would not match"
            );
        }
    }

    #[test]
    fn connect_block_says_what_is_missing_while_booting() {
        let mut f = c16_board();
        f.ips.clear();
        f.lemond_listening = false;
        let text: Vec<String> = connect_block(&f, "|", 80)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        let joined = text.join("\n");
        assert!(joined.contains("waiting for network"), "{joined}");
        assert!(
            joined.contains("press 3"),
            "no network must point at key 3:\n{joined}"
        );

        let mut f = c16_board();
        f.lemond_listening = false;
        let text: Vec<String> = connect_block(&f, "/", 80)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        let joined = text.join("\n");
        assert!(joined.contains("Lemonade starting"), "{joined}");
        assert!(
            joined.contains("http://192.168.50.110:13305"),
            "the URL that will answer is still shown:\n{joined}"
        );
    }

    #[test]
    fn no_line_is_cut_inside_a_word_at_any_width() {
        // "firmware prese" (c16 board) must be impossible
        let f = c16_board();
        for w in [40usize, 60, 78, 80, 100, 160, 240] {
            for l in connect_block(&f, "-", w) {
                let s: String = l.spans.iter().map(|x| x.content.as_ref()).collect();
                assert!(
                    s.chars().count() <= w,
                    "width {w}: {s:?} is {len} wide",
                    len = s.chars().count()
                );
                for word in s.split_whitespace() {
                    if word.len() > 20 {
                        continue; // URLs and paths are single tokens
                    }
                    assert!(
                        ["NPU", "amdxdna", "loaded,", "firmware", "present,", "accel0"]
                            .contains(&word)
                            || !word.ends_with("prese"),
                        "width {w}: cut word {word:?} in {s:?}"
                    );
                }
            }
        }
        // and the panel that carries the phrase: every word survives wrapping
        let npu = panels(&f).into_iter().find(|(t, _)| *t == "NPU").unwrap();
        let joined = npu.1.join("  ");
        for w in [30usize, 34, 40, 60] {
            for line in crate::facts::wrap_words(&joined, w) {
                for word in line.split_whitespace() {
                    assert!(
                        joined.split_whitespace().any(|t| t == word),
                        "width {w}: {word:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_gates_action_labels_are_on_their_own_rows() {
        // vm-check's `U2 menu` row looks for these two strings on tty1, and the owner reads the MENU block, so
        // both sizes carry all nine labels whole, one numbered row each
        let rows = menu_lines();
        assert_eq!(rows.len(), 9, "one row per action: {rows:?}");
        for w in [80usize, 100, 120, 240] {
            for row in &rows {
                assert!(
                    row.chars().count() <= w,
                    "width {w}: {row:?} does not fit on its row"
                );
            }
            let joined = rows.join(" ");
            assert!(joined.contains("Show the API key"), "width {w}: {joined}");
            assert!(joined.contains("Reboot / power off"), "width {w}: {joined}");
            for (i, l) in ACTIONS.iter().enumerate() {
                assert!(
                    joined.contains(&format!("{} {l}", i + 1)),
                    "width {w}: action {} missing",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn the_compact_status_keeps_every_panel_and_never_splits_a_field() {
        let f = c16_board();
        for w in [80usize, 90, 99] {
            let lines: Vec<String> = compact_status(&f, w)
                .iter()
                .map(|l| {
                    l.spans
                        .iter()
                        .map(|s| s.content.as_ref())
                        .collect::<String>()
                })
                .collect();
            assert!(lines.len() <= 2, "width {w}: {lines:?}");
            for l in &lines {
                assert!(l.chars().count() <= w, "width {w}: {l:?} is over");
                assert!(
                    !l.trim_end().ends_with('|'),
                    "width {w}: field split: {l:?}"
                );
                assert!(
                    !l.trim_start().starts_with('|'),
                    "width {w}: field split: {l:?}"
                );
            }
            let joined = lines.join(" ");
            for name in ["LEMONADE", "GPU", "NPU", "MODELS", "FANS", "DATA"] {
                assert!(joined.contains(name), "width {w}: {name} is gone: {joined}");
            }
        }
    }

    #[test]
    fn the_menu_sits_between_the_facts_and_the_log() {
        // the owner, cycle 20: "can we place the menu above the scrolling logs and below the technical
        // information". This is the layout, asserted off a rendered frame rather than off the code's intent.
        for (w, h) in [(240u16, 67u16), (80, 25)] {
            let text = render_snapshot(&app(c16_board()), w, h);
            let lines: Vec<&str> = text.lines().collect();
            let at = |needle: &str| {
                lines
                    .iter()
                    .position(|l| l.contains(needle))
                    .unwrap_or_else(|| panic!("{w}x{h}: {needle:?} is not on screen:\n{text}"))
            };
            let (header, connect, menu, logs) =
                (at("1bit OS"), at("CONNECT"), at("MENU"), at("LOGS"));
            assert!(header < connect, "{w}x{h}: CONNECT is not below the header");
            assert!(
                connect < menu,
                "{w}x{h}: the menu is not below the technical information"
            );
            assert!(
                menu < logs,
                "{w}x{h}: the menu is not above the scrolling log"
            );
            // the nine rows are the nine lines right under the MENU rule, all of them above the log
            for i in 1..=9 {
                let row = menu + i;
                assert!(
                    row < logs,
                    "{w}x{h}: action {i} is not above the log:\n{text}"
                );
                assert!(
                    lines[row].starts_with(&format!("{i} ")),
                    "{w}x{h}: action {i} is not on its own row (row {row} reads {:?}):\n{text}",
                    lines[row]
                );
            }
        }
    }

    #[test]
    fn at_80x25_the_log_pane_still_gets_three_rows() {
        let text = render_snapshot(&app(c16_board()), 80, 25);
        let lines: Vec<&str> = text.lines().collect();
        let log_at = lines
            .iter()
            .position(|l| l.contains("LOGS"))
            .expect("a log pane");
        // the last line is the footer; everything between the LOGS title and it is the pane
        let pane = &lines[log_at + 1..lines.len() - 1];
        let filled = pane.iter().filter(|l| !l.trim().is_empty()).count();
        assert!(
            filled >= 3,
            "the log pane keeps only {filled} rows at 80x25:\n{text}"
        );
        // and the pane reaches the footer: a row counted twice leaves the last console row blank (measured before
        // the log title was made part of the pane)
        assert!(
            lines.last().is_some_and(|l| !l.trim().is_empty()),
            "the last console row at 80x25 is blank:\n{text}"
        );
    }

    #[test]
    fn item_1_keeps_the_gate_string_in_both_key_modes() {
        // pi-prs on #935: the appliance ships BLANK (cycle-20 item 7), so a mode-dependent item 1 turns `U2 menu`
        // red on every shipped image - `vm-check.sh:417` greps `Show the API key` off tty1. The label is therefore
        // the same string whatever the key mode, and the blank-key offer lives in the CONNECT block and in item 1's
        // own dialog (the item-7 work on os/v3-c20).
        let keyed = c16_board();
        let mut blank = c16_board();
        blank.key = None;
        // menu_lines() takes no facts at all any more, which is the structural half of the guarantee
        let rows = menu_lines();
        assert_eq!(rows.len(), 9, "{rows:?}");
        assert_eq!(rows[0], "1 Show the API key");
        for (name, f) in [("keyed", keyed), ("blank", blank)] {
            for (w, h) in [(240u16, 67u16), (80, 25)] {
                let text = render_snapshot(&app(f.clone()), w, h);
                for want in ["Show the API key", "Reboot / power off"] {
                    assert!(
                        text.contains(want),
                        "{name} box at {w}x{h}: the U2 menu row goes red - {want:?} is missing"
                    );
                }
            }
        }
    }

    #[test]
    fn the_header_says_which_system_is_running() {
        // #711: after an install the console must not still look like a stick. `init.sh` decides it at boot and leaves
        // `installed` or `live` in /etc/1bit; the header says the same word, beside the edition's own name.
        let mut f = c16_board();
        f.mode = Mode::Installed;
        let text = render_snapshot(&app(f.clone()), 240, 67);
        assert!(
            text.lines()
                .next()
                .unwrap_or("")
                .contains("1bit OS INSTALLED"),
            "the header does not say INSTALLED:\n{text}"
        );
        f.mode = Mode::Live;
        let text = render_snapshot(&app(f.clone()), 240, 67);
        assert!(
            text.lines().next().unwrap_or("").contains("1bit OS LIVE"),
            "the header does not say LIVE:\n{text}"
        );
        // a box that has not said (a console started by hand, or an init that could not write the marker) invents
        // nothing: the word is absent rather than guessed
        f.mode = Mode::Unknown;
        let text = render_snapshot(&app(f), 240, 67);
        let head = text.lines().next().unwrap_or("");
        assert!(
            !head.contains("LIVE") && !head.contains("INSTALLED"),
            "a mode was invented:\n{text}"
        );
    }

    #[test]
    fn the_header_drops_a_fact_rather_than_letting_it_be_clipped() {
        // The mode cost the header five cells, and at 80 columns it is the temperature and then Secure Boot that give
        // way - both are readable elsewhere (`1bit-board status`, `mokutil --sb-state`) and neither may be clipped to
        // `SoC 4` / `Secure Boot ena`. The mode, the release, the host and the address never go.
        let f = c16_board();
        // 65 is this fixture's mandatory prefix (mode + release + host + address + uptime): below that the line is
        // longer than the window and ratatui clips its tail, exactly as it already did before the mode was added
        for w in 65..=256usize {
            let text = span_text(&header(&f, w));
            assert!(
                text.chars().count() <= w,
                "width {w}: the header is {} cells: {text:?}",
                text.chars().count()
            );
            assert!(
                text.contains("LIVE"),
                "width {w}: the mode was dropped: {text:?}"
            );
            assert!(
                text.contains("rc-2026-11-03-c16"),
                "width {w}: the release was dropped: {text:?}"
            );
            assert!(
                text.contains("192.168.50.110"),
                "width {w}: the address was dropped: {text:?}"
            );
        }
        // the appliance's own tty1 has room for everything, so nothing is lost where the owner looks
        let full = span_text(&header(&f, 240));
        assert!(
            full.contains("Secure Boot enabled") && full.contains("SoC 43C"),
            "240 columns lost a fact: {full}"
        );
    }

    #[test]
    fn the_first_frame_is_the_menu_before_any_probe() {
        // Cycle 31, measured on the take stick `666c8ab5`: two boots of the same bytes, 76 PASS / 1 FAIL each, and
        // the failing row was `U2 tty1 … tty1-screen-empty` on one and `U2 menu …` on the other - a scrape that ar-
        // rived before the first frame landed. The first `App` used to be built from `Facts::collect()`, which waits
        // on lemond's three HTTP gets, `llama-server --list-devices` and `ps`/`netstat`; now it is `Facts::idle()`
        // (file reads plus one `ip`). This renders exactly that frame, at both sizes, and holds the two gate rows:
        // non-empty tty1 (`U2 tty1`), and both labels `U2 menu` greps.
        for (w, h) in [(80u16, 25u16), (240, 67)] {
            let text = render_snapshot(&app(Facts::idle()), w, h);
            assert!(
                text.chars().any(|c| !c.is_whitespace()),
                "{w}x{h}: the first frame is blank, which is the row that reds"
            );
            assert!(text.contains("1bit OS"), "{w}x{h}: no header\n{text}");
            for want in ["Show the API key", "Reboot / power off"] {
                assert!(
                    text.contains(want),
                    "{w}x{h}: `{want}` is not on the first frame\n{text}"
                );
            }
            let rows = text.lines().filter(|l| {
                let t = l.trim_start();
                t.len() > 2 && t.as_bytes()[0].is_ascii_digit() && t.as_bytes()[1] == b' '
            });
            assert_eq!(rows.count(), 9, "{w}x{h}: not the nine actions\n{text}");
            // ...and it must not state a fact it has not asked for. "LEMONADE DOWN" / "DATA not mounted"
            // about a box that is still starting is the same defect as `web UI http://-:13305`.
            assert!(
                text.contains("STARTING"),
                "{w}x{h}: the first frame does not say it is starting\n{text}"
            );
            for lie in [
                "DOWN",
                "DATA not mounted",
                "amdxdna absent",
                "no HRX device",
            ] {
                assert!(
                    !text.contains(lie),
                    "{w}x{h}: the first frame asserts `{lie}` before probing\n{text}"
                );
            }
        }
    }

    #[test]
    fn main_paints_an_idle_frame_before_the_first_collect() {
        // The frame test above holds what the screen says; this holds that `main` actually paints it
        // FIRST. No behavioural test can see this without a boot, and the regression is silent: a
        // `Facts::collect()` here costs seconds at boot (engine#1318, cycle 31's `tty1-screen-empty`)
        // and every other test still passes, because the frame is correct - just late.
        let src = include_str!("main.rs");
        assert!(
            src.contains("let mut app = App::new(Facts::idle());"),
            "main() no longer paints the first frame from Facts::idle(): the first screen now waits on the probes"
        );
        let first_app = src.split("let mut app = App::new(").nth(1).unwrap_or("");
        assert!(
            first_app.starts_with("Facts::idle())"),
            "the first App is built from something other than Facts::idle(): {first_app:.40}"
        );
    }

    #[test]
    fn renders_at_the_two_console_sizes() {
        let want_240 = include_str!("../tests/snapshots/240x67.txt");
        let want_80 = include_str!("../tests/snapshots/80x25.txt");
        let got_240 = render_snapshot(&app(c16_board()), 240, 67);
        let got_80 = render_snapshot(&app(c16_board()), 80, 25);
        assert_eq!(got_240, want_240, "\n--- got 240x67 ---\n{got_240}");
        assert_eq!(got_80, want_80, "\n--- got 80x25 ---\n{got_80}");
    }

    #[test]
    fn the_connect_block_says_what_a_blank_key_means() {
        // Cycle 20: blank is the shipped default, so the block has to say that anyone on the network can use the
        // appliance, at BOTH console sizes - the 80x25 render is what a person standing at the box sees.
        let mut f = c16_board();
        f.key = None;
        for (w, h) in [(80u16, 25u16), (240, 67)] {
            let text = render_snapshot(&app(f.clone()), w, h);
            let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                flat.contains("key none - anyone here can use it; press 1 to require one"),
                "{w}x{h}: {text}"
            );
            assert!(
                !flat.contains("API key: press 1"),
                "{w}x{h}: keyed wording in blank mode"
            );
            // the extra row must not push the nine actions off the screen - that is the cycle-14 failure in a
            // different renderer, and the TUI is the surface this image actually puts on tty1
            for action in ACTIONS {
                assert!(
                    flat.contains(action),
                    "{w}x{h}: action {action:?} not drawn"
                );
            }
        }
        // and keyed mode still carries the strings vm-check's U2 menu connect row greps for
        let text = render_snapshot(&app(c16_board()), 80, 25)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            text.contains("api http://192.168.50.110:13305/api/v1 (API key: press 1)"),
            "{text}"
        );
    }

    #[test]
    fn the_footer_tells_the_truth_about_q_in_both_places() {
        // over ssh `q` is what it is in every full-screen program; on tty1 the console must not promise a way to
        // a bare prompt. The line is the only place the console says which of the two you are looking at.
        let mut ssh = app(c16_board());
        ssh.ssh = true;
        let text = render_snapshot(&ssh, 80, 25);
        assert!(
            text.trim_end()
                .lines()
                .last()
                .unwrap_or("")
                .contains("q quits"),
            "{text}"
        );
        let text = render_snapshot(&app(c16_board()), 80, 25);
        assert!(
            text.trim_end()
                .lines()
                .last()
                .unwrap_or("")
                .contains("q stays"),
            "{text}"
        );
    }

    #[test]
    fn the_mdns_name_is_on_screen_at_both_sizes() {
        // the owner, 2026-11-04: "show the mDNS name too". pi-os3 makes the box announce it; the c16 sample this
        // fixture records shipped no avahi, so the row must say that rather than promise a name that is not there.
        for (w, h) in [(240u16, 67u16), (80, 25)] {
            let text = render_snapshot(&app(c16_board()), w, h);
            assert!(
                text.contains("name http://1bit-os.local:13305"),
                "{w}x{h}: no mDNS name:\n{text}"
            );
            assert!(
                text.contains("avahi not running"),
                "{w}x{h}: a dead name is promised silently:\n{text}"
            );
            let mut up = c16_board();
            up.mdns = true;
            let text = render_snapshot(&app(up), w, h);
            assert!(text.contains("(mDNS)"), "{w}x{h}:\n{text}");
            assert!(!text.contains("avahi not running"), "{w}x{h}:\n{text}");
        }
    }

    #[test]
    fn every_hyperlink_covers_the_url_that_is_drawn_there() {
        // the links are written AFTER the frame, at these very cells, so the only thing that makes them safe is
        // that the text on screen and the link target are the same string. Asserted off the rendered frame.
        for (w, h) in [(240u16, 67u16), (80, 25)] {
            let (text, links) = render_snapshot_with_links(&app(c16_board()), w, h);
            let lines: Vec<&str> = text.lines().collect();
            let mut seen: Vec<String> = Vec::new();
            for l in &links {
                let row = lines[l.y as usize];
                let at: String = row
                    .chars()
                    .skip(l.x as usize)
                    .take(l.text.chars().count())
                    .collect();
                assert_eq!(
                    at, l.text,
                    "{w}x{h}: link at {},{} does not cover its own URL",
                    l.x, l.y
                );
                seen.push(l.text.clone());
            }
            for want in [
                "http://192.168.50.110:13305",
                "http://1bit-os.local:13305",
                "http://192.168.50.110:13305/api/v1",
            ] {
                assert!(
                    seen.iter().any(|s| s == want),
                    "{w}x{h}: {want} is not a link: {seen:?}"
                );
            }
        }
    }

    #[test]
    fn a_url_that_wrapping_split_is_not_a_link() {
        // half a URL as the link text would open the wrong thing (or nothing), so only whole tokens are returned
        assert_eq!(
            urls_in("  web UI http://10.0.0.5:13305", 80),
            vec![(9, "http://10.0.0.5:13305".to_string())]
        );
        assert_eq!(
            urls_in(
                "  api  http://10.0.0.5:13305/api/v1   (API key: press 1)",
                80
            ),
            vec![(7, "http://10.0.0.5:13305/api/v1".to_string())]
        );
        assert_eq!(urls_in("no links here", 80), Vec::new());
        // a token that fills the pane to its last column may be half a URL: no link rather than a broken one
        let row = "  web UI http://10.0.0.5:13305";
        let w = row.chars().count() + 1;
        assert_eq!(
            urls_in(row, w),
            vec![(9, "http://10.0.0.5:13305".to_string())]
        );
        assert_eq!(urls_in(row, w - 1), Vec::new());
    }

    #[test]
    fn the_qr_is_drawn_beside_the_block_only_where_it_fits() {
        // 240x67: 33 columns and 17 rows to the right of the CONNECT rows, on paper (black on white)
        let text = render_snapshot(&app(c16_board()), 240, 67);
        assert!(text.contains('█'), "no QR at 240x67:\n{text}");
        // 80x25: 33 free columns do not exist next to the block, and squeezing the code in would push the nine
        // menu rows or the three log rows off the screen. The URL stays plain text.
        let text = render_snapshot(&app(c16_board()), 80, 25);
        assert!(!text.contains('█'), "the QR took the 80x25 screen:\n{text}");
    }

    #[test]
    fn the_qr_never_costs_another_block_a_row() {
        // Measured on the shipped mode (blank key: six CONNECT rows, the widest block) at 104x29 - a plain ssh
        // window - the code fit the width test, the layout then had one row too many, and ratatui gave the panel
        // grid's rows to the code: `LEMONADE` was gone and the code's bottom quiet zone painted over the MENU
        // rule (it came back as `MENU ----      -`). The frame is the terminal exactly, so the code is drawn only
        // when every block still gets its rows.
        let mut f = c16_board();
        f.key = None;
        for (w, h) in [
            (101u16, 29u16),
            (104, 29),
            (104, 33),
            (104, 39),
            (110, 34),
            (120, 40),
            (140, 40),
            (240, 67),
        ] {
            let text = render_snapshot(&app(f.clone()), w, h);
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(
                lines.len(),
                h as usize,
                "{w}x{h}: frame is not the terminal"
            );
            // the footer is the last block on the screen, whole: at 40 columns it wraps to two lines
            let foot = lines
                .iter()
                .position(|l| l.contains("1-9 actions"))
                .unwrap_or_else(|| panic!("{w}x{h}: no footer:\n{text}"));
            let tail = lines[foot..].join(" ");
            assert!(
                tail.contains("q stays") || tail.contains("q quits"),
                "{w}x{h}: the footer is cut ({tail:?}):\n{text}"
            );
            // a rule is its label and then nothing but dashes: a hole in one means the code was drawn over it
            for label in ["CONNECT", "MENU"] {
                let row = lines
                    .iter()
                    .find(|l| l.trim_start().starts_with(label))
                    .unwrap_or_else(|| panic!("{w}x{h}: no {label} rule:\n{text}"));
                let tail = row.trim_start().split_once(label).unwrap().1;
                assert!(
                    tail.trim().chars().all(|c| c == '-'),
                    "{w}x{h}: the {label} rule has a hole in it ({row:?}):\n{text}"
                );
            }
            // the panel grid is the block the old row budget used to eat: whenever the code IS on the frame it
            // did not cost another block its rows. (A frame too short for the panels drops them by design - the
            // nine actions win - so the assertion is conditional on the code, which is what this test is about.)
            if text.contains('█') {
                assert!(
                    text.contains("LEMONADE"),
                    "{w}x{h}: the status panels lost their rows to the code:\n{text}"
                );
            }
            // and the nine actions are whole, at every size
            for n in 1..=9 {
                assert!(
                    text.contains(&format!("{n} ")),
                    "{w}x{h}: action {n} missing"
                );
            }
        }
        // the code still IS drawn where it fits - the fix must not have deleted it
        assert!(render_snapshot(&app(c16_board()), 240, 67).contains('█'));
    }

    #[test]
    fn renders_a_booting_box_and_a_modal_without_panicking() {
        let mut f = c16_board();
        f.ips.clear();
        f.lemond_listening = false;
        f.logs.clear();
        f.data = None;
        f.gpu = Gpu::default();
        f.models = Models::default();
        let mut a = app(f);
        for (w, h) in [(240u16, 67u16), (80, 25), (40, 12), (240, 8)] {
            let out = render_snapshot(&a, w, h);
            assert!(!out.is_empty(), "{w}x{h}");
        }
        a.modal = Some(Modal::Power);
        let _ = render_snapshot(&a, 80, 25);
        a.modal = Some(Modal::Info {
            title: "API key".into(),
            body: vec!["a".repeat(300)],
        });
        let _ = render_snapshot(&a, 80, 25);
    }

    #[test]
    fn a_short_window_keeps_the_nine_actions_and_the_footer_whole() {
        // Measured before this: a window shorter than the full screen clipped the MENU itself. 40x15 drew actions
        // 1-5 with a footer that still promised "1-9 actions" and an empty `LOGS newest` title; 80x18 and 100x20
        // lost `8 Logs`/`9 Reboot / power off` - the last one is a gate string (`vm-check.sh` greps tty1 for it).
        // Now the blocks above the menu give way whole, in a fixed order (status facts, then the CONNECT block's
        // tail rows, then the log pane), and nothing is drawn half.
        let mut f = c16_board();
        f.key = None;
        for (w, h) in [
            (40u16, 15u16),
            (60, 15),
            (80, 18),
            (80, 20),
            (80, 24),
            (100, 20),
            (120, 16),
            (160, 13),
            (240, 12),
            (240, 67),
        ] {
            let text = render_snapshot(&app(f.clone()), w, h);
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(
                lines.len(),
                h as usize,
                "{w}x{h}: frame is not the terminal"
            );
            for n in 1..=9 {
                assert!(
                    text.contains(&format!("{n} ")),
                    "{w}x{h}: action {n} is off the screen:\n{text}"
                );
            }
            // the footer is the last block on the screen, whole: at 40 columns it wraps to two lines
            let foot = lines
                .iter()
                .position(|l| l.contains("1-9 actions"))
                .unwrap_or_else(|| panic!("{w}x{h}: no footer:\n{text}"));
            let tail = lines[foot..].join(" ");
            assert!(
                tail.contains("q stays") || tail.contains("q quits"),
                "{w}x{h}: the footer is cut ({tail:?}):\n{text}"
            );
            // no half blocks in the pane that carries the log: a log title always has a log row under it
            for (i, l) in lines.iter().enumerate() {
                let t = l.trim_start();
                if t.starts_with("LOGS") {
                    let tail = &lines[i + 1..lines.len() - 1];
                    assert!(
                        tail.iter().any(|n| !n.trim().is_empty()),
                        "{w}x{h}: a log title with no log row under it:\n{text}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_modal_hint_is_never_the_row_that_goes() {
        // A frame too short for the whole dialog used to clip its END - the hint, i.e. the only row that says how to
        // answer. Measured on the shipped binary in ptys: at 40x11, 40x10, 30x12 and 24x12 the blank-key confirm
        // (`API key` / `Require one from now on?` / `y mints one and restarts Lemonade, n cancels`) drew no hint at
        // all, and at 40x12 it drew `y mints one and restarts`. For the reboot confirm that is
        // `y reboots the box, n cancels` off the screen while `y` still reboots.
        let dialogs = [
            (
                Modal::Confirm {
                    title: "Reboot the box?".into(),
                    question: "The console ends and the box comes back on its own.".into(),
                    yes: Yes::Reboot,
                },
                "y reboots the box, n cancels",
            ),
            (
                Modal::Confirm {
                    title: "API key".into(),
                    question: "No API key (blank mode): this appliance is stock Lemonade, so anyone on this network can use it.\nRequire one from now on? A key is minted into /etc/1bit/lemond.conf and Lemonade restarts.".into(),
                    yes: Yes::MintKey,
                },
                "y mints one and restarts Lemonade, n cancels",
            ),
            (
                Modal::Power,
                "r reboots, p powers off, any other key cancels",
            ),
            (
                Modal::Info {
                    title: "Lemonade API key".into(),
                    body: vec!["API key      0f3a".into(), String::new(), "as a header  Authorization: Bearer 0f3a".into()],
                },
                "any key returns to the dashboard",
            ),
        ];
        for (modal, hint) in dialogs {
            for (w, h) in [
                (24u16, 12u16),
                (30, 12),
                (40, 10),
                (40, 11),
                (40, 12),
                (60, 12),
                (80, 25),
                (240, 67),
            ] {
                let mut a = app(c16_board());
                a.modal = Some(modal.clone());
                let text = render_snapshot(&a, w, h);
                let lines: Vec<&str> = text.lines().collect();
                assert_eq!(
                    lines.len(),
                    h as usize,
                    "{w}x{h}: frame is not the terminal:\n{text}"
                );
                // the box's borders sit between the wrapped words: drop them before looking for a phrase
                let flat = text
                    .replace(
                        [
                            '\u{2502}', '\u{250c}', '\u{2510}', '\u{2514}', '\u{2518}', '\u{2500}',
                        ],
                        " ",
                    )
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                assert!(
                    flat.contains(hint),
                    "{w}x{h}: the dialog does not say how to answer it ({hint:?}):\n{text}"
                );
            }
        }
    }

    #[test]
    fn the_menu_rule_gives_way_before_an_action_does() {
        // Measured on the shipped binary before this: 80x10, 56x11, 40x12 and 30x16 are all exactly one row short
        // of the MENU rule plus the nine actions plus the footer, and the rule was charged as fixed while ratatui
        // clipped the END of the menu - `9 Reboot / power off`, a gate string - with the footer still promising
        // "1-9 actions". These are ordinary pane sizes (a tmux split, an ssh window on a phone). The rule is
        // decoration; the actions are the screen, so the rule goes first.
        let mut f = c16_board();
        f.key = None;
        for (w, h) in [(80u16, 10u16), (56, 11), (40, 12), (30, 16)] {
            let text = render_snapshot(&app(f.clone()), w, h);
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(
                lines.len(),
                h as usize,
                "{w}x{h}: frame is not the terminal"
            );
            for n in 1..=9 {
                assert!(
                    text.contains(&format!("{n} ")),
                    "{w}x{h}: action {n} is off the screen:\n{text}"
                );
            }
            assert!(
                text.contains("stays") || text.contains("quits"),
                "{w}x{h}: the footer was paid for the rule ({text})"
            );
            assert!(
                !text.contains("MENU"),
                "{w}x{h}: the rule is still on a frame with no row for it:\n{text}"
            );
        }
        // ...and where a row IS free, the rule is still drawn: the fix must not have deleted it
        for (w, h) in [
            (80u16, 11u16),
            (56, 12),
            (40, 13),
            (30, 17),
            (80, 25),
            (240, 67),
        ] {
            let text = render_snapshot(&app(f.clone()), w, h);
            assert!(
                text.contains("MENU"),
                "{w}x{h}: the MENU rule is missing where it fits:\n{text}"
            );
        }
    }

    #[test]
    fn a_message_gives_way_before_an_action_does() {
        // Issue #1015, reproduced on the shipped binary first: a 40-column window whose pane is 12 rows is exactly
        // the nine actions plus the two rows the legend wraps to. One keypress added the message's own two rows,
        // the frame was over-subscribed, and ratatui clipped the END of the MENU - `9 Reboot / power off`, a
        // `vm-check.sh` / `menugate` string - for the message's 12 s, with the footer still promising
        // `1-9 actions`. The legend is words, the actions are rows: the legend gives way, whole, for the message.
        const SAID: &str = "this console never exits: 7 opens a shell, 9 reboots";
        let mut a = app(c16_board());
        a.say(SAID);
        for (w, h) in [
            (40u16, 12u16), // the reproduced case: the message fits only once the legend is gone
            (40, 13),
            (56, 12),
            (80, 10), // nine actions leave one row, and that row is the message
            (80, 11),
            (100, 13),
            (30, 16),
            (80, 25),
            (240, 67),
        ] {
            let text = render_snapshot(&a, w, h);
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(
                lines.len(),
                h as usize,
                "{w}x{h}: frame is not the terminal:\n{text}"
            );
            for n in 1..=9 {
                assert!(
                    text.contains(&format!("{n} ")),
                    "{w}x{h}: action {n} is off the screen for the message:\n{text}"
                );
            }
            // the message is never CUT: it is drawn whole, or not at all
            let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                flat.contains(SAID),
                "{w}x{h}: the message was cut instead of the legend:\n{text}"
            );
        }
        // where the legend still has the rows it is untouched: the fix is not a permanent diet (80x25, 240x67 are
        // the sizes the gates read)
        for (w, h) in [(80u16, 25u16), (240, 67), (100, 13)] {
            let text = render_snapshot(&a, w, h);
            assert!(
                text.contains("1-9 actions") && text.contains("q stays"),
                "{w}x{h}: the whole legend fits here and was trimmed:\n{text}"
            );
        }
        // ...and where not even the message fits beside the nine actions (40x11: ten menu rows, one footer row,
        // the message wraps to two), the message is the thing that goes - the actions and the gate string stay
        let text = render_snapshot(&a, 40, 11);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 11, "40x11: frame is not the terminal");
        for n in 1..=9 {
            assert!(
                text.contains(&format!("{n} ")),
                "40x11: action {n} is off the screen:\n{text}"
            );
        }
        assert!(
            text.contains("1-9 actions"),
            "40x11: no gate string:\n{text}"
        );
        assert!(
            !text.contains("this console never exits"),
            "40x11: the message is on a frame with no row for it:\n{text}"
        );
    }
}
