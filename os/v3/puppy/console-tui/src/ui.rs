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

//! What the screen looks like. Two layouts, chosen by width, over ONE set of facts:
//!
//!   * 240x67 (the appliance's tty1 on a monitor) - the CONNECT block, then six panels in a 3x2 grid, then the
//!     log pane, then the nine actions.
//!   * 80x25 (a serial/VGA default console, and what the QEMU rows read) - the same block, the same panels one
//!     line each, a shorter log pane, and the SAME action labels: the c15 acceptance row looks for
//!     "Show the API key" and "Reboot / power off" on tty1, so they are never abbreviated.
//!
//! Colours are the 16-colour VT palette by name only (no RGB: a bare Linux console has no truecolour) and no
//! glyph outside what every console font has - titles are ASCII rules, so a VGA text mode with a plain code page
//! still draws the screen. Nothing is clipped: every string goes through `wrap_words` before it reaches a widget,
//! because the c16 board showed "firmware prese".

use crate::facts::{human_bytes, human_uptime, wrap_words, Facts, SecureBoot};
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

fn header(f: &Facts) -> Line<'static> {
    let ip = f.ip().unwrap_or("no address yet");
    let sb = match f.secure_boot {
        SecureBoot::Enabled => ("Secure Boot enabled", Some(Color::Green)),
        SecureBoot::Disabled => ("Secure Boot disabled", Some(Color::Yellow)),
        SecureBoot::Unknown => ("Secure Boot unknown", Some(Color::DarkGray)),
    };
    let mut spans = vec![
        Span::styled(
            format!("1bit OS {} ", f.release),
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
        Span::styled(sb.0, Style::default().fg(sb.1.unwrap_or(Color::Gray))),
    ];
    if let Some(t) = f.temp_c {
        spans.push(Span::styled(
            format!("  SoC {t}C"),
            Style::default().fg(Color::Gray),
        ));
    }
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

    let host = f.ip().unwrap_or("-");
    let rows = [
        format!("web UI http://{host}:13305"),
        format!("api  http://{host}:13305/api/v1   (API key: press 1)"),
        format!("ssh  root@{host}   (your key in /data/ssh/authorized_keys)"),
        format!("app  Lemonade Mobile: server http://{host}:13305 + that key"),
    ];
    for value in rows {
        let indent = IND.len();
        let style = Style::default().fg(Color::Gray);
        let avail = width.saturating_sub(indent).max(8);
        // a row that fits is emitted VERBATIM: its two spaces are part of what the gate greps for (`api  http://`),
        // and wrap_words normalises runs of spaces by design
        let pieces = if value.chars().count() <= avail { vec![value.clone()] } else { wrap_words(&value, avail) };
        for (i, w) in pieces.into_iter().enumerate() {
            if i == 0 {
                out.push(Line::from(vec![
                    Span::styled(IND, Style::default().fg(Color::Cyan)),
                    Span::styled(w, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]));
            } else {
                out.push(Line::from(vec![Span::raw(" ".repeat(indent)), Span::styled(w, style)]));
            }
        }
    }

    if f.ips.is_empty() {
        out.push(Line::from(Span::styled(
            format!("{IND}no address yet - press 3 (Network); DHCP is automatic once a cable is in"),
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
    lemonade.push(format!("http://{}:13305", f.ip().unwrap_or("-")));
    lemonade.push(if f.key.is_some() {
        "API key: set, press 1".to_string()
    } else {
        "API key: none in /etc/1bit/lemond.conf".to_string()
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

/// The nine actions, never abbreviated, packed whole into as few rows as the width allows.
pub fn action_lines(width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, label) in ACTIONS.iter().enumerate() {
        let item = format!("{} {label}", i + 1);
        let add = if cur.is_empty() {
            item.chars().count()
        } else {
            item.chars().count() + 3
        };
        if cur.chars().count() + add > width && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            cur = item;
        } else {
            if !cur.is_empty() {
                cur.push_str("   ");
            }
            cur.push_str(&item);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
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
        return vec![styled(
            " (no Lemonade log and no kernel GPU/NPU lines yet)",
            Some(Color::DarkGray),
            false,
        )];
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
        let block: Vec<(Option<Color>, String)> = wrap_words(rec, width)
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

fn panel_compact(f: &Facts, width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for (title, values) in panels(f) {
        let text = format!("{title}  {}", values.join("  "));
        for (i, l) in wrap_words(&text, width).into_iter().enumerate() {
            if i == 0 {
                out.push(styled(l, Some(Color::Gray), false));
            } else {
                out.push(styled(format!("  {l}"), Some(Color::DarkGray), false));
            }
        }
    }
    out
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

pub fn draw(f: &mut Frame, a: &App) {
    let area = f.area();
    let w = area.width as usize;
    let h = area.height as usize;
    let wide = w >= 100;

    let body: Vec<Line<'static>> = if a.full_logs {
        let inner = h.saturating_sub(2);
        let mut v = vec![log_title(a, w, true)];
        v.extend(log_lines(a, inner, w));
        v
    } else {
        let mut v = vec![header(&a.facts)];
        v.extend(connect_block(&a.facts, crate::facts::token(a.spin), w));
        v.extend(if wide {
            panel_grid(&a.facts, w)
        } else {
            panel_compact(&a.facts, w)
        });
        v
    };

    let actions = action_lines(w);
    let mut footer: Vec<String> = vec![if a.full_logs {
        "q back   arrows scroll   f newest   r refresh   1-9 actions".to_string()
    } else {
        "1-9 actions   arrows scroll   f newest   r refresh   8 logs   q stays on this screen"
            .to_string()
    }];
    if let Some((m, _)) = &a.message {
        footer.push(m.clone());
    }
    let footer: Vec<Line> = footer
        .iter()
        .flat_map(|s| wrap_words(s, w))
        .map(|s| styled(s, Some(Color::DarkGray), false))
        .collect();

    // the log pane gets what is left, which is the only flexible row
    let fixed = body.len() + if a.full_logs { 0 } else { actions.len() } + footer.len();
    let log_h = h.saturating_sub(fixed).max(1);

    let mut constraints: Vec<Constraint> = Vec::new();
    if a.full_logs {
        constraints.push(Constraint::Length(log_h as u16));
        constraints.push(Constraint::Length(footer.len() as u16));
    } else {
        constraints.push(Constraint::Length(body.len() as u16));
        constraints.push(Constraint::Length(log_h as u16));
        constraints.push(Constraint::Length(actions.len() as u16));
        constraints.push(Constraint::Length(footer.len() as u16));
    }
    let rows = Layout::vertical(constraints).split(area);

    if a.full_logs {
        render_lines(f, rows[0], &body);
        render_lines(f, rows[1], &footer);
    } else {
        render_lines(f, rows[0], &body);
        let mut logs = vec![log_title(a, w, false)];
        logs.extend(log_lines(a, log_h.saturating_sub(1), w));
        render_lines(f, rows[1], &logs);
        let acts: Vec<Line> = actions
            .iter()
            .map(|s| styled(s, Some(Color::Cyan), false))
            .collect();
        render_lines(f, rows[2], &acts);
        render_lines(f, rows[3], &footer);
    }

    if let Some(m) = &a.modal {
        let (title, text, hint) = modal_box(m);
        let lines: Vec<String> = std::iter::once(title)
            .chain(text)
            .chain(std::iter::once(hint))
            .collect();
        // the box is the longest line plus its own padding: sizing it to the longest line and then wrapping
        // inside it made every hint wrap for no reason (measured: "any other key / cancels")
        let width = (lines.iter().map(|l| l.chars().count()).max().unwrap_or(10) + 4)
            .min(w.saturating_sub(4))
            .max(27);
        let body: Vec<Line> = lines
            .iter()
            .flat_map(|l| wrap_words(l, width.saturating_sub(4)))
            .map(|l| styled(l, Some(Color::White), false))
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
    let h = header(f);
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
    for l in action_lines(240) {
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
    let backend = TestBackend::new(width, height);
    let mut t = Terminal::new(backend).expect("test terminal");
    t.draw(|f| draw(f, a)).expect("draw");
    buffer_text(t.backend().buffer())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{Gpu, Models};

    use crate::facts::c16_board;

    fn app(f: Facts) -> App {
        App::new(f)
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
    fn the_c18_gate_row_finds_its_strings_on_this_screen() {
        // vm-check.sh's `U2 menu connect` row (pi-os3, cycle 18) greps these fixed strings off tty1: if the TUI
        // is what tty1 gets, that row must keep passing, so the block uses the dialog menu's own wording.
        for (w, h) in [(240u16, 67u16), (80, 25)] {
            let text = render_snapshot(&app(c16_board()), w, h);
            for want in ["CONNECT", "web UI http://", "api  http://", "ssh  root@", "Lemonade Mobile"] {
                assert!(text.contains(want), "{w}x{h}: the U2 menu connect row would go red - {want:?} is missing");
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
    fn the_gates_action_labels_survive_at_every_width() {
        // vm-check's `U2 menu` row looks for these two strings on tty1
        for w in [80usize, 100, 120, 240] {
            let joined = action_lines(w).join(" ");
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
    fn renders_at_the_two_console_sizes() {
        let want_240 = include_str!("../tests/snapshots/240x67.txt");
        let want_80 = include_str!("../tests/snapshots/80x25.txt");
        let got_240 = render_snapshot(&app(c16_board()), 240, 67);
        let got_80 = render_snapshot(&app(c16_board()), 80, 25);
        assert_eq!(got_240, want_240, "\n--- got 240x67 ---\n{got_240}");
        assert_eq!(got_80, want_80, "\n--- got 80x25 ---\n{got_80}");
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
}
