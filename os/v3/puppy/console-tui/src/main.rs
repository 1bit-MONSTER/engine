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

//! 1bit-console-tui - what /etc/inittab respawns on tty1 of the 1bit OS appliance, instead of the `dialog` menu.
//!
//! The owner, on the cycle-18 board boot: "when you first boot the headless there is no data to display on how to
//! connect, just the menu". So the first block on this screen is CONNECT: the IP, the web UI, the OpenAI-compatible
//! base, the key, ssh and Lemonade Mobile, kept current every two seconds until the box answers - and the same
//! facts the `dialog` menu's status block shows, read from the same places (see facts.rs).
//!
//! Three promises this binary keeps, each one paid for by an earlier cycle:
//!   * the console never goes dead. `q` does not exit; Ctrl-C does not exit; a paint failure redraws; if this
//!     process does die, `1bit-console` falls back to the `dialog` menu and /etc/inittab respawns it.
//!   * an action that needs a typed answer is not re-implemented here. Keys 2-7 hand the terminal to
//!     `1bit-console-menu --action N`, i.e. the exact code the QEMU rows (`U2 menu actions`, `--selftest`) already
//!     exercise, so the TUI cannot drift from the menu and the gates keep covering both.
//!   * nothing is clipped and no probe runs in a bare environment: every probe that asks about the GPU is spawned
//!     with /etc/1bit/libhsa.env sourced, because an inittab respawn inherits neither LD_LIBRARY_PATH nor the
//!     no-SVM trio (the cycle-13 board boot: "GPU no HRX device" while chat-hrx served 303 tok/s).
//!
//! Usage:
//!   1bit-console-tui                     the dashboard on this terminal (what tty1 gets)
//!   1bit-console-tui --status            the same facts as plain text, no terminal needed, exit 0
//!   1bit-console-tui --snapshot WxH      render one frame off-screen (WxH or "W x H"), print it, exit 0
//!   1bit-console-tui --snapshot WxH --fixture   the same, from the recorded c16 board sample
//!   1bit-console-tui --version           print the version
//!
//! Exit: 0 clean (1bit-console then shows the dialog menu), 1 the screen could not be set up.

mod facts;
mod qr;
mod ui;

use std::io::stdout;
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use facts::Facts;
use ui::{App, Modal, Yes};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => run(),
        Some("--help" | "-h") => {
            for l in [
                "1bit-console-tui - the 1bit OS appliance console (tty1)",
                "",
                "  1bit-console-tui                     dashboard on this terminal",
                "  1bit-console-tui --status            the facts as plain text",
                "  1bit-console-tui --snapshot WxH      render one frame and print it",
                "  1bit-console-tui --version           print the version",
                "",
                "keys: 1-9 actions, arrows/PgUp/PgDn scroll the log, f follow, r refresh, 8 full-screen logs,",
                "      q quits over ssh and stays on tty1 (the console never drops to a bare prompt).",
            ] {
                println!("{l}");
            }
            ExitCode::SUCCESS
        }
        Some("--version") => {
            println!("1bit-console-tui {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("--status") => {
            for l in ui::status_text(&Facts::collect()) {
                println!("{l}");
            }
            ExitCode::SUCCESS
        }
        Some("--snapshot") => {
            let size = args.get(1).map(String::as_str).unwrap_or("240x67");
            let Some((w, h)) = parse_size(size) else {
                eprintln!("--snapshot wants WxH, for example 240x67 (got '{size}')");
                return ExitCode::FAILURE;
            };
            let f = if args.iter().any(|a| a == "--fixture") {
                facts::c16_board()
            } else {
                Facts::collect()
            };
            print!("{}", ui::render_snapshot(&App::new(f), w, h));
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("1bit-console-tui: unknown argument '{other}' (try --help)");
            ExitCode::FAILURE
        }
    }
}

/// "240x67", "240 x 67", "240X67".
fn parse_size(s: &str) -> Option<(u16, u16)> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let (w, h) = cleaned.split_once(['x', 'X'])?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

fn suspend() {
    let _ = disable_raw_mode();
    let _ = execute!(stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
}

/// Was this process started from an ssh session? `sshd` sets both of these, and neither exists on tty1, which is
/// what makes them the honest question (rather than "is there a terminal", which tty1 answers yes to).
fn ssh_session(conn: Option<&str>, tty: Option<&str>) -> bool {
    conn.is_some_and(|v| !v.is_empty()) || tty.is_some_and(|v| !v.is_empty())
}

fn over_ssh() -> bool {
    ssh_session(
        std::env::var("SSH_CONNECTION").ok().as_deref(),
        std::env::var("SSH_TTY").ok().as_deref(),
    )
}

/// OSC 8 hyperlinks are a terminal-emulator feature. The VGA console (TERM=linux) has no clickable anything and
/// would print the escapes; JetKVM and a serial terminal are the same case.
fn hyperlinks_wanted(term: Option<&str>) -> bool {
    matches!(term, Some(t) if !t.is_empty() && t != "linux" && t != "dumb")
}

/// Underline what was drawn and wrap it in OSC 8 with its own text as the target. Written AFTER the frame: ratatui
/// has no hyperlink style, and the two rules that make this safe are that the text is identical to the cells it
/// overwrites and that the link is re-stated on every frame (the next paint would otherwise drop it).
///
/// The style is not `\x1b[4m` alone: an SGR with no colour would inherit whatever the last drawn cell left current,
/// so the link is set to the CONNECT rows' own white+bold through the same crossterm commands ratatui uses, and
/// reset after - the cell keeps the pixel it had, plus the underline that says it can be clicked.
fn emit_links(links: &[ui::Link]) {
    if links.is_empty() {
        return;
    }
    use crossterm::cursor::MoveTo;
    use crossterm::style::{Attribute, Color, Print, SetAttribute, SetForegroundColor};
    let mut out = stdout();
    for l in links {
        let _ = execute!(
            out,
            MoveTo(l.x, l.y),
            SetForegroundColor(Color::White),
            SetAttribute(Attribute::Bold),
            SetAttribute(Attribute::Underlined),
            Print(format!("\x1b]8;;{u}\x1b\\{u}\x1b]8;;\x1b\\", u = l.text)),
            SetAttribute(Attribute::Reset),
        );
    }
}

fn resume(t: &mut Terminal<CrosstermBackend<std::io::Stdout>>) {
    let _ = execute!(stdout(), EnterAlternateScreen);
    let _ = enable_raw_mode();
    let _ = t.clear();
}

fn run() -> ExitCode {
    // a panic must leave a usable terminal behind: 1bit-console reads a non-zero exit and shows the dialog menu
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        suspend();
        hook(p);
    }));

    let mut term = match Terminal::new(CrosstermBackend::new(stdout())) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("1bit-console-tui: this terminal cannot be driven ({e}); the dialog menu is on tty1");
            return ExitCode::FAILURE;
        }
    };
    let _ = execute!(stdout(), EnterAlternateScreen, crossterm::cursor::Hide);
    if enable_raw_mode().is_err() {
        let _ = execute!(stdout(), LeaveAlternateScreen);
        eprintln!("1bit-console-tui: raw mode refused; the dialog menu is on tty1");
        return ExitCode::FAILURE;
    }

    // the facts are collected off the drawing thread: `llama-server --list-devices` takes seconds, and a screen
    // that stops answering keys is not a console. 2 s while the box is still coming up, 3 s after, and `r` asks
    // for one now.
    let (tx, rx) = mpsc::channel::<Facts>();
    let (req_tx, req_rx) = mpsc::channel::<()>();
    std::thread::spawn(move || loop {
        let f = Facts::collect();
        let wait = if f.reachable() { 3 } else { 2 };
        if tx.send(f).is_err() {
            break;
        }
        match req_rx.recv_timeout(Duration::from_secs(wait)) {
            Ok(()) => continue,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    });

    let mut app = App::new(Facts::collect());
    // asked once: sshd sets SSH_CONNECTION/SSH_TTY for the whole session, tty1 respawn sets neither
    app.ssh = over_ssh();
    let mut last_spin = Instant::now();
    let rc = loop {
        while let Ok(f) = rx.try_recv() {
            let keep = app.log_scroll;
            app.facts = f;
            app.log_scroll = keep.min(app.facts.logs.len().saturating_sub(1));
        }
        if let Some((_, at)) = &app.message {
            if at.elapsed() > Duration::from_secs(12) {
                app.message = None;
            }
        }
        if !app.facts.reachable() && last_spin.elapsed() >= Duration::from_millis(250) {
            app.spin = app.spin.wrapping_add(1);
            last_spin = Instant::now();
        }
        let mut links = Vec::new();
        if term.draw(|f| ui::draw(f, &app, &mut links)).is_err() {
            break 1;
        }
        // over ssh the screen is a terminal emulator, so the CONNECT URLs can be clickable (OSC 8). On the VGA
        // console (TERM=linux) the same bytes would print as garbage - or worse, as text nobody can click.
        if hyperlinks_wanted(std::env::var("TERM").ok().as_deref()) {
            emit_links(&links);
        }

        if !event::poll(Duration::from_millis(250)).unwrap_or(false) {
            continue;
        }
        let Ok(ev) = event::read() else { continue };
        let key = match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => k,
            _ => continue,
        };
        if let Some(rc) = handle(&mut app, &mut term, key, &req_tx) {
            break rc;
        }
    };

    suspend();
    ExitCode::from(rc)
}

/// Returns Some(exit code) when the screen should end. Everything else returns None and redraws.
fn handle(
    app: &mut App,
    term: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    key: KeyEvent,
    req: &mpsc::Sender<()>,
) -> Option<u8> {
    // a modal owns the keyboard
    if let Some(m) = app.modal.clone() {
        match m {
            Modal::Info { .. } => app.modal = None,
            Modal::Busy { .. } => {}
            Modal::Power => {
                app.modal = None;
                match key.code {
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        app.modal = Some(Modal::Confirm {
                            title: "Reboot".into(),
                            question: "Reboot the appliance now?".into(),
                            yes: Yes::Reboot,
                        })
                    }
                    KeyCode::Char('p') | KeyCode::Char('P') => {
                        app.modal = Some(Modal::Confirm {
                            title: "Power off".into(),
                            question: "Power the appliance off now?".into(),
                            yes: Yes::Poweroff,
                        })
                    }
                    _ => app.say("nothing was changed"),
                }
            }
            Modal::Confirm { yes, .. } => {
                app.modal = None;
                let confirmed = matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y'));
                if !confirmed {
                    app.say("cancelled");
                    return None;
                }
                match yes {
                    Yes::ShowKey => {
                        let k = app.facts.key.clone().unwrap_or_default();
                        let base = format!("http://{}:13305", app.facts.ip().unwrap_or("-"));
                        app.modal = Some(Modal::Info {
                            title: "Lemonade API key".into(),
                            body: vec![
                                format!("API key      {k}"),
                                String::new(),
                                format!("as a header  Authorization: Bearer {k}"),
                                format!("try it       curl -H \"Authorization: Bearer {k}\" {base}/api/v1/models"),
                                String::new(),
                                "Anyone who can read this screen can use the appliance.".into(),
                            ],
                        });
                    }
                    Yes::MintKey => handoff(app, term, 1),
                    Yes::Desktop => handoff(app, term, 6),
                    Yes::Reboot | Yes::Poweroff => {
                        let what = if yes == Yes::Reboot {
                            "reboot"
                        } else {
                            "poweroff"
                        };
                        suspend();
                        let _ = std::process::Command::new("sync").status();
                        let _ = std::process::Command::new(what).status();
                        resume(term);
                        app.say(format!("{what} returned - the box did not go down"));
                    }
                }
            }
        }
        return None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
    {
        app.say("this console never exits: 7 opens a shell, 9 reboots");
        return None;
    }

    match key.code {
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
            if app.full_logs {
                app.full_logs = false;
            } else if app.ssh {
                // the owner, 2026-11-04: the TUI is used through ryzen over ssh, where q is the normal way out
                // of a full-screen program. tty1 keeps the old promise: q never drops that console to a prompt.
                return Some(0);
            } else {
                app.say("q stays: no bare prompt here (7 shell, 9 reboot)");
            }
        }
        KeyCode::Char('8') | KeyCode::Char('l') => {
            app.full_logs = !app.full_logs;
            app.log_scroll = 0;
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            let _ = req.send(());
            app.say("refreshing");
        }
        KeyCode::Char('f') | KeyCode::Char('F') => {
            app.log_scroll = 0;
            app.message = None;
        }
        KeyCode::Up => app.log_scroll = (app.log_scroll + 1).min(app.facts.logs.len().saturating_sub(1)),
        KeyCode::Down => app.log_scroll = app.log_scroll.saturating_sub(1),
        KeyCode::PageUp => {
            app.log_scroll = (app.log_scroll + 10).min(app.facts.logs.len().saturating_sub(1))
        }
        KeyCode::PageDown => app.log_scroll = app.log_scroll.saturating_sub(10),
        KeyCode::Home => app.log_scroll = app.facts.logs.len().saturating_sub(1),
        KeyCode::End => app.log_scroll = 0,
        KeyCode::Char(c @ '1'..='9') => match c {
            '1' => {
                if app.facts.key.is_some() {
                    app.modal = Some(Modal::Confirm {
                        title: "API key".into(),
                        question: "Show the Lemonade API key on this screen?\nIt is a bearer token: anyone who can\nread this screen can use the appliance."
                            .into(),
                        yes: Yes::ShowKey,
                    })
                } else {
                    // Cycle 20, owner's "blank key": no key is the SHIPPED state now, not a first-boot accident, so
                    // this asks whether to require one instead of sending the reader to Lemonade's log.
                    app.modal = Some(Modal::Confirm {
                        title: "API key".into(),
                        question: "No API key (blank mode): this appliance is stock\nLemonade, so anyone on this network can use it.\n\nRequire one from now on? A key is minted into\n/etc/1bit/lemond.conf and Lemonade restarts."
                            .into(),
                        yes: Yes::MintKey,
                    })
                }
            }
            '6' => app.modal = Some(Modal::Confirm {
                title: "Desktop".into(),
                question: "Start the desktop (JWM) for this session?\n\nThe boot menu's '1bit OS (desktop, JWM)' entry is the\nproven path. From here the session is logged to /data/logs\nand this console keeps running underneath."
                    .into(),
                yes: Yes::Desktop,
            }),
            '9' => app.modal = Some(Modal::Power),
            '8' => {
                app.full_logs = true;
                app.log_scroll = 0;
            }
            n => handoff(app, term, n as u8 - b'0'),
        },
        _ => {}
    }
    None
}

/// Keys 2,3,4,5,6,7: the `dialog` menu's own item, in the foreground, on this terminal. One code path for the
/// action and for the gate that re-checks it.
fn handoff(app: &mut App, term: &mut Terminal<CrosstermBackend<std::io::Stdout>>, n: u8) {
    if !std::path::Path::new(facts::MENU).exists() {
        app.say(format!(
            "no {} in this image - key {n} needs it",
            facts::MENU
        ));
        return;
    }
    app.modal = Some(Modal::Busy {
        title: format!("action {n}"),
    });
    let mut links = Vec::new();
    let _ = term.draw(|f| ui::draw(f, &*app, &mut links));
    suspend();
    let status = std::process::Command::new(facts::MENU)
        .arg("--action")
        .arg(n.to_string())
        .status();
    resume(term);
    app.modal = None;
    app.say(match status {
        Ok(s) => format!("action {n} finished ({s})"),
        Err(e) => format!("action {n} could not run: {e}"),
    });
}

#[cfg(test)]
mod tests {
    use super::{hyperlinks_wanted, parse_size, ssh_session};

    #[test]
    fn q_is_the_way_out_only_where_a_prompt_is_harmless() {
        // sshd sets both; a tty1 respawn sets neither, and that console must never fall back to a bare prompt
        assert!(ssh_session(Some("10.0.0.2 51234 10.0.0.5 22"), None));
        assert!(ssh_session(None, Some("/dev/pts/3")));
        assert!(!ssh_session(None, None));
        assert!(!ssh_session(Some(""), Some("")));
    }

    #[test]
    fn hyperlinks_are_for_terminal_emulators_not_the_vga_console() {
        for term in [
            "xterm-256color",
            "screen-256color",
            "tmux-256color",
            "alacritty",
            "kitty",
        ] {
            assert!(hyperlinks_wanted(Some(term)), "{term}");
        }
        // TERM=linux is the appliance's own VGA console and a serial line: the escapes would be printed, not used
        for term in ["linux", "dumb", ""] {
            assert!(!hyperlinks_wanted(Some(term)), "{term}");
        }
        assert!(!hyperlinks_wanted(None));
    }

    #[test]
    fn snapshot_sizes() {
        assert_eq!(parse_size("240x67"), Some((240, 67)));
        assert_eq!(parse_size(" 80 x 25 "), Some((80, 25)));
        assert_eq!(parse_size("80X25"), Some((80, 25)));
        assert_eq!(parse_size("big"), None);
    }
}
