<!-- Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 1bit-console-tui - the appliance console on tty1

What `/etc/inittab` respawns on tty1 of the 1bit OS Puppy appliance, instead of the `dialog` menu. One Rust
binary, ratatui + crossterm, `x86_64-unknown-linux-musl`, no runtime dependency of any kind.

Why it exists, in the owner's words: *"can we make that a tui instead of this style"* (the dialog-box console
menu), then, on the cycle-18 board boot, *"when you first boot the headless there is no data to display on how to
connect, just the menu"*. The first block on the screen is therefore **CONNECT**, and it is composed from live
facts on every draw. **Before DHCP there is no address**, so the block says the one true thing it can -
`web UI: waiting for network`, with the row that says what to do (press 3) - instead of `http://-:13305`, which is
what the first seconds of a boot used to show. The address rows come back with the lease.

**The LEMONADE panel names the engine that is serving** - `engine: llamacpp (gpu)`, `engine: flm (npu)` - composed from
what Lemonade reports (`recipe` in `/api/v1/models`, the device in `/api/v1/health`, falling back to the model's
`recipe_options` backend). **Which model it names is the chat engine, not the first thing loaded**: Lemonade keeps one
model per type in `all_models_loaded[]` (`max_models.llm = 1`), so the entry it types `llm` is the answer, with
`model_loaded` ("most recently accessed", which an embedding request moves) only as the fallback when the type is not
reported - `the_engine_is_the_model_answering_not_the_first_loaded`. No marketing map: if the box says `llamacpp`, the
screen says `llamacpp`. At 80x25 the same sentence rides on the compact `LEMONADE` field, because a fact field of its
own would push the status block to a third row and cost the log pane one (owner decision 2026-11-04 20:19, item 1;
`the_console_says_which_engine_is_serving`). A hybrid value (`engine: ryzenai-llm (gpu npu)`, 83 columns with the rest
of the field) re-wraps the facts block inside its existing two rows at 80x25 and costs the log pane nothing - measured,
not assumed.

## The screen

```
1bit OS rc-2026-11-03-c21 strix  192.168.50.110  up 2h 13m  Secure Boot enabled  SoC 43C
 CONNECT -------------------------------------------------------------------------------
  web UI http://192.168.50.110:13305                                        █▀▀▀▀▀█ ▄ █ ▀█▄▄
  name http://1bit-os.local:13305   (mDNS)                                 █ ███ █ ▄▄▄ █▄ █▄
  api  http://192.168.50.110:13305/api/v1   (API key: press 1)             █ ▀▀▀ █ ▄█ ▄▄ ▀▄▄ █
  ssh  root@192.168.50.110   (your key in /data/ssh/authorized_keys)       ▀▀▀▀▀▀▀ ▀▄█▄▀ █▄█ ▀▀▀
  app  Lemonade Mobile: server http://192.168.50.110:13305 + that key      (the web UI URL as a QR)
 LEMONADE -------------------- GPU ---------------------------- NPU ----------------------
 MODELS ---------------------- FANS / TEMPS ------------------- DATA ---------------------
 MENU -----------------------------------------------------------------------------------
 1 Show the API key
 2 Quick chat test (GPU and NPU)
 ...
 9 Reboot / power off
 LOGS newest ----------------------------------------------------------------------------
```

Top to bottom, in this order and with nothing between (the owner, cycle 20: *"can we place the menu above the
scrolling logs and below the technical information"*): **header, CONNECT, the status facts, the MENU, the
scrolling log**, then one footer line.

Two layouts off ONE set of facts, chosen by width:

| size | what you get |
|---|---|
| 240x67 (tty1 on a monitor) | CONNECT block, six panels in a 3x2 grid, the nine menu rows, then the log pane |
| 80x25 (serial/VGA console, what the QEMU rows read) | the same block, the same six facts on two compact rows, all nine menu rows, and the log pane left four rows - `Show the API key` and `Reboot / power off` are never abbreviated, because `vm-check.sh`'s `U2 menu` row looks for exactly those strings on tty1 |

The MENU is whole and always on screen at both sizes; the log pane gets whatever rows are left (at least three at
80x25). Key **8** is the full-screen, scrollable view.

**On a window shorter than the layout** nothing is drawn half: the nine menu rows are never trimmed, and everything
above them gives way in a fixed order - the status facts first (they are the "technical information"),
then the log pane (a title needs a row under it), then the CONNECT block's own tail, whole rows at a time. So an
ssh window at 80x18 still shows all nine actions and `Reboot / power off`; before this the menu itself was clipped
and the footer went on promising "1-9 actions" (`a_short_window_keeps_the_nine_actions_and_the_footer_whole` sweeps
ten sizes).

**The same rule inside a dialog.** A confirmation box is `title | the question | the hint`, and the hint is the only
row that says how to answer, so a frame too short for the whole box loses the question first, whole rows at a time,
and then the title - never the hint. Before this the box was clamped to the frame and ratatui clipped its end: at
40x11, 30x12 and 24x12 the key-confirm dialog drew no hint at all (`y mints one and restarts Lemonade, n cancels`),
and at 40x12 half of one, while `y` still worked - for the reboot confirm that is `y reboots the box, n cancels`
missing from the screen that is about to reboot
(`the_modal_hint_is_never_the_row_that_goes` sweeps four dialogs at eight sizes).

**When the frame is one row short of the menu itself** - 80x10, 56x11, 40x12, 30x16: a tmux split or an ssh window on
a phone - the `MENU` rule goes before any action does. It is decoration, and it used to be charged first, so ratatui
clipped the end of the menu: `9 Reboot / power off` (a `vm-check.sh` string) was off the screen while the footer still
promised "1-9 actions". Swept over 13 widths x 19 heights, 72 sizes gained that row and none lost the
frame-is-the-terminal invariant (`the_menu_rule_gives_way_before_an_action_does`).

**And when something is said** (a keypress puts a message under the legend for twelve seconds) the words give way to
the rows, in that same order of importance: the legend's tail words first, then the legend line, and only if even the
message alone cannot share the frame with the nine actions does the message go. Before this, in a 40-column window
whose pane is 12 rows, one keypress cost `9 Reboot / power off` for twelve seconds - the footer had grown by the two
rows the message wraps to (`a_message_gives_way_before_an_action_does`). At 80x25 and 240x67, where the gates read the
screen, nothing is trimmed.

**A frame is rebuilt in full once a second, because a tty cannot be made private.** Anything holding this terminal open
can write to it, and ratatui writes only the cells whose *content* changed since its own last frame - correct only while
the screen still holds that frame. So one stray line leaves a wrong screen that **a keypress does not fix**: measured on
the shipped c25 artifact (`2fc6909c…`) in real ptys, three printk-shaped lines were still there at +2.2 s and after a
keypress, at 80x25 and at 240x67 - and when the write ends on the last row the VT scrolls, which is how the log pane's
own lines end up painted across the CONNECT block and the MENU (the cycle-25 screenshot: "`engint: o ebit`"). Who writes:
kernel messages while `console_loglevel` allows them, `1bit-release-check`'s "holding the NPU" line and
`puppy-check.sh`'s console note (both `/dev/tty1` on purpose), and any program a handoff runs. So the frame is rebuilt in
full, from a cleared screen, at most once a second: the same write is gone at **+1.0 s**, and the cost is measured -
`+3.5 KiB/s` at 240x67 (4.5 KiB/s, was 1.0). The `Clear` is one escape and the repaint lands right behind it, i.e. well
inside a display frame by arithmetic - **not measured on the appliance's own VT**, which is the one thing here a pty
cannot answer. This is insurance, not the fix for a noisy writer: a writer that paints this screen is still a defect in
the writer.

## Keys

| key | what it does |
|---|---|
| **1** | show the API key (confirmed first: it is a bearer token and this screen can be on a KVM recording). The label is the same whether or not the box has a key - `vm-check.sh`'s `U2 menu` row greps it - and on a blank box the CONNECT block says anyone on the network can use Lemonade (cycle 20's blank-key default, item 7) |
| **2** | quick chat test on GPU and NPU -> `1bit-console-menu --action 2` |
| **3** | network: DHCP, static address, hostname -> `--action 3` |
| **4** | models: list, pull, delete -> `--action 4` |
| **5** | install 1bit OS to a disk (guarded; the installer asks for the whole device name) -> `--action 5` |
| **6** | start the desktop (JWM) for this session, logged by `1bit-desktop` -> `--action 6` |
| **7** | bash; the console comes back on exit -> `--action 7` |
| **8** | full-screen logs; **8** or **q** returns |
| **9** | reboot / power off, with a confirmation |
| arrows, PgUp/PgDn, Home/End | scroll the log (one keypress = one record, oldest to newest) |
| **f** | follow the newest line again |
| **r** | re-read the facts now (they refresh every 2 s while the box is coming up, every 3 s after) |
| **q** | stays on this screen: this console never drops to a bare prompt. Over ssh (`ssh -t root@<box> /usr/local/bin/1bit-console-tui`) it quits, as `q` does in every full-screen program - the footer says which of the two you are looking at |

Keys 2-7 hand the terminal to `1bit-console-menu --action N` - the *same* code `vm-check.sh`'s `U2 menu actions`
and `--selftest` rows exercise. Nothing that needs a typed answer is re-implemented here, so the two surfaces
cannot drift, and an action the gate proves is an action the screen has.

## Reaching it from another machine (ssh, links, QR)

The owner, 2026-11-04: *"due to being headless, can you make the links clickable, because i'm on kvm and i want use
it through ryzen"*. So the same binary runs over ssh - it makes no tty1 assumption - and three things follow from
the terminal it finds itself on:

* **Clickable URLs** (OSC 8 hyperlinks) on the CONNECT rows when `TERM` is a terminal emulator (`xterm-256color`,
  `tmux`, `kitty`, ...). ratatui has no hyperlink style, so `main` writes them *after* the frame, at the cells
  `ui::draw` reports in `Link`, with the drawn text as the target: the text and the link cannot disagree, and
  `every_hyperlink_covers_the_url_that_is_drawn_there` asserts exactly that off the rendered frame. On the VGA
  console (`TERM=linux`), JetKVM and a serial line they are **not** written - there they would be printed as
  garbage - and no link is emitted for a URL that wrapping split, because half a URL opens the wrong thing.
* **A QR code** of the web UI URL, to the right of the CONNECT rows, two modules per cell (17 rows for the
  appliance's 27-character URL: version 2, ECC L, `qrcodegen`, the only new dependency and one with no
  transitive ones), drawn **black on white** because a scanner wants dark-on-light. The drawn code was read back
  with zxing-cpp from a PNG of the 240x67 frame and decoded to `http://192.168.50.110:13305` (2026-11-04, by hand);
  `a_known_url_renders_to_these_exact_rows` pins the 17 rows literally, so a change to the encoder, the ECC level or
  the quiet zone fails the build instead of shipping a code nobody can read. It is only drawn where it
  fits: at 80x25 there are no 33 free columns beside the block and squeezing it in would push the menu or the log
  off the screen, so there the URL stays plain text - which a phone camera reads too.
* **The mDNS name** `http://1bit-os.local:13305`, next to the address, with `(mDNS)` when the box's own
  `avahi-daemon` is running and `(mDNS - avahi not running)` when it is not: the row never promises a name that
  will not answer. The responder itself is pi-os3's half of the request (engine #953).

## What it reads, and why from there

Every number comes from where the rest of the image reads it - no second source of truth (`src/facts.rs`):

| panel | source |
|---|---|
| header | `/data/RELEASE-TAG` or `payload=` in `/etc/1bit/release-identity`, `/proc/sys/kernel/hostname`, `ip -4`, `/proc/uptime`, the `SecureBoot-*` efivar (the value byte is the fifth: efivarfs refuses partial reads, and a `dd skip=4` reads as "disabled" on a Secure Boot box) |
| Lemonade | the process table, an actual TCP connect to 127.0.0.1:13305, `LEMONADE_API_KEY=` in `/etc/1bit/lemond.conf`, and `/api/v1/health` for the loaded models and the engine (`all_models_loaded[]` `type`/`device`, `model_loaded`) |
| GPU | `llama-server --list-devices`, i.e. the runtime's own answer, plus `/sys/class/drm/card*/device/mem_info_{vram,gtt}_{used,total}` and hwmon |
| NPU | `/proc/modules` for `amdxdna`, `/lib/firmware/amdnpu`, `/dev/accel/*`, and Lemonade's health for busy/idle |
| Models | `GET /api/v1/models` (downloaded only) and `GET /api/v1/health` (loaded) |
| last request | `GET /api/v1/stats` - `tokens_per_second`, the same number the release checks quote |
| Fans / temps | `/usr/sbin/1bit-board status`, else hwmon fans; max hwmon temp |
| Data | `df -B1 /data` |
| Logs | the tail of `/var/log/lemond-onebit.log` (by offset: a 40 MB log costs 256 KiB) plus `dmesg` filtered to amdgpu/amdxdna/hrx/accel |

**The GPU probe runs with Lemonade's environment**, and that is the point of `facts::lemonade_env()`: an
`/etc/inittab` respawn inherits neither `LD_LIBRARY_PATH` nor the no-SVM trio, which is why the dialog menu said
"GPU no HRX device" on a board serving 303 tok/s (cycle 13, measured). The environment is obtained by *sourcing
`/etc/1bit/libhsa.env`* rather than by re-implementing its `${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}` expansion, so
there is exactly one place that decides what HRX needs.

Nothing is clipped. Every string the widgets see has already been through `wrap_words`, which is why
`firmware present` cannot become `firmware prese` (cycle 16's board) at any width - there is a test that walks
widths 8..60 and fails on any output word that is not an input word.

## Build and install

```sh
os/v3/puppy/console-tui/build-static.sh        # cargo test + musl release build, prints the binary path
CONSOLE_TUI_BIN=<that path> bash os/v3/puppy/build.sh resolute64
```

`container-build.sh` installs the binary as `/usr/local/bin/1bit-console-tui`, refuses a `CONSOLE_TUI_BIN` that
does not *run* (`--version` must print `1bit-console-tui ...` - a program init cannot exec is how a console
becomes a dead screen), refuses a packed copy that is not executable, and merely notes the binary's absence:
then tty1 keeps the dialog menu, which is the documented fallback. `1bit-console` tries the TUI first and execs
the menu on any exit, zero or not, so there is no path from this screen to a bare prompt.

There is no Rust toolchain on the appliance; the binary is static on purpose and `build-static.sh` refuses a
dynamic one (`readelf` PT_INTERP, `ldd`).

## Tests

```sh
cargo test                                    # 52 tests: the probe parsers, the QR drawing (incl. a literal known answer), the links, the short-window degradation, the two renders, the engine row's model choice
cargo run -- --snapshot 240x67 --fixture      # the frame tests/snapshots/240x67.txt holds
cargo run -- --snapshot 80x25 --fixture
cargo run -- --status                         # the same facts as plain text (what a shell or a gate row can read)
```

The parsers are tested against recorded samples, not invented ones: the board's own `HRX0: AMD Radeon 8060S
Graphics (Node 1) (gfx1151) (122880 MiB, 74158 MiB free)`, the `(none)` of runs A and D in the libhsa
measurement, the efivar bytes `06 00 00 00 01`, the documented `/v1/models`, `/v1/health` and `/v1/stats` bodies,
chunked and plain HTTP, `df`, the c16 status lines. `tests/snapshots/*.txt` are the rendered frames for the c16
board sample: changing the layout without meaning to fails the build.
