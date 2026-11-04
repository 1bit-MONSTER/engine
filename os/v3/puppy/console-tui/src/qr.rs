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

//! The CONNECT block's QR code: the web UI URL in a form a phone can read (the owner, 2026-11-04: "for phones /
//! Lemonade Mobile"). The appliance is headless, so the phone is pointed at the console's own screen.
//!
//! The encoding is `qrcodegen` - Nayuki's reference implementation, one dependency with no transitive ones, so the
//! static binary grows by the encoder and by nothing else. What this module owns is the DRAWING, and it has two
//! rules of its own:
//!
//!   * two modules per cell, so the code is half as tall as it is wide: the block must fit beside the CONNECT
//!     lines on a 67-row console, and 17 rows is what that costs for a 27-character URL (version 2, ECC L).
//!   * the code is drawn on PAPER - black modules on a white background - because a scanner wants dark-on-light,
//!     and a terminal's dark background is precisely the inversion that many phone scanners refuse.
//!
//! Nothing here is guessed: the matrix is the encoder's, and what this module draws from it is pinned by two
//! known-answer tests - `blocks_draws_two_module_rows_per_text_row` (a hand-drawn 2x2) and
//! `a_known_url_renders_to_these_exact_rows` (the appliance's own URL, all 17 rows, literal). That the drawn code
//! really scans is not something a test can assert cheaply: it was measured once, on this box, by rendering the
//! 240x67 frame to a PNG and reading it back with zxing-cpp - it returned `http://192.168.50.110:13305` (2026-11-04).
//! Re-run that by hand if the encoder, the ECC level or the quiet zone is ever changed; the literal test below will
//! fail first, which is the point.

use qrcodegen::{QrCode, QrCodeEcc};

/// The quiet zone the standard asks for, in modules, on each side. It is part of the code, not padding: a scanner
/// needs the light border to find the code's edges.
pub const QUIET: usize = 4;

/// The URL as rows of terminal text, or `None` when the URL does not fit a QR code at all (it cannot: a URL this
/// appliance prints is 30 characters, and version 40 holds 2953 bytes).
pub fn code(url: &str) -> Option<Vec<String>> {
    let q = QrCode::encode_text(url, QrCodeEcc::Low).ok()?;
    let n = q.size();
    let m: Vec<Vec<bool>> = (0..n)
        .map(|y| (0..n).map(|x| q.get_module(x, y)).collect())
        .collect();
    Some(blocks(&m))
}

/// The module matrix as text rows, two module rows per text row: `█` both dark, `▀` upper only, `▄` lower only,
/// a space neither. The light cells are spaces and the drawing side paints them on white (`ui::qr_panel`).
pub fn blocks(m: &[Vec<bool>]) -> Vec<String> {
    let n = m.len();
    let side = n + 2 * QUIET;
    let at = |x: usize, y: usize| -> bool {
        x >= QUIET && y >= QUIET && x < QUIET + n && y < QUIET + n && m[y - QUIET][x - QUIET]
    };
    let mut rows = Vec::new();
    let mut y = 0;
    while y < side {
        let mut row = String::with_capacity(side);
        for x in 0..side {
            // an odd module count leaves the last text row with an empty lower half: paper, not ink
            let top = at(x, y);
            let bottom = y + 1 < side && at(x, y + 1);
            row.push(match (top, bottom) {
                (true, true) => '█',
                (true, false) => '▀',
                (false, true) => '▄',
                (false, false) => ' ',
            });
        }
        rows.push(row);
        y += 2;
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_draws_two_module_rows_per_text_row() {
        // a 2x2 matrix, drawn by hand: the quiet zone, then `▀` (dark over light) and `▄` (light over dark)
        let m = vec![vec![true, false], vec![false, true]];
        let rows = blocks(&m);
        assert_eq!(rows.len(), 5, "{rows:?}"); // 2 modules + 4 quiet on each side, halved
        assert_eq!(rows[0], " ".repeat(10));
        assert_eq!(rows[1], " ".repeat(10));
        assert_eq!(rows[2], "    ▀▄    ");
        assert_eq!(rows[3], " ".repeat(10));
        assert_eq!(rows[4], " ".repeat(10));
    }

    #[test]
    fn a_known_url_renders_to_these_exact_rows() {
        // The known answer, not a fit check: the appliance's URL in, these 17 rows of 33 cells out. Every rule this
        // module owns is visible here - 4 modules of quiet zone on each side (the two blank rows at the top and the
        // two at the bottom), two module rows per text row (`▀` upper only, `▄` lower only, `█` both), spaces for
        // light. If the encoder, the ECC level (L), the version (2 for 27 characters) or the drawing changes, this
        // fails and somebody has to look at a scanner again rather than ship a code nobody can read.
        let want = [
            "                                 ",
            "                                 ",
            "    █▀▀▀▀▀█ ▄ █ ▀█▄▄  █▀▀▀▀▀█    ",
            "    █ ███ █ ▄▄▄ █▄ █▄ █ ███ █    ",
            "    █ ▀▀▀ █ ▄█ ▄▄ ▀▄▄ █ ▀▀▀ █    ",
            "    ▀▀▀▀▀▀▀ ▀▄█▄▀ █▄█ ▀▀▀▀▀▀▀    ",
            "    ▀███▀ ▀▀█▀▄▀█▀▄▀▀█ ▀ ▀ █     ",
            "    ▀▀█▄▀▄▀ █ ▀   ▄▀▄ ▀▀▄▀ ▀█    ",
            "    ██▄ ▀▀▀█ ▀▀▄▄▀▄   ▀▀▀▄▀█▀    ",
            "    █ ▄ ▀ ▀▄▄   █  █ ▀ ▀ █ ▀█    ",
            "    ▀ ▀▀  ▀ █ █▄▄█▄ █▀▀▀█▄█      ",
            "    █▀▀▀▀▀█ ▀ ▄▄▄▀ ▄█ ▀ ██▀█▀    ",
            "    █ ███ █ █▀▄▄▄▀▄██▀███▀█▄▄    ",
            "    █ ▀▀▀ █ █▀ ▄▄ ▄ ▄▀▄▄▄▄▀ █    ",
            "    ▀▀▀▀▀▀▀ ▀▀   ▀ ▀ ▀▀▀▀▀▀▀▀    ",
            "                                 ",
            "                                 ",
        ];
        let got = code("http://192.168.50.110:13305").expect("the URL fits");
        assert_eq!(got, want);
    }

    #[test]
    fn the_appliances_own_url_fits_the_block_the_layout_reserves() {
        // measured: 27 characters of URL is a version-2 symbol (25 modules), so 33 wide and 17 rows tall
        for url in ["http://192.168.50.110:13305", "http://1bit-os.local:13305"] {
            let rows = code(url).expect("the URL fits");
            assert_eq!(rows.len(), 17, "{url}: {} rows", rows.len());
            for r in &rows {
                assert_eq!(r.chars().count(), 33, "{url}: {r:?} is not square");
            }
            // the quiet zone really is quiet, and the code really is inked
            assert!(rows[0].chars().all(|c| c == ' '), "{url}: {}", rows[0]);
            assert!(rows
                .iter()
                .all(|r| r.starts_with("    ") && r.ends_with("    ")));
            let ink: usize = rows
                .iter()
                .map(|r| r.chars().filter(|c| *c != ' ').count())
                .sum();
            assert!(ink > 80, "{url}: only {ink} inked cells");
        }
    }

    #[test]
    fn a_url_too_long_for_any_code_is_absent_rather_than_wrong() {
        assert!(code(&"x".repeat(5000)).is_none());
        assert!(code("http://192.168.50.110:13305").is_some());
    }
}
