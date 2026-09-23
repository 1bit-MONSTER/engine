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
//
// C ABI over Hugging Face `tokenizers` (include/hf_tokenizer.h). Every function
// is panic-free at the boundary: errors come back as NULL / -1, and the message
// is kept for onebit_tok_last_error().
use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::slice;

use tokenizers::Tokenizer;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(msg.replace('\0', " ")).unwrap_or_default());
}

/// Parses a tokenizer.json held in memory. Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn onebit_tok_from_json(json: *const u8, len: usize) -> *mut Tokenizer {
    if json.is_null() {
        set_error("null json".into());
        return std::ptr::null_mut();
    }
    let bytes = slice::from_raw_parts(json, len);
    match Tokenizer::from_bytes(bytes) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(e) => {
            set_error(e.to_string());
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn onebit_tok_free(tok: *mut Tokenizer) {
    if !tok.is_null() {
        drop(Box::from_raw(tok));
    }
}

/// Encodes UTF-8 text. Writes up to `cap` ids to `out` and returns the total
/// number of ids (call again with a bigger buffer if it exceeds `cap`), or -1.
#[no_mangle]
pub unsafe extern "C" fn onebit_tok_encode(
    tok: *const Tokenizer,
    text: *const u8,
    len: usize,
    add_special: bool,
    out: *mut u32,
    cap: usize,
) -> i64 {
    if tok.is_null() || (text.is_null() && len > 0) {
        set_error("null argument".into());
        return -1;
    }
    let s = match std::str::from_utf8(if len == 0 { &[] } else { slice::from_raw_parts(text, len) }) {
        Ok(s) => s,
        Err(e) => {
            set_error(e.to_string());
            return -1;
        }
    };
    match (*tok).encode(s, add_special) {
        Ok(enc) => {
            let ids = enc.get_ids();
            if !out.is_null() {
                let n = ids.len().min(cap);
                std::ptr::copy_nonoverlapping(ids.as_ptr(), out, n);
            }
            ids.len() as i64
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

/// Decodes ids to UTF-8. Writes up to `cap` bytes (not NUL-terminated) and
/// returns the total byte length, or -1.
#[no_mangle]
pub unsafe extern "C" fn onebit_tok_decode(
    tok: *const Tokenizer,
    ids: *const u32,
    n: usize,
    skip_special: bool,
    out: *mut u8,
    cap: usize,
) -> i64 {
    if tok.is_null() || (ids.is_null() && n > 0) {
        set_error("null argument".into());
        return -1;
    }
    let v = if n == 0 { &[][..] } else { slice::from_raw_parts(ids, n) };
    match (*tok).decode(v, skip_special) {
        Ok(s) => {
            if !out.is_null() {
                let m = s.len().min(cap);
                std::ptr::copy_nonoverlapping(s.as_ptr(), out, m);
            }
            s.len() as i64
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

/// Vocabulary size including added tokens.
#[no_mangle]
pub unsafe extern "C" fn onebit_tok_vocab_size(tok: *const Tokenizer) -> i64 {
    if tok.is_null() { -1 } else { (*tok).get_vocab_size(true) as i64 }
}

/// The last error on this thread, valid until the next failing call.
#[no_mangle]
pub extern "C" fn onebit_tok_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}
