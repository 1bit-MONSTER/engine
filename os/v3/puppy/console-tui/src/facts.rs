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

//! Everything the screen shows, read where the rest of the image reads it: /proc, /sys, Lemonade's own API and
//! the two files the boot writes. No second source of truth, and nothing that a stale file can fake.
//!
//! The parsers are free functions on purpose: they are the part that has to keep working when the appliance
//! changes shape (a new `llama-server` line, a chunked HTTP body, a firmware path), so each one is fed a
//! recorded sample in the tests below rather than trusted.
//!
//! Two rules inherited from the cycle-13/16 board boots, both paid for on hardware:
//!   * the GPU question is asked of the RUNTIME (`llama-server --list-devices`), not of `/dev/kfd`: a device node
//!     with no HRX0 registered is exactly the failure "no menu system ... GPU no HRX device" while chat-hrx
//!     served 303 tok/s. So that probe is spawned with the environment `/etc/1bit/libhsa.env` gives lemond,
//!     because an inittab respawn inherits neither `LD_LIBRARY_PATH` nor the no-SVM trio.
//!   * no line is cut in the middle of a word ("firmware prese"). `wrap_words` is the only place text is
//!     shortened and the renderer never clips what it produces.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

pub const API_HOST: &str = "127.0.0.1:13305";
pub const CONF: &str = "/etc/1bit/lemond.conf";
pub const LEMLOG: &str = "/var/log/lemond-onebit.log";
pub const MENU: &str = "/usr/local/bin/1bit-console-menu";
pub const ENVFILE: &str = "/etc/1bit/libhsa.env";
pub const LIBHSA_FILE: &str = "/etc/1bit/libhsa";
const HRX_SERVER: &str = "/opt/1bit-llama/llama-server";
const GPU_CACHE: &str = "/tmp/1bit-console-tui-gpu";
const GPU_CACHE_S: u64 = 120;
const LOG_LINES: usize = 400;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SecureBoot {
    Enabled,
    Disabled,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default)]
pub struct Gpu {
    /// false until the runtime has answered (the probe is cached and may still be running on the first draw)
    pub asked: bool,
    pub device: Option<String>,
    pub free_mib: Option<u64>,
    pub note: String,
    pub vram_used: u64,
    pub vram_total: u64,
    pub gtt_used: u64,
    pub gtt_total: u64,
    pub temp_c: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Npu {
    pub module: bool,
    pub firmware: bool,
    pub device: Option<String>,
    pub busy: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Models {
    pub downloaded: Vec<String>,
    pub loaded: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub tps: Option<f64>,
    pub ttft_s: Option<f64>,
    pub requests: Option<u64>,
    pub active: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct Facts {
    pub release: String,
    pub hostname: String,
    pub uptime_s: u64,
    pub secure_boot: SecureBoot,
    pub ips: Vec<String>,
    pub key: Option<String>,
    /// avahi-daemon is running, i.e. the `1bit-os.local` name the CONNECT block prints can answer
    pub mdns: bool,
    pub lemond_pid: Option<u32>,
    pub lemond_listening: bool,
    /// the port is listed in `netstat -lnt` (true while the socket exists, even if nothing answers on it)
    pub lemond_listed: bool,
    pub gpu: Gpu,
    pub npu: Npu,
    pub models: Models,
    pub fans: String,
    pub temp_c: Option<u32>,
    /// (total, free) bytes of the data partition
    pub data: Option<(u64, u64)>,
    pub stats: Stats,
    /// "LEM " / "KRN " prefixed, oldest first
    pub logs: Vec<String>,
}

impl Facts {
    /// The CONNECT block's own question: can a person reach this box yet?
    pub fn reachable(&self) -> bool {
        !self.ips.is_empty() && self.lemond_listening
    }

    pub fn ip(&self) -> Option<&str> {
        self.ips.first().map(String::as_str)
    }

    pub fn collect() -> Facts {
        let mut f = Facts {
            release: release_tag(),
            hostname: hostname(),
            uptime_s: uptime_s(),
            secure_boot: secure_boot(),
            ips: ips(),
            key: api_key(),
            mdns: false,
            lemond_pid: None,
            lemond_listening: lemond_listening(),
            lemond_listed: false,
            gpu: Gpu::default(),
            npu: Npu::default(),
            models: Models::default(),
            fans: String::new(),
            temp_c: max_temp(),
            data: data_partition(),
            stats: Stats::default(),
            logs: Vec::new(),
        };
        // one `ps` for both questions: lemond and the mDNS responder are looked up in the same snapshot
        let ps = ps_snapshot();
        f.lemond_pid = pid_of(&ps, "lemond");
        f.mdns = pid_of(&ps, "avahi-daemon").is_some();
        f.lemond_listed = Command::new("netstat")
            .args(["-lnt"])
            .output()
            .map(|o| listens_on_13305(&String::from_utf8_lossy(&o.stdout)))
            .unwrap_or(false);
        f.gpu = gpu_probe();
        f.gpu.temp_c = f.temp_c;
        let key = f.key.clone();

        if f.lemond_listening {
            if let Some(body) = get("/api/v1/models", key.as_deref(), Duration::from_secs(2)) {
                f.models = models_from(&body);
            }
            if let Some(body) = get("/api/v1/health", key.as_deref(), Duration::from_secs(2)) {
                let (loaded, busy_npu) = health_from(&body);
                if !loaded.is_empty() {
                    f.models.loaded = loaded;
                }
                f.npu.busy = busy_npu;
            }
            if let Some(body) = get("/api/v1/stats", key.as_deref(), Duration::from_secs(2)) {
                f.stats = stats_from(&body);
            }
        }

        f.npu.module = fs::read_to_string("/proc/modules")
            .map(|m| m.lines().any(|l| l.starts_with("amdxdna ")))
            .unwrap_or(false);
        f.npu.firmware = Path::new("/lib/firmware/amdnpu").exists();
        f.npu.device = fs::read_dir("/dev/accel").ok().and_then(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .next()
        });

        f.fans = fans_line();
        f.logs = logs();
        f
    }
}

// ── the pieces, each reading what the image already writes ────────────────────────────────────────────────

fn first_line(path: &str) -> Option<String> {
    fs::read_to_string(path)
        .ok()?
        .lines()
        .next()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
}

pub fn release_tag() -> String {
    for p in ["/data/RELEASE-TAG", "/etc/1bit/release-identity"] {
        if let Some(v) = first_line(p) {
            if p.ends_with("release-identity") {
                if let Some(v) = v.strip_prefix("payload=") {
                    return v.to_string();
                }
            } else {
                return v;
            }
        }
    }
    "unknown".to_string()
}

pub fn hostname() -> String {
    first_line("/proc/sys/kernel/hostname").unwrap_or_else(|| "1bit-os".to_string())
}

pub fn uptime_s() -> u64 {
    fs::read_to_string("/proc/uptime")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map(|v| v as u64)
        .unwrap_or(0)
}

/// efivarfs does not support partial reads, so the whole variable is read and the value byte (index 4, after
/// the four attribute bytes) is taken. The measured variable on a Secure Boot machine is `06 00 00 00 01`.
pub fn secure_boot_from(bytes: &[u8]) -> SecureBoot {
    match bytes.get(4) {
        Some(1) => SecureBoot::Enabled,
        Some(_) => SecureBoot::Disabled,
        None => SecureBoot::Unknown,
    }
}

fn secure_boot() -> SecureBoot {
    let dir = Path::new("/sys/firmware/efi/efivars");
    let Ok(rd) = fs::read_dir(dir) else {
        return SecureBoot::Unknown;
    };
    for e in rd.flatten() {
        if e.file_name().to_string_lossy().starts_with("SecureBoot-") {
            if let Ok(b) = fs::read(e.path()) {
                return secure_boot_from(&b);
            }
        }
    }
    SecureBoot::Unknown
}

pub fn ips_from(ip_out: &str) -> Vec<String> {
    let mut v: Vec<String> = ip_out
        .lines()
        .filter_map(|l| l.split_whitespace().nth(3))
        .map(|c| c.split('/').next().unwrap_or(c).to_string())
        .collect();
    v.dedup();
    v
}

fn ips() -> Vec<String> {
    let out = Command::new("ip")
        .args(["-4", "-o", "addr", "show", "scope", "global"])
        .output();
    match out {
        Ok(o) => ips_from(&String::from_utf8_lossy(&o.stdout)),
        Err(_) => Vec::new(),
    }
}

pub fn api_key_from(conf: &str) -> Option<String> {
    conf.lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("LEMONADE_API_KEY=")
                .map(|k| k.trim().to_string())
        })
        .filter(|k| !k.is_empty())
}

fn api_key() -> Option<String> {
    let target = fs::read_link(CONF)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| CONF.to_string());
    let text = fs::read_to_string(&target)
        .or_else(|_| fs::read_to_string(CONF))
        .ok()?;
    api_key_from(&text)
}

pub fn pid_of(ps_out: &str, name: &str) -> Option<u32> {
    ps_out.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        let pid = it.next()?;
        let cmd = it.next()?;
        let base = cmd.rsplit('/').next().unwrap_or(cmd);
        if base == name {
            pid.parse().ok()
        } else {
            None
        }
    })
}

/// `ps -eo pid=,comm=` once per refresh; a box without `ps` gets an empty string, not a panic.
fn ps_snapshot() -> String {
    Command::new("ps")
        .args(["-eo", "pid=,comm="])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

pub fn listens_on_13305(netstat: &str) -> bool {
    netstat
        .lines()
        .any(|l| matches!(l.split_whitespace().nth(3), Some(a) if a.ends_with(":13305")))
}

fn lemond_listening() -> bool {
    // the connect IS the question: a listed socket nothing can talk to is not "up", and on a host with no
    // netstat the listing is not even available (measured on this box: lemond running, netstat absent, the panel
    // said "not listening" while curl answered 200)
    connect_ok()
}

fn connect_ok() -> bool {
    let Ok(mut addrs) = API_HOST.to_socket_addrs() else {
        return false;
    };
    match addrs.next() {
        Some(a) => TcpStream::connect_timeout(&a, Duration::from_millis(400)).is_ok(),
        None => false,
    }
}

// ── GPU: the runtime's own answer, with Lemonade's environment ────────────────────────────────────────────

/// `llama-server --list-devices` prints one `HRXn: <name> (<node>) (<isa>) (<total> MiB, <free> MiB free)` line
/// per device. The measured board line is
/// `HRX0: AMD Radeon 8060S Graphics (Node 1) (gfx1151) (122880 MiB, 74158 MiB free)`.
pub fn hrx_device(text: &str) -> Option<(String, Option<u64>)> {
    let line = text.lines().map(str::trim).find(|l| {
        l.starts_with("HRX") && l.contains(':') && !l.to_ascii_uppercase().contains("INCOMPATIBLE")
    })?;
    let id = line.split(':').next()?.to_string();
    let rest = line.split_once(':')?.1.trim();
    // "<name> (<node>) (<isa>) (<total> MiB, <free> MiB free)" -> name is what precedes the first " (" that
    // opens a group; take the free figure from the last "N MiB free" so a name in parentheses cannot fool it
    let name = rest.split(" (").next().unwrap_or(rest).trim().to_string();
    let free = rest
        .rsplit_once("MiB free")
        .and_then(|(head, _)| {
            head.rsplit(|c: char| !c.is_ascii_digit())
                .find(|s| !s.is_empty())
        })
        .and_then(|n| n.parse::<u64>().ok());
    Some((format!("{id}: {name}"), free))
}

/// The environment lemond gets, obtained by sourcing the very file it reads rather than by re-implementing it:
/// the file uses `${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`, and a parser that guessed would be a second source of
/// truth for the thing the cycle-13 board boot got wrong.
pub fn lemonade_env() -> Vec<(String, String)> {
    let script = format!(". {ENVFILE} 2>/dev/null; [ -r {LIBHSA_FILE} ] && export IREE_HAL_AMDGPU_LIBHSA_PATH=$([ -s {LIBHSA_FILE} ] && cat {LIBHSA_FILE}); env -0");
    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c").arg(script);
    cmd.env_clear();
    cmd.env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin");
    match cmd.output() {
        Ok(o) => parse_env0(&o.stdout).into_iter().collect(),
        Err(_) => Vec::new(),
    }
}

pub fn parse_env0(bytes: &[u8]) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for field in bytes.split(|&b| b == 0) {
        let Ok(s) = std::str::from_utf8(field) else {
            continue;
        };
        if let Some((k, v)) = s.split_once('=') {
            if !k.is_empty() {
                m.insert(k.to_string(), v.to_string());
            }
        }
    }
    m
}

pub fn sysfs_u64(path: &str) -> u64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// The first DRM card that reports VRAM at all. The cards are READ from the directory rather than assumed to be
/// `card0..card2`: the node's number depends on probe order, and a machine with a display controller that reports
/// nothing (or a fourth card) would otherwise show the wrong device's memory or none (pi-prs' first focus area).
pub fn gpu_mem_in(drm: &str, gpu: &mut Gpu) {
    let Ok(rd) = fs::read_dir(drm) else { return };
    let mut cards: Vec<std::path::PathBuf> = rd
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("card"))
        .map(|e| e.path())
        .collect();
    cards.sort();
    for card in cards {
        let base = card.join("device");
        let p = |n: &str| base.join(n).to_string_lossy().into_owned();
        let total = sysfs_u64(&p("mem_info_vram_total"));
        if total == 0 {
            continue;
        }
        gpu.vram_total = total;
        gpu.vram_used = sysfs_u64(&p("mem_info_vram_used"));
        gpu.gtt_total = sysfs_u64(&p("mem_info_gtt_total"));
        gpu.gtt_used = sysfs_u64(&p("mem_info_gtt_used"));
        return;
    }
}

fn gpu_mem(gpu: &mut Gpu) {
    gpu_mem_in("/sys/class/drm", gpu);
}

fn gpu_probe() -> Gpu {
    let mut g = Gpu::default();
    gpu_mem(&mut g);

    // cached for two minutes: the probe starts a runtime and the screen redraws every couple of seconds
    let fresh = fs::metadata(GPU_CACHE)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .map(|age| age.as_secs() < GPU_CACHE_S)
        .unwrap_or(false);
    if fresh {
        if let Ok(text) = fs::read_to_string(GPU_CACHE) {
            apply_hrx(&mut g, &text);
            return g;
        }
    }

    if !Path::new("/dev/kfd").exists() {
        g.asked = true;
        g.note = "no /dev/kfd on this machine".to_string();
        return g;
    }
    if !Path::new(HRX_SERVER).exists() {
        g.asked = true;
        g.note = format!("no llama-server at {HRX_SERVER}");
        return g;
    }
    let env = lemonade_env();
    let mut cmd = Command::new(HRX_SERVER);
    cmd.arg("--list-devices");
    cmd.env_clear();
    for (k, v) in &env {
        cmd.env(k, v);
    }
    if !env.iter().any(|(k, _)| k == "PATH") {
        cmd.env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin");
    }
    let text = match cmd.output() {
        Ok(o) => {
            let mut t = String::from_utf8_lossy(&o.stdout).into_owned();
            t.push_str(&String::from_utf8_lossy(&o.stderr));
            t
        }
        Err(e) => format!("llama-server did not run: {e}"),
    };
    let _ = fs::write(GPU_CACHE, &text);
    apply_hrx(&mut g, &text);
    g
}

fn apply_hrx(g: &mut Gpu, text: &str) {
    g.asked = true;
    match hrx_device(text) {
        Some((name, free)) => {
            g.device = Some(name);
            g.free_mib = free;
            g.note = match free {
                Some(mib) => format!("{mib} MiB free"),
                None => String::new(),
            };
        }
        None => {
            // the two shapes the cycle-13 board produced, named instead of guessed at
            g.note = if text.trim().is_empty() {
                "the runtime printed nothing".to_string()
            } else if text.contains("(none)") {
                "the runtime registered no device (no libhsa in the probe's environment)"
                    .to_string()
            } else {
                "no HRX device (the runtime registered none)".to_string()
            };
        }
    }
}

// ── NPU, fans, temps, data ──────────────────────────────────────────────────────────────────────────────────

fn max_temp() -> Option<u32> {
    let mut max = 0u32;
    let rd = fs::read_dir("/sys/class/hwmon").ok()?;
    for h in rd.flatten() {
        let Ok(inner) = fs::read_dir(h.path()) else {
            continue;
        };
        for f in inner.flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if name.starts_with("temp") && name.ends_with("_input") {
                let t = sysfs_u64(&f.path().to_string_lossy()) / 1000;
                if t as u32 > max {
                    max = t as u32;
                }
            }
        }
    }
    (max > 0).then_some(max)
}

pub fn fans_from(board: &str, hwmon: &[(String, u64)]) -> String {
    if let Some(l) = board
        .lines()
        .next()
        .map(str::trim)
        .filter(|l| !l.is_empty())
    {
        return l.to_string();
    }
    if hwmon.is_empty() {
        return "not readable (no EC driver, no hwmon fan)".to_string();
    }
    let list: Vec<String> = hwmon.iter().map(|(n, v)| format!("{n}={v}")).collect();
    format!("fans: {}", list.join(" "))
}

fn fans_line() -> String {
    let board = Command::new("/usr/sbin/1bit-board")
        .arg("status")
        .output()
        .ok();
    let board = board
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let mut hwmon = Vec::new();
    if let Ok(rd) = fs::read_dir("/sys/class/hwmon") {
        for h in rd.flatten() {
            let Ok(inner) = fs::read_dir(h.path()) else {
                continue;
            };
            for f in inner.flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                if name.starts_with("fan") && name.ends_with("_input") {
                    let stem = name.trim_end_matches("_input").to_string();
                    hwmon.push((stem, sysfs_u64(&f.path().to_string_lossy())));
                }
            }
        }
    }
    fans_from(&board, &hwmon)
}

pub fn df_from(df: &str) -> Option<(u64, u64)> {
    // `df -B1 /data` -> header + one row: Filesystem 1B-blocks Used Available Use% Mounted
    let row = df.lines().nth(1)?;
    let mut f = row.split_whitespace();
    f.next()?;
    let total: u64 = f.next()?.parse().ok()?;
    f.next()?;
    let free: u64 = f.next()?.parse().ok()?;
    Some((total, free))
}

fn data_partition() -> Option<(u64, u64)> {
    let out = Command::new("df").args(["-B1", "/data"]).output().ok()?;
    df_from(&String::from_utf8_lossy(&out.stdout))
}

// ── Lemonade's API: models, health, stats ─────────────────────────────────────────────────────────────────

/// The `{...}` objects inside the array that follows `"key":`, at depth 1. Enough JSON for three read-only
/// endpoints without pulling a parser (and its dependencies) into a console that must fit on a 9 MB image.
pub fn json_objects(json: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\"");
    let Some(start) = json.find(&needle) else {
        return Vec::new();
    };
    let Some(open) = json[start..].find('[').map(|i| start + i) else {
        return Vec::new();
    };
    let b = json.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut obj_start = None;
    let mut in_str = false;
    let mut esc = false;
    for i in open + 1..b.len() {
        let c = b[i] as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => {
                depth += 1;
                if depth == 1 {
                    obj_start = Some(i);
                }
            }
            '}' => {
                if depth == 1 {
                    if let Some(s) = obj_start.take() {
                        out.push(json[s..=i].to_string());
                    }
                }
                depth -= 1;
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }
    out
}

/// The value after `"key":` - a quoted string decoded, or a bare token trimmed. Small on purpose: the three
/// endpoints below are read-only and flat, and a JSON crate would cost more than the whole screen.
pub fn json_raw(obj: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let at = obj.find(&needle)? + needle.len();
    let rest = obj[at..].trim_start().strip_prefix(':')?.trim_start();
    if rest.is_empty() {
        return None;
    }
    let Some(s) = rest.strip_prefix('"') else {
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        return Some(rest[..end].trim().to_string());
    };
    let mut out = String::new();
    let mut esc = false;
    for c in s.chars() {
        if esc {
            out.push(match c {
                'n' => '\n',
                't' => '\t',
                other => other,
            });
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

pub fn json_str(obj: &str, key: &str) -> Option<String> {
    json_raw(obj, key)
}

pub fn json_num(obj: &str, key: &str) -> Option<f64> {
    json_raw(obj, key)?.parse().ok()
}

pub fn json_bool(obj: &str, key: &str) -> Option<bool> {
    match json_raw(obj, key)?.as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// `/api/v1/models`: keep what is on the machine. The catalogue lists undownloaded entries too, and a picker
/// that offered them would name a model this box cannot run.
pub fn models_from(json: &str) -> Models {
    let mut m = Models::default();
    for obj in json_objects(json, "data") {
        let Some(id) = json_str(&obj, "id") else {
            continue;
        };
        if json_bool(&obj, "downloaded").unwrap_or(false) {
            m.downloaded.push(id);
        }
    }
    m
}

/// `/api/v1/health`: the models actually LOADED, and whether an NPU one is busy (that is the NPU panel's
/// busy/idle question - the module being loaded says nothing about whether the device is working).
pub fn health_from(json: &str) -> (Vec<String>, bool) {
    let mut loaded = Vec::new();
    let mut npu_busy = false;
    for obj in json_objects(json, "all_models_loaded") {
        let name = json_str(&obj, "model_name").or_else(|| json_str(&obj, "checkpoint"));
        if let Some(n) = name {
            loaded.push(n);
        }
        let device = json_str(&obj, "device").unwrap_or_default();
        let busy = json_bool(&obj, "is_busy").unwrap_or(false)
            || json_bool(&obj, "is_streaming").unwrap_or(false);
        if device.contains("npu") && busy {
            npu_busy = true;
        }
    }
    (loaded, npu_busy)
}

/// `/api/v1/stats`: the last request's rate. This is the number release-check's chat rows quote, so the screen
/// and the acceptance run cannot disagree about how fast the box is.
pub fn stats_from(json: &str) -> Stats {
    Stats {
        tps: json_num(json, "tokens_per_second").filter(|v| *v > 0.0),
        ttft_s: json_num(json, "time_to_first_token"),
        requests: json_num(json, "request_count_total").map(|v| v as u64),
        active: json_num(json, "active_requests").map(|v| v as u64),
    }
}

// ── a GET that cannot hang the screen ─────────────────────────────────────────────────────────────────────

pub fn parse_http(buf: &[u8]) -> Option<String> {
    let split = buf.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = String::from_utf8_lossy(&buf[..split]).to_ascii_lowercase();
    let body = &buf[split + 4..];
    let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
    if status != 200 {
        return None;
    }
    if head.contains("transfer-encoding: chunked") {
        return Some(String::from_utf8_lossy(&dechunk(body)).into_owned());
    }
    Some(String::from_utf8_lossy(body).into_owned())
}

pub fn dechunk(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < body.len() {
        let Some(nl) = body[i..].windows(2).position(|w| w == b"\r\n") else {
            break;
        };
        let line = String::from_utf8_lossy(&body[i..i + nl]).into_owned();
        let size_hex = line.split(';').next().unwrap_or("").trim();
        let Ok(size) = usize::from_str_radix(size_hex, 16) else {
            break;
        };
        i += nl + 2;
        if size == 0 {
            break;
        }
        let end = (i + size).min(body.len());
        out.extend_from_slice(&body[i..end]);
        i = end + 2;
    }
    out
}

pub fn get(path: &str, key: Option<&str>, timeout: Duration) -> Option<String> {
    let addr = API_HOST.to_socket_addrs().ok()?.next()?;
    let mut s = TcpStream::connect_timeout(&addr, timeout).ok()?;
    s.set_read_timeout(Some(timeout)).ok()?;
    s.set_write_timeout(Some(timeout)).ok()?;
    let mut req = format!("GET {path} HTTP/1.1\r\nHost: {API_HOST}\r\nAccept: application/json\r\nConnection: close\r\n");
    if let Some(k) = key {
        req.push_str(&format!("Authorization: Bearer {k}\r\n"));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes()).ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > 1 << 20 {
                    break;
                }
            }
            // a slow or half-closed server is not worth blocking the screen for: keep what arrived
            Err(_) => break,
        }
    }
    parse_http(&buf)
}

// ── the log pane ──────────────────────────────────────────────────────────────────────────────────────────

/// The tail of Lemonade's log and the kernel's own amdgpu/amdxdna lines. Read by offset, so a 40 MB log costs
/// what the last 256 KiB costs.
pub fn tail_file(path: &str, bytes: u64, max_lines: usize) -> Vec<String> {
    let Ok(mut f) = fs::File::open(path) else {
        return Vec::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(bytes);
    use std::io::Seek;
    if f.seek(std::io::SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let mut lines: Vec<&str> = text.lines().collect();
    if start > 0 && lines.len() > 1 {
        lines.remove(0); // first line is a partial one
    }
    let skip = lines.len().saturating_sub(max_lines);
    lines[skip..]
        .iter()
        .map(|l| l.trim_end().to_string())
        .collect()
}

pub fn kernel_lines(dmesg: &str) -> Vec<String> {
    dmesg
        .lines()
        .filter(|l| {
            let l = l.to_ascii_lowercase();
            l.contains("amdgpu")
                || l.contains("amdxdna")
                || l.contains("hrx")
                || l.contains("accel")
        })
        .map(|l| l.trim_end().to_string())
        .collect()
}

fn logs() -> Vec<String> {
    let mut out = Vec::new();
    let dmesg = Command::new("dmesg").output().ok();
    if let Some(o) = dmesg {
        for l in kernel_lines(&String::from_utf8_lossy(&o.stdout)) {
            out.push(format!("KRN {l}"));
        }
    }
    for l in tail_file(LEMLOG, 256 * 1024, LOG_LINES) {
        out.push(format!("LEM {l}"));
    }
    let keep = out.len().saturating_sub(LOG_LINES);
    out.split_off(keep)
}

// ── formatting helpers, all of them word-safe ─────────────────────────────────────────────────────────────

/// Wrap on spaces so nothing is ever cut inside a word. A single word longer than `width` is cut as a last
/// resort (it is a path or a token, not prose).
pub fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if word.chars().count() > width {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            let mut w = word.chars();
            loop {
                let piece: String = w.by_ref().take(width).collect();
                if piece.is_empty() {
                    break;
                }
                let more = w.clone().next().is_some();
                if more {
                    out.push(piece);
                } else {
                    cur = piece;
                    break;
                }
            }
            continue;
        }
        let extra = if cur.is_empty() {
            word.chars().count()
        } else {
            word.chars().count() + 1
        };
        if cur.chars().count() + extra > width {
            out.push(std::mem::take(&mut cur));
            cur = word.to_string();
        } else {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(word);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

pub fn human_uptime(s: u64) -> String {
    let d = s / 86400;
    let h = (s % 86400) / 3600;
    let m = (s % 3600) / 60;
    if d > 0 {
        format!("{d}d {h}h {m}m")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else {
        format!("{m}m")
    }
}

pub fn human_bytes(b: u64) -> String {
    const G: f64 = 1024.0 * 1024.0 * 1024.0;
    const M: f64 = 1024.0 * 1024.0;
    let b = b as f64;
    if b >= G {
        format!("{:.0}G", b / G)
    } else if b >= M {
        format!("{:.0}M", b / M)
    } else {
        format!("{b}B")
    }
}

pub fn token(page: usize) -> &'static str {
    const SPIN: [&str; 4] = ["|", "/", "-", "\\"];
    SPIN[page % SPIN.len()]
}

/// The recorded c16 board boot (the brief's own lines, plus the two measured strings from TOS3-STATE.md). Used by
/// the snapshot regression tests and by `--snapshot ... --fixture`, so the two renders in a PR are the same frame
/// the tests compare - not a hand-typed mock that drifts.
pub fn c16_board() -> Facts {
    Facts {
        release: "rc-2026-11-03-c16".into(),
        hostname: "strix".into(),
        uptime_s: 2 * 3600 + 13 * 60,
        secure_boot: SecureBoot::Enabled,
        ips: vec!["192.168.50.110".into()],
        key: Some("93112a22cafee2ad".into()),
        // the c16 payload shipped no avahi, which is why the recorded sample shows the name as not answering
        mdns: false,
        lemond_pid: Some(8882),
        lemond_listening: true,
        lemond_listed: true,
        gpu: Gpu {
            asked: true,
            device: Some("HRX0: AMD Radeon 8060S Graphics".into()),
            free_mib: Some(74158),
            note: "74158 MiB free".into(),
            vram_used: 2 * 1024 * 1024 * 1024,
            vram_total: 122880 * 1024 * 1024,
            gtt_used: 512 * 1024 * 1024,
            gtt_total: 61 * 1024 * 1024 * 1024,
            temp_c: Some(43),
        },
        npu: Npu { module: true, firmware: true, device: Some("accel0".into()), busy: false },
        models: Models {
            downloaded: vec!["Qwen3-0.6B-NPU-1bit".into(), "qwen3-0.6b-q4_k_m-1bit".into()],
            loaded: vec!["qwen3-0.6b-q4_k_m-1bit".into()],
        },
        fans: "fans: controlled (fan1 0 rpm fan2 0 rpm)".into(),
        temp_c: Some(43),
        data: Some((62277025792, 55800000000)),
        stats: Stats { tps: Some(330.75), ttft_s: Some(0.0103), requests: Some(4), active: Some(0) },
        logs: vec![
            "KRN [    1.100000] amdgpu 0000:c7:00.0: HRX runtime registered HRX0".into(),
            "KRN [    1.200000] amdxdna 0000:c7:00.1: firmware loaded".into(),
            "LEM 2026-11-03 12:07:38 llama-server: listening on 127.0.0.1:8001".into(),
            "LEM 2026-11-03 12:07:41 lemond: POST /api/v1/chat/completions 200 (327 tokens, 330.75 tok/s)".into(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every fixture below is either a sample the board produced (quoted in ~/briefs/TCONSOLE-TUI.md and
    // TOS3-STATE.md) or the payload's own documented response body. None of them was invented to make a parser
    // pass: each is the shape the appliance really answers.

    #[test]
    fn hrx_line_from_the_board() {
        // measured on strixhalo: `llama-server --list-devices` with libhsa.env in the environment
        let text = "\nAvailable devices:\n  HRX0: AMD Radeon 8060S Graphics (Node 1) (gfx1151) (122880 MiB, 74158 MiB free)\n";
        let (name, free) = hrx_device(text).expect("the board's own line");
        assert_eq!(name, "HRX0: AMD Radeon 8060S Graphics");
        assert_eq!(free, Some(74158));
    }

    #[test]
    fn hrx_absent_when_nothing_loaded() {
        // run A and run D of the measurement in TOS3-STATE.md (no libhsa, then the no-SVM trio alone)
        assert!(hrx_device("Available devices:\n  (none)\n").is_none());
    }

    #[test]
    fn hrx_ignores_incompatible_and_takes_the_first_device() {
        let text = "HRX1: AMD Radeon 8060S Graphics (INCOMPATIBLE)\nHRX0: AMD Radeon 8060S Graphics (Node 1) (gfx1151) (122880 MiB, 74158 MiB free)\n";
        let (name, free) = hrx_device(text).expect("second line");
        assert_eq!(name, "HRX0: AMD Radeon 8060S Graphics");
        assert_eq!(free, Some(74158));
    }

    #[test]
    fn env0_from_the_shipped_file() {
        // `sh -c '. /etc/1bit/libhsa.env; env -0'` on the appliance: the exported trio plus the loader path
        let raw = b"HSA_USE_SVM=0\0HSA_USERPTR_FOR_PAGED_MEM=0\0HRX_ALLOW_NO_SVM=1\0LD_LIBRARY_PATH=/opt/rocm-therock/lib\0PATH=/usr/bin:/bin:/usr/sbin:/sbin\0";
        let env = parse_env0(raw);
        assert_eq!(
            env.get("LD_LIBRARY_PATH").map(String::as_str),
            Some("/opt/rocm-therock/lib")
        );
        assert_eq!(env.get("HRX_ALLOW_NO_SVM").map(String::as_str), Some("1"));
        assert_eq!(env.len(), 5);
    }

    #[test]
    fn secure_boot_byte_is_the_fifth() {
        // measured efivarfs content of SecureBoot-* on a Secure Boot machine (rc.sysinit's own check agrees)
        assert_eq!(secure_boot_from(&[6, 0, 0, 0, 1]), SecureBoot::Enabled);
        assert_eq!(secure_boot_from(&[6, 0, 0, 0, 0]), SecureBoot::Disabled);
        assert_eq!(secure_boot_from(&[]), SecureBoot::Unknown);
    }

    #[test]
    fn models_keep_only_downloaded() {
        // the shape /api/v1/models documents, with the catalogue's undownloaded entry alongside
        let json = r#"{"data":[
            {"id":"Qwen3-0.6B-NPU-1bit","created":0,"object":"model","recipe":"flm","downloaded":true,"recipe_options":{"onebit_backend":"npu"}},
            {"id":"Qwen3-8B-1bit","created":0,"object":"model","recipe":"llamacpp","downloaded":false,"recipe_options":{}},
            {"id":"qwen3-0.6b-q4_k_m-1bit","created":0,"object":"model","recipe":"llamacpp","downloaded":true,"recipe_options":{"onebit_backend":"gpu"}}
        ],"object":"list"}"#;
        let m = models_from(json);
        assert_eq!(
            m.downloaded,
            vec!["Qwen3-0.6B-NPU-1bit", "qwen3-0.6b-q4_k_m-1bit"]
        );
    }

    #[test]
    fn health_loads_and_npu_busy() {
        let json = r#"{"status":"ok","version":"2026.43.0","model_loaded":"Qwen3-0.6B-NPU-1bit","all_models_loaded":[
            {"model_name":"Qwen3-0.6B-NPU-1bit","device":"npu","is_busy":false,"is_streaming":false,"status":"ready"},
            {"model_name":"qwen3-0.6b-q4_k_m-1bit","device":"gpu","is_busy":true,"is_streaming":true,"status":"in_use"}
        ],"websocket_port":9001}"#;
        let (loaded, npu_busy) = health_from(json);
        assert_eq!(loaded.len(), 2);
        assert!(!npu_busy, "the busy model is the GPU one");
    }

    #[test]
    fn stats_reads_the_last_request() {
        // the documented /v1/stats body, which carries the same rate the c17 board measured (330.75 tok/s)
        let json = r#"{"time_to_first_token":0.010258,"tokens_per_second":330.7515771574438,"output_tokens":327,"request_count_total":4,"live":{"active_requests":2,"estimated_tokens_per_second":0.0}}"#;
        let s = stats_from(json);
        assert_eq!(s.tps.map(|v| v.round()), Some(331.0));
        assert_eq!(s.requests, Some(4));
        assert_eq!(s.active, Some(2));
    }

    #[test]
    fn http_body_plain_and_chunked() {
        let plain = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        assert_eq!(parse_http(plain).as_deref(), Some("hello"));
        let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        assert_eq!(parse_http(chunked).as_deref(), Some("hello world"));
        // the appliance answers 401 without the key; that is "no data", not "no models"
        let unauth = b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\n\r\n{}";
        assert!(parse_http(unauth).is_none());
    }

    #[test]
    fn ips_from_the_c16_board() {
        let out = "2: enp1s0    inet 192.168.50.110/24 brd 192.168.50.255 scope global dynamic enp1s0\\       valid_lft 86390sec preferred_lft 86390sec\n";
        assert_eq!(ips_from(out), vec!["192.168.50.110"]);
    }

    #[test]
    fn api_key_from_lemond_conf() {
        let conf = "LEMONADE_PORT=13305\nLEMONADE_API_KEY=93112a22cafee2ad0f3\n";
        assert_eq!(api_key_from(conf).as_deref(), Some("93112a22cafee2ad0f3"));
        assert_eq!(api_key_from("LEMONADE_API_KEY=\n"), None);
    }

    #[test]
    fn netstat_listing_is_only_a_hint() {
        let net = "Active Internet connections\ntcp        0      0 127.0.0.1:13305         0.0.0.0:*               LISTEN\n";
        assert!(listens_on_13305(net));
        assert!(!listens_on_13305("tcp 0 0 127.0.0.1:22 0.0.0.0:* LISTEN\n"));
    }

    #[test]
    fn pid_of_lemond() {
        let ps = "  1 systemd\n 8882 lemond\n 9001 llama-server\n";
        assert_eq!(pid_of(ps, "lemond"), Some(8882));
    }

    #[test]
    fn df_row_is_total_and_free() {
        let df = "Filesystem      1B-blocks       Used   Available Use% Mounted on\n/dev/sda2    62277025792 6000000000 55800000000  10% /data\n";
        assert_eq!(df_from(df), Some((62277025792, 55800000000)));
    }

    #[test]
    fn gpu_mem_reads_whichever_card_reports_vram() {
        // a machine whose first DRM node reports nothing (the appliance's own layout: card0 is the display
        // controller, the APU reports VRAM) - a hardcoded card0..card2 list would show 0 there
        let root = std::env::temp_dir().join(format!("1bit-tui-drm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("card0/device")).unwrap();
        fs::create_dir_all(root.join("card1/device")).unwrap();
        fs::write(root.join("card0/device/mem_info_vram_total"), "0\n").unwrap();
        fs::write(
            root.join("card1/device/mem_info_vram_total"),
            "128849018880\n",
        )
        .unwrap();
        fs::write(root.join("card1/device/mem_info_vram_used"), "2147483648\n").unwrap();
        fs::write(
            root.join("card1/device/mem_info_gtt_total"),
            "65498251264\n",
        )
        .unwrap();
        fs::write(root.join("card1/device/mem_info_gtt_used"), "536870912\n").unwrap();
        let mut g = Gpu::default();
        gpu_mem_in(&root.to_string_lossy(), &mut g);
        assert_eq!(g.vram_total, 128849018880);
        assert_eq!(g.vram_used, 2147483648);
        assert_eq!(g.gtt_total, 65498251264);
        // and a directory with no card at all leaves the struct alone rather than inventing numbers
        let empty = root.join("none");
        fs::create_dir_all(&empty).unwrap();
        let mut g2 = Gpu::default();
        gpu_mem_in(&empty.to_string_lossy(), &mut g2);
        assert_eq!(g2.vram_total, 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn fans_prefer_the_board() {
        // the c16 board's own line
        let board = "fans: controlled (fan1 0 rpm fan2 0 rpm)\n";
        assert_eq!(
            fans_from(board, &[]),
            "fans: controlled (fan1 0 rpm fan2 0 rpm)"
        );
        assert_eq!(fans_from("", &[("fan1".into(), 1234)]), "fans: fan1=1234");
        assert!(fans_from("", &[]).contains("not readable"));
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human_uptime(2 * 3600 + 13 * 60), "2h 13m");
        assert_eq!(human_uptime(3 * 86400 + 60), "3d 0h 1m");
        assert_eq!(human_bytes(62277025792), "58G");
    }

    #[test]
    fn wrap_never_cuts_a_word() {
        // the c16 board's defect: "NPU amdxdna loaded, firmware prese|" - "present" must survive at any width
        let text = "NPU amdxdna loaded, firmware present, accel0";
        for w in 8..=60 {
            for line in wrap_words(text, w) {
                assert!(
                    line.chars().count() <= w.max(1),
                    "width {w}: {line:?} too long"
                );
                for word in line.split_whitespace() {
                    assert!(
                        text.split_whitespace().any(|t| t == word),
                        "width {w}: {word:?} is a cut word"
                    );
                }
            }
            let lines = wrap_words(text, w);
            let joined: Vec<&str> = lines.iter().flat_map(|l| l.split_whitespace()).collect();
            assert_eq!(joined.join(" "), text, "width {w} lost or reordered text");
        }
    }

    #[test]
    fn kernel_filter_keeps_the_gpu_and_npu() {
        let dmesg = "[    1.1] amdgpu: HRX runtime registered no device\n[    1.2] usb 1-1: new high-speed USB device\n[    1.3] amdxdna 0000:c7:00.1: bound\n";
        let kept = kernel_lines(dmesg);
        assert_eq!(kept.len(), 2);
        assert!(kept[0].contains("amdgpu") && kept[1].contains("amdxdna"));
    }
}
