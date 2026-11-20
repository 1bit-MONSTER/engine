#include "trace.h"

#include <cstdio>
#include <ctime>
#include <fstream>
#include <unistd.h>
#include <sys/syscall.h>

#include <nlohmann/json.hpp>

namespace onebit::v2::trace {

uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

uint32_t thread_id() {
    thread_local const uint32_t tid = uint32_t(::syscall(SYS_gettid));
    return tid;
}

Recorder& Recorder::global() {
    static Recorder r;
    return r;
}

void Recorder::complete(std::string name, std::string cat, uint64_t ts_ns, uint64_t dur_ns, std::string args_json) {
    if (!enabled()) return;
    Event e;
    e.name = std::move(name);
    e.cat = std::move(cat);
    e.args_json = std::move(args_json);
    e.ph = 'X';
    e.ts_ns = ts_ns;
    e.dur_ns = dur_ns;
    e.tid = thread_id();
    std::lock_guard lk(mu_);
    events_.push_back(std::move(e));
}

void Recorder::instant(std::string name, std::string cat, std::string args_json) {
    if (!enabled()) return;
    Event e;
    e.name = std::move(name);
    e.cat = std::move(cat);
    e.args_json = std::move(args_json);
    e.ph = 'i';
    e.ts_ns = now_ns();
    e.tid = thread_id();
    std::lock_guard lk(mu_);
    events_.push_back(std::move(e));
}

void Recorder::counter(std::string name, double value) {
    if (!enabled()) return;
    Event e;
    e.name = std::move(name);
    e.cat = "counter";
    e.ph = 'C';
    e.ts_ns = now_ns();
    e.tid = thread_id();
    e.value = value;
    std::lock_guard lk(mu_);
    events_.push_back(std::move(e));
}

std::vector<Event> Recorder::snapshot() const {
    std::lock_guard lk(mu_);
    return events_;
}

size_t Recorder::size() const {
    std::lock_guard lk(mu_);
    return events_.size();
}

void Recorder::clear() {
    std::lock_guard lk(mu_);
    events_.clear();
}

bool Recorder::write_json(const std::string& path) const {
    using nlohmann::json;
    const auto events = snapshot();
    const int pid = int(::getpid());
    json out;
    out["displayTimeUnit"] = "ns";
    out["otherData"] = {{"clock", "CLOCK_BOOTTIME"}, {"engine", "1bit v2"}, {"ts_unit", "us (CLOCK_BOOTTIME ns / 1000)"}};
    json arr = json::array();
    for (const auto& e : events) {
        json j;
        j["name"] = e.name;
        j["cat"] = e.cat;
        j["ph"] = std::string(1, e.ph);
        j["ts"] = double(e.ts_ns) / 1000.0;
        j["pid"] = pid;
        j["tid"] = e.tid;
        if (e.ph == 'X') j["dur"] = double(e.dur_ns) / 1000.0;
        if (e.ph == 'i') j["s"] = "t";
        if (e.ph == 'C') {
            j["args"] = {{"value", e.value}};
        } else if (!e.args_json.empty()) {
            j["args"] = json::parse(e.args_json, nullptr, false);
            if (j["args"].is_discarded()) j["args"] = {{"raw", e.args_json}};
        }
        arr.push_back(std::move(j));
    }
    out["traceEvents"] = std::move(arr);
    std::ofstream f(path);
    if (!f) return false;
    f << out.dump() << '\n';
    return bool(f);
}

Span::Span(const char* name, const char* cat, std::string args_json)
    : name_(name), cat_(cat), args_(std::move(args_json)), on_(Recorder::global().enabled()) {
    if (on_) t0_ = now_ns();
}

Span::~Span() {
    if (!on_) return;
    const uint64_t t1 = now_ns();
    Recorder::global().complete(name_, cat_, t0_, t1 - t0_, std::move(args_));
}

}  // namespace onebit::v2::trace
