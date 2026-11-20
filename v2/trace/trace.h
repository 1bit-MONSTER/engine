#pragma once

// [OBS] One clock, one trace (design D12): every V2 layer records spans on CLOCK_BOOTTIME, the
// clock the KFD SMI events and the #315 tools already use, so host spans can later be merged
// with HRX dispatch profiles, NPU RECORD_TIMER values and kernel events on one timeline. The
// sink is Chrome/Perfetto JSON ("traceEvents", ph X/i/C), which ui.perfetto.dev and TraceLens open.
//
// Recording is off until enabled; a disabled Span costs one relaxed load.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace onebit::v2::trace {

// Nanoseconds on CLOCK_BOOTTIME (monotonic, counts suspend).
uint64_t now_ns();

struct Event {
    std::string name, cat, args_json;  // args_json: a JSON object or empty
    char ph = 'X';                     // X complete, i instant, C counter
    uint64_t ts_ns = 0, dur_ns = 0;
    uint32_t tid = 0;
    double value = 0;                  // C only
};

class Recorder {
public:
    static Recorder& global();
    void enable(bool on) { on_.store(on, std::memory_order_relaxed); }
    bool enabled() const { return on_.load(std::memory_order_relaxed); }
    void complete(std::string name, std::string cat, uint64_t ts_ns, uint64_t dur_ns, std::string args_json = "");
    void instant(std::string name, std::string cat, std::string args_json = "");
    void counter(std::string name, double value);
    std::vector<Event> snapshot() const;
    size_t size() const;
    void clear();
    // Writes {"traceEvents": [...], "otherData": {"clock": "CLOCK_BOOTTIME", ...}}; false on an
    // I/O error.
    bool write_json(const std::string& path) const;

private:
    std::atomic<bool> on_{false};
    mutable std::mutex mu_;
    std::vector<Event> events_;
};

// RAII span on the global recorder.
class Span {
public:
    explicit Span(const char* name, const char* cat = "v2", std::string args_json = "");
    ~Span();
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;

private:
    const char* name_;
    const char* cat_;
    std::string args_;
    uint64_t t0_ = 0;
    bool on_ = false;
};

uint32_t thread_id();

}  // namespace onebit::v2::trace
