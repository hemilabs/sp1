// Optional per-kernel GPU timing, for finding where a proof's GPU time goes on backends without a
// profiler at hand (HIP and SYCL on Windows).
//
// SP1_GPU_KERNEL_PROFILE=<file> turns it on: every kernel launch (named by its getter, see
// SP1_KERNEL_PTR) and every async copy is timed on the GPU, and at exit <file> gets one line per
// name, by total time: total ms, launches, mean us, name. Off, a launch pays one branch.
//
// The runtimes (lib/runtime/stream.cu, lib/sycl/runtime.cpp) time the work; this aggregates it.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sp1prof {

inline const char* output_path() {
    static const char* path = std::getenv("SP1_GPU_KERNEL_PROFILE");
    return path && *path ? path : nullptr;
}

inline bool enabled() {
    static const bool on = output_path() != nullptr;
    return on;
}

class Totals {
  public:
    // A kernel's name from its launch pointer, as its getter registered it.
    void name(void* kernel, const char* name) {
        std::lock_guard<std::mutex> lock(m_);
        names_[kernel] = name;
    }

    const char* name_of(void* kernel) {
        std::lock_guard<std::mutex> lock(m_);
        auto it = names_.find(kernel);
        return it == names_.end() ? "<unnamed kernel>" : it->second;
    }

    void add(const char* name, double ms) {
        std::lock_guard<std::mutex> lock(m_);
        auto& t = totals_[name];
        if (t.count == 0 || ms < t.min_ms) t.min_ms = ms;
        if (ms > t.max_ms) t.max_ms = ms;
        t.count += 1;
        t.ms += ms;
    }

    void write() {
        std::vector<std::pair<std::string, Row>> rows;
        {
            std::lock_guard<std::mutex> lock(m_);
            rows.assign(totals_.begin(), totals_.end());
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return a.second.ms > b.second.ms;
        });
        FILE* f = std::fopen(output_path(), "w");
        if (!f) return;
        double all = 0;
        for (auto& r : rows) all += r.second.ms;
        std::fprintf(f, "# total GPU ms %.1f (summed over streams)\n", all);
        std::fprintf(
            f,
            "# %12s %10s %10s %6s %10s %10s  name\n",
            "total_ms",
            "count",
            "mean_us",
            "pct",
            "min_us",
            "max_us");
        for (auto& r : rows) {
            std::fprintf(
                f,
                "%14.2f %10llu %10.1f %5.1f%% %10.1f %10.1f  %s\n",
                r.second.ms,
                (unsigned long long)r.second.count,
                1000.0 * r.second.ms / (double)r.second.count,
                all > 0 ? 100.0 * r.second.ms / all : 0.0,
                1000.0 * r.second.min_ms,
                1000.0 * r.second.max_ms,
                r.first.c_str());
        }
        std::fclose(f);
    }

  private:
    struct Row {
        uint64_t count = 0;
        double ms = 0;
        double min_ms = 0;
        double max_ms = 0;
    };

    std::mutex m_;
    std::unordered_map<void*, const char*> names_;
    std::map<std::string, Row> totals_;
};

inline Totals& totals() {
    static Totals* t = new Totals();  // never destroyed: written from an atexit handler
    return *t;
}

}  // namespace sp1prof

// Time GPU work queued on `stream` between the two calls, under `name` (HIP and CUDA; the SYCL
// runtime times each submission's own event instead). `begin` returns null when profiling is off.
extern "C" void* sp1_profile_begin(void* stream);
extern "C" void sp1_profile_end(void* token, const char* name, void* stream);

namespace sp1prof {

// Times the GPU work a scope queues on a stream.
class Scope {
  public:
    Scope(void* stream, const char* name)
        : stream_(stream), name_(name), token_(sp1_profile_begin(stream)) {}
    ~Scope() { sp1_profile_end(token_, name_, stream_); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

  private:
    void* stream_;
    const char* name_;
    void* token_;
};

}  // namespace sp1prof
