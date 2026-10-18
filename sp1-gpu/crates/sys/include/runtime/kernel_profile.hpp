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
#include <chrono>
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

    // With the operation's absolute start and end (ms on one GPU clock), for the busy-time union.
    void add(const char* name, double ms, double start_ms, double end_ms) {
        {
            std::lock_guard<std::mutex> lock(m_);
            intervals_.push_back(Interval{start_ms, end_ms, name});
        }
        add(name, ms);
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
        std::vector<Interval> iv;
        {
            std::lock_guard<std::mutex> lock(m_);
            iv = intervals_;
        }
        // SP1_GPU_PROFILE_WINDOW_MS=<n>: the busy time and gaps of the last n ms only (e.g. the
        // proof, without the setup before it).
        if (const char* w = std::getenv("SP1_GPU_PROFILE_WINDOW_MS"); w && !iv.empty()) {
            const double window = std::atof(w);
            double last_end = 0;
            for (auto& v : iv) last_end = v.end > last_end ? v.end : last_end;
            std::vector<Interval> kept;
            for (auto& v : iv) {
                if (v.start >= last_end - window) kept.push_back(v);
            }
            iv.swap(kept);
            std::fprintf(f, "# (busy time and gaps over the last %.0f ms only)\n", window);
        }
        if (!iv.empty()) {
            std::sort(iv.begin(), iv.end(), [](const Interval& a, const Interval& b) {
                return a.start < b.start;
            });
            struct Gap {
                double ms, at;
                const char* before;
                const char* after;
            };
            std::vector<Gap> gaps;
            double busy = 0, cur_start = iv[0].start, cur_end = iv[0].end, last = iv[0].end;
            const char* cur_name = iv[0].name;
            for (auto& v : iv) {
                if (v.end > last) last = v.end;
                if (v.start > cur_end) {
                    busy += cur_end - cur_start;
                    gaps.push_back(Gap{v.start - cur_end, cur_end - iv[0].start, cur_name, v.name});
                    cur_start = v.start;
                    cur_end = v.end;
                    cur_name = v.name;
                } else if (v.end > cur_end) {
                    cur_end = v.end;
                    cur_name = v.name;
                }
            }
            busy += cur_end - cur_start;
            const double span = last - iv[0].start;
            // Idle time by the operation it follows, and the longest gaps.
            std::map<std::string, std::pair<double, uint64_t>> idle_after;
            for (auto& g : gaps) {
                auto& e = idle_after[g.before];
                e.first += g.ms;
                e.second += 1;
            }
            std::vector<std::pair<std::string, std::pair<double, uint64_t>>> idle(
                idle_after.begin(),
                idle_after.end());
            std::sort(idle.begin(), idle.end(), [](const auto& a, const auto& b) {
                return a.second.first > b.second.first;
            });
            for (size_t i = 0; i < idle.size() && i < 25; i++) {
                std::fprintf(
                    f,
                    "# idle %10.1f ms in %6llu gaps after  %s\n",
                    idle[i].second.first,
                    (unsigned long long)idle[i].second.second,
                    idle[i].first.c_str());
            }
            std::sort(gaps.begin(), gaps.end(), [](const Gap& a, const Gap& b) {
                return a.ms > b.ms;
            });
            for (size_t i = 0; i < gaps.size() && i < 25; i++) {
                std::fprintf(
                    f,
                    "# gap %8.1f ms at %9.1f ms: %s -> %s\n",
                    gaps[i].ms,
                    gaps[i].at,
                    gaps[i].before,
                    gaps[i].after);
            }
            std::fprintf(
                f,
                "# GPU busy %.1f ms of a %.1f ms span (%.0f%%), first timed operation to last\n",
                busy,
                span,
                span > 0 ? 100.0 * busy / span : 0.0);
        }
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
    struct Interval {
        double start, end;
        const char* name;
    };
    std::vector<Interval> intervals_;
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

// Times a scope on the host (allocator calls and the like), listed with the GPU operations.
class HostTimer {
  public:
    explicit HostTimer(const char* name) : name_(name) {
        if (enabled()) start_ = std::chrono::steady_clock::now();
    }
    ~HostTimer() {
        if (!enabled()) return;
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
                .count();
        totals().add(name_, ms);
    }
    HostTimer(const HostTimer&) = delete;
    HostTimer& operator=(const HostTimer&) = delete;

  private:
    const char* name_;
    std::chrono::steady_clock::time_point start_{};
};

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
