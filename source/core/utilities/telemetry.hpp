/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */
#ifndef TELEMETRY_HPP
#define TELEMETRY_HPP
#include "da_logger.hpp"
#include "macros.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/* Telemetry instrumentation
 * -------------------------
 *
 * Lightweight telemetry for timing sections or whole functions.
 * Timers are stored in Telemetry which should result in zero overhead when not
 * enabled via -DTELEMETRY.
 * 
 * Usage:
 *  1) (main) Add "SCOPED_TIMER;" at the top of a function.
 *     This macro wraps a class with a constructor that starts a timer and a
 *     destructor that stops it. The macros automatically add the function name.
 * 
 *  2) For timing sections wrap code block with
 *      SCOPED_TIMER_BLOCK(NAME) macro WITHIN A BRACE BLOCK
 *      {
 *          SCOPED_TIMER_BLOCK("Section/Title");
 *          <code>
 *      }
 * 
 *  These macros are defined in context.hpp and use the context's Telemetry to
 *  store all the timings. 
 * 
 *  3) Alternatively, manually starting / stopping the timer:
 *      telemetry::Telemetry t("example");
 *      t.start("Section/Title"); // OR
 *      size_t t_ind = t.start("Section/Title");
 *      <code> 
 *      t.stop("Section/Title"); // or
 *      t.stop(t_idx); // this call is cheaper that the string lookup
 *      ...
 *      t.print();
 *
 *      start() also returns an index that can be passed to stop() on the
 *      same instance. Indices survive new registrations and reset(), but not
 *      destruction of the instance. An out-of-range index throws std::out_of_range.
 *      Names are copied into the registry on first use; input views need only
 *      remain valid for the call. ScopedTimer stores the index, not the name.
 * 
 *      In practical terms, the context's Telemetry instance is used, so
 *      telemetry::Telemetry &t = context::get_context()->get_telemetry();
 *      t.start("Section/Title");
 *      <code> 
 *      t.stop("Section/Title");
 * 
 *  When destroying the context, the telemetry data is printed.
 */

// Helper macro for the context's telemetry instrumentation
#define SCOPED_TIMER_HELPER(T, NAME) telemetry::ScopedTimer CONCAT(__t, __LINE__)(T, NAME)

namespace telemetry {

namespace pretty_print {
[[nodiscard]] inline std::string prettify(std::string s);
}

class Telemetry {

  public:
    Telemetry() = default;
    Telemetry(const Telemetry &) = delete;
    Telemetry &operator=(const Telemetry &) = delete;
    Telemetry(Telemetry &&) = delete;
    Telemetry &operator=(Telemetry &&) = delete;
    virtual ~Telemetry() {}

    virtual size_t start(std::string_view name) = 0;
    virtual void stop(std::string_view name) = 0;
    virtual void stop(size_t index) = 0;
    virtual void reset() = 0;
    virtual void print() const = 0;
};

class TelemetryDummy final : public Telemetry {
  public:
    explicit TelemetryDummy(const std::string &) : Telemetry() {}
    size_t start(std::string_view) override { return 0; }
    void stop(std::string_view) override {}
    void stop(size_t) override {}
    void reset() override {}
    void print() const override {}
};

class TelemetryImplementation final : public Telemetry {
  private:
    static constexpr size_t maxn{10};
    static constexpr int lenwd{45};
    static constexpr int lennd{15};
    using Clock_t = std::chrono::time_point<std::chrono::steady_clock>;
    using duration_t = std::chrono::nanoseconds;
    using display_duration_t = std::chrono::duration<double, std::milli>;

    class TimeDuration {
        enum class State { STOPPED, STARTED };
        Clock_t _start{}, _stop{};
        State _state{State::STOPPED};
        bool is_valid{false};
        size_t aggregation_count{0};
        duration_t _duration{0};

      public:
        TimeDuration() = default;
        ~TimeDuration() = default;
        void start() noexcept {
            _start = std::chrono::steady_clock::now();
            is_valid = _state == State::STOPPED;
            _state = State::STARTED;
        }
        void stop() noexcept {
            _stop = std::chrono::steady_clock::now();
            is_valid &= _state == State::STARTED;
            is_valid &= _start <= _stop;
            _state = State::STOPPED;
            if (is_valid) {
                ++aggregation_count;
                _duration += std::chrono::duration_cast<duration_t>(_stop - _start);
            }
        }
        void reset() noexcept {
            _state = State::STOPPED;
            is_valid = false;
            aggregation_count = 0;
            _duration = duration_t{0};
        }
        bool valid() const noexcept { return is_valid && _state == State::STOPPED; }
        size_t aggregated() const noexcept { return aggregation_count; }
        duration_t duration() const noexcept {
            return is_valid ? _duration : duration_t{0};
        }
    };

    std::string name;
    std::vector<std::pair<std::string, TimeDuration>> times;

  public:
    explicit TelemetryImplementation(const std::string &name) : Telemetry(), name(name) {
        times.reserve(maxn);
    }
    ~TelemetryImplementation() {
        try {
            print();
        } catch (...) {
            // Suppress all exceptions in the destructor
        }
    }
    const std::string &get_name() const { return name; }
    size_t start(std::string_view name) override {
        for (size_t i = 0; i < times.size(); ++i) {
            auto &time = times[i];
            if (time.first == name) {
                time.second.start();
                return i;
            }
        }
        times.emplace_back(name, TimeDuration{});
        times.back().second.start();
        return times.size() - 1;
    }
    void stop(std::string_view name) override {
        for (auto &time : times) {
            if (time.first == name) {
                time.second.stop();
                return;
            }
        }
    }
    void stop(size_t index) override { times.at(index).second.stop(); }
    void reset() override {
        for (auto &time : times) {
            time.second.reset();
        }
    }
    void print() const override {
        // For MT cases output may get mangled.
        da_logger::logger out;
        if (times.empty()) {
            out << "No telemetry data for \"" << name << "\"." << std::endl;
            return;
        }
        out << "Begin telemetry [" << name << " @ "
            << std::to_string(reinterpret_cast<std::uintptr_t>(this)) << "]\n";
        std::vector<std::string> sections;
        sections.reserve(times.size());
        int ilenwd{lenwd};
        for (const auto &time : times) {
            sections.emplace_back(pretty_print::prettify(time.first));
            const size_t name_length = sections.back().length();
            ilenwd = std::max(ilenwd, static_cast<int>(name_length));
        }
        out << "   " << std::left << std::setw(ilenwd) << "Section"
            << " " << std::setw(lennd) << std::right << "Duration (ms)"
            << "\n";
        for (size_t i = 0; i < times.size(); ++i) {
            const auto &time = times[i];
            const std::string &section = sections[i];
            const bool valid = time.second.valid();
            const size_t aggregated = time.second.aggregated();
            const display_duration_t duration =
                valid ? std::chrono::duration_cast<display_duration_t>(
                            time.second.duration())
                      : display_duration_t{0};

            out << "   " << std::left << std::setw(ilenwd) << section << " " << std::right
                << std::setw(lennd);
            if (valid) {
                out << std::fixed << std::setprecision(6) << duration.count() << " "
                    << (aggregated > 1 ? " (count: " + std::to_string(aggregated) + ")"
                                       : "");
            } else {
                out << " "
                    << "-invalid-";
            }
            out << "\n";
        }
        out << "End telemetry [" << name << "]\n";
        out << std::flush;
    }
};

[[nodiscard]] inline Telemetry &get_telemetry(const std::string &name) {
#ifdef TELEMETRY
    static thread_local TelemetryImplementation instance(name);
    if (instance.get_name() != name) {
        da_logger::logger out;
        out << "Warning: Telemetry instance name mismatch. Instance name is \""
            << instance.get_name() << "\"\n";
        out << "Additionally, get_telemetry should only be called once per thread."
            << std::endl;
    }
#else
    static TelemetryDummy instance(name);
#endif
    return instance;
}

class ScopedTimer final {
    Telemetry &t;
    size_t index;

  public:
    ScopedTimer(Telemetry &t, std::string_view name) : t(t), index(t.start(name)) {}
    ~ScopedTimer() { t.stop(index); }
};

namespace pretty_print {

// Try to be somewhat smart at shortening __PRETTY_FUNCTION__
[[nodiscard]] inline std::string prettify(std::string s) {

    if (s.length() <= 45) {
        return s; // short enough
    }

    const auto open = s.find('('), space = s.find(' ');
    // Drop "(args)", then the return type -- constructors have none, and their
    // first space, if any, comes after "(" rather than before it
    if (open != std::string::npos && space < open) {
        s.erase(open, s.rfind(')') - open + 1);
        s.erase(0, space + 1);
    }
    // signature (da_status = da_status_, da_int = int, ...); keep the first entry
    if (const auto semicolon = s.find(';'); semicolon != std::string::npos)
        s.replace(semicolon, std::string::npos, "]");
    // Shorted context namespace
    if (const auto pos = s.find("da_dynamic_dispatch_"); pos != std::string::npos)
        s.erase(pos, 20);
    // "[with T = float]" -> "[T = float]"
    if (const auto pos = s.find("[with "); pos != std::string::npos)
        s.erase(pos + 1, 5);
    return s;
}
} // namespace pretty_print

} // namespace telemetry
#endif
