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

#include "context.hpp"
#include "macros.h"
#include "telemetry.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

namespace {

std::string capture(const telemetry::Telemetry &telemetry) {
    testing::internal::CaptureStdout();
    telemetry.print();
    return testing::internal::GetCapturedStdout();
}

std::string section_line(const std::string &output, const std::string &section) {
    const auto section_pos = output.find(section);
    if (section_pos == std::string::npos)
        return {};

    const auto line_end = output.find('\n', section_pos);
    return output.substr(section_pos, line_end - section_pos);
}

// Returns the duration reported for a section, or -1 if it is absent or invalid
double section_duration(const std::string &output, const std::string &section) {
    const auto line = section_line(output, section);
    if (line.empty() || line.find("-invalid-") != std::string::npos)
        return -1;
    try {
        return std::stod(line.substr(section.size()));
    } catch (const std::exception &) {
        return -1;
    }
}

void time_scoped_section() {
    SCOPED_TIMER;
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
}

void time_scoped_block() {
    SCOPED_TIMER_BLOCK("Test/ScopedBlock");
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
}

} // namespace

TEST(telemetry, type_traits) {
    EXPECT_TRUE((std::is_constructible_v<telemetry::TelemetryImplementation,
                                         const std::string &>));
    EXPECT_TRUE(
        (std::is_constructible_v<telemetry::TelemetryImplementation, const char *>));
    EXPECT_FALSE(std::is_default_constructible_v<telemetry::TelemetryImplementation>);
    EXPECT_FALSE(std::is_copy_constructible_v<telemetry::TelemetryImplementation>);
    EXPECT_FALSE(std::is_copy_assignable_v<telemetry::TelemetryImplementation>);
    EXPECT_FALSE(std::is_move_constructible_v<telemetry::TelemetryImplementation>);
    EXPECT_FALSE(std::is_move_assignable_v<telemetry::TelemetryImplementation>);
    EXPECT_FALSE(
        (std::is_convertible_v<std::string, telemetry::TelemetryImplementation>));
}

TEST(telemetry, empty_output) {
    telemetry::TelemetryImplementation telemetry("empty");

    EXPECT_EQ(capture(telemetry), "No telemetry data for \"empty\".\n");
}

TEST(telemetry, timer_states) {
    telemetry::TelemetryImplementation telemetry("transitions");
    telemetry.start("valid");
    telemetry.stop("valid");
    telemetry.start("running");
    telemetry.start("duplicate-start");
    telemetry.start("duplicate-start");
    telemetry.stop("duplicate-start");
    telemetry.start("duplicate-stop");
    telemetry.stop("duplicate-stop");
    telemetry.stop("duplicate-stop");
    telemetry.stop("unknown");

    const auto output = capture(telemetry);
    EXPECT_THAT(section_line(output, "valid"),
                ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_THAT(section_line(output, "running"), ::testing::HasSubstr("-invalid-"));
    EXPECT_THAT(section_line(output, "duplicate-start"),
                ::testing::HasSubstr("-invalid-"));
    EXPECT_THAT(section_line(output, "duplicate-stop"),
                ::testing::HasSubstr("-invalid-"));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("unknown")));
}

TEST(telemetry, reset_reuse) {
    telemetry::TelemetryImplementation telemetry("reset");
    telemetry.start("reused");
    telemetry.stop("reused");
    telemetry.start("left-reset");
    telemetry.stop("left-reset");
    telemetry.reset();
    telemetry.start("reused");
    telemetry.stop("reused");

    const auto output = capture(telemetry);
    EXPECT_THAT(section_line(output, "reused"),
                ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_THAT(section_line(output, "left-reset"), ::testing::HasSubstr("-invalid-"));
    EXPECT_EQ(output.find("reused"), output.rfind("reused"));
}

TEST(telemetry, duration_format) {
#ifdef _WIN32
    GTEST_SKIP() << "Test skipped in Windows";
#endif
    telemetry::TelemetryImplementation telemetry("duration");
    telemetry.start("timed");
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
    telemetry.stop("timed");

    EXPECT_THAT(section_line(capture(telemetry), "timed"),
                ::testing::MatchesRegex("timed +[0-9]+\\.[0-9]{6} *"));
}

TEST(telemetry, duration_units) {
    telemetry::TelemetryImplementation telemetry("units");
    const auto start = std::chrono::steady_clock::now();
    telemetry.start("timed");
    const auto inner_start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::microseconds{2250});
    const auto inner_stop = std::chrono::steady_clock::now();
    telemetry.stop("timed");
    const auto stop = std::chrono::steady_clock::now();

    const auto output = capture(telemetry);
    EXPECT_THAT(output, ::testing::HasSubstr("Duration (ms)"));
    const auto printed_ms = section_duration(output, "timed");
    ASSERT_GT(printed_ms, 0);
    // Bracket the measured interval to allow scheduling delays without masking unit errors.
    EXPECT_GE(
        printed_ms,
        (std::chrono::duration<double, std::milli>(inner_stop - inner_start).count()) -
            0.000001);
    EXPECT_LE(printed_ms,
              (std::chrono::duration<double, std::milli>(stop - start).count()) +
                  0.000001);
}

TEST(telemetry, sample_count) {
    telemetry::TelemetryImplementation telemetry("aggregation");
    for (int samples = 1; samples <= 12; ++samples) {
        SCOPED_TRACE(samples);
        telemetry.start("repeated");
        telemetry.stop("repeated");

        const auto output = capture(telemetry);
        const auto line = section_line(output, "repeated");
        EXPECT_THAT(line, ::testing::Not(::testing::HasSubstr("-invalid-")));
        if (samples == 1) {
            EXPECT_THAT(line, ::testing::Not(::testing::HasSubstr("(count:")));
        } else {
            EXPECT_THAT(line,
                        ::testing::HasSubstr("(count: " + std::to_string(samples) + ")"));
        }
        EXPECT_EQ(capture(telemetry), output);
    }
}

TEST(telemetry, sample_reset) {
    telemetry::TelemetryImplementation telemetry("reset-aggregation");
    telemetry.start("repeated");
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
    telemetry.stop("repeated");
    telemetry.start("repeated");
    telemetry.stop("repeated");
    telemetry.reset();
    const auto reset_output = capture(telemetry);
    EXPECT_THAT(section_line(reset_output, "repeated"),
                ::testing::HasSubstr("-invalid-"));
    EXPECT_THAT(reset_output, ::testing::Not(::testing::HasSubstr("(count:")));
    telemetry.start("repeated");
    telemetry.stop("repeated");

    EXPECT_THAT(section_line(capture(telemetry), "repeated"),
                ::testing::Not(::testing::HasSubstr("(count:")));
    telemetry.start("repeated");
    telemetry.stop("repeated");
    EXPECT_THAT(section_line(capture(telemetry), "repeated"),
                ::testing::HasSubstr("(count: 2)"));
}

TEST(telemetry, invalid_samples) {
    telemetry::TelemetryImplementation telemetry("invalid-samples");
    telemetry.stop("unknown");
    for (int i = 0; i < 2; ++i) {
        telemetry.start("repeated");
        telemetry.stop("repeated");
    }

    telemetry.start("repeated");
    const auto running_output = capture(telemetry);
    EXPECT_THAT(section_line(running_output, "repeated"),
                ::testing::HasSubstr("-invalid-"));
    EXPECT_THAT(running_output, ::testing::Not(::testing::HasSubstr("(count:")));
    telemetry.start("repeated");
    telemetry.stop("repeated");
    EXPECT_THAT(section_line(capture(telemetry), "repeated"),
                ::testing::HasSubstr("-invalid-"));

    telemetry.start("repeated");
    telemetry.stop("repeated");
    EXPECT_THAT(section_line(capture(telemetry), "repeated"),
                ::testing::HasSubstr("(count: 3)"));

    telemetry.stop("repeated");
    EXPECT_THAT(section_line(capture(telemetry), "repeated"),
                ::testing::HasSubstr("-invalid-"));
    telemetry.start("repeated");
    telemetry.stop("repeated");
    const auto output = capture(telemetry);
    EXPECT_THAT(section_line(output, "repeated"), ::testing::HasSubstr("(count: 4)"));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("unknown")));
}

TEST(telemetry, section_counts) {
    telemetry::TelemetryImplementation telemetry("section-counts");
    for (int i = 0; i < 3; ++i) {
        telemetry.start("first");
        telemetry.stop("first");
        if (i < 2) {
            telemetry.start("second");
            telemetry.stop("second");
        }
    }
    const auto output = capture(telemetry);
    EXPECT_THAT(section_line(output, "first"), ::testing::HasSubstr("(count: 3)"));
    EXPECT_THAT(section_line(output, "second"), ::testing::HasSubstr("(count: 2)"));
}

TEST(telemetry, scoped_samples) {
    telemetry::TelemetryImplementation telemetry("scoped-samples");
    for (int i = 0; i < 3; ++i) {
        telemetry::ScopedTimer timer(telemetry, "repeated-scope");
    }
    EXPECT_THAT(section_line(capture(telemetry), "repeated-scope"),
                ::testing::HasSubstr("(count: 3)"));
}

TEST(telemetry, section_order) {
    telemetry::TelemetryImplementation telemetry("capacity");
    for (int i = 0; i < 12; i++) {
        const auto section = "section-" + std::to_string(i);
        telemetry.start(section);
        telemetry.stop(section);
    }

    const auto output = capture(telemetry);
    std::string::size_type previous_pos = 0;
    for (int i = 0; i < 12; i++) {
        const auto section = "section-" + std::to_string(i);
        const auto section_pos = output.find(section);
        ASSERT_NE(section_pos, std::string::npos);
        EXPECT_GE(section_pos, previous_pos);
        previous_pos = section_pos;
    }
}

TEST(telemetry, dummy_nop) {
    telemetry::TelemetryDummy telemetry("null");
    const auto index = telemetry.start("section");
    telemetry.stop("section");
    telemetry.stop(index);
    {
        telemetry::ScopedTimer timer(telemetry, std::string(80, 'x'));
    }
    telemetry.reset();

    EXPECT_TRUE(capture(telemetry).empty());
}

TEST(telemetry, stream_restoration) {
    telemetry::TelemetryImplementation telemetry("flags");
    telemetry.start("section");
    telemetry.stop("section");
    const auto original_flags = std::cout.flags();
    const auto original_precision = std::cout.precision();
    std::cout << std::left;
    const auto expected_flags = std::cout.flags();

    capture(telemetry);
    const auto actual_flags = std::cout.flags();
    const auto actual_precision = std::cout.precision();
    std::cout.flags(original_flags);
    std::cout.precision(original_precision);

    EXPECT_EQ(actual_flags, expected_flags);
    EXPECT_EQ(actual_precision, original_precision);
}

TEST(telemetry, prettify) {
    using telemetry::pretty_print::prettify;

    // Signatures that already fit the column are handed back untouched
    EXPECT_EQ(prettify("void ns::f(int)"), "void ns::f(int)");

    // Return type, argument list and dispatch prefix all go
    EXPECT_EQ(prettify("void da_dynamic_dispatch_zen3::da_utils::helper(int)"),
              "zen3::da_utils::helper");

    // GCC's expansion for linear_model<T>::fit: the typedef substitutions are
    // dropped but the template argument survives, so float and double stay apart
    const auto fit = [](const char *type) {
        return prettify("da_status da_dynamic_dispatch_zen5::da_linmod::linear_model<T>"
                        "::fit(da_int, const T*) [with T = " +
                        std::string(type) + "; da_status = da_status_; da_int = int]");
    };
    EXPECT_EQ(fit("float"), "zen5::da_linmod::linear_model<T>::fit [T = float]");
    EXPECT_EQ(fit("double"), "zen5::da_linmod::linear_model<T>::fit [T = double]");
}

TEST(telemetry, pretty_output) {
    telemetry::TelemetryImplementation telemetry("pretty-output");
    const std::string original =
        "void da_dynamic_dispatch_zen3::da_utils::long_helper_function_name(int)";
    const std::string pretty = "zen3::da_utils::long_helper_function_name";
    telemetry.start(original);
    telemetry.stop(original);
    telemetry.start("brief");
    telemetry.stop("brief");

    const auto output = capture(telemetry);
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr(original)));
    const auto pretty_line = section_line(output, pretty);
    const auto brief_line = section_line(output, "brief");
    ASSERT_FALSE(pretty_line.empty());
    ASSERT_FALSE(brief_line.empty());
    ASSERT_NE(pretty_line.find('.'), std::string::npos);
    EXPECT_EQ(pretty_line.find('.'), brief_line.find('.'));
    EXPECT_EQ(capture(telemetry), output);

    telemetry.start(original);
    telemetry.stop(original);
    const auto repeated_output = capture(telemetry);
    EXPECT_THAT(section_line(repeated_output, pretty),
                ::testing::HasSubstr("(count: 2)"));
    EXPECT_EQ(repeated_output.find(pretty), repeated_output.rfind(pretty));
}

TEST(telemetry, function_macro) {
    telemetry::Telemetry &telemetry = context::get_context()->get_telemetry();
    time_scoped_section();

    const auto output = capture(telemetry);
#ifdef TELEMETRY
    const auto line = section_line(output, "time_scoped_section");
    // SCOPED_TIMER names the section after the enclosing function
    EXPECT_FALSE(line.empty());
    EXPECT_THAT(line, ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_THAT(line, ::testing::Not(::testing::HasSubstr("(count:")));
#else
    // Without -DTELEMETRY the macro expands to nothing and the context holds a
    // no-op instance, so nothing is recorded and nothing is printed
    EXPECT_TRUE(output.empty());
#endif
}

TEST(telemetry, block_macro) {
    telemetry::Telemetry &telemetry = context::get_context()->get_telemetry();
    time_scoped_block();

    const auto output = capture(telemetry);
#ifdef TELEMETRY
    const auto line = section_line(output, "Test/ScopedBlock");
    // SCOPED_TIMER_BLOCK uses the name it is given rather than the function name
    EXPECT_FALSE(line.empty());
    EXPECT_THAT(line, ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("time_scoped_block")));
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST(telemetry, scope_exit) {
    telemetry::TelemetryImplementation telemetry("object");
    {
        telemetry::ScopedTimer timer(telemetry, "timed-scope");
        // The section is still running inside the scope
        EXPECT_THAT(section_line(capture(telemetry), "timed-scope"),
                    ::testing::HasSubstr("-invalid-"));
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }

    const auto output = capture(telemetry);
    EXPECT_THAT(section_line(output, "timed-scope"),
                ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_GE(section_duration(output, "timed-scope"), 5);
}

TEST(telemetry, nested_scopes) {
    telemetry::TelemetryImplementation telemetry("nested");
    {
        telemetry::ScopedTimer outer(telemetry, "outer-scope");
        {
            telemetry::ScopedTimer inner(telemetry, "inner-scope");
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }
    }

    const auto output = capture(telemetry);
    EXPECT_THAT(section_line(output, "outer-scope"),
                ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_THAT(section_line(output, "inner-scope"),
                ::testing::Not(::testing::HasSubstr("-invalid-")));
    // The outer section encloses the inner one, so it cannot be shorter
    EXPECT_GE(section_duration(output, "outer-scope"),
              section_duration(output, "inner-scope"));
    // Sections are reported in the order they were first started
    EXPECT_LT(output.find("outer-scope"), output.find("inner-scope"));
}

TEST(telemetry, exception_unwind) {
    telemetry::TelemetryImplementation telemetry("unwind");
    try {
        telemetry::ScopedTimer timer(telemetry, "thrown");
        throw std::runtime_error("telemetry test");
    } catch (const std::runtime_error &) {
        // The timer must have been stopped while the stack unwound
    }

    EXPECT_THAT(section_line(capture(telemetry), "thrown"),
                ::testing::Not(::testing::HasSubstr("-invalid-")));
}

TEST(telemetry, polymorphic_dispatch) {
    telemetry::TelemetryImplementation implementation("polymorphic");
    telemetry::TelemetryDummy dummy("polymorphic");
    telemetry::Telemetry *instances[]{&implementation, &dummy};

    for (auto *instance : instances) {
        instance->start("dispatch");
        instance->stop("dispatch");
        instance->reset();
        instance->start("dispatch");
        instance->stop("dispatch");
    }

    EXPECT_THAT(capture(implementation), ::testing::HasSubstr("dispatch"));
    EXPECT_TRUE(capture(dummy).empty());
}

TEST(telemetry, duration_sum) {
    telemetry::TelemetryImplementation telemetry("accumulate");
    telemetry.start("summed");
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    telemetry.stop("summed");
    const auto after_first = section_duration(capture(telemetry), "summed");

    telemetry.start("summed");
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    telemetry.stop("summed");
    const auto after_second = section_duration(capture(telemetry), "summed");

    EXPECT_GE(after_first, 5);
    EXPECT_GE(after_second, after_first + 5);
}

TEST(telemetry, sub_millisecond) {
    telemetry::TelemetryImplementation telemetry("sub-ms");

    // A single interval well below the millisecond display unit
    const auto before = std::chrono::steady_clock::now();
    telemetry.start("brief");
    std::this_thread::sleep_for(std::chrono::microseconds{250});
    telemetry.stop("brief");
    const auto after = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration<double, std::milli>(after - before).count();

    const auto printed = section_duration(capture(telemetry), "brief");
    // Accumulating in whole milliseconds would floor this to zero
    EXPECT_GE(printed, 0.25 - 1e-6);
    EXPECT_LE(printed, elapsed + 1e-6);
    // Confirm the interval really was sub-millisecond. A loaded machine can
    // overshoot the sleep, so only assert it when the measurement agrees.
    if (elapsed < 1.0) {
        EXPECT_LT(printed, 1.0);
    }
}

TEST(telemetry, sub_millisecond_accumulation) {
    constexpr int repeats = 53;
    telemetry::TelemetryImplementation telemetry("sub-ms-sum");

    const auto before = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; i++) {
        telemetry.start("brief");
        telemetry.stop("brief");
    }
    const auto after = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration<double, std::milli>(after - before).count();

    const auto output = capture(telemetry);
    const auto printed = section_duration(output, "brief");
    // Every interval here is a small fraction of a millisecond. Truncating each
    // one to the display unit would discard all 53 and report nothing.
    EXPECT_GT(printed, 0.0);
    EXPECT_LE(printed, elapsed + 1e-6);
    EXPECT_THAT(section_line(output, "brief"),
                ::testing::HasSubstr("(count: " + std::to_string(repeats) + ")"));
}

TEST(telemetry, section_growth) {
    telemetry::TelemetryImplementation telemetry("growth");
    telemetry.start("grow-first");
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    telemetry.stop("grow-first");
    telemetry.start("grow-first");
    telemetry.stop("grow-first");

    // Push well past the reserved capacity so the underlying storage reallocates
    for (int i = 0; i < 32; i++) {
        const auto section = "grow-filler-" + std::to_string(i);
        telemetry.start(section);
        telemetry.stop(section);
    }

    const auto output = capture(telemetry);
    EXPECT_THAT(section_line(output, "grow-first"),
                ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_GE(section_duration(output, "grow-first"), 5);
    EXPECT_THAT(section_line(output, "grow-first"), ::testing::HasSubstr("(count: 2)"));
    EXPECT_THAT(output, ::testing::HasSubstr("grow-filler-31"));
}

TEST(telemetry, scoped_growth) {
    telemetry::TelemetryImplementation telemetry("scoped-growth");
    const std::string name(80, 'x');
    {
        telemetry::ScopedTimer outer(telemetry, std::string(80, 'x'));
        for (int i = 0; i < 32; ++i) {
            telemetry::ScopedTimer inner(telemetry, "temporary-" + std::to_string(i));
        }
    }
    const auto output = capture(telemetry);
    EXPECT_THAT(output, ::testing::HasSubstr(name));
    EXPECT_THAT(output, ::testing::HasSubstr("temporary-31"));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("-invalid-")));
}

TEST(telemetry, name_mutation) {
    telemetry::TelemetryImplementation telemetry("mutation");
    const std::string original(80, 'x');
    {
        std::string name = original;
        telemetry::ScopedTimer timer(telemetry, name);
        name.assign(200, 'y');
    }
    const auto output = capture(telemetry);
    EXPECT_THAT(output, ::testing::HasSubstr(original));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("-invalid-")));
}

TEST(telemetry, scoped_reset) {
    telemetry::TelemetryImplementation telemetry("scoped-reset");
    const std::string name(80, 'x');
    {
        telemetry::ScopedTimer timer(telemetry, name);
        telemetry.reset();
    }
    EXPECT_THAT(capture(telemetry), ::testing::HasSubstr("-invalid-"));
    {
        telemetry::ScopedTimer timer(telemetry, name);
    }
    const auto output = capture(telemetry);
    EXPECT_THAT(output, ::testing::HasSubstr(name));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_EQ(output.find(name), output.rfind(name));
}

TEST(telemetry, index_reuse) {
    telemetry::TelemetryImplementation telemetry("indices");
    telemetry::Telemetry &base = telemetry;
    const std::string storage = "section-not-part-of-name";
    const std::string_view name(storage.data(), 7);
    const auto index = base.start(name);
    base.stop(index);
    EXPECT_EQ(base.start(std::string("section")), index);
    base.stop(name);
    EXPECT_THAT(section_line(capture(telemetry), "section"),
                ::testing::HasSubstr("(count: 2)"));
    base.reset();
    EXPECT_EQ(base.start(name), index);
    base.stop(index);
    const auto output = capture(telemetry);
    EXPECT_THAT(output, ::testing::HasSubstr("section"));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("-not-part-of-name")));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("-invalid-")));
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("(count:")));
}

TEST(telemetry, invalid_index) {
    telemetry::TelemetryImplementation telemetry("invalid-index");
    EXPECT_THROW(telemetry.stop(0), std::out_of_range);
    const auto index = telemetry.start("section");
    EXPECT_THROW(telemetry.stop(index + 1), std::out_of_range);
    telemetry.stop(index);
    EXPECT_THAT(capture(telemetry), ::testing::Not(::testing::HasSubstr("-invalid-")));
}

TEST(telemetry, long_name) {
    const std::string long_name{std::string(61, 'x').insert(5, ": y")};
    const std::string short_name{std::string(21, 'x').insert(5, ": y")};
    // Actual __PRETTY_FUNCTION__ values from linear_model<float>::fit (LP64, Zen 5).
    const std::string gcc_name{
        "da_status da_dynamic_dispatch_zen5::da_linmod::linear_model<T>::fit(da_int, "
        "const T*) [with T = float; da_status = da_status_; da_int = int]"};
    const std::string clang_name{
        "da_status da_dynamic_dispatch_zen5::da_linmod::linear_model<float>::fit(da_int, "
        "const T *) [T = float]"};
    telemetry::TelemetryImplementation telemetry("long-name");
    telemetry.start(short_name);
    telemetry.start(long_name);
    telemetry.start(gcc_name);
    telemetry.start(clang_name);
    telemetry.stop(clang_name);
    telemetry.stop(gcc_name);
    telemetry.stop(long_name);
    telemetry.stop(short_name);
    const auto output = capture(telemetry);
    // The column width is a minimum, so a long name must not be truncated
    EXPECT_THAT(output, ::testing::HasSubstr(long_name));
    EXPECT_THAT(output, ::testing::HasSubstr(short_name));
    EXPECT_THAT(output, ::testing::HasSubstr(
                            "zen5::da_linmod::linear_model<T>::fit [T = float]"));
    EXPECT_THAT(output, ::testing::HasSubstr(
                            "zen5::da_linmod::linear_model<float>::fit [T = float]"));
}

TEST(telemetry, unstarted_stop) {
    telemetry::TelemetryImplementation telemetry("no-sections");
    telemetry.reset();
    telemetry.stop("never-started");

    // Stopping an unknown section must not register it
    EXPECT_EQ(capture(telemetry), "No telemetry data for \"no-sections\".\n");
}

TEST(telemetry, destructor_output) {
    testing::internal::CaptureStdout();
    {
        telemetry::TelemetryImplementation telemetry("destructor");
        telemetry.start("section");
        telemetry.stop("section");
    }
    const auto output = testing::internal::GetCapturedStdout();

    EXPECT_THAT(output, ::testing::HasSubstr("Begin telemetry [destructor"));
    EXPECT_THAT(output, ::testing::HasSubstr("End telemetry [destructor]"));
}

TEST(telemetry, context_reuse) {
    // get_telemetry hands back the same instance on every call from a thread,
    // and that is the instance the context exposes
    telemetry::Telemetry &first = telemetry::get_telemetry("context");
    telemetry::Telemetry &second = telemetry::get_telemetry("context");

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(&first, &context::get_context()->get_telemetry());
}

TEST(telemetry, thread_instances) {
    telemetry::Telemetry *main_instance = &telemetry::get_telemetry("main");
    bool first_instance{false};
    bool second_instance{false};
    main_instance->start("main-section");
    std::thread first_thread([&] {
        telemetry::Telemetry &t1 = telemetry::get_telemetry("first");
        t1.print();
        first_instance = &t1 == main_instance;
    });
    std::thread second_thread([&] {
        telemetry::Telemetry &t2 = telemetry::get_telemetry("second");
        t2.print();
        second_instance = &t2 == main_instance;
    });
    main_instance->stop("main-section");
    first_thread.join();
    main_instance->print();
    second_thread.join();

#ifdef TELEMETRY
    EXPECT_FALSE(first_instance);
    EXPECT_FALSE(second_instance);
#else
    EXPECT_TRUE(first_instance);
    EXPECT_TRUE(second_instance);
#endif
}
