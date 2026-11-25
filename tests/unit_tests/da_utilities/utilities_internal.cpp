/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "context.hpp"
#include "da_logger.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <sstream>
#include <string>
#include <type_traits>

namespace {

using namespace std::string_literals;
using logger = da_logger::logger;

TEST(Logger, TypeTraits) {
    static_assert(std::is_same_v<logger, da_logger::logger>);
    static_assert(!std::is_copy_constructible_v<logger>);
    static_assert(!std::is_copy_assignable_v<logger>);
    static_assert(!std::is_move_constructible_v<logger>);
    static_assert(!std::is_move_assignable_v<logger>);
}

TEST(Logger, DefaultStream) {
    logger out;
    EXPECT_EQ(&out.stream(), &std::cout);
}

TEST(Logger, ThrowingStream) {
    std::ostringstream stream;
    stream.exceptions(std::ios::badbit);

    EXPECT_NO_THROW({ logger out(stream); });
}

TEST(Logger, Forwarding) {
    std::ostringstream stream;
    logger out(stream);

    logger &result = out << "value=" << 42 << ' ' << std::boolalpha << true << std::endl;

    EXPECT_EQ(&result, &out);
    EXPECT_EQ(stream.str(), "value=42 true\n");
}

TEST(Logger, StreamAndReset) {
    std::ostringstream stream;
    stream << std::hex << std::setprecision(7);
    const auto saved_flags = stream.flags();

    logger out(stream);
    out.stream() << std::dec << std::setprecision(2);
    EXPECT_NE(stream.flags(), saved_flags);
    EXPECT_EQ(stream.precision(), 2);

    out.reset();
    EXPECT_EQ(stream.flags(), saved_flags);
    EXPECT_EQ(stream.precision(), 7);
}

TEST(Logger, StateRestoration) {
    std::ostringstream stream;
    std::ostringstream tied_stream;
    stream << std::hex << std::showbase << std::setprecision(7) << std::setfill('#')
           << std::setw(11);
    stream.tie(&tied_stream);
    stream.exceptions(std::ios::failbit);

    const auto saved_flags = stream.flags();
    const auto saved_precision = stream.precision();
    const auto saved_width = stream.width();
    const auto saved_fill = stream.fill();
    const auto saved_exceptions = stream.exceptions();

    {
        logger out(stream);
        out << std::dec << std::noshowbase << std::setprecision(2) << std::setfill('*')
            << std::setw(6) << 42 << std::endl;
        out.stream().tie(nullptr);
        out.stream().exceptions(std::ios::badbit);
    }

    EXPECT_EQ(stream.flags(), saved_flags);
    EXPECT_EQ(stream.precision(), saved_precision);
    EXPECT_EQ(stream.width(), saved_width);
    EXPECT_EQ(stream.fill(), saved_fill);
    EXPECT_EQ(stream.tie(), &tied_stream);
    EXPECT_EQ(stream.exceptions(), saved_exceptions);
    EXPECT_EQ(stream.str(), "****42\n");
}

namespace dynamic_dispatch {
TEST(UtilitiesTest, DynamicDispatchEnv) {

    if (!context::get_context()->is_dynamic)
        GTEST_SKIP() << "AOCL_DA_ARCH only applies to dynamic dispatch builds.";

    char arch[64]{""};
    char ns[64]{""};
    da_int len{64};
    da_int tmp{0};

    // map of arch found
    std::map<std::string, bool> archs;
    std::vector<std::string> arch_list{"generic", "generic_avx512", "zen2", "zen3",
                                       "zen4",    "zen5",           "zen6"};

    // make sure its empty
    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", "", 1));

    // Get the architecture:
    std::string ok_arch{""};
    da_int count{0};
    std::cout << "This build supports the following archs:\n";
    std::cout << "  " << std::setw(12) << "requested"
              << " " << std::setw(12) << "set"
              << "    " << std::setw(30) << "namespace"
              << "   "
              << "notes" << std::endl;
    da_status status = da_status_success;
    for (auto &arch_env : arch_list) {
        arch[0] = '\0';
        ns[0] = '\0';
        EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", arch_env.c_str(), 1));
        status = da_get_arch_info(&len, arch, ns);
        switch (status) {
        case da_status_success:
            std::cout << "  " << std::setw(12) << arch_env << " " << std::setw(12) << arch
                      << "    " << std::setw(30) << ns;
            archs[arch_env] = std::string(ns) == "da_dynamic_dispatch_"s + arch_env;
            if (archs[arch_env]) {
                std::cout << "   AVAILABLE";
                ++count;
                ok_arch = arch_env;
                std::cout << " selected";
            } else {
                std::cout << "   (unavailable/ignored ns da_dynamic_dispatch_" << arch_env
                          << ")";
            }
            std::cout << std::endl;
            break;
        default:
            std::cout << "  " << arch_env << ":"
                      << "    "
                      << "unexpected error" << std::endl;
            break;
        }
    }

    // Get an available arch for the rest of the test
    EXPECT_NE(ok_arch, "");

    // Try to set the highest returned arch (for the previous loop)
    // and check the codepath (namespace)
    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", ok_arch.c_str(), 1));
    EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));
    if (count > 1) {
        // -DARCH=dynamic, so we know the highest Zen version string is of the form zenX
        std::string znver_flag = "znver" + std::string(1, arch[strlen(arch) - 1]);
        const char *version_char = da_get_version();
        std::size_t found = std::string(version_char).find(znver_flag);
        // Only check we have the zenX namespace if the znverX flag was used during compilation, else check the namespace is undefined
        if (found != std::string::npos) {
            EXPECT_EQ(std::string(ns), "da_dynamic_dispatch_"s + std::string(arch));
        } else {
            EXPECT_EQ(std::string(ns), "<arch not supported>"s);
        }
    } else {
        // count == 1
        // 1. -DARCH=zenX uses zenX
        // 2. -DARCH=native
        // 3. -DARCH=generic or generic_avx512
        // In both cases arch and ns should match
        EXPECT_EQ(std::string(ns), "da_dynamic_dispatch_"s + std::string(arch));
    }

    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", ok_arch.c_str(), 1));
    EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));

    // try an invalid arch and test
    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", "invalid_arch", 1));
    EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));
    EXPECT_STREQ(arch, ok_arch.c_str());

    // check len
    EXPECT_EQ(da_status_invalid_array_dimension, da_get_arch_info(&tmp, arch, ns));
    EXPECT_GT(tmp, 0);

    // arch needs to be zen* or generic
    EXPECT_THAT(arch,
                testing::AnyOf(testing::StartsWith("zen"), testing::StrCaseEq("generic"),
                               testing::StrCaseEq("generic_avx512")));
    // arch needs to match with ns
    std::string ns2arch{"da_dynamic_dispatch_"s + std::string(arch)};
    EXPECT_STREQ(ns2arch.c_str(), ns);

    if (archs["generic"]) {
        // test generic <-> zen1 alias
        EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", "zen1", 1));
        EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));
        EXPECT_EQ(arch, "generic"s);
    } else {
        std::cout << "SKIP Test: zen1/generic alias test cannot be performed on this node"
                  << std::endl;
    }
}

TEST(UtilitiesTest, FixedDispatchIgnoresEnv) {
    const auto *ctx = context::get_context();
    if (ctx->is_dynamic)
        GTEST_SKIP() << "Fixed dispatch behavior only applies to single-target builds.";

    char arch[64]{""};
    char ns[64]{""};
    da_int len{64};

    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", "", 1));
    EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));
    const std::string expected_arch{arch};
    const std::string expected_ns{ns};
    const char *requested_arch = expected_arch == "generic" ? "zen6" : "generic";

    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", requested_arch, 1));
    EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));
    EXPECT_EQ(arch, expected_arch);
    EXPECT_EQ(ns, expected_ns);
    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", "", 1));
}

// Try to set an architecture that is newer than the local cpu.
// This test is only meaningful for dynamic dispatch builds where zen4
// code was actually compiled into the library.
TEST(UtilitiesTest, DynamicDispatchTryArch) {

    if (!context::get_context()->is_dynamic)
        GTEST_SKIP() << "Architecture compatibility is checked only for dynamic builds.";

    const char *version = da_get_version();
    std::string version_str(version);

    char arch[64]{""};
    char ns[64]{""};
    da_int len{64};

    // make sure its empty
    EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", "", 1));
    // get the architecture
    EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));
    // save
    std::string a{arch};

    if (a == "zen2"s || a == "zen3"s) {
        // Only request zen4 if this build actually includes znver4 objects.
        if (version_str.find("znver4") != std::string::npos) {
            EXPECT_EQ(0, da_test::da_setenv("AOCL_DA_ARCH", "zen4", 1));
            EXPECT_EQ(da_status_success, da_get_arch_info(&len, arch, ns));
            // Local arch is older, so request should be ignored.
            EXPECT_EQ(std::string(arch), a);
        }
    }
}

} // namespace dynamic_dispatch

// setter / getter for hidden context setting, this is used to pass
// hidden options to the solvers, i.e. to force a certain ISA and
// bypass auto-tuning for full code coverage, benchmarks, etc.
TEST(UtilitiesTest, ContextHiddenSettings) {
    // Exersize getter and setter
    std::string skey = "Unit   Test";
    const char charvalue[]{" Foo   Bar"};
    char charans[100]{""};
    EXPECT_EQ(da_debug_get(nullptr, -1, nullptr), da_status_success);
    EXPECT_EQ(da_debug_set(" unit Test ", "Bar Foo BAZ "), da_status_success);
    EXPECT_EQ(da_debug_get(skey.c_str(), 100, charans), da_status_success);
    EXPECT_THAT(std::string(charans), testing::StrCaseEq("bar foo baz"));
    EXPECT_EQ(da_debug_set(skey.c_str(), charvalue), da_status_success);
    EXPECT_EQ(da_debug_get(skey.c_str(), 100, charans), da_status_success);
    EXPECT_THAT(std::string(charans), testing::StrCaseEq("foo bar"));
    EXPECT_EQ(da_debug_set(" new key", " new value "), da_status_success);
    EXPECT_EQ(da_debug_get(nullptr, -1, nullptr), da_status_success);

    // erase "Unit Test" key
    EXPECT_EQ(da_debug_set(" unit Test ", ""), da_status_success);
    EXPECT_EQ(da_debug_get(" unit Test ", 100, charans), da_status_option_not_found);
    EXPECT_EQ(da_debug_get("new key", 100, charans), da_status_success);
    EXPECT_THAT(std::string(charans), testing::StrCaseEq("new value"));

    // Error returns
    EXPECT_EQ(da_debug_set("    ", "Foo"), da_status_invalid_input);
    EXPECT_EQ(da_debug_set(nullptr, "Foo"), da_status_invalid_input);
    EXPECT_EQ(da_debug_set(charvalue, nullptr), da_status_invalid_input);
    EXPECT_EQ(da_debug_get(skey.c_str(), 10, charans), da_status_invalid_input);
    EXPECT_EQ(da_debug_get(skey.c_str(), -1, charans), da_status_invalid_input);
    EXPECT_EQ(da_debug_get(nullptr, -1, charans), da_status_success);
    EXPECT_EQ(da_debug_get(charvalue, -1, nullptr), da_status_success);

    EXPECT_EQ(da_debug_get("NonExisting!", 100, charans), da_status_option_not_found);
}

TEST(UtilitiesTest, intInfo) {
    std::string int_lib = INT_LIB;
    std::string expected_int_str;
    if (int_lib == "LP64")
        expected_int_str = "32";
    else if (int_lib == "ILP64")
        expected_int_str = "64";
    else
        expected_int_str = "?";

    // Valid call
    size_t len = 3;
    char int_type[3];
    EXPECT_EQ(da_get_int_info(&len, int_type), da_status_success);
    EXPECT_EQ(expected_int_str, std::string(int_type));

    // wrong length
    len = 0;
    EXPECT_EQ(da_get_int_info(&len, int_type), da_status_invalid_array_dimension);
    // Second call has valid length
    EXPECT_EQ(da_get_int_info(&len, int_type), da_status_success);
    EXPECT_EQ(expected_int_str, std::string(int_type));

    // wrong pointer
    len = 3;
    EXPECT_EQ(da_get_int_info(&len, nullptr), da_status_invalid_input);
}

} // namespace