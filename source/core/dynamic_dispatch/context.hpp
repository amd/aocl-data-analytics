/* ************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef context_HPP
#define context_HPP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include "Au/Cpuid/X86Cpu.hh"
#pragma GCC diagnostic pop
#include "aoclda_types.h"
#include "macros.h"
#include "telemetry.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#ifdef TELEMETRY
// Add macros to ease the invocation of the context's telemetry instrumentation
#define SCOPED_TIMER                                                                     \
    SCOPED_TIMER_HELPER(context::get_context()->get_telemetry(), __PRETTY_FUNCTION__)
#define SCOPED_TIMER_BLOCK(NAME)                                                         \
    SCOPED_TIMER_HELPER(context::get_context()->get_telemetry(), NAME)
#else
#define SCOPED_TIMER
#define SCOPED_TIMER_BLOCK(NAME)
#endif

enum dispatch_architecture {
    generic = 1,
    generic_avx512, // for non-Zen AVX512 enabled machines
    zen2,
    zen3,
    zen4,
    zen5,
    zen6,
    zen_new = zen6, // Replace with next generation; needs to be AVX512F/DQ/VL compatible
};

// ISA context preference
enum class context_isa_t {
    UNSET = 0, // Not set (default)
    GENERIC = 1,
    AVX2 = 2,
    AVX512F = 3,
    AVX512DQ = 4,
    AVX512VL = 5,
    AVX512IFMA = 6,
    AVX512CD = 7,
    AVX512BW = 8,
    AVX512_BF16 = 9,
    AVX512_VBMI = 10,
    AVX512_VNNI = 11,
    AVX512_VPOPCNTDQ = 12,
    AVX512_FP16 = 13,
    LENGTH = 14
};

/* This function is borrowed from aoclsparse and used to query the environment variable and return
it. In case of int, it converts the string into an integer. This function can only be used for int
and string type inputs.
*/
template <typename T> T env_get_var(const char *env, const T fallback) {
    T r_val;
    char *str;

    // Query the environment variable and store the result in str.
    str = getenv(env);
    // Set the return value based on the string obtained from getenv().
    if (str != nullptr) {
        if constexpr (std::is_same_v<T, da_int>) {
            // If there was no error, convert the char[] to an integer and
            // return that integer.
            r_val = static_cast<da_int>(strtol(str, nullptr, 10));
            return r_val;
        } else if constexpr (std::is_same_v<T, std::string>) {
            // If there was no error, convert the char[] to a std::string
            return std::string(str);
        }
    }

    // If there was an error, use the "fallback" as the return value.
    return fallback;
}

/* Singleton class containing details of the system for dynamic dispatch */
class context {
  private:
    dispatch_architecture local_arch = generic;

    // Set max_target_arch to the maximum Zen generation that was compiled, set by the ZNVER_MAX compile option
    // For a dynamic dispatch build, this will likely be zen3, zen4, zen5 or zen6 depending on the compiler
    // For a native or specific build this could be generic, generic_avx512, zen2, zen3, zen4, zen5 or zen6
    dispatch_architecture max_target_arch = ZNVER_MAX;

    bool cpuflags[static_cast<int>(context_isa_t::LENGTH)];

    static bool dynamic_dispatch_enabled();

    // Check environmental variable and update on-the-fly arch
    // if this is a dynamic dispatch build
    void check_env() {
        if (!is_dynamic)
            return;

        const std::string env_arch{env_get_var<std::string>("AOCL_DA_ARCH", "")};
        if (!env_arch.empty()) {
            // if requested arch is not build in the lib,
            // expect a da_status_arch_not_supported
            // Also, don't assign arch if the local arch can't handle the
            // request, avoid illegal cpu instructions
            if (env_arch == "zen1" || env_arch == "generic") {
                this->arch = generic;
            } else if (env_arch == "generic_avx512" &&
                       (local_arch == generic_avx512 || local_arch >= zen4)) {
                this->arch = generic_avx512;
            } else if (env_arch == "zen2" && local_arch >= zen2) {
                this->arch = zen2;
            } else if (env_arch == "zen3" && local_arch >= zen3) {
                this->arch = zen3;
            } else if (env_arch == "zen4" && local_arch >= zen4) {
                this->arch = zen4;
            } else if (env_arch == "zen5" && local_arch >= zen5) {
                this->arch = zen5;
            } else if (env_arch == "zen6" && local_arch >= zen6) {
                this->arch = zen6;
            }
            // don't change if "invalid"
        }
    }

    // Private constructor ensures direct calls to constructor not possible
    context() : is_dynamic(dynamic_dispatch_enabled()) {

        Au::X86Cpu Cpu = {0};
        Au::EUarch uarch = Cpu.getUarch();

        for (int f = 0; f < static_cast<int>(context_isa_t::LENGTH); ++f)
            cpuflags[f] = false;

        // Check for the list of flags supported
        this->cpuflags[static_cast<int>(context_isa_t::AVX2)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx2);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512F)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512f);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512DQ)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512dq);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512VL)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512vl);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512IFMA)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512ifma);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512CD)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512cd);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512BW)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512bw);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_BF16)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512_bf16);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_VBMI)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512vbmi);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_VNNI)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512_vnni);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_VPOPCNTDQ)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512_vpopcntdq);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_FP16)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512_fp16);

        has_avx512 = this->cpuflags[static_cast<int>(context_isa_t::AVX512F)] &&
                     this->cpuflags[static_cast<int>(context_isa_t::AVX512DQ)] &&
                     this->cpuflags[static_cast<int>(context_isa_t::AVX512VL)];

        has_avx512_bf16 =
            has_avx512 && this->cpuflags[static_cast<int>(context_isa_t::AVX512_BF16)];

        has_avx512_fp16 =
            has_avx512 && this->cpuflags[static_cast<int>(context_isa_t::AVX512_FP16)];

        has_avx512_vnni =
            has_avx512 && this->cpuflags[static_cast<int>(context_isa_t::AVX512_VNNI)];

        if (!is_dynamic) {
            arch = max_target_arch;
            return;
        }

        //LCOV_EXCL_START
        switch (uarch) {
        case Au::EUarch::Zen:
        case Au::EUarch::ZenPlus:
            local_arch = generic;
            break;
        case Au::EUarch::Zen2:
            local_arch = zen2;
            break;
        case Au::EUarch::Zen3:
            local_arch = zen3;
            break;
        case Au::EUarch::Zen4:
            local_arch = zen4;
            break;
        case Au::EUarch::Zen5:
            local_arch = zen5;
            break;
        case Au::EUarch::Zen6:
            local_arch = zen6;
            break;
        default:
            // Check to see if it is a new Zen model
            if (Cpu.isAMD()) {
                if (has_avx512) {
                    local_arch = zen_new; // Assume new model
                } else {
                    local_arch = zen3; // Fall-back to latest known avx2 model
                }
            } else if (has_avx512) {
                local_arch = generic_avx512;
            } else
                local_arch = generic; // Assume avx2 for non-AMD
        }

        /*
         * max target arch can be:
         * generic, generic_avx512, zen2, zen3, zen4, zen5, zen6
         * local arch can be:
         * generic, generic_avx512, zen2, zen3, zen4, zen5, zen6
         */

        // Deal with the special case of Zen 4+ with AVX-512 support disabled by treating it as Zen 3
        if ((local_arch >= zen4) && (has_avx512 == false)) {
            local_arch = zen3;
        }

        if (local_arch <= max_target_arch) {
            // For dynamic dispatch, there is a build that matches local arch
            arch = local_arch;
        } else if (max_target_arch == generic_avx512 && !has_avx512) {
            // Special case where the max target architecture has AVX512 but local CPU does not
            arch = generic;
        } else {
            // The maximum target architecture will work fine on this CPU
            arch = max_target_arch;
        }
        //LCOV_EXCL_STOP
        check_env(); // update arch if AOCL_DA_ARCH is set
    }

  public:
    // Note: Ensure direct calls to destructor are avoided with delete
    ~context() {}

    // Delete the copy constructor of the context class to ensure it's a singleton
    context(context &t) = delete;

    // Runtime CPU capabilities used for kernel selection
    bool has_avx512 = false;

    bool has_avx512_bf16 = false;

    bool has_avx512_fp16 = false;

    bool has_avx512_vnni = false;

    // True when multiple architecture namespaces were compiled for runtime dispatch
    const bool is_dynamic;

    dispatch_architecture arch = generic;

    // Delete the assignment operator of the context class to ensure it's a singleton
    void operator=(const context &) = delete;

    // Returns a reference to the global context
    static context *get_context();

    void refresh() {
        check_env(); // Apply a dynamic-build AOCL_DA_ARCH override if present
    }

    // Dictionary to store hidden / debug settings
    thread_local static std::unordered_map<std::string, std::string> hidden_settings;
    void set_hidden_setting(const std::string &key, const std::string &value) {
        // empty value implies delete the key
        if (value.empty()) {
            hidden_settings.erase(key);
            return;
        }
        try {
            hidden_settings[key] = value;
        } catch (const std::exception &) {
            // silently ignore...
        }
    }

    // access hidden settings
    static std::unordered_map<std::string, std::string> &get_hidden_settings();

    // Access telemetry instrumentation
    telemetry::Telemetry &get_telemetry() { return telemetry::get_telemetry("context"); }
};
#endif //context_HPP
