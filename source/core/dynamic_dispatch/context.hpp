/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef context_HPP
#define context_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "Au/Cpuid/X86Cpu.hh"
#include "aoclda_types.h"
#include "macros.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>

enum dispatch_architecture {
    generic = 0, // alias to zen1
    zen2 = 2,
    zen3 = 3,
    zen4 = 4,
    zen5 = 5,
    zen_new = zen5, // Replace with next generation; needs to be AVX512F/DQ/VL compatible
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
    LENGTH = 13
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
#if !defined(_WIN32)
    // On Windows we use Meyers' singleton class rather than mutex-based singleton as it interacts better with Python
    static context *global_obj;
    static std::mutex global_lock;
#endif

    dispatch_architecture local_arch = generic;

    // Set max_target_arch to the maximum Zen generation that was compiled, set by the ZNVER_MAX compile option
    dispatch_architecture max_target_arch = ZNVER_MAX;

    bool cpuflags[static_cast<int>(context_isa_t::LENGTH)];

    // Check environmental variable and update on-the-fly arch
    // if build is dynamic (generic is not aliased)
    void check_env() {
        if (max_target_arch != generic) {
            const std::string env_arch{env_get_var<std::string>("AOCL_DA_ARCH", "")};
            if (!env_arch.empty()) {
                // if requested arch is not build in the lib,
                // expect a da_status_arch_not_supported
                // Also, don't assign arch if the local arch can't handle the
                // request, avoid illegal cpu instructions
                if (env_arch == "zen1" || env_arch == "generic") {
                    this->arch = generic;
                } else if (env_arch == "zen2" && local_arch >= zen2) {
                    this->arch = zen2;
                } else if (env_arch == "zen3" && local_arch >= zen3) {
                    this->arch = zen3;
                } else if (env_arch == "zen4" && local_arch >= zen4) {
                    this->arch = zen4;
                } else if (env_arch == "zen5" && local_arch >= zen5) {
                    this->arch = zen5;
                }
                // don't change if "invalid"
            }
        }
    }

    // Private constructor ensures direct calls to constructor not possible
    context() {

        Au::X86Cpu Cpu = {0};
        Au::EUarch uarch = Cpu.getUarch();

        for (int f = 0; f < static_cast<int>(context_isa_t::LENGTH); ++f)
            cpuflags[f] = false;

        // Check for the list of flags supported
        // Note: Utils does not support BF16 flag lookup
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

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_VBMI)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512vbmi);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_VNNI)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512_4vnniw);

        this->cpuflags[static_cast<int>(context_isa_t::AVX512_VPOPCNTDQ)] =
            Cpu.hasFlag(Au::ECpuidFlag::avx512_vpopcntdq);

        bool has_avx512 = this->cpuflags[static_cast<int>(context_isa_t::AVX512F)] &&
                          this->cpuflags[static_cast<int>(context_isa_t::AVX512DQ)] &&
                          this->cpuflags[static_cast<int>(context_isa_t::AVX512VL)];

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
        default:
            // Check to see if it is a new Zen model
            if (Cpu.isAMD()) {
                if (has_avx512) {
                    local_arch = zen_new; // Assume new model
                } else {
                    local_arch = zen3; // Fall-back to latest known avx2 model
                }
            } else {
                local_arch = generic; // Assume avx2 for non-AMD
            }
        }

        if (local_arch <= max_target_arch) {
            // there is a build that matches local arch
            arch = local_arch;
        } else if (max_target_arch == generic) {
            // generic catches native/non-dynamic builds using the generic namespace
            arch = generic;
        } else if (has_avx512 && max_target_arch >= zen4) {
            // local arch seems to have AVX512* but is newer than the
            // library build, set to a AVX512 variant build
            arch = max_target_arch;
        } else {
            // set to last AVX2-only cpu
            arch = zen3;
        }
        check_env(); // update arch if AOCL_DA_ARCH is set
    }

  protected:
    // Ensure direct calls to destructor are avoided with delete
    ~context() {}

  public:
    // Delete the copy constructor of the context class to ensure it's a singleton
    context(context &t) = delete;

    dispatch_architecture arch = generic;

    // Delete the assignment operator of the context class to ensure it's a singleton
    void operator=(const context &) = delete;

    // Returns a reference to the global context
    static context *get_context();

    void refresh() {
        check_env(); // Check AOCL_DA_ARCH and update arch if needed
    }
};
#endif //context_HPP
