/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef MACROS_H
#define MACROS_H

// Auxiliary macro for token concatenation
#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

// Define ARCHITECTURE
#if defined ARCH_znver
#define ARCHITECTURE CONCAT(zen, ARCH_znver)
#elif defined ARCH_generic
#define ARCHITECTURE generic
#endif

// Define a macro that creates the dynamic_dispatch_ARCHITECTURE namespace
#define ARCH CONCAT(da_dynamic_dispatch_, ARCHITECTURE)

// Macros to wrap calls in public APIs to dynamic dispatch
// SFINAE and template meta-programming could be an option to achieve this, but this is cleaner and
// more readable since 'code' is pretty arbitrary and not just a single function call
#if defined(generic_AVAILABLE)
#define DYNAMIC_DISPATCH_GENERIC(err_buffer, code)                                       \
    {                                                                                    \
        using namespace da_dynamic_dispatch_generic;                                     \
        code;                                                                            \
    }
#else
#define DYNAMIC_DISPATCH_GENERIC(err_buffer, code)                                       \
    return da_error_bypass(err_buffer, da_status_arch_not_supported,                     \
                           "Generic architecture not supported.");
#endif

#if defined(znver2_AVAILABLE)
#define DYNAMIC_DISPATCH_ZEN2(err_buffer, code)                                          \
    {                                                                                    \
        using namespace da_dynamic_dispatch_zen2;                                        \
        code;                                                                            \
    }
#else
#define DYNAMIC_DISPATCH_ZEN2(err_buffer, code)                                          \
    return da_error_bypass(err_buffer, da_status_arch_not_supported,                     \
                           "Zen 2 architecture not supported.");
#endif

#if defined(znver3_AVAILABLE)
#define DYNAMIC_DISPATCH_ZEN3(err_buffer, code)                                          \
    {                                                                                    \
        using namespace da_dynamic_dispatch_zen3;                                        \
        code;                                                                            \
    }
#else
#define DYNAMIC_DISPATCH_ZEN3(err_buffer, code)                                          \
    return da_error_bypass(err_buffer, da_status_arch_not_supported,                     \
                           "Zen 3 architecture not supported.");
#endif

#if defined(znver4_AVAILABLE)
#define DYNAMIC_DISPATCH_ZEN4(err_buffer, code)                                          \
    {                                                                                    \
        using namespace da_dynamic_dispatch_zen4;                                        \
        code;                                                                            \
    }
#else
#define DYNAMIC_DISPATCH_ZEN4(err_buffer, code)                                          \
    return da_error_bypass(err_buffer, da_status_arch_not_supported,                     \
                           "Zen 4 architecture not supported.");
#endif

#if defined(znver5_AVAILABLE)
#define DYNAMIC_DISPATCH_ZEN5(err_buffer, code)                                          \
    {                                                                                    \
        using namespace da_dynamic_dispatch_zen5;                                        \
        code;                                                                            \
    }
#else
#define DYNAMIC_DISPATCH_ZEN5(err_buffer, code)                                          \
    return da_error_bypass(err_buffer, da_status_arch_not_supported,                     \
                           "Zen 5 architecture not supported.");
#endif

#define DISPATCHER(err_buffer, code)                                                     \
    {                                                                                    \
        switch (context::get_context()->arch) {                                          \
        case generic:                                                                    \
            DYNAMIC_DISPATCH_GENERIC(err_buffer, code)                                   \
            break;                                                                       \
        case zen2:                                                                       \
            DYNAMIC_DISPATCH_ZEN2(err_buffer, code)                                      \
            break;                                                                       \
        case zen3:                                                                       \
            DYNAMIC_DISPATCH_ZEN3(err_buffer, code)                                      \
            break;                                                                       \
        case zen4:                                                                       \
            DYNAMIC_DISPATCH_ZEN4(err_buffer, code)                                      \
            break;                                                                       \
        case zen5:                                                                       \
            DYNAMIC_DISPATCH_ZEN5(err_buffer, code)                                      \
            break;                                                                       \
        default:                                                                         \
            return da_error_bypass(                                                      \
                err_buffer, da_status_internal_error,                                    \
                "dynamic dispatcher trying to call an unknown architecture?");           \
        }                                                                                \
    }

#endif