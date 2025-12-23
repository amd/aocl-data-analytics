/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/* This file is a place for miscellaneous utility functions that do not belong with
 * any particular classes and do not fit elsewhere. */

#include "aoclda.h"
#include "context.hpp"
#include "da_omp.hpp"

// Check that parallel builds of AOCL-DA work. Once we have further OpenMP functionality, this can probably be removed
da_status da_parallel_check() {

    da_int max_threads = omp_get_max_threads();

    da_int n_threads = 1;

// We could do anything here really - we just want to check we are linking omp.h correctly
#pragma omp parallel reduction(max : n_threads) num_threads(max_threads) default(none)
    { n_threads = omp_get_thread_num() + 1; }

    if (n_threads != max_threads)
        return da_status_internal_error;

    return da_status_success;
}

static const char *da_version = AOCLDA_VERSION_STRING;

// Return the version string of AOCL-DA.
const char *da_get_version() { return da_version; }

void context_set_hidden_settings(const std::string &key,
                                 const std::string &value) noexcept {
    context::get_context()->set_hidden_setting(key, value);
}