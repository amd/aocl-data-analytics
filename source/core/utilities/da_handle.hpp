/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DA_HANDLE_HPP
#define DA_HANDLE_HPP

#include <new>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "basic_handle.hpp"
#include "context.hpp"
#include "csv_reader.hpp"
#include "da_error.hpp"
#include "dynamic_dispatch.hpp"

/**
 * @brief Handle structure containing input / output data required for functions such as fit and predict
 */
struct _da_handle {
  public:
    // Pointer to error trace and related methods
    da_errors::da_error_t *err = nullptr;
    // Pointer for each sub-handle
    da_csv::csv_reader *csv_parser = nullptr;
    da_precision precision = da_double;
    da_handle_type handle_type = da_handle_uninitialized;

    // subhandle
    basic_handle<double> *alg_handle_d = nullptr;
    basic_handle<float> *alg_handle_s = nullptr;

    // Clear telemetry, for now it only clears the error stack.
    // vector<>.clear() is linear in cost wrt the number of elements to erase.
    void clear(void) {
        if (err)
            err->clear();
    };

    da_status get_current_opts(da_options::OptionRegistry **opts, bool refresh = false);

    template <typename T> basic_handle<T> *get_alg_handle();

    // Serialize and save the complete handle state (model and options).
    // The buffer overload is the main implementation.
    // The file_name overload is a wrapper that calls the buffer version
    // and writes the result to disk.
    da_status save_handle(std::vector<char> &buffer);
    da_status save_handle(const std::string &file_name);

    // Deserialize and load the complete handle state (model and options).
    // The buffer overload is the main implementation.
    // The file_name overload is a wrapper that reads from disk
    // and calls the buffer version.
    static da_status load_handle(da_handle &handle, const char *buffer_data,
                                 const size_t data_size);
    static da_status load_handle(da_handle &handle, const std::string &file_name);
};

#endif
