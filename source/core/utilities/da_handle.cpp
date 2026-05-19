/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "da_handle.hpp"
#include "aoclda.h"

/*
 * Get pointer to the option member of the currently active sub-handle, also
 * if refresh is true, the it calls the appropriate sub-handle's refresh()
 * member to indicate that substantial changes have occurred in the handle.
 * E.g. options changes that alter the model requiring re-training, etc...
 */
da_status _da_handle::get_current_opts(da_options::OptionRegistry **opts, bool refresh) {
    const std::string msg = "handle seems to be corrupted.";

    switch (precision) {
    case da_double:
        if (alg_handle_d == nullptr)
            return da_error(this->err, da_status_invalid_pointer, msg);
        *opts = &alg_handle_d->opts;
        if (refresh)
            alg_handle_d->refresh();
        break;
    case da_single:
        if (alg_handle_s == nullptr)
            return da_error(this->err, da_status_invalid_pointer, msg);
        *opts = &alg_handle_s->opts;
        if (refresh)
            alg_handle_s->refresh();
        break;
    default:
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "handle precision was not correctly set");
        break;
    }
    return da_status_success;
}

da_status _da_handle::save_handle(std::vector<char> &buffer_data) {
    da_model_persistence::serialization_buffer buffer(this->handle_type);
    da_status status = buffer.set_buffer_data(&buffer_data);
    if (status != da_status_success)
        return status;

    if (this->precision == da_single) {
        status = this->alg_handle_s->save_model(buffer);
    } else if (this->precision == da_double) {
        status = this->alg_handle_d->save_model(buffer);
    }
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing handle.");

    return status;
}

da_status _da_handle::save_handle(const std::string &file_name) {
    da_status status = da_status_success;
    std::vector<char> buffer_data;

    status = this->save_handle(buffer_data);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing handle.");

    std::ofstream file;
    file.open(file_name, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return da_error(this->err, da_status_io_error,              // LCOV_EXCL_LINE
                        "Error while opening/creating save file."); // LCOV_EXCL_LINE
    }
    file.write(buffer_data.data(), buffer_data.size());
    if (!file) {
        return da_error(this->err, da_status_io_error, // LCOV_EXCL_LINE
                        "Failed to write to file.");   // LCOV_EXCL_LINE
    }
    file.close();

    return status;
}

da_status _da_handle::load_handle(da_handle &handle, const char *buffer_data,
                                  const size_t data_size) {
    da_int precision;
    da_model_persistence::serialization_buffer buffer(da_handle_uninitialized);
    da_status status = buffer.set_buffer_data(buffer_data, data_size);
    if (status != da_status_success)
        return status;

    status = buffer.deserialize_metadata(precision);
    if (status != da_status_success)
        return status;

    if (precision == (da_int)sizeof(float)) {
        status = da_handle_init_s(&handle, buffer.get_handle_type());
        if (status != da_status_success)
            return status;

        status = handle->alg_handle_s->load_model(buffer);
    } else if (precision == (da_int)sizeof(double)) {
        status = da_handle_init_d(&handle, buffer.get_handle_type());
        if (status != da_status_success)
            return status;

        status = handle->alg_handle_d->load_model(buffer);
    }
    if (status != da_status_success ||
        (handle->alg_handle_s == nullptr && handle->alg_handle_d == nullptr)) {
        da_status return_status =
            status != da_status_success ? status : da_status_invalid_file_data;
        return da_error_trace(handle->err, return_status, "Failure deserializing model.");
    }

    return status;
}

da_status _da_handle::load_handle(da_handle &handle, const std::string &file_name) {
    std::ifstream file(file_name, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return da_status_io_error; // LCOV_EXCL_LINE
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size <= 0) {
        return da_status_io_error;
    }

    std::vector<char> buffer_data;
    try {
        buffer_data.resize(size);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    if (!file.read(buffer_data.data(), size))
        return da_status_io_error;

    file.close();

    da_status status = load_handle(handle, buffer_data.data(), (size_t)size);
    if (status != da_status_success && handle) {
        return da_error_trace(handle->err, status,
                              "Failure deserializing handle from file.");
    }

    return status;
}

template <> basic_handle<double> *_da_handle::get_alg_handle<double>() {
    return alg_handle_d;
}
template <> basic_handle<float> *_da_handle::get_alg_handle<float>() {
    return alg_handle_s;
}
