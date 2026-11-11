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

#ifndef INTERNAL_UTILITIES_R_HPP
#define INTERNAL_UTILITIES_R_HPP

#include "aoclda.h"
#include <Rcpp.h>
#include <cstdlib>
#include <sstream>

// R integers are always 32-bit. The R interface relies on da_int being 32-bit
// (int32_t) so that IntegerVector / IntegerMatrix can be passed directly to the
// AOCL-DA C API without overflow or silent truncation.  If AOCL-DA was built
// with ILP64 (da_int == int64_t) this interface will not work correctly.
static_assert(sizeof(da_int) == 4,
              "AOCL-DA was built with 64-bit da_int (ILP64). "
              "The R interface requires a 32-bit da_int (LP32/ILP32) build. "
              "Rebuild AOCL-DA without the ILP64 option.");
#include "aoclda_cpp_overloads.hpp"

/*
 * rda_handle: base class for all R wrapper classes that own a da_handle.
 *
 * Mirrors pyda_handle from internal_utilities_py.hpp but without the
 * Python/numpy-specific concerns:
 *   - No precision tracking: R numeric is always double.
 *   - No order tracking: R matrices are always column-major.
 *
 * Subclasses inherit the handle, a virtual destructor (no need to write
 * their own), exception_check() which surfaces the handle's stored
 * diagnostic message, and save_data()/load_data() hooks for future
 * model serialisation (mirrors the py::dict-based hooks in pyda_handle).
 */
class rda_handle {
  protected:
    da_handle handle = nullptr;

    virtual void save_data([[maybe_unused]] Rcpp::List &state) {}
    virtual void load_data([[maybe_unused]] Rcpp::List &state) {}

  public:
    rda_handle() = default;
    virtual ~rda_handle() { da_handle_destroy(&handle); }

    void print_error_message() { da_handle_print_error_message(handle); }

    void print_model_versions() {
        std::ostringstream oss;
        std::streambuf *old = std::cout.rdbuf(oss.rdbuf());
        da_status status = da_handle_print_model_versions(handle);
        std::cout.rdbuf(old);
        Rcpp::Rcout << oss.str();
        exception_check(status);
    }

    Rcpp::List save_model() {
        std::vector<char> buffer_data;
        da_status status = da_handle_save_model(handle, buffer_data);
        exception_check(status);

        Rcpp::List state;
        state["data"] = Rcpp::RawVector(buffer_data.begin(), buffer_data.end());
        this->save_data(state);
        return state;
    }

    void load_model(Rcpp::List state) {
        if (!state.containsElementNamed("data"))
            Rcpp::stop("Invalid or incomplete model state: missing 'data' field.");

        Rcpp::RawVector raw = Rcpp::as<Rcpp::RawVector>(state["data"]);
        if (raw.size() == 0)
            Rcpp::stop("Model state contains empty data buffer.");

        da_handle_destroy(&handle);
        handle = nullptr;

        this->load_data(state);

        da_status status = da_handle_load_model(
            &handle, reinterpret_cast<const char *>(raw.begin()), raw.size());

        if (handle == nullptr)
            Rcpp::stop("Failed to load model.");

        exception_check(status);
    }

    /*
     * exception_check: mirrors pyda_handle::exception_check().
     *
     * If a non-empty message is supplied it is used verbatim (useful when the
     * handle is not yet fully initialised, e.g. during da_handle_init_d).
     * Otherwise the diagnostic stored inside the handle is retrieved:
     *   DA_ERROR   -> Rcpp::stop  (throws R error)
     *   DA_WARNING -> Rcpp::warning (non-fatal, execution continues)
     */
    void exception_check(da_status status, const std::string &message = "") {
        if (status == da_status_success)
            return;

        if (!message.empty()) {
            Rcpp::stop(message);
        }

        char *msg = nullptr;
        da_severity severity = DA_NOTSET;
        da_handle_get_error_message(handle, &msg);
        da_handle_get_error_severity(handle, &severity);

        std::string s(msg ? msg : "Unknown AOCL-DA error");
        free(msg);

        if (severity == DA_ERROR)
            Rcpp::stop(s);
        else
            Rcpp::warning(s);
    }
};

// Free-function variant for use in wrappers that do not own a handle
// (e.g. basic stats helpers). Handle-owning classes should use the
// rda_handle::exception_check() member instead.
// Mirrors Python's status_to_exception() in internal_utilities_py.hpp.
inline void status_to_exception(da_status status) {
    switch (status) {
    case da_status_success:
        return;
    case da_status_memory_error:
        throw std::bad_alloc();
    case da_status_invalid_array_dimension:
        throw std::length_error("One of the input arrays was too small.");
    case da_status_invalid_input:
    case da_status_negative_data:
        throw std::invalid_argument(
            "One of the options passed to the function had an invalid value.");
    default:
        Rcpp::stop("AOCL-DA operation failed with status: %d", (int)status);
    }
}

#endif // INTERNAL_UTILITIES_R_HPP
