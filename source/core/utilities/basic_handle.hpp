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

#ifndef BASIC_HANDLE_HPP
#define BASIC_HANDLE_HPP
#include "aoclda_error.h"
#include "aoclda_result.h"
#include "aoclda_types.h"
#include "da_error.hpp"
#include "da_utils.hpp"
#include "model_persistence.hpp"
#include "options.hpp"

#include <string>

/*
 * Base handle class (basic_handle) that contains members that
 * are common for all specialized handle types, pca, linear
 * models, etc.
 *
 * This handle is inherited by all specialized (internal) handles.
 */
template <typename T> class basic_handle {
  protected:
    // Version of the serialization format supported by this algorithm.
    // During loading, the saved version must exactly match this value.
    // Algorithms may override this in their constructor to track format changes independently.
    // Format: Major * 10'000 + Minor * 100 + Patch
    da_int serialization_version = 0;

    // Version of the build of the library that was used to save the algorithm.
    std::string saved_aoclda_version = "NA";

  public:
    basic_handle(da_errors::da_error_t *err = nullptr);
    basic_handle(da_errors::da_error_t &err);
    virtual ~basic_handle();

    /*
     * Generic interface to extract all data stored
     * in handle via the da_get_result_X C API
     */
    virtual da_status get_result(da_result query, da_int *dim, T *result) = 0;
    virtual da_status get_result(da_result query, da_int *dim, da_int *result) = 0;
    // Get the results objects that are stored in basic_handle
    da_status get_result_common(da_result query, da_int *dim, da_int *result);

    /*
     * Function to inform that something related to the (sub)handle has
     * changed and to mark as update-required. E.g. options changed and potentially
     * the underlying model is different and a new call to fit is required.
     * Each (sub)handle is responsible to implement this function if required.
     */
    virtual void refresh();

    /*
     * Model serialization function used to specify all handle-specific members
     * which will be saved/loaded. This function is called during save_model()/
     * load_model() and should be overridden by derived classes
     * to correctly serialize their internal state.
     */
    virtual da_status
    serialize([[maybe_unused]] da_model_persistence::serialization_buffer &buffer) {
        return da_status_not_implemented;
    };

    /*
     * Functions to facilitate serialization (saving/loading) of the handle state.
     * Derived classes should override these to verify the model is in the required
     * state, and perform any additional functionalities to get the model to the
     * initial state while also calling the parent function.
     */
    virtual da_status save_model(da_model_persistence::serialization_buffer &buffer);
    virtual da_status load_model(da_model_persistence::serialization_buffer &buffer);

    /*
     * Helpers to reserve, save and load options from the registry of the model.
     */
    da_status serialize_registry(da_model_persistence::serialization_buffer &buffer);
    da_status deserialize_registry(da_model_persistence::serialization_buffer &buffer);

    void set_serialization_version(da_int serialization_version) {
        this->serialization_version = serialization_version;
    }

    void set_saved_aoclda_version(const std::string &saved_aoclda_version) {
        this->saved_aoclda_version = saved_aoclda_version;
    }

    /* Print saved versions of AOCL-DA and model serialization. */
    da_status print_model_versions();

    /*
     * Calculates the total buffer size required for serialization and preallocates the memory.
     */
    da_status preallocate_buffer(da_model_persistence::serialization_buffer &buffer);

    // user's data storage: row or column major order
    // all 2d arrays passed from/to user are in this order, internally
    // these arrays can and are converted. But when passing data back to user
    // this is the order used.
    da_order order{da_order::column_major};

    // Pointer to error trace
    da_errors::da_error_t *err{nullptr};

    // Options registry
    da_options::OptionRegistry opts;

    da_options::OptionRegistry &get_opts();

    // Whether the model is trained/computed
    bool model_trained{false};

    // Whether model was loaded from memory
    bool model_loaded{false};

    // Argument checking for a 1D input array, including NaN check if option is set
    // @tparam T Type of the data array: float, double, da_int
    da_status check_1D_array(da_int n, const T *data, const std::string &n_name,
                             const std::string &data_name, da_int n_min = 1);
    da_status check_1D_array(da_int n, const da_int *data, const std::string &n_name,
                             const std::string &data_name, da_int n_min = 1);

    /**
     * @brief Argument checking for a 2D input array, including NaN check if option is set
     *
     * This function performs checks for invalid dimensions, null pointers, and NaN values (if enabled).
     * It is intended to be used for validating 2D arrays passed to the handle for those routines which
    * handle row or column major storage internally.
     *
     * @param order The storage order of the input data (row-major or column-major).
     * @param n_rows The number of rows in the 2D array.
     * @param n_cols The number of columns in the 2D array.
     * @param data A pointer to the input data.
     * @param lddata The leading dimension of the input data.
     * @param n_rows_name The name of the variable representing the number of rows (for error messages).
     * @param n_cols_name The name of the variable representing the number of columns (for error messages).
     * @param data_name The name of the variable representing the data (for error messages).
     * @param lddata_name The name of the variable representing the leading dimension (for error messages).
     * @param n_rows_min The minimum allowed value for the number of rows.
     * @param n_cols_min The minimum allowed value for the number of columns.
     */
    da_status check_2D_array(da_order order, da_int n_rows, da_int n_cols, const T *data,
                             da_int lddata, const std::string &n_rows_name,
                             const std::string &n_cols_name, const std::string &data_name,
                             const std::string &lddata_name, da_int n_rows_min = 1,
                             da_int n_cols_min = 1);

    void copy_2D_results_array(da_int n_rows, da_int n_cols, const T *data, da_int lddata,
                               T *results_arr);
};

#endif