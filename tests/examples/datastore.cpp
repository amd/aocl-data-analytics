/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include <iostream>
#include <string>
#include <vector>

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

int main() {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Load data from csv file" << std::endl;

    da_datastore store;
    std::string filename = std::string(DATA_DIR) + "/" + "datastore_ex.csv";
    da_status status;

    // Load data
    da_datastore_init(&store);
    da_datastore_options_set_string(store, "CSV datatype", "double");
    da_datastore_options_set_int(store, "CSV use header row", 1);
    status = da_data_load_from_csv(store, filename.c_str());
    if (status != da_status_success) {
        std::cout << "Data loading unsuccessful" << std::endl;
        return 1;
    }

    // Select the first 2 columns as the feature matrix and the last one as the response
    std::vector<double> features(10), rhs(5);
    da_data_select_columns(store, "features", 0, 1);
    da_data_select_columns(store, "rhs", 2, 2);
    da_data_extract_selection_real_d(store, "features", 5, features.data());
    da_data_extract_selection_real_d(store, "rhs", 5, rhs.data());

    // define the regression problem to solve
    da_handle handle;
    da_handle_init_d(&handle, da_handle_linmod);
    da_linmod_d_select_model(handle, linmod_model_mse);
    da_linmod_d_define_features(handle, 2, 5, features.data(), rhs.data());

    // solve the problem
    status = da_linmod_d_fit(handle);

    int exit = 0;
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        da_int nx = 2;
        std::vector<double> x(2);
        da_handle_get_result_d(handle, da_result::da_linmod_coeff, &nx, x.data());
        std::cout << "Coefficients: " << x[0] << " " << x[1] << std::endl;
        std::cout << "(Expected   : " << 0.199256 << " " << 0.130354 << ")" << std::endl;
    } else
        exit = 1;

    da_datastore_destroy(&store);
    da_handle_destroy(&handle);

    return exit;
}