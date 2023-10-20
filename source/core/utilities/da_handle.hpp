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

#ifndef DA_HANDLE_HPP
#define DA_HANDLE_HPP

#include <new>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "csv_reader.hpp"
#include "da_error.hpp"
#include "decision_forest.hpp"
#include "linear_model.hpp"
#include "pca.hpp"
#include "decision_forest.hpp"

/**
 * @brief Handle structure containing input / output data required for functions such as fit and predict
 *
 */
struct _da_handle {
  public:
    da_csv::csv_reader *csv_parser = nullptr;
    linear_model<double> *linreg_d = nullptr;
    linear_model<float> *linreg_s = nullptr;
    da_handle_type handle_type = da_handle_uninitialized;
    // Pointer to error trace and related methods
    da_errors::da_error_t *err = nullptr;
    da_precision precision = da_double;
    da_pca::da_pca<double> *pca_d = nullptr;
    da_pca::da_pca<float> *pca_s = nullptr;
    decision_tree<double> *dt_d = nullptr;
    decision_tree<float> *dt_s = nullptr;
    decision_forest<double> *df_d = nullptr;
    decision_forest<float> *df_s = nullptr;

    da_status get_current_opts(da_options::OptionRegistry **opts);
};

#endif
