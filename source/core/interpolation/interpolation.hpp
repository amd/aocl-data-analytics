/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "interpolation_generic.hpp"
#include "interpolation_options.hpp"
#include "macros.h"
#include <vector>

#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

namespace ARCH {

namespace da_interpolation {

template <typename T> class interpolation_p : public basic_handle<T> {

  private:
    std::unique_ptr<interpolation_generic<T>> interp = nullptr;

  protected:
    da_interpolation_model model = da_interpolation_model::interpolation_unset;

  public:
    interpolation_p(da_errors::da_error_t *err = nullptr);
    interpolation_p(da_errors::da_error_t &err);
    void refresh();

    da_status select_model(da_interpolation_model model);

    // Sites and values setters
    da_status set_sites(da_int n, const T *x);
    da_status set_sites_uniform(da_int n, T x_start, T x_end);
    da_status set_values(da_int n, da_int dim, const T *y_data, da_int ldy, da_int order);

    // Search for cells containing query points
    da_status search_cells(da_int n_eval, const T *x_eval, da_int *cells);

    // Set the boundary conditions
    da_status set_boundary_conditions(da_int dim, da_int left_order, const T *left_values,
                                      da_int right_order, const T *right_values);

    // Interpolate using stored sites and values
    da_status interpolate();

    // Evaluate the piecewise polynomial at given points
    da_status evaluate(da_int n_eval, const T *x_eval, T *y_eval, da_int n_orders = 1,
                       const da_int *order = 0);

    // Get results
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result(da_result query, da_int *dim, da_int *result);
};

} // namespace da_interpolation
} // namespace ARCH

#endif // INTERPOLATION_HPP