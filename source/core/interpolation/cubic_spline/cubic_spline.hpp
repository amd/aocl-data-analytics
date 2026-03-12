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
#include "cubic_spline_types.hpp"
#include "da_error.hpp"
#include "interpolation_generic.hpp"
#include "macros.h"
#include <vector>

#ifndef CUBIC_SPLINE_HPP
#define CUBIC_SPLINE_HPP

namespace ARCH {

namespace da_interpolation {

using namespace da_cubic_spline;

template <typename T> class cubic_spline : public interpolation_generic<T> {
  private:
    // Spline type
    da_int spline_type = (da_int)spline_type_t::natural;

    // Interpolation sites (x_i, y_i)

    // Boundary conditions
    bool custom_bc_set = false;
    da_int left_order = 2, right_order = 2;
    std::vector<T> left_bc, right_bc;

    // Working memory
    // diag*: vectors to build the tridiagonal matrix
    // z: used for both the right hand side and the solution of the system
    std::vector<T> diag, diag_up, diag_lo;

    std::vector<T> z;

    // Spline output
    // coeffs(4*(n_sites-1)*dim_y): coefficients of the computed splines
    std::vector<T> coeffs;

  private:
    da_status initialize_system();
    void coeffs_from_second_derivatives();

    da_status compute_natural_spline();
    da_status compute_clamped_spline();
    da_status compute_custom_bc_spline();
    da_status compute_hermite_spline();

  public:
    cubic_spline(da_errors::da_error_t &err, da_options::OptionRegistry &opts);
    ~cubic_spline() = default;
    void refresh();

    // Set the boundary conditions
    da_status set_boundary_conditions(da_int dim, da_int left_order, const T *left_values,
                                      da_int right_order, const T *right_values);
    da_status interpolate();
    da_status evaluate(da_int n_eval, const T *x_eval, T *y_eval, da_int n_orders = 1,
                       const da_int *order = 0);

    // Get results
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result(da_result query, da_int *dim, da_int *result);

    // Test getters
    const T *get_x_sites() const { return this->x_sites; }
    const T *get_y() const { return this->y; }
};

} // namespace da_interpolation
} // namespace ARCH

#endif // CUBIC_SPLINE_HPP