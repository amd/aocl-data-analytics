/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "aoclda.h"
#include "context.hpp"
#include "da_error.hpp"
#include "da_utils.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

da_errors::error_bypass_t *nosave_utils(nullptr);

da_status da_check_data_d(da_order order, da_int n_rows, da_int n_cols, const double *X,
                          da_int ldx) {
    DISPATCHER(nosave_utils,
               return (da_utils::check_data(order, n_rows, n_cols, X, ldx)));
}

da_status da_check_data_s(da_order order, da_int n_rows, da_int n_cols, const float *X,
                          da_int ldx) {
    DISPATCHER(nosave_utils,
               return (da_utils::check_data(order, n_rows, n_cols, X, ldx)));
}

da_status da_switch_order_copy_d(da_order order, da_int n_rows, da_int n_cols,
                                 const double *X, da_int ldx, double *Y, da_int ldy) {
    DISPATCHER(nosave_utils, return (da_utils::switch_order_copy(order, n_rows, n_cols, X,
                                                                 ldx, Y, ldy)));
}
da_status da_switch_order_copy_s(da_order order, da_int n_rows, da_int n_cols,
                                 const float *X, da_int ldx, float *Y, da_int ldy) {
    DISPATCHER(nosave_utils, return (da_utils::switch_order_copy(order, n_rows, n_cols, X,
                                                                 ldx, Y, ldy)));
}

da_status da_switch_order_in_place_d(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     double *X, da_int ldx_in, da_int ldx_out) {
    DISPATCHER(nosave_utils, return (da_utils::switch_order_in_place(
                                 order_X_in, n_rows, n_cols, X, ldx_in, ldx_out)));
}

da_status da_switch_order_in_place_s(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     float *X, da_int ldx_in, da_int ldx_out) {
    DISPATCHER(nosave_utils, return (da_utils::switch_order_in_place(
                                 order_X_in, n_rows, n_cols, X, ldx_in, ldx_out)));
}

da_status da_get_arch_info(da_int *len = nullptr, char *arch = nullptr,
                           char *ns = nullptr) {
    std::string NS{"?"};

    context::get_context()->refresh();

    auto try_get_namespace = []([[maybe_unused]] std::string &NS) -> da_status {
        const char *ns{nullptr};
        DISPATCHER(nosave_utils, ns = da_arch::get_namespace());
        // this section is executed only if dispatcher hasn't return'd yet
        // so ns is set to the corresponding namespace
        NS = ns;
        return da_status_success;
    };

    da_status status = try_get_namespace(NS);
    // either success or da_status_arch_not_supported (here tmp == nullptr)
    switch (status) {
    case da_status_success:
        // NS = correct namespace
        break;
    case da_status_arch_not_supported:
        NS = "<arch not supported>";
        break;
    default:
        // internal error?
        return status;
    }

    std::string Arch;
    switch (context::get_context()->arch) {
    case generic:
        Arch = "generic";
        break;
    case zen2:
    case zen3:
    case zen4:
    case zen5:
        // assuming ->arch matches with zen generation...
        Arch = "zen" + std::to_string(context::get_context()->arch);
        break;
    default:
        // Probably new zen model
        return da_status_internal_error;
        break;
    }
    if (arch != nullptr && ns != nullptr && len != nullptr) {
        da_int mlen = da_int(std::max(Arch.size(), NS.size()) + 1);
        if (*len < mlen) {
            *len = mlen;
            return da_status_invalid_array_dimension;
        }
        std::copy(NS.begin(), NS.end(), ns);
        ns[NS.size()] = '\0';
        std::copy(Arch.begin(), Arch.end(), arch);
        arch[Arch.size()] = '\0';
    } else {
        std::cout << "AOCL-DA local architecture: " << Arch << std::endl;
        std::cout << "AOCL-DA running architecture: " << NS << std::endl;
    }
    return da_status_success;
}
