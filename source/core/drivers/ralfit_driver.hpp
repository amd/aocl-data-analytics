/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#define ral_int da_int
#ifdef SINGLE_PRECISION
#define SINGLE_PREC_TURNED_ON
#endif
#define SINGLE_PRECISION
#undef ral_real
#undef ral_nlls_h
#undef PREC
#include "ral_nlls.h" // Add declarations for single precision
#undef SINGLE_PRECISION
#undef ral_real
#undef ral_nlls_h
#undef PREC
#include "ral_nlls.h" // Add declarations for double precision
// clean up
#ifndef SINGLE_PREC_TURNED_ON
// turn back off
#undef SINGLE_PRECISION
#endif
#undef SINGLE_PREC_TURNED_ON
// clean name-space
#undef ral_real
#undef ral_int
#undef ral_nlls_default_options
#undef ral_nlls_options
#undef ral_nlls_inform
#undef nlls_solve
#undef ral_nlls_init_workspace
#undef ral_nlls_iterate
#undef nlls_strerror
#undef ral_nlls_free_workspace
#undef PREC

#include "macros.h"

#include <cassert>
#include <functional>
#include <type_traits>

namespace ARCH {

namespace ralfit {

using namespace std::literals::string_literals;

// define ral_nlls types depending on typename T
template <typename T>
using ral_nlls_options_t =
    typename std::conditional_t<std::is_same_v<T, double>, struct ral_nlls_options_d,
                                struct ral_nlls_options_s>;
template <typename T>
using ral_nlls_inform_t =
    typename std::conditional_t<std::is_same_v<T, double>, struct ral_nlls_inform_d,
                                struct ral_nlls_inform_s>;

template <typename T>
using ral_nlls_eval_r_type_t =
    typename std::conditional_t<std::is_same_v<T, double>, ral_nlls_eval_r_type_d,
                                ral_nlls_eval_r_type_s>;

template <typename T>
using ral_nlls_eval_j_type_t =
    typename std::conditional_t<std::is_same_v<T, double>, ral_nlls_eval_j_type_d,
                                ral_nlls_eval_j_type_s>;

template <typename T>
using ral_nlls_eval_hf_type_t =
    typename std::conditional_t<std::is_same_v<T, double>, ral_nlls_eval_hf_type_d,
                                ral_nlls_eval_hf_type_s>;

template <typename T>
using ral_nlls_eval_hp_type_t =
    typename std::conditional_t<std::is_same_v<T, double>, ral_nlls_eval_hp_type_d,
                                ral_nlls_eval_hp_type_s>;

const da_int RAL_NLLS_CB_DUMMY{-3024};
const da_int RAL_NLLS_CB_FD{-45544554};
// Dummy call-back headers
template <typename T>
da_int da_nlls_eval_j_dummy([[maybe_unused]] da_int n, [[maybe_unused]] da_int m,
                            [[maybe_unused]] void *params, [[maybe_unused]] const T *x,
                            [[maybe_unused]] T *j) {
    return RAL_NLLS_CB_FD;
}
template <typename T>
da_int da_nlls_eval_hf_dummy([[maybe_unused]] da_int n, [[maybe_unused]] da_int m,
                             [[maybe_unused]] void *params, [[maybe_unused]] const T *x,
                             [[maybe_unused]] const T *f, [[maybe_unused]] T *hf) {
    return RAL_NLLS_CB_DUMMY;
}

// Copy RALFit's inform into DA's info array
template <typename T>
void copy_inform(ral_nlls_inform_t<T> &inform, std::vector<T> &info) {
    info[da_optim_info_t::info_iter] = T(inform.iter);
    info[da_optim_info_t::info_nevalf] = T(inform.f_eval);
    info[da_optim_info_t::info_nevalg] = T(inform.g_eval);
    info[da_optim_info_t::info_nevalh] = T(inform.h_eval);
    info[da_optim_info_t::info_nevalhp] = T(inform.hp_eval);
    info[da_optim_info_t::info_nevalfd] = T(inform.fd_f_eval);
    info[da_optim_info_t::info_objective] = T(inform.obj);
    info[da_optim_info_t::info_grad_norm] = T(inform.norm_g);
    info[da_optim_info_t::info_scl_grad_norm] = T(inform.scaled_g);
}

// Get RALFit's exit status/message and copy into DA's status
template <typename T>
da_status get_exit_status(ral_nlls_inform_t<T> &inform, da_errors::da_error_t &err) {
    // Exit status: ral_int inform.status;
    // Error message: char inform.error_message[81]
    // See ral_nlls_workspaces.f90 / module

    if (inform.status == 0)
        return da_status_success;

    std::string errmsg{inform.error_message};
    da_options::OptionUtils::prep_str(errmsg); // prepare string
    std::string msg;
    bool warn{false}; // Exit status is a warning or error?
    da_status status{da_status_internal_error};

    switch (inform.status) {
    case -1: // Warning + solution is usable
        status = da_status_maxit;
        warn = true;
        break;
    case -2:
    case -4:
        status = da_status_optimization_usrstop;
        warn = true;
        break;
    case -7:
    case -8:
    case -11:
    case -201:
    case -202:
    case -301:
    case -302:
    case -303:
    case -501:
        status = da_status_numerical_difficulties;
        warn = true;
        break;
    case -3: // Error + no usable output
    case -5:
    case -10:
    case -12:
    case -14:
    case -15:
    case -17:
    case -101:
    case -401:
    case -900:
        status = da_status_invalid_option;
        break;
    case -16:
        status = da_status_operation_failed;
        break;
    case -18:
        status = da_status_option_invalid_bounds;
        break;
    case -19:
        status = da_status_bad_derivatives;
        break;
    case -20:
        status = da_status_invalid_input;
        break;
    case -6:
    case -13:
    case -999:
        status = da_status_memory_error;
        break;
    default:
        msg = "Unexpected exit status from RALFit solver. return="s +
              std::to_string(inform.status) + " ("s + errmsg + ")."s;
        return da_error(&err, da_status_internal_error, msg);
        break;
    }

    if (warn) {
        // Compose and return a warning
        msg = "RALFit solver warning message: "s + errmsg + " (return="s +
              std::to_string(inform.status) + ").";
        return da_warn(&err, status, msg);
    }

    // Compose and return an error
    msg = "RALFit solver error message: "s + errmsg + " (return="s +
          std::to_string(inform.status) + ").";
    return da_error(&err, status, msg);
}

template <typename T>
da_status copy_options_to_ralfit(da_options::OptionRegistry &opts,
                                 ral_nlls_options_t<T> &options,
                                 da_errors::da_error_t &err, bool ok_eval_HF) {
    da_status status;
    const std::string msg{" option not found in the registry?"};

    // ===========================================================================
    // INTEGER OPTIONS
    // ===========================================================================
    da_int debug;
    status = opts.get("debug", debug);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<debug>"s + msg);
    }
    if (debug)
        options.print_options = true;

    da_int prlvl;
    status = opts.get("print level", prlvl);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<print level>"s + msg);
    }
    options.print_level = prlvl;

    da_int maxit;
    status = opts.get("ralfit iteration limit", maxit);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit iteration limit>"s + msg);
    }
    options.maxit = maxit;
    // ===========================================================================
    // REAL OPTIONS
    // ===========================================================================
    T derivative_test_tol;
    status = opts.get("derivative test tol", derivative_test_tol);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<derivative test tol>"s + msg);
    }
    options.derivative_test_tol = derivative_test_tol;
    T fd_step;
    status = opts.get("finite differences step", fd_step);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<finite differences step>"s + msg);
    }
    options.fd_step = fd_step;
    T bigbnd;
    status = opts.get("infinite bound size", bigbnd);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<infinite bound size>"s + msg);
    }
    options.box_bigbnd = bigbnd;

    T atolf;
    status = opts.get("ralfit convergence abs tol fun", atolf);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit convergence abs tol fun>"s + msg);
    }
    options.stop_f_absolute = atolf;

    T rtolf;
    status = opts.get("ralfit convergence rel tol fun", rtolf);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit convergence rel tol fun>"s + msg);
    }
    options.stop_f_relative = rtolf;

    T atolg;
    status = opts.get("ralfit convergence abs tol grd", atolg);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit convergence abs tol grd>"s + msg);
    }
    options.stop_g_absolute = atolg;

    T rtolg;
    status = opts.get("ralfit convergence rel tol grd", rtolg);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit convergence rel tol grd>"s + msg);
    }
    options.stop_g_relative = rtolg;

    T stol;
    status = opts.get("ralfit convergence step size", stol);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit convergence step size>"s + msg);
    }
    options.stop_s = stol;

    T reg_term;
    status = opts.get("regularization term", reg_term);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<regularization term>"s + msg);
    }
    options.regularization_term = reg_term;

    // ===========================================================================
    // STRING OPTIONS
    // ===========================================================================
    std::string chkder;
    da_int ichkder;
    status = opts.get("check derivatives", chkder, ichkder);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<check derivatives>"s + msg);
    }
    options.check_derivatives = ichkder;

    std::string model;
    da_int imodel;
    status = opts.get("ralfit model", model, imodel);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit model>"s + msg);
    }
    options.model = imodel;

    std::string nlls_method;
    da_int inlls_method;
    status = opts.get("ralfit nlls method", nlls_method, inlls_method);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit nlls method>"s + msg);
    }
    options.nlls_method = inlls_method;

    std::string glob_method;
    da_int iglob_method;
    status = opts.get("ralfit globalization method", glob_method, iglob_method);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<ralfit globalization method>"s + msg);
    }
    options.type_of_method = iglob_method;

    std::string storage;
    da_int istorage;
    status = opts.get("storage order", storage, istorage);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<storage order>"s + msg);
    }
    if (istorage == column_major) {
        options.Fortran_Jacobian = true;
    } else {
        options.Fortran_Jacobian = false;
    }

    std::string reg_power;
    da_int ireg_power;
    status = opts.get("regularization power", reg_power, ireg_power);
    if (status != da_status_success) {
        return da_error(&err, da_status_option_not_found, // LCOV_EXCL_LINE
                        "<regularization power>"s + msg);
    }
    switch (ireg_power) {
    case da_optim_types::regularization::quadratic:
        options.regularization_power = 2.0;
        break;
    case da_optim_types::regularization::cubic:
        options.regularization_power = 3.0;
        break;
    default:
        return da_error(&err, da_status_option_invalid_value, // LCOV_EXCL_LINE
                        "<regularization power> option has an invalid value?");
        break;
    }

    // Set up automatic options
    if (reg_term > 0)
        options.regularization = 1;

    // Exact second derivative -> user provided HF?
    options.exact_second_derivatives = ok_eval_HF;

    return da_status_success;
}

// Entry point to RALFit (via ral_nlls.h)
template <typename T>
da_status ralfit_driver(da_options::OptionRegistry &opts, da_int nvar, da_int nres, T *x,
                        resfun_t<T> eval_r, resgrd_t<T> eval_J, reshes_t<T> eval_HF,
                        reshp_t<T> eval_HP, T *lower_bounds, T *upper_bounds, T *weights,
                        void *usrdata, std::vector<T> &info, da_errors::da_error_t &err) {
    ral_nlls_options_t<T> options;
    ral_nlls_inform_t<T> inform;

    // Initialize option values
    if constexpr (std::is_same_v<T, double>) {
        ral_nlls_default_options_d(&options);
    } else {
        ral_nlls_default_options_s(&options);
    }

    if (copy_options_to_ralfit<T>(opts, options, err, bool(eval_HF)) != da_status_success)
        return da_error_trace(&err, da_status_internal_error,
                              "Could not copy the options into the RALFit struct.");

    // Initialize the workspace
    void *workspace;
    void *inner_workspace;

    // init_workspace allocates and links together workspace with inner_workspace
    if constexpr (std::is_same_v<T, double>) {
        ral_nlls_init_workspace_d(&workspace, &inner_workspace);
    } else {
        ral_nlls_init_workspace_s(&workspace, &inner_workspace);
    }

    // Get address of eval_r
    assert(typeid(ral_nlls_eval_r_type_t<T>) == eval_r.target_type());

    ral_nlls_eval_r_type_t<T> ral_nlls_eval_r =
        *(eval_r.template target<ral_nlls_eval_r_type_t<T>>());

    ral_nlls_eval_j_type_t<T> ral_nlls_eval_J{nullptr};
    ral_nlls_eval_hf_type_t<T> ral_nlls_eval_HF{nullptr};
    ral_nlls_eval_hp_type_t<T> ral_nlls_eval_HP{nullptr};

    // Get address of eval_J or dummy
    if (eval_J) {
        assert(typeid(ral_nlls_eval_j_type_t<T>) == eval_J.target_type());
        ral_nlls_eval_J = *(eval_J.template target<ral_nlls_eval_j_type_t<T>>());
    } else {
        // Instantiate
        da_nlls_eval_j_dummy<T>(0, 0, nullptr, nullptr, nullptr);
        // Assign
        ral_nlls_eval_J = da_nlls_eval_j_dummy<T>;
    }

    // Get address of eval_HF or dummy
    if (eval_HF) {
        assert(typeid(ral_nlls_eval_hf_type_t<T>) == eval_HF.target_type());
        ral_nlls_eval_HF = *(eval_HF.template target<ral_nlls_eval_hf_type_t<T>>());
    } else {
        // Instantiate
        da_nlls_eval_hf_dummy<T>(0, 0, nullptr, nullptr, nullptr, nullptr);
        // Assign
        ral_nlls_eval_HF = da_nlls_eval_hf_dummy<T>;
    }

    // Get address of eval_HP or nullptr
    if (eval_HP) {
        assert(typeid(ral_nlls_eval_hp_type_t<T>) == eval_HP.target_type());
        ral_nlls_eval_HP = *(eval_HP.template target<ral_nlls_eval_hp_type_t<T>>());
    }

    if constexpr (std::is_same_v<T, double>) {
        nlls_solve_d(nvar, nres, x, ral_nlls_eval_r, ral_nlls_eval_J, ral_nlls_eval_HF,
                     usrdata, &options, &inform, weights, ral_nlls_eval_HP, lower_bounds,
                     upper_bounds);
        ral_nlls_free_workspace_d(&workspace);
        ral_nlls_free_workspace_d(&inner_workspace);
    } else {
        nlls_solve_s(nvar, nres, x, ral_nlls_eval_r, ral_nlls_eval_J, ral_nlls_eval_HF,
                     usrdata, &options, &inform, weights, ral_nlls_eval_HP, lower_bounds,
                     upper_bounds);
        ral_nlls_free_workspace_s(&workspace);
        ral_nlls_free_workspace_s(&inner_workspace);
    }

    copy_inform(inform, info);

    // Translate exit status -> severity
    da_status status = get_exit_status<T>(inform, err);

    return status; // err stack populated
}
} // namespace ralfit

} // namespace ARCH
