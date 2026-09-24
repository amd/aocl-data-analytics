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

#ifndef DA_R_NLLS_CB_HPP
#define DA_R_NLLS_CB_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

// ---------------------------------------------------------------------------
// Callback data structure - holds references to the R callback functions
// and user-supplied data. Passed as void* udata to da_nlls_fit_d.
// R is single-threaded so there is no thread-safety concern with this pattern.
// ---------------------------------------------------------------------------
struct NLLSCallbackData {
    Function fun;
    Nullable<Function> jac;
    Nullable<Function> hes;
    Nullable<Function> hp;
    SEXP user_data; // any R object, or R_NilValue
};

// ---------------------------------------------------------------------------
// Static C-style trampoline callbacks that delegate to the R functions.
//
// R callback contracts (return-based; no in-place modification required):
//   fun(x, data)        -> numeric vector of length n_res   [residuals]
//   jac(x, data)        -> numeric matrix n_res x n_coef    [Jacobian, row-major]
//   hes(x, wr, data)    -> numeric matrix n_coef x n_coef   [Hessian]
//   hp(x, y, data)      -> numeric matrix n_coef x n_res    [Hessian-vector product]
// data is omitted when it is NULL / R_NilValue.
// All callbacks must return 0 on success, non-zero on failure.
// ---------------------------------------------------------------------------

static da_int nlls_resfun(da_int n_coef, da_int n_res, void *data, const double *x,
                          double *res) {
    NLLSCallbackData *cbdata = static_cast<NLLSCallbackData *>(data);
    try {
        NumericVector x_r(x, x + n_coef);
        RObject result;
        if (cbdata->user_data == R_NilValue) {
            result = cbdata->fun(x_r);
        } else {
            result = cbdata->fun(x_r, cbdata->user_data);
        }
        NumericVector res_r = as<NumericVector>(result);
        if ((da_int)res_r.size() != n_res)
            return 1;
        std::copy(res_r.begin(), res_r.end(), res);
        return 0;
    } catch (...) {
        return 1;
    }
}

static da_int nlls_resgrd(da_int n_coef, da_int n_res, void *data, const double *x,
                          double *jac_out) {
    NLLSCallbackData *cbdata = static_cast<NLLSCallbackData *>(data);
    try {
        NumericVector x_r(x, x + n_coef);
        RObject result;
        if (cbdata->user_data == R_NilValue) {
            Function f(cbdata->jac.get());
            result = f(x_r);
        } else {
            Function f(cbdata->jac.get());
            result = f(x_r, cbdata->user_data);
        }
        NumericMatrix jac_r = as<NumericMatrix>(result);
        // Expected n_res rows x n_coef cols; R stores matrices in column-major order.
        // The NLLS C API uses column-major storage by default (set via "storage order").
        std::copy(jac_r.begin(), jac_r.end(), jac_out);
        return 0;
    } catch (...) {
        return 1;
    }
}

static da_int nlls_reshes(da_int n_coef, da_int n_res, void *data, const double *x,
                          const double *wr, double *hes_out) {
    NLLSCallbackData *cbdata = static_cast<NLLSCallbackData *>(data);
    try {
        NumericVector x_r(x, x + n_coef);
        NumericVector wr_r(wr, wr + n_res);
        RObject result;
        if (cbdata->user_data == R_NilValue) {
            Function f(cbdata->hes.get());
            result = f(x_r, wr_r);
        } else {
            Function f(cbdata->hes.get());
            result = f(x_r, wr_r, cbdata->user_data);
        }
        NumericMatrix hes_r = as<NumericMatrix>(result);
        std::copy(hes_r.begin(), hes_r.end(), hes_out);
        return 0;
    } catch (...) {
        return 1;
    }
}

// Note: reshp has a different argument order from the other callbacks - void* data is last.
static da_int nlls_reshp(da_int n_coef, da_int n_res, const double *x, const double *y,
                         double *hp_out, void *data) {
    NLLSCallbackData *cbdata = static_cast<NLLSCallbackData *>(data);
    try {
        NumericVector x_r(x, x + n_coef);
        NumericVector y_r(y, y + n_coef);
        RObject result;
        if (cbdata->user_data == R_NilValue) {
            Function f(cbdata->hp.get());
            result = f(x_r, y_r);
        } else {
            Function f(cbdata->hp.get());
            result = f(x_r, y_r, cbdata->user_data);
        }
        NumericMatrix hp_r = as<NumericMatrix>(result);
        std::copy(hp_r.begin(), hp_r.end(), hp_out);
        return 0;
    } catch (...) {
        return 1;
    }
}

// ---------------------------------------------------------------------------
// NLLSModel Rcpp class
// ---------------------------------------------------------------------------
class NLLSModel : public rda_handle {
  private:
    da_int n_coef;
    da_int n_res;

  public:
    NLLSModel(da_int n_coef, da_int n_res, Nullable<NumericVector> weights = R_NilValue,
              Nullable<NumericVector> lower_bounds = R_NilValue,
              Nullable<NumericVector> upper_bounds = R_NilValue,
              std::string model = "hybrid", std::string method = "galahad",
              std::string glob_strategy = "tr", std::string reg_power = "quadratic",
              std::string check_derivatives = "no", da_int verbose = 0,
              bool check_data = false)
        : n_coef(n_coef), n_res(n_res) {

        da_status status = da_handle_init_d(&handle, da_handle_nlls);
        exception_check(status);

        status = da_nlls_define_residuals_d(handle, n_coef, n_res, &nlls_resfun,
                                            &nlls_resgrd, nullptr, nullptr);
        exception_check(status);

        status = da_options_set(handle, "ralfit model", model.c_str());
        exception_check(status);
        status = da_options_set(handle, "ralfit nlls method", method.c_str());
        exception_check(status);
        status =
            da_options_set(handle, "ralfit globalization method", glob_strategy.c_str());
        exception_check(status);
        status = da_options_set(handle, "regularization power", reg_power.c_str());
        exception_check(status);
        status = da_options_set(handle, "check derivatives", check_derivatives.c_str());
        exception_check(status);

        if (verbose < 0 || verbose > 5)
            exception_check(da_status_option_invalid_value,
                            "Option 'verbose' must be between 0 and 5.");
        status = da_options_set(handle, "print level", verbose);
        exception_check(status);
        if (verbose > 1) {
            status = da_options_set(handle, "print options", "yes");
            exception_check(status);
        }

        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }

        if (weights.isNotNull()) {
            SEXP w_sexp = weights.get();
            if (!Rf_isNull(Rf_getAttrib(w_sexp, R_DimSymbol)))
                exception_check(da_status_invalid_input,
                                "'weights' must be a 1D vector, not a matrix.");
            NumericVector w = as<NumericVector>(w_sexp);
            status = da_nlls_define_weights_d(handle, (da_int)w.size(), w.begin());
            exception_check(status);
        }

        double *lower_ptr = nullptr, *upper_ptr = nullptr;
        NumericVector lb, ub;
        if (lower_bounds.isNotNull()) {
            SEXP lb_sexp = lower_bounds.get();
            if (!Rf_isNull(Rf_getAttrib(lb_sexp, R_DimSymbol)))
                exception_check(da_status_invalid_input,
                                "'lower_bounds' must be a 1D vector, not a matrix.");
            lb = as<NumericVector>(lb_sexp);
            if (lb.size() == 0)
                exception_check(da_status_invalid_input,
                                "'lower_bounds' must not be an empty vector.");
            lower_ptr = lb.begin();
        }
        if (upper_bounds.isNotNull()) {
            SEXP ub_sexp = upper_bounds.get();
            if (!Rf_isNull(Rf_getAttrib(ub_sexp, R_DimSymbol)))
                exception_check(da_status_invalid_input,
                                "'upper_bounds' must be a 1D vector, not a matrix.");
            ub = as<NumericVector>(ub_sexp);
            if (ub.size() == 0)
                exception_check(da_status_invalid_input,
                                "'upper_bounds' must not be an empty vector.");
            upper_ptr = ub.begin();
        }
        if (lower_ptr || upper_ptr) {
            if (lower_ptr && upper_ptr && lb.size() != ub.size())
                exception_check(
                    da_status_invalid_input,
                    "'lower_bounds' and 'upper_bounds' must be the same size.");
            da_int bound_n = lower_ptr ? (da_int)lb.size() : (da_int)ub.size();
            status = da_nlls_define_bounds_d(handle, bound_n, lower_ptr, upper_ptr);
            exception_check(status);
        }
    }

    void fit(NumericVector x, Function fun, Nullable<Function> jac = R_NilValue,
                      Nullable<Function> hes = R_NilValue,
                      Nullable<Function> hp = R_NilValue, SEXP data = R_NilValue,
                      double ftol = 1.0e-8, double abs_ftol = 1.0e-8,
                      double gtol = 1.0e-8, double abs_gtol = 1.0e-5,
                      double xtol = 2.22e-16, double reg_term = 0.0, da_int maxit = 100,
                      double fd_step = 1.0e-7, double fd_ttol = 1.0e-4,
                      double maxtime = 1.0e6) {

        if (!handle)
            exception_check(da_status_handle_not_initialized,
                            "Handle could not be initialized.");

        if (Rf_isMatrix(x))
            exception_check(da_status_invalid_array_dimension,
                            "``x`` must be a 1D array of size " + std::to_string(n_coef) +
                                ".");
        if ((da_int)x.size() != n_coef)
            exception_check(da_status_invalid_array_dimension,
                            "``x`` must be a 1D array of size " + std::to_string(n_coef) +
                                ".");

        da_status status;

        // Register C callbacks
        da_resfun_t_d *fun_cb{&nlls_resfun};
        da_resgrd_t_d *jac_cb{&nlls_resgrd};
        da_reshes_t_d *hes_cb{&nlls_reshes};
        da_reshp_t_d *hp_cb{&nlls_reshp};
        if (jac.isNull())
            jac_cb = nullptr;
        if (hes.isNull())
            hes_cb = nullptr;
        if (hp.isNull())
            hp_cb = nullptr;

        status = da_nlls_define_residuals_d(handle, this->n_coef, this->n_res, fun_cb,
                                            jac_cb, hes_cb, hp_cb);
        exception_check(status);

        // Pass the options
        status = da_options_set(handle, "ralfit convergence rel tol fun", ftol);
        exception_check(status);
        status = da_options_set(handle, "ralfit convergence rel tol grd", gtol);
        exception_check(status);
        status = da_options_set(handle, "ralfit convergence abs tol fun", abs_ftol);
        exception_check(status);
        status = da_options_set(handle, "ralfit convergence abs tol grd", abs_gtol);
        exception_check(status);
        status = da_options_set(handle, "ralfit convergence step size", xtol);
        exception_check(status);
        status = da_options_set(handle, "ralfit iteration limit", maxit);
        exception_check(status);
        status = da_options_set(handle, "regularization term", reg_term);
        exception_check(status);
        status = da_options_set(handle, "finite differences step", fd_step);
        exception_check(status);
        status = da_options_set(handle, "derivative test tol", fd_ttol);
        exception_check(status);
        status = da_options_set(handle, "time limit", maxtime);
        exception_check(status);

        // Set up callback data
        NLLSCallbackData cbdata{fun, jac, hes, hp, data};

        // Call solver (modifies x in-place, like Python)
        void *udata = &cbdata;
        status = da_nlls_fit_d(handle, this->n_coef, x.begin(), udata);

        exception_check(status);
    }

    void get_info(da_int &iter, da_int &f_eval, da_int &g_eval, da_int &h_eval,
                  da_int &hp_eval, double &obj, double &norm_g, double &scaled_g,
                  da_int &fd_f_eval) {
        da_int dim = 100;
        std::vector<double> rinfo(dim);
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo.data());
        exception_check(status);

        iter = (da_int)rinfo[da_optim_info_t::info_iter];
        f_eval = (da_int)rinfo[da_optim_info_t::info_nevalf];
        g_eval = (da_int)rinfo[da_optim_info_t::info_nevalg];
        h_eval = (da_int)rinfo[da_optim_info_t::info_nevalh];
        hp_eval = (da_int)rinfo[da_optim_info_t::info_nevalhp];
        fd_f_eval = (da_int)rinfo[da_optim_info_t::info_nevalfd];
        obj = rinfo[da_optim_info_t::info_objective];
        norm_g = rinfo[da_optim_info_t::info_grad_norm];
        scaled_g = rinfo[da_optim_info_t::info_scl_grad_norm];
    }

    // Mirrors Python's get_info_iter() / get_info_evals() / get_info_optim()
    da_int get_info_iter() {
        da_int iter, f_eval, g_eval, h_eval, hp_eval, fd_f_eval;
        double obj, norm_g, scaled_g;
        get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g, fd_f_eval);
        return iter;
    }

    List get_info_evals() {
        da_int iter, f_eval, g_eval, h_eval, hp_eval, fd_f_eval;
        double obj, norm_g, scaled_g;
        get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g, fd_f_eval);
        return List::create(Named("f") = f_eval, Named("j") = g_eval, Named("h") = h_eval,
                            Named("hp") = hp_eval, Named("fd_f") = fd_f_eval);
    }

    List get_info_optim() {
        da_int iter, f_eval, g_eval, h_eval, hp_eval, fd_f_eval;
        double obj, norm_g, scaled_g;
        get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g, fd_f_eval);
        return List::create(Named("obj") = obj, Named("norm_g") = norm_g,
                            Named("scl_norm_g") = scaled_g);
    }
};

#endif // DA_R_NLLS_CB_HPP
