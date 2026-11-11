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

#ifndef LINMOD_R_HPP
#define LINMOD_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class LinearModel : public rda_handle {
  private:
    da_int n_samples, n_feat;
    bool intercept, warm_start;
    linmod_model mod_enum;
    std::string logreg_constraint_str, solver_str;
    bool fitted = false;

  public:
    LinearModel(std::string mod, Nullable<da_int> max_iter, bool intercept = false,
                std::string solver = "auto", std::string scaling = "auto",
                std::string constraint = "ssc", bool warm_start = false,
                bool mixed_precision = false,
                Nullable<da_int> low_precision_max_iter = 500, bool check_data = false,
                Nullable<da_int> optim_coord_skip_min = 2,
                Nullable<da_int> optim_coord_skip_max = 100,
                Nullable<da_int> print_level = R_NilValue)
        : intercept(intercept), warm_start(warm_start), logreg_constraint_str(constraint),
          solver_str(solver) {

        // Initialize handle
        da_status status = da_handle_init_d(&handle, da_handle_linmod);
        exception_check(status);

        // Set model type correctly
        if (mod == "mse") {
            mod_enum = linmod_model_mse;
        } else if (mod == "logistic") {
            mod_enum = linmod_model_logistic;
        } else {
            mod_enum = linmod_model_undefined;
        }

        status = da_linmod_select_model_d(handle, mod_enum);
        exception_check(status);

        if (intercept) {
            status = da_options_set(handle, "intercept", 1);
            exception_check(status);
        }

        status = da_options_set(handle, "optim method", solver.c_str());
        exception_check(status);
        status = da_options_set(handle, "scaling", scaling.c_str());
        exception_check(status);

        if (max_iter.isNotNull()) {
            status = da_options_set_int(handle, "optim iteration limit",
                                        Rcpp::as<da_int>(max_iter));
            exception_check(status);
        }

        status = da_options_set(handle, "logistic constraint", constraint.c_str());
        exception_check(status);

        if (low_precision_max_iter.isNotNull()) {
            status = da_options_set_int(handle, "low precision iteration limit",
                                        Rcpp::as<da_int>(low_precision_max_iter));
            exception_check(status);
        }

        if (optim_coord_skip_min.isNotNull()) {
            status = da_options_set_int(handle, "optim coord skip min",
                                        Rcpp::as<da_int>(optim_coord_skip_min));
            exception_check(status);
        }

        if (optim_coord_skip_max.isNotNull()) {
            status = da_options_set_int(handle, "optim coord skip max",
                                        Rcpp::as<da_int>(optim_coord_skip_max));
            exception_check(status);
        }

        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }

        exception_check(status);

        if (mixed_precision == true) {
            status = da_options_set(handle, "mixed precision", "yes");
            exception_check(status);
        }

        if (print_level.isNotNull() && Rcpp::as<da_int>(print_level) > 0) {
            status =
                da_options_set(handle, "print level", Rcpp::as<da_int>(print_level));
            exception_check(status);
            status = da_options_set(handle, "print options", "yes");
            exception_check(status);
        }
    }

    void fit(NumericMatrix X, NumericVector y, Nullable<NumericVector> x0 = R_NilValue,
             Nullable<double> progress_factor = R_NilValue, double reg_lambda = 0.0,
             double reg_alpha = 0.0, double tol = 0.0001,
             double low_precision_tol = 0.001, Nullable<double> dual_tol = 0.0001,
             Nullable<double> optim_coord_skip_tol = 1.0e-4) {
        n_samples = X.nrow();
        n_feat = X.ncol();

        da_status status = da_status_success;
        std::vector<double> warm_coefficients;
        da_int dim;

        if (this->fitted && this->warm_start) {
            da_int n_iter, nrow_prev, ncol_prev, n_samples_prev, n_feat_prev,
                n_class_prev;
            bool well_determined_prev;
            double loss, time, nrm_gradient_loss;

            status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples_prev,
                               &n_feat_prev, &n_class_prev, &nrow_prev, &ncol_prev,
                               &well_determined_prev);
            exception_check(status);

            // Calculate current properties
            da_int n_class = static_cast<da_int>(
                std::round(*std::max_element(y.begin(), y.end())) + 1);
            da_int ncol = intercept ? n_feat + 1 : n_feat;
            bool is_well_determined =
                n_samples >= n_feat + static_cast<da_int>(intercept);

            if (n_class != n_class_prev && mod_enum == linmod_model_logistic)
                exception_check(
                    da_status_invalid_input,
                    "The number of classes has changed since the last fit. Cannot cast "
                    "previous coefficients. Try again with warm_start=False.");

            if (ncol != ncol_prev)
                exception_check(
                    da_status_invalid_input,
                    "The number of features has changed since the last fit. Cannot cast "
                    "previous coefficients. Try again with warm_start=False.");
            if (!is_well_determined &&
                (solver_str == "cg" || solver_str == "sparse_cg")) {
                if (well_determined_prev)
                    exception_check(
                        da_status_invalid_input,
                        "Previously fit data was well determined, cannot obtain dual "
                        "coefficients to warm start. Try again with warm_start=False.");
                else {
                    dim = n_samples;
                    warm_coefficients.resize(dim);
                    status = da_handle_get_result_d(handle, da_linmod_dual_coef, &dim,
                                                    warm_coefficients.data());
                    exception_check(status);
                }
            } else {
                dim = ncol_prev * nrow_prev;
                warm_coefficients.resize(dim);
                status = da_handle_get_result_d(handle, da_linmod_coef, &dim,
                                                warm_coefficients.data());
                exception_check(status);
            }
        }

        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_linmod_define_features_d(handle, n_samples, n_feat, X.begin(),
                                             n_samples, y.begin());
        exception_check(status);

        status = da_options_set(handle, "lambda", reg_lambda);
        exception_check(status);

        status = da_options_set(handle, "alpha", reg_alpha);
        exception_check(status);

        status = da_options_set(handle, "optim convergence tol", tol);
        exception_check(status);

        status =
            da_options_set(handle, "low precision convergence tol", low_precision_tol);
        exception_check(status);

        if (progress_factor.isNotNull()) {
            double pf = as<double>(progress_factor);
            status = da_options_set(handle, "optim progress factor", pf);
            exception_check(status);
        }

        if (dual_tol.isNotNull()) {
            status = da_options_set(handle, "optim dual gap tol", as<double>(dual_tol));
            exception_check(status);
        }

        if (optim_coord_skip_tol.isNotNull()) {
            status = da_options_set(handle, "optim coord skip tol",
                                    as<double>(optim_coord_skip_tol));
            exception_check(status);
        }

        if (!x0.isNull()) {
            const NumericVector &x0_ref = x0.get();
            status = da_linmod_fit_start_d(handle, x0_ref.length(), x0_ref.begin());
            exception_check(status);
        } else if (fitted && warm_start) {
            // Use warm start coefficients
            status = da_linmod_fit_start_d(handle, dim, warm_coefficients.data());
            exception_check(status);
        } else {
            status = da_linmod_fit_d(handle);
            exception_check(status);
        }

        this->fitted = status == da_status_success ? true : false;
    }

    List predict(NumericMatrix X, Nullable<NumericVector> y = R_NilValue) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        NumericVector predictions(n_samples);
        double loss = 0.0;
        double *loss_ptr = nullptr;
        const double *obs_ptr = nullptr;

        if (y.isNotNull()) {
            const NumericVector &y_ref = y.get();
            if (y_ref.size() != n_samples) {
                exception_check(
                    da_status_invalid_input,
                    "Observations array has invalid shape. It should be a 1D "
                    "contiguous array with the same number elements as rows of X.");
            }
            obs_ptr = y_ref.begin();
            loss_ptr = &loss;
        }

        da_status status;
        status =
            da_linmod_evaluate_model_d(handle, n_samples, n_features, X.begin(),
                                       n_samples, predictions.begin(), obs_ptr, loss_ptr);

        exception_check(status);

        if (y.isNotNull())
            return List::create(Named("predictions") = predictions, Named("loss") = loss);
        return List::create(Named("predictions") = predictions);
    }

    bool get_model_trained() {
        da_status status = da_status_success;
        da_int dim = 1;
        da_int trained = 0;
        switch (mod_enum) {
        case linmod_model_mse:
        case linmod_model_logistic: {
            status = da_handle_get_result_int(handle, da_trained, &dim, &trained);
            exception_check(status);
            return trained != 0;
        } break;
        default:
            exception_check(da_status_internal_error,
                            "Model type was not correctly defined.");
            return false;
        }
    }

    NumericVector get_coef() {
        da_status status = da_status_success;
        da_int dim = 0;

        switch (mod_enum) {
        case linmod_model_mse: {
            dim = intercept ? n_feat + 1 : n_feat;
            NumericVector coef(dim);
            status = da_handle_get_result_d(handle, da_linmod_coef, &dim, coef.begin());
            exception_check(status);
            return coef;
        } break;
        case linmod_model_logistic: {
            da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
            bool is_well_determined;
            double loss, time, nrm_gradient_loss;

            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);

            dim = ncol_coef * nrow_coef;
            NumericVector coef(dim);
            status = da_handle_get_result_d(handle, da_linmod_coef, &dim, coef.begin());
            exception_check(status);
            return coef;
        } break;
        default:
            exception_check(da_status_internal_error,
                            "Model type was not correctly defined.");
            return NumericVector();
        }
    }

    da_status get_rinfo(double *loss, double *nrm_gradient_loss, da_int *n_iter,
                        double *time, da_int *n_samples, da_int *n_feat, da_int *n_class,
                        da_int *nrow_coef, da_int *ncol_coef, bool *is_well_determined) {
        da_status status;
        da_int dim = 100;
        double rinfo[100];

        status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);

        // Use exact same indexing as Python bindings
        *loss = static_cast<double>((rinfo[da_linmod_info_t_::linmod_info_objective]));
        *nrm_gradient_loss =
            static_cast<double>((rinfo[da_linmod_info_t_::linmod_info_grad_norm]));
        *n_iter = static_cast<da_int>((rinfo[da_linmod_info_t_::linmod_info_iter]));
        *time = static_cast<double>((rinfo[da_linmod_info_t_::linmod_info_time]));
        *n_samples =
            static_cast<da_int>((rinfo[da_linmod_info_t_::linmod_info_nsamples]));
        *n_feat = static_cast<da_int>((rinfo[da_linmod_info_t_::linmod_info_nfeat]));
        *n_class = static_cast<da_int>((rinfo[da_linmod_info_t_::linmod_info_nclass]));
        *nrow_coef = static_cast<da_int>(rinfo[da_linmod_info_t_::linmod_info_nrow_coef]);
        *ncol_coef =
            static_cast<da_int>((rinfo[da_linmod_info_t_::linmod_info_ncol_coef]));
        *is_well_determined =
            static_cast<bool>((rinfo[da_linmod_info_t_::linmod_info_well_determined]));

        return da_status_success;
    }

    NumericVector get_dual_coef() {
        da_status status;

        da_int dim = n_samples;
        NumericVector dual_coef(dim);
        status =
            da_handle_get_result_d(handle, da_linmod_dual_coef, &dim, dual_coef.begin());
        exception_check(status);

        return dual_coef;
    }

    double get_loss() {
        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        double loss, nrm_gradient_loss, time;
        da_status status;

        status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                           &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
        exception_check(status);

        return loss;
    }

    double get_norm_gradient_loss() {
        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        double loss, nrm_gradient_loss, time;
        da_status status;

        status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                           &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
        exception_check(status);

        return nrm_gradient_loss;
    }

    double get_time() {
        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        double loss, nrm_gradient_loss, time;
        da_status status;

        status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                           &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
        exception_check(status);

        return time;
    }

    da_int get_n_iter() {
        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        double loss, nrm_gradient_loss, time;
        da_status status;

        status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                           &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
        exception_check(status);

        return n_iter;
    }

    da_int get_lp_n_iter() {
        da_int dim = 100;
        double rinfo[100];
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);
        return static_cast<da_int>(rinfo[da_linmod_info_t_::linmod_info_lp_n_iter]);
    }

    void save_data(Rcpp::List &state) override {
        state["n_samples"] = (int)n_samples;
        state["n_feat"] = (int)n_feat;
        state["intercept"] = (int)intercept;
        state["warm_start"] = (int)warm_start;
        state["fitted"] = (int)fitted;
        state["logreg_constraint_str"] = logreg_constraint_str;
        state["solver_str"] = solver_str;
        state["mod_enum"] = (int)mod_enum;
    }

    void load_data(Rcpp::List &state) override {
        n_samples = Rcpp::as<int>(state["n_samples"]);
        n_feat = Rcpp::as<int>(state["n_feat"]);
        intercept = (bool)Rcpp::as<int>(state["intercept"]);
        warm_start = (bool)Rcpp::as<int>(state["warm_start"]);
        fitted = (bool)Rcpp::as<int>(state["fitted"]);
        logreg_constraint_str = Rcpp::as<std::string>(state["logreg_constraint_str"]);
        solver_str = Rcpp::as<std::string>(state["solver_str"]);
        mod_enum = (linmod_model)Rcpp::as<int>(state["mod_enum"]);
    }
};

#endif // LINMOD_R_HPP
