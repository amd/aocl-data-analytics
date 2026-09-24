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

#ifndef SVM_R_HPP
#define SVM_R_HPP

#include <Rcpp.h>
#include <memory>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class BaseSVM : public rda_handle {
  protected:
    da_int n_samples, n_features;

  public:
    BaseSVM(da_svm_model model, std::string kernel = "rbf", da_int degree = 3,
            da_int max_iter = -1, da_int probability = 0, da_int seed = 0,
            da_int max_ws_size = -1, bool mixed_precision = false,
            da_int low_precision_max_iter = -1, bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_svm);
        exception_check(status);

        status = da_svm_select_model_d(handle, model);
        exception_check(status);

        status = da_options_set(handle, "kernel", kernel.c_str());
        exception_check(status);
        status = da_options_set(handle, "degree", degree);
        exception_check(status);
        status = da_options_set(handle, "max_iter", max_iter);
        exception_check(status);
        status = da_options_set(handle, "predict probabilities", probability);
        exception_check(status);
        status = da_options_set(handle, "seed", seed);
        exception_check(status);
        status = da_options_set(handle, "max_ws_size", max_ws_size);
        exception_check(status);
        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }

        status = da_options_set(handle, "low precision max_iter", low_precision_max_iter);
        exception_check(status);

        if (mixed_precision) {
            status = da_options_set(handle, "mixed precision", "yes");
            exception_check(status);
        }
    }

    void common_fit(const NumericMatrix &X, const NumericVector &y, double gamma,
                    double coef0, double tol, double low_precision_tol, double cache_size,
                    Nullable<double> tau = R_NilValue) {

        n_samples = X.nrow();
        n_features = X.ncol();

        da_status status;

        status = da_options_set(handle, "gamma", gamma);
        exception_check(status);
        status = da_options_set(handle, "coef0", coef0);
        exception_check(status);
        status = da_options_set(handle, "tolerance", tol);
        exception_check(status);
        status = da_options_set(handle, "low precision convergence tolerance",
                                low_precision_tol);
        exception_check(status);
        status = da_options_set(handle, "cache size", cache_size);
        exception_check(status);
        if (tau.isNotNull()) {
            status = da_options_set(handle, "tau", as<double>(tau));
            exception_check(status);
        }

        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_svm_set_data_d(handle, n_samples, n_features, X.begin(), n_samples,
                                   y.begin());
        exception_check(status);

        status = da_svm_compute_d(handle);
        exception_check(status);
    }

    NumericVector predict(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        NumericVector predictions(n_samples);

        da_status status = da_svm_predict_d(handle, n_samples, n_features, X.begin(),
                                            n_samples, predictions.begin());
        exception_check(status);

        return predictions;
    }

    NumericMatrix decision_function(NumericMatrix X, std::string shape = "ovr") {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();
        da_int nclass = get_n_classes();
        da_int nclassifiers = nclass * (nclass - 1) / 2;

        da_svm_decision_function_shape shape_enum;
        if (shape == "ovo") {
            shape_enum = da_svm_decision_function_shape::ovo;
        } else if (shape == "ovr") {
            shape_enum = da_svm_decision_function_shape::ovr;
        } else {
            stop("Given decision function shape does not exist. Available choices are: "
                 "'ovo', 'ovr'.");
        }

        da_status status;

        da_int n_col;
        if (nclass > 2) {
            n_col = (shape_enum == da_svm_decision_function_shape::ovo) ? nclassifiers
                                                                        : nclass;
        } else {
            n_col = 1;
        }
        NumericMatrix decision_values(n_samples, n_col);
        status = da_svm_decision_function_d(handle, n_samples, n_features, X.begin(),
                                            n_samples, shape_enum,
                                            decision_values.begin(), n_samples);
        exception_check(status);
        return decision_values;
    }

    NumericMatrix predict_proba(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();
        da_int nclass = get_n_classes();

        NumericMatrix probabilities(n_samples, nclass);

        da_status status =
            da_svm_predict_proba_d(handle, n_samples, n_features, X.begin(), n_samples,
                                   probabilities.begin(), n_samples);
        exception_check(status);

        return probabilities;
    }

    NumericMatrix predict_log_proba(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_int nclass = get_n_classes();

        NumericMatrix log_probabilities(n_samples, nclass);

        da_status status =
            da_svm_predict_log_proba_d(handle, n_samples, n_features, X.begin(),
                                       n_samples, log_probabilities.begin(), n_samples);
        exception_check(status);

        return log_probabilities;
    }

    double score(NumericMatrix X, NumericVector y) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        double score = 0.0;

        da_status status = da_svm_score_d(handle, n_samples, n_features, X.begin(),
                                          n_samples, y.begin(), &score);
        exception_check(status);

        return score;
    }

    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_classes) {
        da_int dim = 100;
        std::vector<double> rinfo(dim);
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo.data());
        exception_check(status);

        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
        *n_classes = static_cast<da_int>(rinfo[2]);
    }

    da_int get_n_samples() {
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        return n_samples;
    }

    da_int get_n_features() {
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        return n_features;
    }

    da_int get_n_classes() {
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        return n_classes;
    }

    da_int get_n_sv() {
        da_int one = 1;
        da_int n_sv;
        da_status status =
            da_handle_get_result_int(handle, da_svm_n_support_vectors, &one, &n_sv);
        exception_check(status);
        return n_sv;
    }

    IntegerVector get_n_sv_per_class() {
        da_int nclass = get_n_classes();

        IntegerVector n_sv_per_class(nclass);
        da_status status = da_handle_get_result_int(
            handle, da_svm_n_support_vectors_per_class, &nclass, n_sv_per_class.begin());
        exception_check(status);

        return n_sv_per_class;
    }

    NumericVector get_bias() {
        da_int nclass = get_n_classes();
        da_int n_classifiers = nclass * (nclass - 1) / 2;

        NumericVector bias(n_classifiers);
        da_status status =
            da_handle_get_result_d(handle, da_svm_bias, &n_classifiers, bias.begin());
        exception_check(status);

        return bias;
    }

    NumericVector get_probA() {
        da_int nclass = get_n_classes();
        da_int n_classifiers = nclass * (nclass - 1) / 2;

        NumericVector probA(n_classifiers);
        da_status status =
            da_handle_get_result_d(handle, da_svm_probaA, &n_classifiers, probA.begin());
        exception_check(status);

        return probA;
    }

    NumericVector get_probB() {
        da_int nclass = get_n_classes();
        da_int n_classifiers = nclass * (nclass - 1) / 2;

        NumericVector probB(n_classifiers);
        da_status status =
            da_handle_get_result_d(handle, da_svm_probaB, &n_classifiers, probB.begin());
        exception_check(status);

        return probB;
    }

    IntegerVector get_n_iterations() {
        da_int nclass = get_n_classes();
        da_int n_classifiers = nclass * (nclass - 1) / 2;

        IntegerVector n_iterations(n_classifiers);
        da_status status = da_handle_get_result_int(handle, da_svm_n_iterations,
                                                    &n_classifiers, n_iterations.begin());
        exception_check(status);

        return n_iterations;
    }

    IntegerVector get_lp_n_iterations() {
        da_int nclass = get_n_classes();
        da_int n_classifiers = nclass * (nclass - 1) / 2;

        IntegerVector lp_n_iterations(n_classifiers);
        da_status status = da_handle_get_result_int(handle, da_svm_lp_n_iterations,
                                                    &n_classifiers,
                                                    lp_n_iterations.begin());
        exception_check(status);

        return lp_n_iterations;
    }

    NumericMatrix get_dual_coefs() {
        da_int n_sv = get_n_sv();
        da_int nclass = get_n_classes();

        NumericMatrix dual_coefs(nclass - 1, n_sv);
        da_int dim = (nclass - 1) * n_sv;

        da_status status =
            da_handle_get_result_d(handle, da_svm_dual_coef, &dim, dual_coefs.begin());
        exception_check(status);

        return dual_coefs;
    }

    NumericMatrix get_sv() {
        da_int n_sv = get_n_sv();
        da_int n_feat = get_n_features();
        NumericMatrix support_vectors(n_sv, n_feat);
        da_int dim = n_sv * n_feat;

        da_status status = da_handle_get_result_d(handle, da_svm_support_vectors, &dim,
                                                  support_vectors.begin());
        exception_check(status);

        return support_vectors;
    }

    IntegerVector get_support_vectors_idx() {
        da_int n_sv = get_n_sv();
        IntegerVector support_idx(n_sv);

        da_status status = da_handle_get_result_int(handle, da_svm_idx_support_vectors,
                                                    &n_sv, support_idx.begin());
        exception_check(status);

        return support_idx;
    }

    void save_data(Rcpp::List &state) override {
        state["n_samples"] = (int)n_samples;
        state["n_features"] = (int)n_features;
    }

    void load_data(Rcpp::List &state) override {
        n_samples = Rcpp::as<int>(state["n_samples"]);
        n_features = Rcpp::as<int>(state["n_features"]);
    }
};

/*******************/
/*      SVC        */
/*******************/
class SVC : public BaseSVM {
  public:
    SVC(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = -1,
        da_int probability = 0, da_int seed = 0, da_int max_ws_size = -1,
        bool mixed_precision = false, da_int low_precision_max_iter = -1,
        bool check_data = false)
        : BaseSVM(da_svm_model::svc, kernel, degree, max_iter, probability, seed,
                  max_ws_size, mixed_precision, low_precision_max_iter, check_data) {}

    void fit(NumericMatrix X, NumericVector y, Nullable<double> tau, double C = 1.0,
             double gamma = 1.0, double coef0 = 0.0, double tol = 0.001,
             double low_precision_tol = 0.01, double cache_size = 200.0) {

        da_status status = da_options_set(handle, "C", C);
        exception_check(status);

        common_fit(X, y, gamma, coef0, tol, low_precision_tol, cache_size, tau);
    }
};

/*******************/
/*      SVR        */
/*******************/
class SVR : public BaseSVM {
  public:
    SVR(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = -1,
        da_int max_ws_size = -1, bool mixed_precision = false,
        da_int low_precision_max_iter = -1, bool check_data = false)
        : BaseSVM(da_svm_model::svr, kernel, degree, max_iter, 0, 0, max_ws_size,
                  mixed_precision, low_precision_max_iter, check_data) {}

    void fit(NumericMatrix X, NumericVector y, Nullable<double> tau, double C = 1.0,
             double epsilon = 0.1, double gamma = 1.0, double coef0 = 0.0,
             double tol = 0.001, double low_precision_tol = 0.01,
             double cache_size = 200.0) {

        da_status status = da_options_set(handle, "C", C);
        exception_check(status);

        status = da_options_set(handle, "epsilon", epsilon);
        exception_check(status);

        common_fit(X, y, gamma, coef0, tol, low_precision_tol, cache_size, tau);
    }
};

/*******************/
/*     nuSVC       */
/*******************/
class NuSVC : public BaseSVM {
  public:
    NuSVC(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = -1,
          da_int probability = 0, da_int seed = 0, da_int max_ws_size = -1,
          bool mixed_precision = false, da_int low_precision_max_iter = -1,
          bool check_data = false)
        : BaseSVM(da_svm_model::nusvc, kernel, degree, max_iter, probability, seed,
                  max_ws_size, mixed_precision, low_precision_max_iter, check_data) {}

    void fit(NumericMatrix X, NumericVector y, Nullable<double> tau, double nu = 0.5,
             double gamma = 1.0, double coef0 = 0.0, double tol = 0.001,
             double low_precision_tol = 0.01, double cache_size = 200.0) {

        da_status status = da_options_set(handle, "nu", nu);
        exception_check(status);

        common_fit(X, y, gamma, coef0, tol, low_precision_tol, cache_size, tau);
    }
};

/*******************/
/*     nuSVR       */
/*******************/
class NuSVR : public BaseSVM {
  public:
    NuSVR(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = -1,
          da_int max_ws_size = -1, bool mixed_precision = false,
          da_int low_precision_max_iter = -1, bool check_data = false)
        : BaseSVM(da_svm_model::nusvr, kernel, degree, max_iter, 0, 0, max_ws_size,
                  mixed_precision, low_precision_max_iter, check_data) {}

    void fit(NumericMatrix X, NumericVector y, Nullable<double> tau = R_NilValue,
             double nu = 0.5, double C = 1.0, double gamma = 1.0, double coef0 = 0.0,
             double tol = 0.001, double low_precision_tol = 0.01,
             double cache_size = 200.0) {

        da_status status = da_options_set(handle, "nu", nu);
        exception_check(status);

        status = da_options_set(handle, "C", C);
        exception_check(status);

        common_fit(X, y, gamma, coef0, tol, low_precision_tol, cache_size, tau);
    }
};

#endif