/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "macros.h"
#include "options.hpp"
#include "svm_types.hpp"
#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

/*
 * SVM handle class that contains definitions to all the user
 * facing functionalities like set_data(),
 */

namespace ARCH {

// This function returns whether observation is in I_up set
template <typename T> bool is_upper(T &alpha, const T &y, T &C);
// This function returns whether observation is in I_low set
template <typename T> bool is_lower(T &alpha, const T &y, T &C);

// This function returns whether observation is in I_up set and is a positive class
template <typename T> bool is_upper_pos(T &alpha, const T &y, T &C);

// This function returns whether observation is in I_up set and is a negative class
template <typename T> bool is_upper_neg(T &alpha, const T &y);

// This function returns whether observation is in I_low set and is a positive class
template <typename T> bool is_lower_pos(T &alpha, const T &y);

// This function returns whether observation is in I_low set and is a negative class
template <typename T> bool is_lower_neg(T &alpha, const T &y, T &C);

// Functions that return pointer to internal kernel function implementation
template <typename T>
static void rbf_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                        T *x_norm, da_int ldx, const T *Y, T *y_norm, da_int ldy, T *D,
                        da_int ldd, T gamma, da_int /*degree*/, T /*coef0*/, bool X_is_Y);

template <typename T>
static void linear_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                           T * /*x_norm*/, da_int ldx, const T *Y, T * /*y_norm*/,
                           da_int ldy, T *D, da_int ldd, T /*gamma*/, da_int /*degree*/,
                           T /*coef0*/, bool X_is_Y);

template <typename T>
static void sigmoid_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                            T * /*x_norm*/, da_int ldx, const T *Y, T * /*y_norm*/,
                            da_int ldy, T *D, da_int ldd, T gamma, da_int /*degree*/,
                            T coef0, bool X_is_Y);

template <typename T>
static void polynomial_wrapper(da_order order, da_int m, da_int n, da_int k, const T *X,
                               T * /*x_norm*/, da_int ldx, const T *Y, T * /*y_norm*/,
                               da_int ldy, T *D, da_int ldd, T gamma, da_int degree,
                               T coef0, bool X_is_Y);

namespace da_svm {

using namespace da_svm_types;

// This forward declaration is here to allow for "friending" it with base_svm few lines below
template <typename T> class svm;

/*
 * Base SVM handle class that contains members that
 * are common for all SVM models.
 *
 * This handle is inherited by all specialized svm handles.
 *
 * The inheritance scheme is as follows:
 *
 *                          SVM
 *                         /   \
 *                        /     \
 *                   C-SVM       Nu-SVM
 *                  /     \      /     \
 *                 /       \    /       \
 *              SVC       SVR Nu-SVC   Nu-SVR
 */

template <typename T> class base_svm {
  public:
    // User's data
    const T *XUSR;
    const T *yusr;
    // n x p (samples x features)
    da_int n = 0;
    da_int p = 0;
    // Data that we operate on (in SVR / binary classification it is same as user's data)
    T *X = nullptr;
    // Second ldx is here for case when we have multiclass, X is created internally dense, so we want to proceed with ldx_2 = nsamples
    // However in other cases we use user's pointer where we want ldx_2 = ldx
    da_int ldx, ldx_2;
    T *y = nullptr;
    // actual_size = 2n if (SVR or nuSVR), otherwise n
    da_int actual_size;

    // Used in multi-class classification (memory allocated in svm.hpp set_data())
    std::vector<da_int> idx_class; // indexes of observations where class is either i or j
    std::vector<da_int> support_indexes_pos, support_indexes_neg;
    std::vector<bool> idx_is_positive;
    bool ismulticlass = false;
    da_int pos_class, neg_class;

    // Kernel function to use for computation
    da_int kernel_function = rbf;
    kernel_f_type<T> kernel_f = nullptr;
    // Kernel specific parameters
    T gamma = 1.0;
    da_int degree = 3;
    T coef0 = 0.0;
    // Regularisation parameters
    T C = 1, eps = 0.1, nu = 0.5;
    // Working set parameter tau, value of the denominator if kernel is not positive semi definite (safe eps)
    T tau = 2 * std::numeric_limits<T>::epsilon();
    // Convergence tolerance
    T tol = 1.0e-3;
    da_int max_iter;
    da_int iter;

    // Variable is being set at the contructors of spiecialised classes
    da_svm_model mod = svm_undefined;

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // Variables for result handling
    std::vector<T> gradient, alpha, response;
    da_int n_support;                    // Number of support vectors
    std::vector<da_int> support_indexes, // Indexes of support vectors
        n_support_per_class;
    std::vector<T> support_coefficients; // Alphas of support vectors
    T bias;                              // Constant in decision function

    // Internal working variables
    std::vector<T> alpha_diff;
    da_int ws_size; // Size of working set
    std::vector<T> local_alpha, local_gradient, local_response;
    std::vector<T> x_norm_aux, y_norm_aux; // Work array for kernel computation
    std::vector<bool> I_low_p, I_up_p, I_low_n, I_up_n;

    // ws_indexes is array with indexes of outer working set
    // index_aux is array with argsorted gradient array
    std::vector<da_int> ws_indexes, index_aux;
    std::vector<bool> ws_indicator;

  public:
    // This friend is here to allow following "svm" class to set protected members of "base_svm" class, such as kernel_function, X or y.
    friend class svm<T>;
    base_svm(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train);
    virtual ~base_svm(); // Virtual to remove warnings

    // Main functions
    da_status compute();
    da_status predict(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                      T *decision_values);
    da_status decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                da_int ldx_test, T *decision_values);

    // General auxiliary functions
    void update_gradient(std::vector<T> &gradient, std::vector<T> &alpha_diff,
                         da_int &nrow, da_int &ncol, std::vector<T> &kernel_matrix);
    void kernel_compute(std::vector<da_int> &idx, da_int &idx_size,
                        std::vector<T> &X_temp, std::vector<T> &kernel_matrix);
    void compute_ws_size(da_int &ws_size);
    da_int maxpowtwo(da_int &n);
    void wssi(std::vector<bool> &I_up, std::vector<T> &gradient, da_int &i, T &min_grad);
    void wssj(std::vector<bool> &I_low, std::vector<T> &gradient, da_int &i, T &min_grad,
              da_int &j, T &max_grad, std::vector<T> &kernel_matrix, T &delta,
              T &max_fun);

    // Functions that need specialisation
    virtual da_status initialisation(da_int &size, std::vector<T> &gradient,
                                     std::vector<T> &response, std::vector<T> &alpha) = 0;
    virtual void outer_wss(da_int &size, std::vector<da_int> &selected_ws_idx,
                           std::vector<bool> &selected_ws_indicator,
                           da_int &n_selected) = 0;
    virtual void local_smo(da_int &ws_size, std::vector<da_int> &idx,
                           std::vector<T> &kernel_matrix,
                           std::vector<T> &local_kernel_matrix, std::vector<T> &alpha,
                           std::vector<T> &local_alpha, std::vector<T> &gradient,
                           std::vector<T> &local_gradient, std::vector<T> &response,
                           std::vector<T> &local_response, std::vector<bool> &I_low_p,
                           std::vector<bool> &I_up_p, std::vector<bool> &I_low_n,
                           std::vector<bool> &I_up_n, T &first_diff,
                           std::vector<T> &alpha_diff, std::optional<T> tol) = 0;
    virtual da_status set_bias(std::vector<T> &alpha, std::vector<T> &gradient,
                               std::vector<T> &response, da_int &size, T &bias) = 0;
    virtual da_status set_sv(std::vector<T> &alpha, da_int &n_support) = 0;
};

template <typename T> class svm : public basic_handle<T> {
  private:
    // Pointers to SVM problem class that will be specialised
    std::vector<std::unique_ptr<base_svm<T>>> classifiers;

    da_int n_class, n_classifiers;
    // Only used in multi-class classification
    std::vector<da_int> class_sizes;

    // Utility pointer to column major allocated copy of user's data
    T *X_temp = nullptr;

    const T *X = nullptr;
    const T *y = nullptr;
    da_int nrow, ncol;

    da_int ldx_train;

    // Set true when user data is loaded
    bool loadingdone = false;
    // Set true when SVM is computed successfully
    bool iscomputed = false;
    bool ismulticlass = false;

    da_svm_model mod = svm_undefined;

    // Results
    std::vector<bool> is_sv; // only used for multiclass
    da_int n_sv = 0;
    std::vector<T> support_coefficients, support_vectors, bias;
    std::vector<da_int> support_indexes, n_sv_per_class, n_iteration;

  public:
    svm(da_errors::da_error_t &err);
    ~svm();

    // Main functions
    da_status set_data(da_int n_samples, da_int n_features, const T *X, da_int ldx_train,
                       const T *y);
    da_status select_model(da_svm_model mod);
    da_status compute();
    da_status predict(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                      T *predictions);
    da_status decision_function(da_int nsamples, da_int nfeat, const T *X_test,
                                da_int ldx_test, da_svm_decision_function_shape shape,
                                T *decision_values, da_int ldd);
    da_status score(da_int nsamples, da_int nfeat, const T *X_test, da_int ldx_test,
                    const T *y_test, T *score);

    void refresh();

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result(da_result query, da_int *dim, da_int *result);
};

template <typename T> class csvm : public base_svm<T> {
  private:
  public:
    csvm(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train);
    virtual ~csvm(); // Make the destructor virtual to remove warnings
    // Specialised functions
    void outer_wss(da_int &size, std::vector<da_int> &selected_ws_idx,
                   std::vector<bool> &selected_ws_indicator, da_int &n_selected);
    void local_smo(da_int &ws_size, std::vector<da_int> &idx,
                   std::vector<T> &kernel_matrix, std::vector<T> &local_kernel_matrix,
                   std::vector<T> &alpha, std::vector<T> &local_alpha,
                   std::vector<T> &gradient, std::vector<T> &local_gradient,
                   std::vector<T> &response, std::vector<T> &local_response,
                   std::vector<bool> &I_low_p, std::vector<bool> &I_up_p,
                   std::vector<bool> &I_low_n, std::vector<bool> &I_up_n, T &first_diff,
                   std::vector<T> &alpha_diff, std::optional<T> tol);
    da_status set_bias(std::vector<T> &alpha, std::vector<T> &gradient,
                       std::vector<T> &response, da_int &size, T &bias);

    // Inherited functions
    virtual da_status initialisation(da_int &size, std::vector<T> &gradient,
                                     std::vector<T> &response, std::vector<T> &alpha) = 0;
    virtual da_status set_sv(std::vector<T> &alpha, da_int &n_support) = 0;
};

template <typename T> class svc : public csvm<T> {
  private:
  public:
    svc(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train);
    virtual ~svc(); // Make the destructor virtual to remove warnings
    // Specialised functions
    da_status initialisation(da_int &size, std::vector<T> &gradient,
                             std::vector<T> &response, std::vector<T> &alpha);
    da_status set_sv(std::vector<T> &alpha, da_int &n_support);
};

template <typename T> class svr : public csvm<T> {
  private:
  public:
    svr(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train);
    virtual ~svr(); // Make the destructor virtual to remove warnings
    // Specialised functions
    da_status initialisation(da_int &size, std::vector<T> &gradient,
                             std::vector<T> &response, std::vector<T> &alpha);
    da_status set_sv(std::vector<T> &alpha, da_int &n_support);
};

template <typename T> class nusvm : public base_svm<T> {
  private:
  public:
    nusvm(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train);
    virtual ~nusvm(); // Make the destructor virtual to remove warnings
    // Important functions
    void outer_wss(da_int &size, std::vector<da_int> &selected_ws_idx,
                   std::vector<bool> &selected_ws_indicator, da_int &n_selected);
    void local_smo(da_int &ws_size, std::vector<da_int> &idx,
                   std::vector<T> &kernel_matrix, std::vector<T> &local_kernel_matrix,
                   std::vector<T> &alpha, std::vector<T> &local_alpha,
                   std::vector<T> &gradient, std::vector<T> &local_gradient,
                   std::vector<T> &response, std::vector<T> &local_response,
                   std::vector<bool> &I_low_p, std::vector<bool> &I_up_p,
                   std::vector<bool> &I_low_n, std::vector<bool> &I_up_n, T &first_diff,
                   std::vector<T> &alpha_diff, std::optional<T> tol);
    da_status set_bias(std::vector<T> &alpha, std::vector<T> &gradient,
                       std::vector<T> &response, da_int &size, T &bias);

    // Inherited functions
    virtual da_status initialisation(da_int &size, std::vector<T> &gradient,
                                     std::vector<T> &response, std::vector<T> &alpha) = 0;
    virtual da_status set_sv(std::vector<T> &alpha, da_int &n_support) = 0;

    // Auxiliary functions
    da_status initialise_gradient(std::vector<T> &alpha_diff, da_int counter,
                                  std::vector<T> &gradient);
};

template <typename T> class nusvc : public nusvm<T> {
  private:
  public:
    nusvc(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train);
    virtual ~nusvc(); // Make the destructor virtual to remove warnings
    // Specialised functions
    da_status initialisation(da_int &size, std::vector<T> &gradient,
                             std::vector<T> &response, std::vector<T> &alpha);
    da_status set_sv(std::vector<T> &alpha, da_int &n_support);
};

template <typename T> class nusvr : public nusvm<T> {
  private:
  public:
    nusvr(const T *XUSR, const T *yusr, da_int n, da_int p, da_int ldx_train);
    virtual ~nusvr(); // Make the destructor virtual to remove warnings
    // Specialised functions
    da_status initialisation(da_int &size, std::vector<T> &gradient,
                             std::vector<T> &response, std::vector<T> &alpha);
    da_status set_sv(std::vector<T> &alpha, da_int &n_support);
};

} // namespace da_svm

} // namespace ARCH
