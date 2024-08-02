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
#include "da_handle.hpp"
#include "svm.hpp"

da_status da_svm_select_model_d(da_handle handle, da_svm_model mod) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_svm::svm<double> *svm_d =
        dynamic_cast<da_svm::svm<double> *>(handle->alg_handle_d);
    if (svm_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_d->select_model(mod);
}

da_status da_svm_select_model_s(da_handle handle, da_svm_model mod) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_svm::svm<float> *svm_s = dynamic_cast<da_svm::svm<float> *>(handle->alg_handle_s);
    if (svm_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_s->select_model(mod);
}

da_status da_svm_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                            const double *X, da_int ldx_train, const double *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_svm::svm<double> *svm_d =
        dynamic_cast<da_svm::svm<double> *>(handle->alg_handle_d);
    if (svm_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_d->set_data(n_samples, n_features, X, ldx_train, y);
}

da_status da_svm_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                            const float *X, da_int ldx_train, const float *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_svm::svm<float> *svm_s = dynamic_cast<da_svm::svm<float> *>(handle->alg_handle_s);
    if (svm_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_s->set_data(n_samples, n_features, X, ldx_train, y);
}

da_status da_svm_compute_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_svm::svm<double> *svm_d =
        dynamic_cast<da_svm::svm<double> *>(handle->alg_handle_d);
    if (svm_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_d->compute();
}

da_status da_svm_compute_s(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_svm::svm<float> *svm_s = dynamic_cast<da_svm::svm<float> *>(handle->alg_handle_s);
    if (svm_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_s->compute();
}

da_status da_svm_predict_d(da_handle handle, da_int n_samples, da_int n_features,
                           const double *X_test, da_int ldx_test, double *predictions) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_svm::svm<double> *svm_d =
        dynamic_cast<da_svm::svm<double> *>(handle->alg_handle_d);
    if (svm_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_d->predict(n_samples, n_features, X_test, ldx_test, predictions);
}

da_status da_svm_predict_s(da_handle handle, da_int n_samples, da_int n_features,
                           const float *X_test, da_int ldx_test, float *predictions) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_svm::svm<float> *svm_s = dynamic_cast<da_svm::svm<float> *>(handle->alg_handle_s);
    if (svm_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_s->predict(n_samples, n_features, X_test, ldx_test, predictions);
}

da_status da_svm_decision_function_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X_test,
                                     da_int ldx_test, double *decision_vales, da_int ldd,
                                     da_svm_decision_function_shape shape) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than dobule.");
    da_svm::svm<double> *svm_d =
        dynamic_cast<da_svm::svm<double> *>(handle->alg_handle_d);
    if (svm_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_d->decision_function(n_samples, n_features, X_test, ldx_test,
                                    decision_vales, ldd, shape);
}

da_status da_svm_decision_function_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X_test,
                                     da_int ldx_test, float *decision_vales, da_int ldd,
                                     da_svm_decision_function_shape shape) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_svm::svm<float> *svm_s = dynamic_cast<da_svm::svm<float> *>(handle->alg_handle_s);
    if (svm_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_s->decision_function(n_samples, n_features, X_test, ldx_test,
                                    decision_vales, ldd, shape);
}

da_status da_svm_score_d(da_handle handle, da_int n_samples, da_int n_features,
                         const double *X_test, da_int ldx_test, const double *y_test,
                         double *score) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    da_svm::svm<double> *svm_d =
        dynamic_cast<da_svm::svm<double> *>(handle->alg_handle_d);
    if (svm_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_d->score(n_samples, n_features, X_test, ldx_test, y_test, score);
}

da_status da_svm_score_s(da_handle handle, da_int n_samples, da_int n_features,
                         const float *X_test, da_int ldx_test, const float *y_test,
                         float *score) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    da_svm::svm<float> *svm_s = dynamic_cast<da_svm::svm<float> *>(handle->alg_handle_s);
    if (svm_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with "
                        "handle_type=da_handle_svm or "
                        "handle is invalid.");

    return svm_s->score(n_samples, n_features, X_test, ldx_test, y_test, score);
}