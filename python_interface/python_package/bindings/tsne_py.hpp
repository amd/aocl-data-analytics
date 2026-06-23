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
 */

#ifndef TSNE_PY_HPP
#define TSNE_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "internal_utilities_py.hpp"
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class tsne : public pyda_handle {
  public:
    tsne(da_int n_components = 2, da_int max_iter = 1000, std::string init = "pca",
         da_int seed = -1, std::string prec = "double", bool check_data = false,
         bool mixed_precision = false, da_int low_precision_max_iter = 200) {
        if (prec == "double")
            da_handle_init<double>(&handle, da_handle_tsne);
        else if (prec == "single") {
            da_handle_init<float>(&handle, da_handle_tsne);
            precision = da_single;
        }
        da_status status;
        status = da_options_set_int(handle, "n_components", n_components);
        exception_check(status);
        status = da_options_set_int(handle, "max_iter", max_iter);
        exception_check(status);
        status = da_options_set_string(handle, "init", init.c_str());
        exception_check(status);
        status = da_options_set_int(handle, "seed", seed);
        exception_check(status);
        if (mixed_precision == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "mixed precision", yes_str.c_str());
            exception_check(status);
        }
        status =
            da_options_set_int(handle, "low precision max_iter", low_precision_max_iter);
        exception_check(status);
        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.c_str());
            exception_check(status);
        }
    }
    ~tsne() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T> X, T perplexity = 30.0, T learning_rate = -1.0,
             T early_exaggeration = 12.0, T theta = 0.5,
             da_int n_iter_without_progress = 300, T min_grad_norm = (T)1e-7,
             std::optional<py::array_t<T>> init_embedding = std::nullopt,
             T low_precision_min_grad_norm = (T)1e-4) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;
        da_int n_samples, n_features, ldx;
        get_numpy_array_properties(X, n_samples, n_features, ldx);

        status = da_options_set(handle, "perplexity", perplexity);
        exception_check(status);
        status = da_options_set(handle, "learning rate", learning_rate);
        exception_check(status);
        status = da_options_set(handle, "early exaggeration", early_exaggeration);
        exception_check(status);
        status = da_options_set(handle, "theta", theta);
        exception_check(status);
        status = da_options_set_int(handle, "n_iter_without_progress",
                                    n_iter_without_progress);
        exception_check(status);
        status = da_options_set(handle, "min_grad_norm", min_grad_norm);
        exception_check(status);
        status = da_options_set(handle, "low precision min_grad_norm",
                                low_precision_min_grad_norm);
        exception_check(status);
        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        exception_check(status);
        status = da_tsne_set_data(handle, n_samples, n_features, X.data(), ldx);
        exception_check(status);
        // Set initial embedding if supplied (must be after set_data)
        if (init_embedding.has_value()) {
            da_int n_samples_y, n_components_y, ldy;
            get_numpy_array_properties(init_embedding.value(), n_samples_y,
                                       n_components_y, ldy);
            status = da_tsne_set_init_embedding(handle, init_embedding->data(), ldy);
            exception_check(status);
        }
        status = da_tsne_compute<T>(handle);
        exception_check(status);
    }

    template <typename T>
    py::array_t<T>
    fit_transform(py::array_t<T> X, T perplexity = 30.0, T learning_rate = -1.0,
                  T early_exaggeration = 12.0, T theta = 0.5,
                  da_int n_iter_without_progress = 300, T min_grad_norm = (T)1e-7,
                  std::optional<py::array_t<T>> init_embedding = std::nullopt,
                  T low_precision_min_grad_norm = (T)1e-4) {
        fit<T>(X, perplexity, learning_rate, early_exaggeration, theta,
               n_iter_without_progress, min_grad_norm, init_embedding,
               low_precision_min_grad_norm);
        return get_embedding_t<T>();
    }

    py::array get_embedding() {
        if (precision == da_single) {
            return get_embedding_t<float>();
        }
        return get_embedding_t<double>();
    }

    template <typename T> py::array_t<T> get_embedding_t() {
        da_int n_samples, n_features, n_components, n_iter;
        T kl_div;
        da_status status =
            get_rinfo<T>(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);
        da_int dim = n_samples * n_components;
        size_t shape[2]{(size_t)n_samples, (size_t)n_components};
        size_t strides[2];
        if (order == c_contiguous) {
            strides[0] = sizeof(T) * n_components;
            strides[1] = sizeof(T);
        } else {
            strides[0] = sizeof(T);
            strides[1] = sizeof(T) * n_samples;
        }

        auto res = py::array_t<T>(shape, strides);
        status =
            da_handle_get_result(handle, da_tsne_embedding, &dim, res.mutable_data());
        exception_check(status);
        return res;
    }

    double get_kl_divergence() {
        da_status status;
        da_int n_samples, n_features, n_components, n_iter;
        if (precision == da_single) {
            float kl_div;
            status = get_rinfo<float>(&n_samples, &n_features, &n_components, &n_iter,
                                      &kl_div);
            exception_check(status);
            return static_cast<double>(kl_div);
        }
        double kl_div;
        status =
            get_rinfo<double>(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);
        return kl_div;
    }

    auto get_n_samples() {
        da_status status;
        da_int n_samples, n_features, n_components, n_iter;
        if (precision == da_single) {
            float kl_div;
            status = get_rinfo<float>(&n_samples, &n_features, &n_components, &n_iter,
                                      &kl_div);
            exception_check(status);
        } else {
            double kl_div;
            status = get_rinfo<double>(&n_samples, &n_features, &n_components, &n_iter,
                                       &kl_div);
            exception_check(status);
        }
        return n_samples;
    }

    auto get_n_features() {
        da_status status;
        da_int n_samples, n_features, n_components, n_iter;
        if (precision == da_single) {
            float kl_div;
            status = get_rinfo<float>(&n_samples, &n_features, &n_components, &n_iter,
                                      &kl_div);
            exception_check(status);
        } else {
            double kl_div;
            status = get_rinfo<double>(&n_samples, &n_features, &n_components, &n_iter,
                                       &kl_div);
            exception_check(status);
        }
        return n_features;
    }

    auto get_n_components() {
        da_status status;
        da_int n_samples, n_features, n_components, n_iter;
        if (precision == da_single) {
            float kl_div;
            status = get_rinfo<float>(&n_samples, &n_features, &n_components, &n_iter,
                                      &kl_div);
            exception_check(status);
        } else {
            double kl_div;
            status = get_rinfo<double>(&n_samples, &n_features, &n_components, &n_iter,
                                       &kl_div);
            exception_check(status);
        }
        return n_components;
    }

    auto get_n_iter() {
        da_status status;
        da_int n_samples, n_features, n_components, n_iter;
        if (precision == da_single) {
            float kl_div;
            status = get_rinfo<float>(&n_samples, &n_features, &n_components, &n_iter,
                                      &kl_div);
            exception_check(status);
        } else {
            double kl_div;
            status = get_rinfo<double>(&n_samples, &n_features, &n_components, &n_iter,
                                       &kl_div);
            exception_check(status);
        }
        return n_iter;
    }

  private:
    template <typename T>
    da_status get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_components,
                        da_int *n_iter, T *kl_divergence) {
        da_int dim = 6;
        T rinfo[6];
        da_status status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
        if (status != da_status_success) {
            return status;
        }
        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
        *n_components = static_cast<da_int>(rinfo[2]);
        *n_iter = static_cast<da_int>(rinfo[3]);
        *kl_divergence = rinfo[4];
        return da_status_success;
    }
};

#endif
