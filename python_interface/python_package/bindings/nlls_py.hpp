
/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DA_PY_NLLS_CB_HPP
#define DA_PY_NLLS_CB_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"

#include <iostream>
#include <optional>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <stdexcept>
#include <string>
#include <typeinfo>

using namespace std::string_literals;

namespace nlls_cb {

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

// Call-back type 1:
// Signature for fun (resfun), jac (resgrd), and hes(reshes) python functions
template <typename T> struct meta_py_cb1_t {
    static_assert(
        std::is_floating_point<T>::value,
        "Template argument for residual call-back (of type 1) must be floating point");
    using type = std::function<da_int(py::array_t<T>, py::array_t<T>, py::object)>;
};
template <typename T> using py_cb1_t = typename meta_py_cb1_t<T>::type;

// Call-back type 2:
// Signature for hp(reshp) python function
template <typename T> struct meta_py_cb2_t {
    static_assert(
        std::is_floating_point<T>::value,
        "Template argument for residual call-back (of type 2) must be floating point");
    using type =
        std::function<da_int(py::array_t<T>, py::array_t<T>, py::array_t<T>, py::object)>;
};
template <typename T> using py_cb2_t = typename meta_py_cb2_t<T>::type;

template <typename T> struct cb_t {
    py_cb1_t<T> *f{nullptr};
    py_cb1_t<T> *j{nullptr};
    py_cb2_t<T> *hf{nullptr};
    py_cb2_t<T> *hp{nullptr};
    void set(py_cb1_t<T> &f, py_cb1_t<T> &j, py_cb2_t<T> &hf, py_cb2_t<T> &hp) {
        this->f = &f;
        this->j = &j;
        this->hf = &hf;
        this->hp = &hp;
    }
    void get(py_cb1_t<T> &f, py_cb1_t<T> &j, py_cb2_t<T> &hf, py_cb2_t<T> &hp) {
        f = *(this->f);
        j = *(this->j);
        hf = *(this->hf);
        hp = *(this->hp);
    }
};

// container for the call-backs
// FIXME: see if cb_t signature can be replaced with py::function
class callbacks_t {
  private:
    struct cb_t<double> cb_d;
    struct cb_t<float> cb_s;

  public:
    void set(py_cb1_t<double> &f, py_cb1_t<double> &j, py_cb2_t<double> &hf,
             py_cb2_t<double> &hp) {
        cb_d.set(f, j, hf, hp);
    };
    void set(py_cb1_t<float> &f, py_cb1_t<float> &j, py_cb2_t<float> &hf,
             py_cb2_t<float> &hp) {
        cb_s.set(f, j, hf, hp);
    };
    void get_f(py_cb1_t<double> *&f) { f = cb_d.f; };
    void get_j(py_cb1_t<double> *&j) { j = cb_d.j; };
    void get_hf(py_cb2_t<double> *&hf) { hf = cb_d.hf; };
    void get_hp(py_cb2_t<double> *&hp) { hp = cb_d.hp; };
    void get_f(py_cb1_t<float> *&f) { f = cb_s.f; };
    void get_j(py_cb1_t<float> *&j) { j = cb_s.j; };
    void get_hf(py_cb2_t<float> *&hf) { hf = cb_s.hf; };
    void get_hp(py_cb2_t<float> *&hp) { hp = cb_s.hp; };
    // actual user data
    py::object data = py::none();
    bool storage_scheme_c;
};

// c++ wrappers with python objects
template <typename T>
da_int py_wrapper_resfun_t(da_int n_coef, da_int n_res, void *cb_data, const T *x, T *r) {
    auto px = py::array_t<T>{n_coef, x, py::cast(x)};
    auto pr = py::array_t<T>{n_res, r, py::cast(r)};
    // ASSUMES cb_data is PROPERLY set up
    callbacks_t *callbacks = (callbacks_t *)cb_data;
    py_cb1_t<T> *fun;
    callbacks->get_f(fun);
    try {
        return (*fun)(px, pr, callbacks->data);
    } catch (py::error_already_set &e) {
        py::print(e.what());
        return -10;
    } catch (std::exception &e) {
        // unexpected error...
        std::cerr << e.what() << std::endl;
        return -20;
    }
}
template <typename T>
da_int py_wrapper_resgrd_t(da_int n_coef, da_int n_res, void *cb_data, const T *x, T *J) {
    auto px = py::array_t<T>{n_coef, x, py::cast(x)};
    // ASSUMES cb_data is PROPERLY set up
    callbacks_t *callbacks = (callbacks_t *)cb_data;
    py_cb1_t<T> *jac;
    callbacks->get_j(jac);
    try {
        if (callbacks->storage_scheme_c) {
            auto pJ = py::array_t<T, py::array::c_style | py::array::forcecast>{
                {n_res, n_coef}, J, py::cast(J)};
            return (*jac)(px, pJ, callbacks->data);
        } else {
            auto pJ = py::array_t<T, py::array::f_style | py::array::forcecast>{
                {n_res, n_coef}, J, py::cast(J)};
            return (*jac)(px, pJ, callbacks->data);
        }
    } catch (py::error_already_set &e) {
        py::print(e.what());
        return -10;
    } catch (std::exception &e) {
        // unexpected error...
        std::cerr << e.what() << std::endl;
        return -20;
    }
}
template <typename T>
da_int py_wrapper_reshes_t(da_int n_coef, da_int n_res, void *cb_data, const T *x,
                           const T *r, T *HF) {
    auto px = py::array_t<T>{n_coef, x, py::cast(x)};
    auto pr = py::array_t<T>{n_res, r, py::cast(r)};
    // ASSUMES cb_data is PROPERLY set up
    callbacks_t *callbacks = (callbacks_t *)cb_data;
    py_cb2_t<T> *hf;
    callbacks->get_hf(hf);
    try {
        if (callbacks->storage_scheme_c) {
            auto pHF = py::array_t<T, py::array::c_style | py::array::forcecast>{
                {n_coef, n_coef}, HF, py::cast(HF)};
            return (*hf)(px, pr, pHF, callbacks->data);
        } else {
            auto pHF = py::array_t<T, py::array::f_style | py::array::forcecast>{
                {n_coef, n_coef}, HF, py::cast(HF)};
            return (*hf)(px, pr, pHF, callbacks->data);
        }
    } catch (py::error_already_set &e) {
        py::print(e.what());
        return -10;
    } catch (std::exception &e) {
        // unexpected error...
        std::cerr << e.what() << std::endl;
        return -20;
    }
}
template <typename T>
da_int py_wrapper_reshp_t(da_int n_coef, da_int n_res, const T *x, const T *y, T *HP,
                          void *cb_data) {
    auto px = py::array_t<T>{n_coef, x, py::cast(x)};
    auto p_y = py::array_t<T>{n_coef, y, py::cast(y)};
    // ASSUMES cb_data is PROPERLY set up
    callbacks_t *callbacks = (callbacks_t *)cb_data;
    py_cb2_t<T> *hp;
    callbacks->get_hp(hp);
    try {
        if (callbacks->storage_scheme_c) {
            auto pHP = py::array_t<T, py::array::c_style | py::array::forcecast>{
                {n_coef, n_res}, HP, py::cast(HP)};
            return (*hp)(px, p_y, pHP, callbacks->data);
        } else {
            auto pHP = py::array_t<T, py::array::f_style | py::array::forcecast>{
                {n_coef, n_res}, HP, py::cast(HP)};
            return (*hp)(px, p_y, pHP, callbacks->data);
        }
    } catch (py::error_already_set &e) {
        py::print(e.what());
        return -10;
    } catch (std::exception &e) {
        // unexpected error...
        std::cerr << e.what() << std::endl;
        return -20;
    }
}
} // namespace nlls_cb

// Declaration of NLLS C++ wrapper call-backs - actual definition are in
// aoclda_pywrappers.cpp
da_int py_wrapper_resfun_d(da_int n_coef, da_int n_res, void *data, const double *x,
                           double *r);
da_int py_wrapper_resfun_s(da_int n_coef, da_int n_res, void *data, const float *x,
                           float *r);
da_int py_wrapper_resgrd_d(da_int n_coef, da_int n_res, void *data, const double *x,
                           double *J);
da_int py_wrapper_resgrd_s(da_int n_coef, da_int n_res, void *data, const float *x,
                           float *J);
da_int py_wrapper_reshes_d(da_int n_coef, da_int n_res, void *data, const double *x,
                           const double *r, double *HF);
da_int py_wrapper_reshes_s(da_int n_coef, da_int n_res, void *data, const float *x,
                           const float *r, float *HF);
da_int py_wrapper_reshp_d(da_int n_coef, da_int n_res, const double *x, const double *y,
                          double *HP, void *data);
da_int py_wrapper_reshp_s(da_int n_coef, da_int n_res, const float *x, const float *y,
                          float *HP, void *data);

class nlls : public pyda_handle {

    da_precision precision{da_unknown};
    da_int ncoef{0};
    da_int nres{0};
    bool storage_scheme_c{true};

    class nlls_cb::callbacks_t callbacks;

  public:
    nlls(da_int n_coef, da_int n_res, std::optional<py::array> weights,
         std::optional<py::array> lower_bounds, std::optional<py::array> upper_bounds,
         std::string order = "c", std::string prec = "double",
         std::string model = "hybrid", std::string method = "galahad",
         std::string glob_strategy = "tr", std::string reg_power = "quadratic",
         da_int verbose = (da_int)0) {

        da_status status{da_status_success};
        // prep precision string
        std::string mesg{prec};
        const std::regex ltrim("^[[:space:]]+");
        const std::regex rtrim("[[:space:]]+$");
        const std::regex squeeze("[[:space:]]+");
        mesg = std::regex_replace(mesg, ltrim, std::string(""));
        mesg = std::regex_replace(mesg, rtrim, std::string(""));
        mesg = std::regex_replace(mesg, squeeze, std::string(" "));
        transform(mesg.begin(), mesg.end(), mesg.begin(), ::tolower);
        if (mesg == "double"s)
            precision = da_double;
        else if (mesg == "single"s)
            precision = da_single;
        mesg = "";

        if (precision == da_double) {
            using T = double;
            da_handle_init<T>(&handle, da_handle_nlls);
            if (handle)
                // just store in the handle n_coef and n_res used for the bounds and weights
                // will be correctly set up by an additional call)
                status =
                    da_nlls_define_residuals_d(handle, n_coef, n_res, py_wrapper_resfun_d,
                                               py_wrapper_resgrd_d, nullptr, nullptr);
            else {
                status = da_status_handle_not_initialized;
                mesg = "Handle could not be initialized.";
            }
        } else if (precision == da_single) {
            using T = float;
            da_handle_init<T>(&handle, da_handle_nlls);
            if (handle)
                // just store in the handle n_coef and n_res used for the bounds and weights
                // will be correctly set up by an additional call)
                status =
                    da_nlls_define_residuals_s(handle, n_coef, n_res, py_wrapper_resfun_s,
                                               py_wrapper_resgrd_s, nullptr, nullptr);
            else {
                status = da_status_handle_not_initialized;
                mesg = "Handle could not be initialized.";
            }
        } else {
            status = da_status_wrong_type;
            mesg = "Invalid floating precision type argument ``prec``, try ''double'' "
                   "(default) or 'single'.";
        }

        exception_check(status, mesg);

        // from here on precision is known and valid

        // Add options
        status = da_options_set(handle, "ralfit model", model.c_str());
        exception_check(status, mesg);
        status = da_options_set(handle, "ralfit nlls method", method.c_str());
        exception_check(status, mesg);
        status =
            da_options_set(handle, "ralfit globalization method", glob_strategy.c_str());
        exception_check(status, mesg);
        status = da_options_set(handle, "regularization power", reg_power.c_str());
        exception_check(status, mesg);
        if (verbose < 0 || verbose > 5) {
            status = da_status_option_invalid_value;
            mesg = "Option ``verbose`` must be between 0 and 5.";
            exception_check(status, mesg);
        }
        status = da_options_set(handle, "print level", verbose);
        exception_check(status, mesg);
        if (verbose > 1) {
            da_options_set(handle, "print options", "yes");
        }

        da_int dim{0};

        // Add optional components
        if (weights.has_value()) {
            auto w = weights.value();
            dim = w.ndim();
            if (dim == 1) {
                da_int wlen = w.shape(0);
                if (w.dtype().is(py::dtype::of<double>())) {
                    double *wptr = py::array_t<double>(w).mutable_data();
                    status = da_nlls_define_weights(handle, wlen, wptr);
                } else if (w.dtype().is(py::dtype::of<float>())) {
                    float *wptr = py::array_t<float>(w).mutable_data();
                    status = da_nlls_define_weights(handle, wlen, wptr);
                } else {
                    status = da_status_wrong_type;
                    mesg = "Vector `weights` is not of the same dtype as the one "
                           "defined in the parameter `prec`.";
                }
            } else {
                status = da_status_invalid_input;
                mesg = "`weights` argument is not a 1D column array (ndim>1).";
            }
        }
        exception_check(status, mesg);

        da_int llen{0}, ulen{0};
        double *lower_d{nullptr}, *upper_d{nullptr};
        float *lower_s{nullptr}, *upper_s{nullptr};
        // Add bounds
        if (lower_bounds.has_value()) {
            auto l = lower_bounds.value();
            dim = l.ndim();
            if (dim != 1) {
                status = da_status_invalid_input;
                mesg = "`lower_bounds` argument is not a 1D column array (ndim>1).";
                exception_check(status, mesg);
            }
            llen = l.shape(0);
            if (llen > 0) {
                if (l.dtype().is(py::dtype::of<double>()) && precision == da_double)
                    lower_d = py::array_t<double>(l).mutable_data();
                else if (l.dtype().is(py::dtype::of<float>()) && precision == da_single)
                    lower_s = py::array_t<float>(l).mutable_data();
                else {
                    status = da_status_wrong_type;
                    mesg = "Vector `lower_bounds` is not of the same dtype as the one "
                           "defined in the parameter `prec`.";
                }
            }
            exception_check(status, mesg);
        }

        if (upper_bounds.has_value()) {
            auto u = upper_bounds.value();
            dim = u.ndim();
            if (dim != 1) {
                status = da_status_invalid_input;
                mesg = "`upper_bounds` argument is not a 1D column array (ndim>1).";
                exception_check(status, mesg);
            }
            ulen = u.shape(0);
            if (ulen > 0) {
                if (u.dtype().is(py::dtype::of<double>()) && precision == da_double)
                    upper_d = py::array_t<double>(u).mutable_data();
                else if (u.dtype().is(py::dtype::of<float>()) && precision == da_single)
                    upper_s = py::array_t<float>(u).mutable_data();
                else {
                    status = da_status_wrong_type;
                    mesg = "Vector `upper_bounds` is not of the same dtype as the one "
                           "defined in the parameter `prec`.";
                }
            }
            exception_check(status, mesg);
        }

        if (llen > 0 || ulen > 0) {
            // make sure that both have the same length if pointers are valid
            da_int len{0};
            bool ok{true};
            if (lower_d || upper_s)
                len = llen;
            if (upper_d || upper_s) {
                if (len > 0)
                    ok = ulen == len;
                else
                    len = ulen;
            }
            if (!ok) {
                status = da_status_invalid_input;
                mesg = "The arrays `lower_bound` and `upper_bound` must either be of "
                       "the same size or empty.";
                exception_check(status, mesg);
            }

            if (precision == da_double)
                status = da_nlls_define_bounds(handle, len, lower_d, upper_d);
            else if (precision == da_single)
                status = da_nlls_define_bounds(handle, len, lower_s, upper_s);
            else {
            }
            exception_check(status, mesg);
        }

        // Options
        // order default is "c"
        status = da_options_set(handle, "storage scheme", order.c_str());
        exception_check(status);
        char opt_order[20];
        da_int lorder;
        da_int okey;
        status = da_options_get(handle, "storage scheme", opt_order, &lorder, &okey);
        exception_check(status);
        // namespace da_optimization_options { enum storage_scheme { fortran = 1, c = 2 };
        // see optiomization_options.hpp
        this->storage_scheme_c = okey == 2;

        this->ncoef = n_coef;
        this->nres = n_res;
    }
    ~nlls() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T> x, nlls_cb::py_cb1_t<T> &fun,
             std::optional<nlls_cb::py_cb1_t<T>> jac,
             std::optional<nlls_cb::py_cb2_t<T>> hes,
             std::optional<nlls_cb::py_cb2_t<T>> hp, std::optional<py::object> data,
             T ftol = 1.0e-8, T abs_ftol = 1.0e-8, T gtol = 1.0e-8, T abs_gtol = 1.0e-5,
             T xtol = 2.22e-16, T reg_term = 0.0, da_int maxit = da_int(100)) {

        da_status status{da_status_success};
        std::string mesg{""};

        if (!handle) {
            status = da_status_handle_not_initialized;
            mesg = "Handle could not be initialized.";
            exception_check(status, mesg);
        }
        if constexpr (std::is_same_v<T, double>) {
            if (this->precision != da_double) {
                status = da_status_wrong_type;
            }
        } else if constexpr (std::is_same_v<T, float>) {
            if (this->precision != da_single) {
                status = da_status_wrong_type;
            }
        } else {
            status = da_status_wrong_type;
        }
        if (status != da_status_success) {
            mesg =
                "Wrong specialization called: T="s + std::string(typeid(T).name()) + "?"s;
            exception_check(status, mesg);
        }

        if (x.ndim() != 1 || x.shape()[0] != this->ncoef) {
            status = da_status_invalid_array_dimension;
            mesg =
                "``x`` must be a 1D array of size " + std::to_string(this->ncoef) + ".";
            exception_check(status, mesg);
        }

        // gather all ptrs to the python functions (user call-backs)
        nlls_cb::py_cb1_t<T> py_jac = jac.value_or(nullptr);
        nlls_cb::py_cb2_t<T> py_hf = hes.value_or(nullptr);
        nlls_cb::py_cb2_t<T> py_hp = hp.value_or(nullptr);

        // gather all ptrs to the cpp (py_wrapper_res*) call-backs
        if constexpr (std::is_same_v<T, double>) {
            da_resfun_t_d *cxx_fun{py_wrapper_resfun_d};
            da_resgrd_t_d *cxx_jac{py_wrapper_resgrd_d};
            da_reshes_t_d *cxx_hf{py_wrapper_reshes_d};
            da_reshp_t_d *cxx_hp{py_wrapper_reshp_d};
            if (!jac.has_value())
                cxx_jac = nullptr;
            if (!hes.has_value())
                cxx_hf = nullptr;
            if (!hp.has_value())
                cxx_hp = nullptr;
            status = da_nlls_define_residuals_d(handle, this->ncoef, this->nres, cxx_fun,
                                                cxx_jac, cxx_hf, cxx_hp);
        } else {
            da_resfun_t_s *cxx_fun{py_wrapper_resfun_s};
            da_resgrd_t_s *cxx_jac{py_wrapper_resgrd_s};
            da_reshes_t_s *cxx_hf{py_wrapper_reshes_s};
            da_reshp_t_s *cxx_hp{py_wrapper_reshp_s};
            if (!jac.has_value())
                cxx_jac = nullptr;
            if (!hes.has_value())
                cxx_hf = nullptr;
            if (!hp.has_value())
                cxx_hp = nullptr;
            status = da_nlls_define_residuals_s(handle, this->ncoef, this->nres, cxx_fun,
                                                cxx_jac, cxx_hf, cxx_hp);
        }
        exception_check(status, mesg);

        // pass the options
        status = da_options_set(handle, "ralfit convergence rel tol fun", ftol);
        exception_check(status, mesg);
        status = da_options_set(handle, "ralfit convergence rel tol grd", gtol);
        exception_check(status, mesg);
        status = da_options_set(handle, "ralfit convergence abs tol fun", abs_ftol);
        exception_check(status, mesg);
        status = da_options_set(handle, "ralfit convergence abs tol grd", abs_gtol);
        exception_check(status, mesg);
        status = da_options_set(handle, "ralfit convergence step size", xtol);
        exception_check(status, mesg);
        status = da_options_set(handle, "ralfit iteration limit", maxit);
        exception_check(status, mesg);
        status = da_options_set(handle, "regularization term", reg_term);
        exception_check(status, mesg);

        callbacks.set(fun, py_jac, py_hf, py_hp);
        callbacks.storage_scheme_c = this->storage_scheme_c;
        // attach optional user data
        callbacks.data = data.value_or(py::none());

        // Call solver
        void *udata = &callbacks;
        status = da_nlls_fit(this->handle, this->ncoef, x.mutable_data(), udata);

        exception_check(status);
    }
    // Query handle for information
    template <typename T>
    void get_info(da_int &iter, da_int &f_eval, da_int &g_eval, da_int &h_eval,
                  da_int &hp_eval, T &obj, T &norm_g, T &scaled_g) {
        da_status status;
        da_int dim = 100;
        T info[100];
        status = da_handle_get_result(handle, da_rinfo, &dim, info);
        exception_check(status);

        iter = da_int(info[info_t::info_iter]);
        f_eval = da_int(info[info_t::info_nevalf]);
        g_eval = da_int(info[info_t::info_nevalg]);
        h_eval = da_int(info[info_t::info_nevalh]);
        hp_eval = da_int(info[info_t::info_nevalhp]);
        obj = info[info_t::info_objective];
        norm_g = info[info_t::info_grad_norm];
        scaled_g = info[info_t::info_scl_grad_norm];
    }

    // Getters for info
    auto get_info_iter() {
        da_int iter, f_eval, g_eval, h_eval, hp_eval;
        if (precision == da_single) {
            using T = float;
            T obj, norm_g, scaled_g;
            get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g);
        } else {
            using T = double;
            T obj, norm_g, scaled_g;
            get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g);
        }
        return iter;
    }
    auto get_info_evals() {
        da_int iter, f_eval, g_eval, h_eval, hp_eval;
        if (precision == da_single) {
            using T = float;
            T obj, norm_g, scaled_g;
            get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g);
            return py::dict("f"_a = f_eval, "j"_a = g_eval, "h"_a = h_eval,
                            "hp"_a = hp_eval);
        } else {
            using T = double;
            T obj, norm_g, scaled_g;
            get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g);
            return py::dict("f"_a = da_int(f_eval), "j"_a = da_int(g_eval),
                            "h"_a = da_int(h_eval), "hp"_a = da_int(hp_eval));
        }
    }
    auto get_info_optim() {
        da_int iter, f_eval, g_eval, h_eval, hp_eval;
        if (precision == da_single) {
            using T = float;
            T obj, norm_g, scaled_g;
            get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g);
            return py::dict("obj"_a = obj, "norm_g"_a = norm_g,
                            "scl_norm_g"_a = scaled_g);
        } else {
            using T = double;
            T obj, norm_g, scaled_g;
            get_info(iter, f_eval, g_eval, h_eval, hp_eval, obj, norm_g, scaled_g);
            return py::dict("obj"_a = obj, "norm_g"_a = norm_g,
                            "scl_norm_g"_a = scaled_g);
        }
    }
};

#endif