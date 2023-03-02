/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef LINMOD_OPTIONS_HPP
#define LINMOD_OPTIONS_HPP

#include "aoclda.h"
#include "linear_model.hpp"
#include "options.hpp"
#include <limits>

template <class T> da_status register_linmod_options(da_options::OptionRegistry &opts) {
    using namespace da_options;
    T safe_eps = (T)2.0 * std::numeric_limits<T>::epsilon();
    T safe_tol = std::sqrt(safe_eps);
    T max_real = std::numeric_limits<T>::max();

    try {
        std::shared_ptr<OptionNumeric<bool>> ob;
        ob = std::make_shared<OptionNumeric<bool>>(OptionNumeric<bool>(
            "linmod intercept", "Add intercept variable to the model", false));
        opts.register_opt(ob);

        std::shared_ptr<OptionNumeric<T>> oT;
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("linmod norm2 reg", "norm2 regularization term", 0.0,
                             da_options::lbound_t::greaterequal, max_real,
                             da_options::ubound_t::lessthan, 0.0));
        opts.register_opt(oT);
        oT = std::make_shared<OptionNumeric<T>>(
            OptionNumeric<T>("linmod norm1 reg", "norm1 regularization term", 0.0,
                             da_options::lbound_t::greaterequal, max_real,
                             da_options::ubound_t::lessthan, 0.0));
        opts.register_opt(oT);

        std::shared_ptr<OptionString> os;
        os = std::make_shared<OptionString>(
            OptionString("linmod optim method", "Select optimization method to use",
                         {{"auto", 0}, {"lbfgs", 1}, {"qr", 2}}, "auto"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    } catch (...) {
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error;
    }

#if 0
    ob = std::make_shared<OptionNumeric<bool>>(OptionNumeric<bool>(
        "linmod regularization", "Add regularization to the model", true));
    opts.register_opt(ob);

    std::shared_ptr<OptionNumeric<da_int>> oi;
    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "iteration limit", "Maximum number of iterations", 1,
        da_options::lbound_t::greaterequal, 1, da_options::ubound_t::p_inf, 100));
    opts.register_opt(oi);
    oi = std::make_shared<OptionNumeric<da_int>>(OptionNumeric<da_int>(
        "monitoring step", "Step number of iterations", 0,
        da_options::lbound_t::greaterequal, 1000, da_options::ubound_t::lessequal, 50));
    opts.register_opt(oi);

    std::shared_ptr<OptionNumeric<T>> oT;
    oT = std::make_shared<OptionNumeric<T>>(OptionNumeric<T>(
        "convergence tol", "tolerance WRT to macheps, see option safe_eps", safe_eps,
        da_options::lbound_t::greaterthan, (T)1.0, da_options::ubound_t::lessthan,
        safe_tol));
    opts.register_opt(oT);

    std::shared_ptr<OptionString> os;
    os = std::make_shared<OptionString>(
        OptionString("precond type", "Select preconditioner to use",
                     {{"ichol", 1}, {"ilu", 2}, {"jacobi", 0}, {"blkjac", 4}}, "blkjac"));
    opts.register_opt(os);

    opts.print_options();
    opts.print_details(false);
    opts.print_details(true);

    // test option getter and setter: ALL WORK OK
    ob->set(false);  // was true
    oi->set(23);     // was 50
    oT->set((T)0.5); // was safe_tol<T>
    os->set("ilu");  //was blkjac
    da_int i;
    T t;
    bool b;
    string s;
    ob->get(b);
    oi->get(i);
    oT->get(t);
    os->get(s);
    os->get(s, i);
    std::cout << "Boolean get ==> " << ob->get_name() << " : " << b << std::endl;
    std::cout << "Integer get ==> " << oi->get_name() << " : " << i << std::endl;
    std::cout << "<T>     get ==> " << oT->get_name() << " : " << std::scientific << t
              << std::endl;
    std::cout << "String  get ==> " << os->get_name() << " : " << s << "id: " << i << std::endl;
    opts.print_details(false); // should report some options are set by_user

    // Registry SETTER and GETTER 

    string option;

    option = "linmod regularization";
    if (opts.set(option, true) != da_status_success)
        std::cout << "SET FAILED" + string(__FILE__) + to_string(__LINE__) << std::endl;

    option = "monitoring step";
    if (opts.set(option, 97) != da_status_success)
        std::cout << "SET FAILED" + string(__FILE__) + to_string(__LINE__) << std::endl;

    option = "convergence tol";
    if (opts.set(option, (T)0.01) != da_status_success)
        std::cout << "SET FAILED" + string(__FILE__) + to_string(__LINE__) << std::endl;

    option = "precond type";
    if (opts.set(option, "ichol", da_options::solver) != da_status_success)
        std::cout << "SET FAILED" + string(__FILE__) + to_string(__LINE__) << std::endl;

    opts.print_details(false);

    bool bret;
    da_status status;
    option = "linmod intercept";
    status = opts.get(option, bret);
    std::cout << "Option Registry: " << option << " val = " << bret
              << "  Return = " << status << std::endl;

    da_int iret;
    option = "monitoring step ";
    status = opts.get(option, iret);
    std::cout << "Option Registry: " << option << " val = " << iret
              << "  Return = " << status << std::endl;

    T Tret;
    option = "convergence tol";
    status = opts.get(option, Tret);
    std::cout << "Option Registry: " << option << " val = " << Tret
              << "  Return = " << status << std::endl;

    string sret;
    option = "precond type";
    status = opts.get(option, sret);
    std::cout << "Option Registry: " << option << " val = " << sret
              << "  Return = " << status << std::endl;

    da_int id;
    status = opts.get(option, sret, id);
    std::cout << "Option Registry: " << option << " val = " << sret << " id = " << id 
              << "  Return = " << status << std::endl;
#endif
    return da_status_success;
}

#endif //LINMOD_OPTIONS_HPP