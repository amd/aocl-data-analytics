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

#include "aoclda.h"
#include "da_handle.hpp"
#include "options.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <limits>
#include <regex>
#include <string>

namespace {

using namespace da_options;

OptionNumeric<da_int> opt_int("integer option", "Preloaded Integer Option", 0,
                              da_options::lbound_t::greaterequal, 10,
                              da_options::ubound_t::lessequal, 10);
std::shared_ptr<OptionNumeric<da_int>> oI;
OptionNumeric<float> opt_float("float option", "Preloaded Float Option", 0.0f,
                               da_options::lbound_t::greaterthan, 10.0f,
                               da_options::ubound_t::lessthan, 8.0f);
std::shared_ptr<OptionNumeric<float>> oF;
OptionNumeric<double> opt_double("double option", "Preloaded Double Option", 1.0f,
                                 da_options::lbound_t::greaterthan, 20.0f,
                                 da_options::ubound_t::lessthan, 16.0f);
std::shared_ptr<OptionNumeric<double>> oD;
OptionNumeric<bool> opt_bool("bool option", "Preloaded bool Option", true);
std::shared_ptr<OptionNumeric<bool>> oB;
// String option with categorical values
OptionString opt_string("string option", "Preloaded Categorical String Option",
                        {{"yes", 1}, {"no", 0}, {"maybe", 2}}, "yes");
// String option with free-form value
OptionString opt_ff_string("free-form string option", "Preloaded Free-Form String Option",
                           {}, "any");
std::shared_ptr<OptionString> oS;

da_status preload(OptionRegistry &r) {
    oS = std::make_shared<OptionString>(opt_string);
    da_status status;
    status = r.register_opt(oS);
    if (status != da_status_success)
        return status;

    oS = std::make_shared<OptionString>(opt_ff_string);
    status = r.register_opt(oS);
    if (status != da_status_success)
        return status;

    oI = std::make_shared<OptionNumeric<da_int>>(opt_int);
    status = r.register_opt(oI);
    if (status != da_status_success)
        return status;

    oF = std::make_shared<OptionNumeric<float>>(opt_float);
    status = r.register_opt(oF);
    if (status != da_status_success)
        return status;

    oD = std::make_shared<OptionNumeric<double>>(opt_double);
    status = r.register_opt(oD);
    if (status != da_status_success)
        return status;

    oB = std::make_shared<OptionNumeric<bool>>(opt_bool);
    status = r.register_opt(oB);
    if (status != da_status_success)
        return status;

    return status;
};

TEST(OpOptionInternal, OpClsCommon) {
    EXPECT_THROW(OptionNumeric<da_int> opt_i("", "Preloaded Integer Option", 0,
                                             da_options::lbound_t::greaterequal, 10,
                                             da_options::ubound_t::lessequal, 10),
                 std::invalid_argument);
    OptionNumeric<da_int> opt_i(" IntegeR    OptiOn    ", "Preloaded Integer Option", 0,
                                da_options::lbound_t::greaterequal, 10,
                                da_options::ubound_t::lessequal, 10);
    EXPECT_STRCASEEQ((opt_i.get_name()).c_str(), "integer option");
    EXPECT_EQ(opt_i.get_option_t(), da_options::option_t::opt_int);
    EXPECT_THROW(OptionString opt_s("      ", "Preloaded String Option",
                                    {{"yes", 1}, {"no", 0}, {"maybe", 2}}, "yes"),
                 std::invalid_argument);
    OptionString opt_s("  str   OPT  ", "Preloaded String Option",
                       {{"yes", 1}, {"no", 0}, {"maybe", 2}}, "yes");
    EXPECT_STRCASEEQ(opt_s.get_name().c_str(), "str opt");
    EXPECT_EQ(opt_s.get_option_t(), da_options::option_t::opt_string);
};

template <typename T> void OpClsNumeric(void) {
    std::string const descr("Preloaded Option");
    OptionNumeric<T> opt(" Placeholder    OptiOn    ", descr, 0,
                         da_options::lbound_t::greaterequal, 10,
                         da_options::ubound_t::lessequal, 10);
    // Call to cover pretty printing
    [[maybe_unused]] string pretty;
    pretty = opt.print_details(true) + opt.print_details(false, true) +
             opt.print_details(false, false);
    bool has_nan = std::numeric_limits<T>::has_quiet_NaN;
    T val = -999;
    opt.get(val);
    EXPECT_EQ(val, (T)10);
    EXPECT_EQ(opt.set((T)1000), da_status_option_invalid_value);
    // check print_detail() grep match Set-by: default
    std::string s_default("Set-by: (default");
    std::string s_user("Set-by: (user");
    std::string s_solver("Set-by: (solver");
    std::regex reg_default(s_default, std::regex::grep);
    std::regex reg_solver(s_solver, std::regex::grep);
    std::regex reg_user(s_user, std::regex::grep);
    std::string det = opt.print_details();
    std::smatch m;
    std::regex_search(det, m, reg_default);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_default.c_str());

    EXPECT_EQ(opt.set(1), da_status_success);
    // check print_detail() grep match Set-by: user
    det = opt.print_details();
    std::regex_search(det, m, reg_user);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_user.c_str());

    EXPECT_EQ(opt.set(2, da_options::setby_t::solver), da_status_success);
    // check print_detail() grep match Set-by: solver
    det = opt.print_details();
    std::regex_search(det, m, reg_solver);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_solver.c_str());

    std::string prn = opt.print_option();
    EXPECT_EQ(prn, " placeholder option = 2\n"s);

    // lower > upper
    EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, 10,
                                      da_options::lbound_t::greaterequal, 1,
                                      da_options::ubound_t::lessequal, 1),
                 std::invalid_argument);
    if (has_nan) {
        // lower = nan
        EXPECT_THROW(OptionNumeric<T> opt("Opt", descr,
                                          std::numeric_limits<T>::quiet_NaN(),
                                          da_options::lbound_t::greaterequal, 10,
                                          da_options::ubound_t::lessequal, 5),
                     std::invalid_argument);
        // upper = nan
        EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, -1,
                                          da_options::lbound_t::greaterequal,
                                          std::numeric_limits<T>::quiet_NaN(),
                                          da_options::ubound_t::lessequal, 5),
                     std::invalid_argument);
        // default = nan
        EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, -9,
                                          da_options::lbound_t::greaterequal, 10,
                                          da_options::ubound_t::lessequal,
                                          std::numeric_limits<T>::quiet_NaN()),
                     std::invalid_argument);
    }
    // default out of range l == u
    EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, 2, da_options::lbound_t::greaterthan,
                                      2, da_options::ubound_t::lessequal, -11),
                 std::invalid_argument);
    // default out of range l <= x <= u < d
    EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterequal,
                                      10, da_options::ubound_t::lessequal, 11),
                 std::invalid_argument);
    // default out of range l <= x < u = d
    EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterequal,
                                      10, da_options::ubound_t::lessthan, 10),
                 std::invalid_argument);
    // default out of range d < l <= x <= u
    EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterequal,
                                      10, da_options::ubound_t::lessequal, -11),
                 std::invalid_argument);
    // default out of range d = l <= x < u
    EXPECT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterthan,
                                      10, da_options::ubound_t::lessthan, 0),
                 std::invalid_argument);
    {
        OptionNumeric<T> pretty_print("Opt", descr, 0, da_options::lbound_t::greaterthan,
                                      10, da_options::ubound_t::lessthan, 5);
        pretty = pretty_print.print_details();
        pretty = pretty_print.print_details(false, true);
        pretty = pretty_print.print_details(false, false);
    }
    {
        OptionNumeric<T> pretty_print("Opt", descr, 0, da_options::lbound_t::m_inf, 10,
                                      da_options::ubound_t::p_inf, 0);
        pretty = pretty_print.print_details();
        pretty = pretty_print.print_details(false, true);
        pretty = pretty_print.print_details(false, false);
    }
}

// Bool specialization
template <> void OpClsNumeric<bool>(void) {
    std::string const descr("Preloaded Option");
    OptionNumeric<bool> opt(" Placeholder    OptiOn    ", descr, true);
    // Call to cover pretty printing
    [[maybe_unused]] string pretty;
    pretty = opt.print_details(true) + opt.print_details(false, true) +
             opt.print_details(false, false);
    bool val;
    opt.get(val);
    EXPECT_EQ(val, true);
    // check print_detail() grep match Set-by: default
    std::string s_default("Set-by: (default");
    std::string s_user("Set-by: (user");
    std::string s_solver("Set-by: (solver");
    std::regex reg_default(s_default, std::regex::grep);
    std::regex reg_solver(s_solver, std::regex::grep);
    std::regex reg_user(s_user, std::regex::grep);
    std::string det = opt.print_details();
    std::smatch m;
    std::regex_search(det, m, reg_default);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_default.c_str());

    EXPECT_EQ(opt.set(false), da_status_success);
    // check print_detail() grep match Set-by: user
    det = opt.print_details();
    std::regex_search(det, m, reg_user);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_user.c_str());

    EXPECT_EQ(opt.set(true, da_options::setby_t::solver), da_status_success);
    // check print_detail() grep match Set-by: solver
    det = opt.print_details();
    std::regex_search(det, m, reg_solver);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_solver.c_str());

    std::string prn = opt.print_option();
    EXPECT_EQ(prn, " placeholder option = true\n"s);

    {
        OptionNumeric<bool> pretty_print("Opt", descr, true);
        pretty = pretty_print.print_details(true);
        pretty = pretty_print.print_details(false, true);
        pretty = pretty_print.print_details(false, false);
    }
}

TEST(OpOptionInternal, OpClsNumericAll) {
    OpClsNumeric<float>();
    OpClsNumeric<double>();
    OpClsNumeric<da_int>();
    OpClsNumeric<bool>();
};

TEST(OpOptionInternal, OpClsStringAll) {

    std::string val;
    da_int id;
    // Categorical String Option
    ::opt_string.get(val);
    EXPECT_EQ(val, "yes");
    ::opt_string.get(val, id);
    EXPECT_EQ(id, 1);
    // Free-form String Option
    ::opt_ff_string.get(val);
    EXPECT_EQ(val, "any");
    EXPECT_THROW(::opt_ff_string.get(val, id), std::runtime_error);
    EXPECT_EQ(::opt_ff_string.set("New Free-Form Value", da_options::setby_t::solver),
              da_status_success);
    ::opt_ff_string.get(val);
    EXPECT_EQ(val, "new free-form value");
    // check print_detail() grep match Set-by: default
    std::string s_default("Set-by: (default");
    std::string s_user("Set-by: (user");
    std::string s_solver("Set-by: (solver");
    std::regex reg_default(s_default, std::regex::grep);
    std::regex reg_solver(s_solver, std::regex::grep);
    std::regex reg_user(s_user, std::regex::grep);
    std::string det = opt_string.print_details();
    std::smatch m;
    std::regex_search(det, m, reg_default);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_default.c_str());

    EXPECT_EQ(::opt_string.set("maybe"), da_status_success);
    // check print_detail() grep match Set-by: user
    det = opt_string.print_details();
    std::regex_search(det, m, reg_user);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_user.c_str());

    EXPECT_EQ(::opt_string.set("no", da_options::setby_t::solver), da_status_success);
    // check print_detail() grep match Set-by: solver
    det = ::opt_string.print_details(true);
    std::regex_search(det, m, reg_solver);
    EXPECT_STRCASEEQ(std::string(m[0]).c_str(), s_solver.c_str());
    [[maybe_unused]] std::string prn;
    prn = opt_string.print_option();
    EXPECT_EQ(prn, " string option = no\n"s);
    prn = opt_string.print_details(false);
    prn = opt_string.print_details(false, false);

    EXPECT_NO_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                            {{"yes", 1}, {"yes", 0}, {"yes", 5}}, "yes"));
    EXPECT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"No", 0}}, "           "),
                 std::invalid_argument);
    EXPECT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"   No  ", 0}}, "no"),
                 std::invalid_argument);
    EXPECT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"", 1}}, "yes"),
                 std::invalid_argument);
    EXPECT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"", 2}}, "yes"),
                 std::invalid_argument);
    EXPECT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"no", 0}, {"maybe", 2}},
                                         "   yes   "),
                 std::invalid_argument);
    EXPECT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"no", 0}, {"maybe", 2}},
                                         "invalid"),
                 std::invalid_argument);
    EXPECT_EQ(::opt_string.set("invalid"), da_status_option_invalid_value);
}

TEST(OpRegistryInternal, OpRegALL) {
    da_options::OptionRegistry reg;
    EXPECT_EQ(preload(reg), da_status_success);
    // test the lock
    reg.lock();
    EXPECT_EQ(reg.register_opt(oI), da_status_option_locked);
    reg.unlock();
    da_status status;
    // add option twice;
    status = reg.register_opt(oI);
    EXPECT_EQ(status, da_status_invalid_input);
    // add option (same name but different type)
    OptionNumeric<bool> opt_over("integer option", "Preloaded bool Option", true);
    std::shared_ptr<OptionNumeric<bool>> over =
        std::make_shared<OptionNumeric<bool>>(opt_over);
    status = reg.register_opt(over);
    EXPECT_EQ(status, da_status_invalid_input);

    // set with locked registry
    reg.lock();
    da_int one = 1;
    EXPECT_EQ(reg.set("integer opt", one), da_status_option_locked);
    reg.unlock();
    // option not found
    EXPECT_EQ(reg.set("nonexistent option", one), da_status_option_not_found);
    // set with the wrong type
    EXPECT_EQ(reg.set("integer option", "wrong"), da_status_option_wrong_type);
    EXPECT_EQ(reg.set("integer option", 3.33f), da_status_option_wrong_type);
    //EXPECT_EQ(reg.set("integer option", true), da_status_option_wrong_type);
    //bool b = false;
    //EXPECT_EQ(reg.get("integer option", &b), da_status_option_wrong_type);
    string ret;
    da_int id;
    EXPECT_EQ(reg.get("wrong string option", ret, id), da_status_option_not_found);
    EXPECT_EQ(reg.get("integer option", ret, id), da_status_option_wrong_type);
    // test string ff and categorical
    EXPECT_EQ(reg.set("string option", "yes"), da_status_success);
    EXPECT_EQ(reg.set("free-form string option", " new   value "), da_status_success);
    EXPECT_EQ(reg.get("free-form string option", ret), da_status_success);
    EXPECT_EQ(ret, "new value");

    reg.print_details(true);
    reg.print_details(false, false);
    reg.print_details(false, true);
    reg.print_options();
}

// Public API Unit-tests

TEST(OpRegistryWrappers, getset_string) {
    da_handle handle;
    OptionRegistry *opts;
    da_int n = 16;
    char sv[] = "yes";
    char str[25];
    char cv[25] = "quite long option value;";
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(handle->get_current_opts(&opts), da_status_success);
    EXPECT_EQ(preload(*opts), da_status_success);
    // String categorical
    EXPECT_EQ(da_options_set_string(nullptr, "string option", sv),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_get_string(nullptr, "string option", str, &n),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_set_string(handle, "string option", sv), da_status_success);
    char value[36];
    EXPECT_EQ(da_options_get_string(handle, "string option", value, &n),
              da_status_success);
    EXPECT_EQ("yes", string(value));
    // String free-form
    EXPECT_EQ(da_options_set_string(handle, "free-form string option", cv),
              da_status_success);
    EXPECT_EQ(da_options_get_string(handle, "free-form string option", value, &n),
              da_status_invalid_input);
    EXPECT_EQ(n, 25);
    EXPECT_EQ(da_options_get_string(handle, "free-form string option", value, &n),
              da_status_success);
    EXPECT_EQ(string(cv), string(value));

    // target char * is too small
    n = 1;
    EXPECT_EQ(da_options_get_string(handle, "string option", value, &n),
              da_status_invalid_input);
    // Try to get wrong option
    EXPECT_EQ(da_options_get_string(handle, "nonexistent option", value, &n),
              da_status_option_not_found);
    // Try to set option with incorrect value
    char invalid[] = "non existent";
    EXPECT_EQ(da_options_set_string(handle, "string option", invalid),
              da_status_option_invalid_value);
    // Try to set option with incorrect value
    EXPECT_EQ(da_options_set_int(handle, "string option", 1),
              da_status_option_wrong_type);
    da_handle_destroy(&handle);
};

TEST(OpRegistryWrappers, getset_int) {
    da_handle handle;
    OptionRegistry *opts;
    da_int value = 5;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(handle->get_current_opts(&opts), da_status_success);
    EXPECT_EQ(preload(*opts), da_status_success);
    EXPECT_EQ(da_options_set_int(nullptr, "integer option", value),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_get_int(nullptr, "integer option", &value),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_set_int(handle, "integer option", value), da_status_success);
    EXPECT_EQ(da_options_get_int(handle, "integer option", &value), da_status_success);
    EXPECT_EQ(5, value);
    // Try to get wrong option
    EXPECT_EQ(da_options_get_int(handle, "nonexistent option", &value),
              da_status_option_not_found);
    // Try to set option with incorrect value
    value = -99;
    EXPECT_EQ(da_options_set_int(handle, "integer option", value),
              da_status_option_invalid_value);
    // Try to set option with incorrect value
    double dv = 1.0;
    EXPECT_EQ(da_options_set_real_d(handle, "integer option", dv),
              da_status_option_wrong_type);
    da_handle_destroy(&handle);
};

// Public API
TEST(OpRegistryWrappers, getset_double) {
    da_handle handle;
    OptionRegistry *opts;
    double value = 5.0;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(handle->get_current_opts(&opts), da_status_success);
    EXPECT_EQ(preload(*opts), da_status_success);
    EXPECT_EQ(da_options_set_real_d(nullptr, "double option", value),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_get_real_d(nullptr, "double option", &value),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_set_real_d(handle, "double option", value), da_status_success);
    EXPECT_EQ(da_options_get_real_d(handle, "double option", &value), da_status_success);
    EXPECT_EQ(5.0, value);
    // Try to get wrong option
    EXPECT_EQ(da_options_get_real_d(handle, "nonexistent option", &value),
              da_status_option_not_found);
    // Try to set option with incorrect value
    value = -99.0;
    EXPECT_EQ(da_options_set_real_d(handle, "double option", value),
              da_status_option_invalid_value);
    // Try to set option with incorrect value
    da_int iv = 1;
    EXPECT_EQ(da_options_set_int(handle, "double option", iv),
              da_status_option_wrong_type);

    float fv;
    EXPECT_EQ(da_options_get_real_s(handle, "double option", &fv), da_status_wrong_type);
    EXPECT_EQ(da_options_set_real_s(handle, "double option", fv), da_status_wrong_type);
    da_handle_destroy(&handle);
};

TEST(OpRegistryWrappers, getset_float) {
    da_handle handle;
    OptionRegistry *opts;
    float value = 5.0f;
    EXPECT_EQ(da_handle_init_s(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(handle->get_current_opts(&opts), da_status_success);
    EXPECT_EQ(preload(*opts), da_status_success);
    EXPECT_EQ(da_options_set_real_s(nullptr, "float option", value),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_get_real_s(nullptr, "float option", &value),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_options_set_real_s(handle, "float option", value), da_status_success);
    EXPECT_EQ(da_options_get_real_s(handle, "float option", &value), da_status_success);
    EXPECT_EQ(5.0f, value);
    // Try to get wrong option
    EXPECT_EQ(da_options_get_real_s(handle, "nonexistent option", &value),
              da_status_option_not_found);
    // Try to set option with incorrect value
    value = 20.0f;
    EXPECT_EQ(da_options_set_real_s(handle, "float option", value),
              da_status_option_invalid_value);
    // Try to set option with incorrect value
    da_int iv = 1;
    EXPECT_EQ(da_options_set_int(handle, "double option", iv),
              da_status_option_wrong_type);
    EXPECT_EQ(da_options_get_real_s(handle, "float option", &value), da_status_success);
    double dv;
    EXPECT_EQ(da_options_get_real_d(handle, "float option", &dv), da_status_wrong_type);
    EXPECT_EQ(da_options_set_real_d(handle, "float option", dv), da_status_wrong_type);
    da_handle_destroy(&handle);
};

// No public boolean API yet
//TEST(OpRegistryWrappers, getset_bool) {
//    EXPECT_EQ(da_options_set_bool(nullptr, "bool option", true),
//              da_status_invalid_pointer);
//    bool value;
//    EXPECT_EQ(da_options_get_bool(nullptr, "bool option", &value),
//              da_status_invalid_pointer);
//};
} // namespace