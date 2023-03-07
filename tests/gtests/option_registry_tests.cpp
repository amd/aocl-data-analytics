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

// Testing for the Options and Registry framework
// Tests:
// 1. Internal
// 1.1 Options class
//     ALL (int, float, bool, string: setby, name(empty), get_name, get_option_t, get(), set(user/solver/default), print_option to just match string length
//     Int and Float: test validate(all bound types)
//     String: options with same entries or empty entry, get(+key)
// 1.2 Registry class
//     same name but different type string/numeric
// 2. Public
//     Get/Set for all types

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
OptionString opt_string("string option", "Preloaded String Option",
                        {{"yes", 1}, {"no", 0}, {"maybe", 2}}, "yes");
std::shared_ptr<OptionString> oS;

da_status preload(OptionRegistry &r) {
    oS = std::make_shared<OptionString>(opt_string);
    da_status status;
    status = r.register_opt(oS);
    if (status != da_status_success)
        return status;

    // status = r.register_opt(opt_int);
    // if (status != da_status_success)
    //     return status;
    // status = r.register_opt(opt_bool);
    // if (status != da_status_success)
    //     return status;
    // status = r.register_opt(opt_int);
    // if (status != da_status_success)
    //     return status;

    return status;
};

TEST(OpOptionInternal, OpClsCommon) {
    ASSERT_THROW(OptionNumeric<da_int> opt_i("", "Preloaded Integer Option", 0,
                                             da_options::lbound_t::greaterequal, 10,
                                             da_options::ubound_t::lessequal, 10),
                 std::invalid_argument);
    OptionNumeric<da_int> opt_i(" IntegeR    OptiOn    ", "Preloaded Integer Option", 0,
                                da_options::lbound_t::greaterequal, 10,
                                da_options::ubound_t::lessequal, 10);
    ASSERT_STRCASEEQ((opt_i.get_name()).c_str(), "integer option");
    ASSERT_EQ(opt_i.get_option_t(), da_options::option_t::opt_int);
    ASSERT_THROW(OptionString opt_s("      ", "Preloaded String Option",
                                    {{"yes", 1}, {"no", 0}, {"maybe", 2}}, "yes"),
                 std::invalid_argument);
    OptionString opt_s("  str   OPT  ", "Preloaded String Option",
                       {{"yes", 1}, {"no", 0}, {"maybe", 2}}, "yes");
    ASSERT_STRCASEEQ(opt_s.get_name().c_str(), "str opt");
    ASSERT_EQ(opt_s.get_option_t(), da_options::option_t::opt_string);
};

template <typename T> void OpClsNumeric(void) {
    std::string const descr("Preloaded Option");
    OptionNumeric<T> opt(" Placeholder    OptiOn    ", descr, 0,
                         da_options::lbound_t::greaterequal, 10,
                         da_options::ubound_t::lessequal, 10);
    bool has_nan = std::numeric_limits<T>::has_quiet_NaN;
    T val = -999;
    opt.get(val);
    ASSERT_EQ(val, (T)10);
    // check print_detail() grep match Set-by: default
    std::string s_default("Set-by: (default");
    std::string s_user("Set-by: (user");
    std::string s_solver("Set-by: (solver");
    std::regex reg_default(s_default, std::regex::grep);
    std::regex reg_solver(s_solver, std::regex::grep);
    std::regex reg_user(s_user, std::regex::grep);
    std::string det = opt.print_details(false);
    std::smatch m;
    std::regex_search(det, m, reg_default);
    ASSERT_STRCASEEQ(std::string(m[0]).c_str(), s_default.c_str());

    ASSERT_EQ(opt.set(1), da_status_success);
    // check print_detail() grep match Set-by: user
    det = opt.print_details(false);
    std::regex_search(det, m, reg_user);
    ASSERT_STRCASEEQ(std::string(m[0]).c_str(), s_user.c_str());

    ASSERT_EQ(opt.set(2, da_options::setby_t::solver), da_status_success);
    // check print_detail() grep match Set-by: solver
    det = opt.print_details(false);
    std::regex_search(det, m, reg_solver);
    ASSERT_STRCASEEQ(std::string(m[0]).c_str(), s_solver.c_str());

    std::string prn = opt.print_option();
    ASSERT_EQ(prn.size(), std::string(" placeholder option = 2\n").size());

    // lower > upper
    ASSERT_THROW(OptionNumeric<T> opt("Opt", descr, 10,
                                      da_options::lbound_t::greaterequal, 1,
                                      da_options::ubound_t::lessequal, 1),
                 std::invalid_argument);
    if (has_nan) {
        // lower = nan
        ASSERT_THROW(OptionNumeric<T> opt("Opt", descr,
                                          std::numeric_limits<T>::quiet_NaN(),
                                          da_options::lbound_t::greaterequal, 10,
                                          da_options::ubound_t::lessequal, 5),
                     std::invalid_argument);
        // upper = nan
        ASSERT_THROW(OptionNumeric<T> opt("Opt", descr, -1,
                                          da_options::lbound_t::greaterequal,
                                          std::numeric_limits<T>::quiet_NaN(),
                                          da_options::ubound_t::lessequal, 5),
                     std::invalid_argument);
        // default = nan
        ASSERT_THROW(OptionNumeric<T> opt("Opt", descr, -9,
                                          da_options::lbound_t::greaterequal, 10,
                                          da_options::ubound_t::lessequal,
                                          std::numeric_limits<T>::quiet_NaN()),
                     std::invalid_argument);
    }
    // default out of range l <= x <= u < d
    ASSERT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterequal,
                                      10, da_options::ubound_t::lessequal, 11),
                 std::invalid_argument);
    // default out of range l <= x < u = d
    ASSERT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterequal,
                                      10, da_options::ubound_t::lessthan, 10),
                 std::invalid_argument);
    // default out of range d < l <= x <= u
    ASSERT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterequal,
                                      10, da_options::ubound_t::lessequal, -11),
                 std::invalid_argument);
    // default out of range d = l <= x < u
    ASSERT_THROW(OptionNumeric<T> opt("Opt", descr, 0, da_options::lbound_t::greaterthan,
                                      10, da_options::ubound_t::lessthan, 0),
                 std::invalid_argument);
}

TEST(OpOptionInternal, OpClsNumericAll) {
    OpClsNumeric<float>();
    OpClsNumeric<double>();
    OpClsNumeric<da_int>();
};

TEST(OpOptionInternal, OpClsStringAll) {

    std::string val;
    da_int id;
    ::opt_string.get(val);
    ASSERT_EQ(val, "yes");
    ::opt_string.get(val, id);
    ASSERT_EQ(id, 1);
    // check print_detail() grep match Set-by: default
    std::string s_default("Set-by: (default");
    std::string s_user("Set-by: (user");
    std::string s_solver("Set-by: (solver");
    std::regex reg_default(s_default, std::regex::grep);
    std::regex reg_solver(s_solver, std::regex::grep);
    std::regex reg_user(s_user, std::regex::grep);
    std::string det = opt_string.print_details(false);
    std::smatch m;
    std::regex_search(det, m, reg_default);
    ASSERT_STRCASEEQ(std::string(m[0]).c_str(), s_default.c_str());

    ASSERT_EQ(::opt_string.set("maybe"), da_status_success);
    // check print_detail() grep match Set-by: user
    det = opt_string.print_details(false);
    std::regex_search(det, m, reg_user);
    ASSERT_STRCASEEQ(std::string(m[0]).c_str(), s_user.c_str());

    ASSERT_EQ(::opt_string.set("no", da_options::setby_t::solver), da_status_success);
    // check print_detail() grep match Set-by: solver
    det = ::opt_string.print_details(false);
    std::regex_search(det, m, reg_solver);
    ASSERT_STRCASEEQ(std::string(m[0]).c_str(), s_solver.c_str());

    std::string prn = opt_string.print_option();
    ASSERT_EQ(prn.size(), std::string(" string option = no\n").size());

    ASSERT_NO_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                            {{"yes", 1}, {"yes", 0}, {"yes", 5}}, "yes"));
    ASSERT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"No", 0}}, "           "),
                 std::invalid_argument);
    ASSERT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"   No  ", 0}}, "no"),
                 std::invalid_argument);
    ASSERT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"", 1}}, "yes"),
                 std::invalid_argument);
    ASSERT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"", 2}}, "yes"),
                 std::invalid_argument);
    ASSERT_THROW(
        OptionString opt_string("string option", "Preloaded String Option", {}, "yes"),
        std::invalid_argument);
    ASSERT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"no", 0}, {"maybe", 2}},
                                         "   yes   "),
                 std::invalid_argument);
    ASSERT_THROW(OptionString opt_string("string option", "Preloaded String Option",
                                         {{"yes", 1}, {"no", 0}, {"maybe", 2}},
                                         "invalid"),
                 std::invalid_argument);
}

TEST(OpRegistryWrappers, get_string) {
    da_handle handle;
    OptionRegistry *opts;
    ASSERT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    ASSERT_EQ(handle->get_current_opts(&opts), da_status_success);
    ASSERT_EQ(preload(*opts), da_status_success);
    ASSERT_EQ(da_options_set_string(handle, "string option", "yes"), da_status_success);
    char value[16];
    ASSERT_EQ(da_options_get_string(handle, "string option", value, 16),
              da_status_success);
    ASSERT_EQ("yes", string(value));
    // target char * is too small
    ASSERT_EQ(da_options_get_string(handle, "string option", value, 1),
              da_status_invalid_input);
    // Try to get wrong option
    ASSERT_EQ(da_options_get_string(handle, "nonexistent option", value, 1),
              da_status_option_not_found);
};

} // namespace