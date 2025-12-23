/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

/* Test related to producing documentation
 * currently test and produced option description tables
 */

#include "aoclda.h"
#include "da_datastore.hpp"
#include "da_error.hpp"
#include "da_handle.hpp"
#include "options.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <regex>
#include <string>

namespace {

// Handle type and descriptive names
// Add new ones here
const static std::map<da_handle_type, std::string> htypes{
    {da_handle_pca, "Principal Component Analysis"},
    {da_handle_linmod, "Linear Models"},
    {da_handle_kmeans, "k-means Clustering"},
    {da_handle_dbscan, "DBSCAN clustering"},
    {da_handle_decision_tree, "Decision Trees"},
    {da_handle_decision_forest, "Decision Forests"},
    {da_handle_knn, "k-Nearest Neighbors"},
#ifndef NO_FORTRAN
    {da_handle_nlls, "Nonlinear Least Squares"},
#endif
    {da_handle_svm, "Support Vector Machines"},
};

void options_print(da_handle_type htype) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, htype), da_status_success);
    EXPECT_EQ(da_options_print(handle), da_status_success);
    // Also print in other formats
    da_options::OptionRegistry *opts;
    EXPECT_EQ(handle->get_current_opts(&opts), da_status_success);
    // Doxygen
    opts->print_details(false, true);
    da_handle_destroy(&handle);
}

void options_print_rst(da_handle_type htype, const std::string caption) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, htype), da_status_success);
    da_options::OptionRegistry *opts;
    EXPECT_EQ(handle->get_current_opts(&opts), da_status_success);
    // ReStructuredText
    opts->print_details(false, false, caption);
    da_handle_destroy(&handle);
}

TEST(DocOptions, handle) {
    // Add new handle types here
    for (auto htype : htypes) {
        std::cout << "Options for da_handle_type::" << htype.second << std::endl;
        options_print(htype.first);
        std::cout << std::endl;
    }
}

std::string cleanstring(std::string s) {
    std::string str{"UNKNOWN"};
    const std::regex ltrim("^[[:space:]]+");
    const std::regex rtrim("[[:space:]]+$");
    const std::regex rm("[[:space:]]+");
    str = std::regex_replace(s, ltrim, std::string(""));
    str = std::regex_replace(str, rtrim, std::string(""));
    str = std::regex_replace(str, rm, std::string(""));
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

// Used to generate doc, name *must* start with ``RST``
TEST(DocOptions, RST_handle) {
    std::cout << ".. AUTO GENERATED. Do not hand edit this file! (see doc_test.cpp)\n\n";
    std::cout << "Supported Optional Parameters\n"
              << "******************************\n\n";
    std::cout << ".. note::\n";
    std::cout << "   This page lists optional parameters for **C APIs** only.\n\n";
    std::cout << "In all the following tables, :math:`\\varepsilon`, refers to "
                 "a *safe* machine precision (twice the actual machine precision) "
                 "for the given floating point data type.\n";
    std::string str;
    for (auto htype : htypes) {
        str = cleanstring(htype.second);
        std::cout << "\n.. _opts_" << str << ":" << std::endl;
        std::cout << "\n" << htype.second << std::endl;
        std::cout << "==============================================\n" << std::endl;
        options_print_rst(htype.first,
                          ":strong:`Table of Options for " + htype.second + ".`");
        std::cout << std::endl;
    }
}

TEST(DocOptions, store) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    std::cout << "Options for da_datastore" << std::endl;
    EXPECT_EQ(da_datastore_options_print(store), da_status_success);
    // Doxygen
    store->opts->print_details(false, true);
    // ReStructuredText (restore std::out)
    store->opts->print_details(false, false);
    da_datastore_destroy(&store);
}

// Used to generate doc, name *must* start with ``RST``
TEST(DocOptions, RST_store) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    // ReStructuredText (restore std::out)
    std::cout << "\n.. _opts_datastore:" << std::endl;
    std::cout << "\nDatastore handle :cpp:type:`da_datastore`" << std::endl;
    std::cout << "=============================================\n" << std::endl;
    store->opts->print_details(
        false, false, ":strong:`Table of options for` :cpp:type:`da_datastore`.");
    std::cout << std::endl;
    da_datastore_destroy(&store);
}

} // namespace
