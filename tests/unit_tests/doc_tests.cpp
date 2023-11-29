/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>

namespace {

// Handle type and descriptive names
// Add new ones here
const static std::map<da_handle_type, std::string> htypes{
    {da_handle_pca, "PCA"}, {da_handle_linmod, "Linear Model"},
    //    {da_handle_decision_tree, "Decision tree"},
    //    {da_handle_decision_forest, "Decision forest"}
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

void options_print_rst(da_handle_type htype) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, htype), da_status_success);
    da_options::OptionRegistry *opts;
    EXPECT_EQ(handle->get_current_opts(&opts), da_status_success);
    // ReStructuredText
    opts->print_details(false, false);
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

// Used to generate doc, name *must* start with ``RST``
TEST(DocOptions, RST_handle) {
    std::cout << "Supported Optional Parameters\n"
              << "**************************************\n\n";
    std::cout << "In all the following tables, :math:`\\varepsilon`, refers to "
                 "the machine precision for the given floating point data "
                 "precision.\n";
    for (auto htype : htypes) {
        std::cout << "\nOptions for " << htype.second << std::endl;
        std::cout << "==============================================\n" << std::endl;
        options_print_rst(htype.first);
        std::cout << std::endl;
    }

    std::cout << ".. _df_options:" << std::endl;
    std::cout << std::endl;
    std::cout << "\nOptions for Decision Forest" << std::endl;
    std::cout << "==============================================\n" << std::endl;
    options_print_rst(da_handle_decision_forest);
    std::cout << std::endl;
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
    std::cout << "\nOptions for datastore" << std::endl;
    std::cout << "=============================================\n" << std::endl;
    store->opts->print_details(false, false);
    std::cout << std::endl;
    da_datastore_destroy(&store);
}

// Used to generate doc, name *must* start with ``RST``
TEST(DocOptionsInternal, RST_optim) {
    // ReStructuredText (restore std::out)
    std::cout << "\n.. only:: internal\n"
              << "\nOptimization Solvers\n"
              << "====================\n"
              << std::endl;
    da_options::OptionRegistry opt;
    da_errors::da_error_t err(da_errors::action_t::DA_THROW);
    EXPECT_EQ(register_optimization_options<double>(err, opt), da_status_success);
    // ReStructuredText
    opt.print_details(false, false);
    std::cout << std::endl;
}

} // namespace
