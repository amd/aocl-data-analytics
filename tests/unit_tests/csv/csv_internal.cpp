/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "char_to_num.hpp"
#include "da_datastore.hpp"
#include "gtest/gtest.h"

TEST(csvtest, char_to_num) {
    // Unit test to exercise some of the more obscure code paths in char_to_num
    double number_d;
    float number_s;
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    char **endptr = nullptr;
    int maybe_int = 1;
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "£", endptr, &number_d,
                                  &maybe_int),
              da_status_parsing_error);
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "1e-100000", endptr,
                                  &number_d, &maybe_int),
              da_status_success);
    EXPECT_EQ(number_d, 0.0);
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "1e100000", endptr,
                                  &number_d, &maybe_int),
              da_status_parsing_error);
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "1e-400", endptr, &number_d,
                                  &maybe_int),
              da_status_success);
    EXPECT_EQ(number_d, 0.0);
    EXPECT_EQ(
        da_csv::char_to_num(store->csv_parser->parser,
                            "1.3948394582957560682857698275827458672847856285728567",
                            endptr, &number_d, &maybe_int),
        da_status_success);
    EXPECT_NEAR(number_d, 1.394839458295756, 1e-14);
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "£", endptr, &number_s,
                                  &maybe_int),
              da_status_parsing_error);
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "1e-100000", endptr,
                                  &number_s, &maybe_int),
              da_status_success);
    EXPECT_EQ(number_s, 0.0);
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "1e100000", endptr,
                                  &number_s, &maybe_int),
              da_status_parsing_error);
    EXPECT_EQ(da_csv::char_to_num(store->csv_parser->parser, "1e-50", endptr, &number_s,
                                  &maybe_int),
              da_status_success);
    EXPECT_EQ(number_s, 0.0f);
    EXPECT_EQ(
        da_csv::char_to_num(store->csv_parser->parser,
                            "1.3948394582957560682857698275827458672847856285728567",
                            endptr, &number_s, &maybe_int),
        da_status_success);
    EXPECT_NEAR(number_s, 1.39483941, 1e-6);
    da_datastore_destroy(&store);
}
