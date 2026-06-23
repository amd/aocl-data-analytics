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
 *
 */

#include "aoclda.h"
#include "gtest/gtest.h"
#include <cmath>

/*
 * Test the pairwise distances C API (double precision).
 * Based on tests/examples/metrics.cpp
 */
TEST(MetricsCAPI, PairwiseDistancesDouble) {
    // X: 3 samples, 2 features (column-major)
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    // Y: 2 samples, 2 features (column-major)
    double Y[4] = {0.0, 1.0, 0.0, 1.0};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = m, ldy = n, ldd = m;
    double D[6]; // m x n distance matrix

    EXPECT_EQ(da_pairwise_distances_d(column_major, m, n, k, X, ldx, Y, ldy, D, ldd, 2.0,
                                      da_euclidean),
              da_status_success);

    // Distance from (1,4) to (0,0) = sqrt(1+16) = sqrt(17)
    EXPECT_NEAR(D[0], std::sqrt(17.0), 1.0e-10);
}

TEST(MetricsCAPI, PairwiseDistancesFloat) {
    float X[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float Y[4] = {0.0f, 1.0f, 0.0f, 1.0f};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = m, ldy = n, ldd = m;
    float D[6];

    EXPECT_EQ(da_pairwise_distances_s(column_major, m, n, k, X, ldx, Y, ldy, D, ldd, 2.0f,
                                      da_euclidean),
              da_status_success);
}

TEST(MetricsCAPI, SqEuclideanDouble) {
    double X[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double Y[4] = {0.0, 1.0, 0.0, 1.0};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = m, ldy = n, ldd = m;
    double D[6];

    EXPECT_EQ(da_pairwise_distances_d(column_major, m, n, k, X, ldx, Y, ldy, D, ldd, 2.0,
                                      da_sqeuclidean),
              da_status_success);

    // Squared distance from (1,4) to (0,0) = 1+16 = 17
    EXPECT_NEAR(D[0], 17.0, 1.0e-10);
}

TEST(MetricsCAPI, SqEuclideanFloat) {
    float X[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float Y[4] = {0.0f, 1.0f, 0.0f, 1.0f};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = m, ldy = n, ldd = m;
    float D[6];

    EXPECT_EQ(da_pairwise_distances_s(column_major, m, n, k, X, ldx, Y, ldy, D, ldd, 2.0f,
                                      da_sqeuclidean),
              da_status_success);
}
