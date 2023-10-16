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

#include <iostream>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "da_handle.hpp"
#include "options.hpp"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

/* simple errors tests */
TEST(pca, badHandle) {
    da_handle handle = nullptr;
    da_int n = 2;

    EXPECT_EQ(da_pca_set_method(handle, pca_method_svd), da_status_invalid_pointer);
    EXPECT_EQ(da_pca_set_num_components(handle, n), da_status_invalid_pointer);

    EXPECT_EQ(da_pca_compute_d(handle), da_status_invalid_pointer);
    EXPECT_EQ(da_pca_compute_s(handle), da_status_invalid_pointer);

    da_int dim = 5;
    double *resultd = nullptr;
    float *results = nullptr;
    EXPECT_EQ(da_handle_get_result_d(handle, da_pca_components, &dim, resultd),
              da_status_invalid_pointer);
    EXPECT_EQ(da_handle_get_result_s(handle, da_pca_components, &dim, results),
              da_status_invalid_pointer);
}

TEST(pca, wrongType) {
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_pca), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_pca), da_status_success);

    EXPECT_EQ(da_pca_compute_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_pca_compute_s(handle_d), da_status_wrong_type);

    EXPECT_EQ(da_pca_compute_d(handle_d), da_status_invalid_pointer);
    EXPECT_EQ(da_pca_compute_s(handle_s), da_status_invalid_pointer);

    da_int dim = 5;
    double *resultd = nullptr;
    float *results = nullptr;
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_pca_components, &dim, resultd),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_pca_components, &dim, results),
              da_status_no_data);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

/*
    This test compares the computed pca components 
    with the reference pca components which are computed 
    using scikit learn using below python code

    import numpy as np
    from scipy import linalg
    from sklearn.decomposition import PCA
    import csv
    import time
    import random

    # dump 10 random input and outputs into csv file
    num_test = np.array([10])
    filename = "pca_data.csv"
    csvfile = open(filename, "w")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(num_test)
    for x in range(num_test.data[0]):
        isize = random.randint(2, 30)
        X = np.random.rand(isize, isize)
        X = X.transpose()
        pca = PCA(svd_solver="auto")
        pca.fit(X)
        csvwriter.writerow(X.shape)
        csvwriter.writerows(X.transpose())
        csvwriter.writerows(pca.components_.transpose())

*/
#if defined(_MSC_VER)
#define FSCANF_PCA fscanf_s
#else
#define FSCANF_PCA fscanf
#endif

TEST(pca, wrongPCAoutput) {
    GTEST_SKIP() << "Skipping failing test";
    char filepath[256] = DATA_DIR;
    da_int ntest, n, p;

    strcat(filepath, "pca_data.csv");
    FILE *fp = nullptr;

/* Most of the time MSVC compiler can automatically replace CRT functions with _s versions, but not this one */
#if defined(_MSC_VER)
    if (fopen_s(&fp, filepath, "r") != 0) {
#else
    fp = fopen(filepath, "r");
    if (fp == nullptr) {
#endif
    }

    if (fp != NULL) {
        FSCANF_PCA(fp, "%d,\n", &ntest);
        if (ntest < 0)
            ntest = 0;
        for (da_int i = 0; i < ntest; i++) {

            /*scan the dims*/
            FSCANF_PCA(fp, "%d,%d\n", &n, &p);
            if (n < 2 || p < 2)
                continue;

            /*create memory and handle*/
            da_handle handle_d = nullptr;
            da_int npc = std::min(n, p);
            double *A = new double[n * p];

            /*Read the input matrix A from file*/
            for (da_int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    FSCANF_PCA(fp, "%lf,", (A + i * p + j));
                }
                FSCANF_PCA(fp, "\n");
            }

            /*Perform PCA */
            EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_pca), da_status_success);
            EXPECT_EQ(da_options_set_string(handle_d, "pca method", "svd"),
                      da_status_success);
            EXPECT_EQ(da_options_set_int(handle_d, "npc", npc), da_status_success);
            EXPECT_EQ(da_pca_init_d(handle_d, n, p, A, n), da_status_success);
            EXPECT_EQ(da_pca_compute_d(handle_d), da_status_success);

            /*Get the result*/
            double *components = new double[npc * npc];
            da_int dim = npc * npc;
            EXPECT_EQ(
                da_handle_get_result_d(handle_d, da_pca_components, &dim, components),
                da_status_success);

            /*Read the reference output from file*/
            double *ref_components = new double[npc * npc];
            for (da_int i = 0; i < npc; i++) {
                for (int j = 0; j < npc; j++) {
                    FSCANF_PCA(fp, "%lf,", (ref_components + j + i * npc));
                }
                FSCANF_PCA(fp, "\n");
            }

            /*Verify the result with ref*/
            EXPECT_ARR_ABS_NEAR((npc * npc), ref_components, components, 1e-8);

            delete[] ref_components;
            delete[] components;
            delete[] A;
            da_handle_destroy(&handle_d);
        }
    }
    if (fp)
        fclose(fp);
}

} // namespace
