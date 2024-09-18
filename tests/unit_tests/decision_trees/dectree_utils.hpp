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

template <typename T> struct test_data_type {
    std::vector<T> X_train;
    std::vector<da_int> y_train;
    std::vector<T> X_test;
    std::vector<da_int> y_test;
    da_int n_samples_train, n_feat, ldx_train, ldx_test;
    da_int n_samples_test;
};

template <typename T> void set_test_data_8x1(test_data_type<T> &data) {

    // idea is that y = 1 with prob 0.75 when x < 0.5
    // and          y = 1 with prob 0.25 with x > 0.5
    data.X_train = {
        (T)0.1, (T)0.2, (T)0.3, (T)0.4, (T)0.6, (T)0.7, (T)0.8, (T)0.9,
    };
    data.y_train = {0, 1, 0, 0, 1, 1, 0, 1};
    data.X_test = {(T)0.1, (T)0.9};
    data.y_test = {0, 1};
    data.n_samples_train = 8, data.n_feat = 1;
    data.n_samples_test = 2;
    data.ldx_train = 8;
    data.ldx_test = 2;
}

template <typename T> void set_test_data_8x2_unique(test_data_type<T> &data) {

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are unique
    data.X_train = {
        (T)0.12, (T)0.11, (T)0.42, (T)0.41,
        (T)0.62, (T)0.61, (T)0.92, (T)0.91, // first column of data
        (T)0.39, (T)0.79, (T)0.38, (T)0.78,
        (T)0.37, (T)0.77, (T)0.36, (T)0.76 // second column of data
    };

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    data.y_train = {0, 1, 0, 1, 1, 1, 1, 1};
    data.X_test = {(T)0.25, (T)0.25, (T)0.75, (T)0.75,
                   (T)0.25, (T)0.75, (T)0.25, (T)0.75};
    data.y_test = {0, 1, 1, 1};
    data.n_samples_train = 8, data.n_feat = 2;
    data.n_samples_test = 4;
    data.ldx_train = 8;
    data.ldx_test = 4;
}

template <typename T> void set_test_data_8x2_ldx(test_data_type<T> &data) {

    // idea is that y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are unique
    data.X_train = {(T)0.12, (T)0.11, (T)0.42, (T)0.41, (T)0.62,  (T)0.61, (T)0.92,
                    (T)0.91, (T)-50., (T)-50., (T)0.39, (T)0.79,  (T)0.38, (T)0.78,
                    (T)0.37, (T)0.77, (T)0.36, (T)0.76, (T)-100., (T)-100.};

    data.y_train = {0, 1, 0, 1, 1, 1, 1, 1};
    data.X_test = {(T)0.25, (T)0.25, (T)0.75, (T)0.75, (T)50., (T)50.,
                   (T)0.25, (T)0.75, (T)0.25, (T)0.75, (T)50., (T)50.};
    data.y_test = {0, 1, 1, 1};
    data.n_samples_train = 8, data.n_feat = 2;
    data.n_samples_test = 4;
    data.ldx_train = 10;
    data.ldx_test = 6;
}

template <typename T> void set_test_data_8x2_nonunique(test_data_type<T> &data) {
    // y = 0 if x1 < 0.5 and x2 < 0.5, otherwise y = 1
    // training data values are not unique
    data.X_train = {0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9, 0.1, 0.6, 0.1, 0.8,  0.2,
                    0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.4, 0.1, 0.45, 0.45};
    data.y_train = {1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0};
    data.X_test = {(T)0.25, (T)0.25, (T)0.75, (T)0.75,
                   (T)0.25, (T)0.75, (T)0.25, (T)0.75};
    data.y_test = {0, 1, 1, 1};
    data.n_samples_train = 12, data.n_feat = 2;
    data.n_samples_test = 4;
    data.ldx_train = 12;
    data.ldx_test = 4;
}

template <typename T> void set_data_identical(test_data_type<T> &data) {
    //  X contains all 1, No splitting should be done
    data.X_train = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    data.y_train = {1, 0, 1};
    data.X_test = {2.0, 3.0, -2.0, -2.5};
    data.y_test = {1, 1};
    data.n_samples_train = 3, data.n_feat = 2;
    data.n_samples_test = 2;
    data.ldx_train = 3;
    data.ldx_test = 2;
}
