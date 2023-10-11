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

#ifndef DF_AUX_HPP_
#define DF_AUX_HPP_

template <typename T>
T no_split_score(uint8_t *y, da_int n, uint8_t &y_pred,
                 std::function<T(T, da_int, T, da_int)> score_fun) {
    float acc_r = 0.0;

    for (da_int idx = 0; idx < n; idx++) {
        acc_r += y[idx];
    }

    if (acc_r > (n / 2)) {
        y_pred = 1;
    } else {
        y_pred = 0;
    }

    T score = score_fun((da_int)0, (da_int)0, acc_r, n);
    return score;
}

template <typename T>
void split(uint8_t *y, da_int n, da_int &split_idx, T &min_score,
           std::function<T(T, da_int, T, da_int)> score_fun) {

    da_int n_r = n, n_l = 0;
    T acc_r = 0.0, acc_l = 0.0;

    for (da_int idx = 0; idx < n; idx++) {
        acc_r += y[idx];
    }

    for (da_int idx = 1; idx < n; idx++) {
        n_r = n - idx;
        n_l = idx;

        acc_r -= y[idx - 1];
        acc_l += y[idx - 1];

        T score = score_fun(acc_l, n_l, acc_r, n_r);

        if (score < min_score) {
            min_score = score;
            split_idx = idx;
        }
    }
}

template <typename T> struct feature_label_idx {
    T x_value;
    uint8_t y_value;
    da_int idx;
};

template <typename T> int compare_floats(const void *a, const void *b) {
    feature_label_idx<T> arg1 = *(const feature_label_idx<T> *)a;
    feature_label_idx<T> arg2 = *(const feature_label_idx<T> *)b;

    if (arg1.x_value < arg2.x_value)
        return -1;
    if (arg1.x_value > arg2.x_value)
        return 1;
    return 0;
}

template <typename T>
void sort_1d_array(uint8_t *y, da_int n_obs, T *x, da_int n_features, da_int col_idx) {
    std::vector<feature_label_idx<T>> x_y_idx(n_obs);

    for (da_int i = 0; i < n_obs; i++) {
        x_y_idx[i].x_value = x[i * n_features + col_idx];
        x_y_idx[i].y_value = y[i];
        x_y_idx[i].idx = i;
    }

    qsort(x_y_idx.data(), n_obs, sizeof(x_y_idx[0]), compare_floats<T>);

    for (da_int i = 0; i < n_obs; i++) {
        y[i] = x_y_idx[i].y_value;
    }
}

/**
 * @brief
 *
 * @param x
 * @param ldx number of columns in array
 * @param m number of rows in array
 */
template <typename T>
void sort_2d_array_by_col(T *x, da_int m, da_int ldx, da_int col_idx) {
    std::vector<std::vector<T>> x_vec(m);
    for (da_int i = 0; i < m; i++) {
        // x_vec[i] should contain ith row of x
        for (da_int j = 0; j < ldx; j++) {
            x_vec[i].resize(ldx);
            x_vec[i][j] = x[i * ldx + j];
        }
    }

    auto cmp_lambda = [col_idx](const std::vector<T> &a, const std::vector<T> &b) {
        return a.at(col_idx) < b.at(col_idx);
    };

    std::sort(x_vec.begin(), x_vec.end(), cmp_lambda);

    for (da_int i = 0; i < m; i++) {
        for (da_int j = 0; j < ldx; j++) {
            x[i * ldx + j] = x_vec[i][j];
        }
    }
}

#endif
