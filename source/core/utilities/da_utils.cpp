/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "da_utils.hpp"
#include "aoclda.h"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "context.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "macros.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <type_traits>

namespace ARCH {

namespace da_arch {
#define STR_(A) #A
#define STR(A) STR_(A)
const char *get_namespace(void) {
    // return the implemented arch
    const char *arch = STR(ARCH);
    return arch;
}
} // namespace da_arch
#undef STR
#undef STR_

namespace da_utils {

template <typename T> T hidden_settings_query(const std::string &key, T default_value) {
    auto &hidden_settings = context::get_context()->hidden_settings;
    if (hidden_settings.find(key) != hidden_settings.end()) {
        std::string val = hidden_settings[key];
        if constexpr (std::is_same_v<T, size_t>) {
            return std::stoull(val);
        } else if constexpr (std::is_same_v<T, unsigned int>) {
            return std::stoul(val);
        } else if constexpr (std::is_same_v<T, da_int>) {
            return std::stol(val);
        } else if constexpr (std::is_same_v<T, float>) {
            return std::stof(val);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::stod(val);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return val;
        } else {
            static_assert(false, "Unsupported type for hidden settings query");
        }
    }
    return default_value;
}

void blocking_scheme(da_int n_samples, da_int block_size, da_int &n_blocks,
                     da_int &block_rem) {
    n_blocks = n_samples / block_size;
    block_rem = n_samples % block_size;
    // Count the remainder in the number of blocks
    if (block_rem > 0)
        n_blocks += 1;
}

/* Generalisation of blocking_scheme.
Determines a blocking scheme for partitioning n_samples into blocks, 
ensuring block_size is at least min_block_size and n_blocks does not exceed
max_blocks. Rounds block_size up to the nearest multiple of 256 if needed, 
and adjusts the final_block_size so it is either its own block, if large enough,
or merged with the previous block.
Used in da_qr and da_syrk to compute blocks for tall skinny algs*/
void tall_skinny_blocking_scheme(da_int n_samples, da_int min_block_size,
                                 da_int max_blocks, da_int min_final_block_size,
                                 da_int &n_blocks, da_int &block_size,
                                 da_int &final_block_size) {
    block_size = std::min(min_block_size, n_samples);
    if (n_samples / block_size > max_blocks) {
        block_size = n_samples / max_blocks;
        //Round up to 256 as long as we don't exceed m
        block_size = std::min(((block_size + 255) >> 8) << 8, n_samples);
    }

    n_blocks = n_samples / block_size;
    final_block_size = n_samples % block_size;

    // If final block is at least min_final_block_size, let it be its own block
    // otherwise append to previous block
    if (final_block_size >= min_final_block_size) {
        n_blocks += 1;
    } else {
        final_block_size += block_size;
        if (n_blocks == 1)
            block_size = final_block_size;
    }
}

/* Return the number of threads to use in a parallel region containing a loop*/
da_int get_n_threads_loop(da_int loop_size) {
    if (omp_get_max_active_levels() == omp_get_level())
        return (da_int)1;

    return std::min((da_int)omp_get_max_threads(), loop_size);
}

template <typename T>
void copy_transpose_2D_array_row_to_column_major(da_int n_rows, da_int n_cols, const T *A,
                                                 da_int lda, T *B, da_int ldb) {
    da_blas::omatcopy('T', n_cols, n_rows, T(1), A, lda, B, ldb);
}

template <typename T>
void copy_transpose_2D_array_column_to_row_major(da_int n_rows, da_int n_cols, const T *A,
                                                 da_int lda, T *B, da_int ldb) {
    da_blas::omatcopy('T', n_rows, n_cols, T(1), A, lda, B, ldb);
}

template <typename T>
da_status check_data(da_order order, da_int n_rows, da_int n_cols, const T *X,
                     da_int ldx) {
    if (n_rows < 1 || n_cols < 1)
        return da_status_invalid_array_dimension;

    if (X == nullptr)
        return da_status_invalid_pointer;

    // For floating point types, check for NaNs
    if (order == row_major) {
        if (ldx < n_cols)
            return da_status_invalid_leading_dimension;
        if constexpr (std::is_floating_point<T>::value) {
            for (da_int i = 0; i < n_rows; i++) {
                for (da_int j = 0; j < n_cols; j++) {
                    // x==x+1 check needed to get round a pybind1 + clang 18 Windows release build bug
                    if (std::isnan(X[i * ldx + j]) ||
                        X[i * ldx + j] == X[i * ldx + j] + (T)1) {
                        return da_status_invalid_input;
                    }
                }
            }
        }
    } else {
        if (ldx < n_rows)
            return da_status_invalid_leading_dimension;
        if constexpr (std::is_floating_point<T>::value) {
            for (da_int j = 0; j < n_cols; j++) {
                for (da_int i = 0; i < n_rows; i++) {
                    if (std::isnan(X[i + j * ldx]) ||
                        X[i + j * ldx] == X[i + j * ldx] + (T)1) {
                        return da_status_invalid_input;
                    }
                }
            }
        }
    }

    return da_status_success;
}

template <typename T>
da_status switch_order_copy(da_order order, da_int n_rows, da_int n_cols, const T *X,
                            da_int ldx, T *Y, da_int ldy) {
    if (n_rows < 1 || n_cols < 1)
        return da_status_invalid_array_dimension;
    if (X == nullptr || Y == nullptr)
        return da_status_invalid_pointer;

    if (order == row_major) {
        if (ldy < n_rows || ldx < n_cols)
            return da_status_invalid_leading_dimension;
        da_blas::omatcopy('T', n_cols, n_rows, T(1), X, ldx, Y, ldy);
    } else {
        if (ldx < n_rows || ldy < n_cols)
            return da_status_invalid_leading_dimension;
        da_blas::omatcopy('T', n_rows, n_cols, T(1), X, ldx, Y, ldy);
    }

    return da_status_success;
}

template <typename T>
da_status switch_order_in_place(da_order order_X_in, da_int n_rows, da_int n_cols, T *X,
                                da_int ldx_in, da_int ldx_out) {
    if (n_rows < 1 || n_cols < 1)
        return da_status_invalid_array_dimension;
    if (X == nullptr)
        return da_status_invalid_pointer;

    if (order_X_in == row_major) {
        if (ldx_out < n_rows || ldx_in < n_cols)
            return da_status_invalid_leading_dimension;
        da_blas::imatcopy('T', n_cols, n_rows, (T)1.0, X, ldx_in, ldx_out);
    } else {
        if (ldx_in < n_rows || ldx_out < n_cols)
            return da_status_invalid_leading_dimension;
        da_blas::imatcopy('T', n_rows, n_cols, (T)1.0, X, ldx_in, ldx_out);
    }

    return da_status_success;
}

template <typename T>
da_status convert_fp_classes(da_int m, da_int precision, const T *classes,
                             std::vector<int64_t> &int_classes) {
    // For classes supplied as type float/double, the function converts them to integers
    // by multiplying them by the precision and then flooring them by converting them to int64.

    if (precision < 1)
        return da_status_invalid_input;

#pragma omp simd
    for (da_int i = 0; i < m; ++i) {
        int_classes[i] = (int64_t)(classes[i] * precision);
    }
    return da_status_success;
}

da_status class_frequency_approximation(da_int n_draws, da_int m, da_int n_classes,
                                        boost::random::mt19937 &rand_engine,
                                        std::vector<da_int> &appx_freq,
                                        const std::vector<da_int> &counts) {
    // Approximate the number of elements (counts) for each class based on `n` random draws.
    // The goal is to approximately preserve the original class frequency distribution.

    // double is used in order to give higher precision in the calculations of the frequency
    // of each class. No templating required as data is da_int in all cases.
    std::vector<double> expected_counts, exp_counts_remainder;
    try {
        expected_counts.resize(n_classes);
        exp_counts_remainder.resize(n_classes);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    // Calculate the approximate frequency of each class based on the total number of draws.
    // The loop computes the floored frequency for each class and the remainder, which will
    // be used later to distribute the remaining draws proportionally.
    da_int floored_sum = 0;
    for (da_int i = 0; i < n_classes; ++i) {
        expected_counts[i] = ((double)counts[i] / m) * n_draws;
        appx_freq[i] = (da_int)expected_counts[i];
        exp_counts_remainder[i] = expected_counts[i] - appx_freq[i];
        floored_sum += appx_freq[i];
    }

    // Get how many draws are left to be made after getting the floored frequency
    da_int left_over = n_draws - floored_sum;

    for (da_int i = 0; i < left_over; ++i) {
        // Get the highest leftover
        double biggest_value =
            *std::max_element(exp_counts_remainder.begin(), exp_counts_remainder.end());

        // Get number of classes with the biggest remainder
        da_int n_classes_big_prob =
            std::count_if(exp_counts_remainder.begin(), exp_counts_remainder.end(),
                          [biggest_value](double i) { return i == biggest_value; });

        if (n_classes_big_prob == 1) {
            auto it = std::find(exp_counts_remainder.begin(), exp_counts_remainder.end(),
                                biggest_value);
            da_int index_b_value = std::distance(exp_counts_remainder.begin(), it);
            // update the counts using the index of the biggest remainder
            // and then null the remainder of that class so that it is not selected again.
            ++appx_freq[index_b_value];
            exp_counts_remainder[index_b_value] = 0;
        } else {
            // If multiple classes have the same remainder select one at random

            // Get indices of the classes with biggest remainder
            std::vector<da_int> inds;
            try {
                std::vector<double>::iterator it = exp_counts_remainder.begin();
                while ((it = std::find_if(it, exp_counts_remainder.end(),
                                          [biggest_value](double x) {
                                              return x == biggest_value;
                                          })) != exp_counts_remainder.end()) {
                    inds.push_back(std::distance(exp_counts_remainder.begin(), it));
                    ++it;
                }
            } catch (std::bad_alloc const &) {
                return da_status_memory_error; // LCOV_EXCL_LINE
            }

            // Create distribution to select a random value (index)
            boost::random::uniform_int_distribution<da_int> dis(0, inds.size() - 1);
            da_int random_index = inds[dis(rand_engine)];

            // update the counts using the randomly selected index
            // and then null the remainder of that class so that it is not selected again.
            ++appx_freq[random_index];
            exp_counts_remainder[random_index] = 0;
        }
    }

    return da_status_success;
}

da_status validate_parameters_stratified_shuffle(da_int m, da_int train_size,
                                                 da_int test_size, da_int n_classes,
                                                 const std::vector<da_int> &counts) {
    if (n_classes < 2)
        return da_status_invalid_input;
    if (train_size + test_size > m)
        return da_status_invalid_input;
    if (train_size < n_classes || test_size < n_classes)
        return da_status_invalid_input;
    for (da_int i = 0; i < n_classes; ++i) {
        if (counts[i] < 2)
            return da_status_invalid_input;
    }

    return da_status_success;
}

template <typename T>
da_status stratified_shuffle(da_int m, boost::random::mt19937 &rand_engine,
                             da_int train_size, da_int test_size, const T *classes,
                             da_int *shuffled_indices) {
    // The function is templated for int32 and int64. If the classes were initially floats/doubles,
    // they are converted to int64. Thus, da_int cannot be used here, because if the library is built
    // with int32, there would be no matching overload for int64.

    // Create a hash map which will store the class as the key and the sample indices
    // coresponding to that class inside a vector as the value
    std::map<T, std::vector<da_int>> class_indices_map;
    try {
        for (da_int i = 0; i < m; ++i) {
            class_indices_map[classes[i]].push_back(i);
        }
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    // Create a counts vector which will store the number of times (counts) each class appear
    da_int n_classes = (da_int)class_indices_map.size();
    std::vector<da_int> counts;
    try {
        counts.resize(n_classes);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    da_int l = 0;
    for (auto it = class_indices_map.begin(); it != class_indices_map.end(); ++it, ++l) {
        counts[l] = (da_int)it->second.size();
    }

    da_status status = validate_parameters_stratified_shuffle(m, train_size, test_size,
                                                              n_classes, counts);
    if (status != da_status_success)
        return status;

    std::vector<da_int> appx_freq_train, appx_freq_test, left_counts;
    try {
        appx_freq_train.resize(n_classes);
        appx_freq_test.resize(n_classes);
        left_counts.resize(n_classes);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    // Approximate the counts for each class for the trian split
    status = class_frequency_approximation(train_size, m, n_classes, rand_engine,
                                           appx_freq_train, counts);

    // Remove samples used in train in order to accurately approximate test
    for (da_int i = 0; i < n_classes; ++i) {
        left_counts[i] = counts[i] - appx_freq_train[i];
    }

    status = class_frequency_approximation(test_size, m - train_size, n_classes,
                                           rand_engine, appx_freq_test, left_counts);

    // class_index keeps track of the current class being processed.
    // train_index keeps track of the position in the shuffled_indices array
    // where the next training sample index will be stored.
    // test_index keeps track of the position in the shuffled_indices array
    // where the next testing sample index will be stored (offset by train_size).
    da_int class_index = 0;
    da_int train_index = 0;
    da_int test_index = train_size;
    for (auto it = class_indices_map.begin(); it != class_indices_map.end(); ++it) {

        // Shuffle the indices of the specific class then split them to train and test
        da_std::shuffle(it->second.data(), it->second.data() + it->second.size(),
                        rand_engine);

        // Fill shuffled_indices with the indices that are going to be part of the trian split
        // by using the first x indices of that class,
        // where x is the counts of appx_freq_trian for that class
        for (da_int j = 0; j < appx_freq_train[class_index]; ++j) {
            shuffled_indices[train_index] = it->second[j];
            ++train_index;
        }

        // Do the above for the test split, however get the indices from the back to
        // avoid adding the same index in both splits
        for (da_int j = 0; j < appx_freq_test[class_index]; ++j) {
            shuffled_indices[test_index] = it->second[it->second.size() - 1 - j];
            ++test_index;
        }
        ++class_index;
    }

    // Shuffle train and test indices so they are not grouped by classes
    da_std::shuffle(shuffled_indices, shuffled_indices + train_size, rand_engine);
    da_std::shuffle(shuffled_indices + train_size,
                    shuffled_indices + train_size + test_size, rand_engine);

    return status;
}

da_status validate_parameters_shuffle_array(da_int m, da_int seed,
                                            da_int *shuffle_array) {
    if (shuffle_array == nullptr)
        return da_status_invalid_pointer;
    if (m <= 1)
        return da_status_invalid_array_dimension;
    if (seed < -1) {
        return da_status_invalid_input;
    }

    return da_status_success;
}

template <typename T>
da_status get_shuffled_indices(da_int m, da_int seed, da_int train_size, da_int test_size,
                               da_int fp_precision, const T *classes,
                               da_int *shuffle_array) {
    da_status status = validate_parameters_shuffle_array(m, seed, shuffle_array);

    if (status != da_status_success)
        return status;

    if (seed == -1) {
        std::random_device r;
        seed = std::abs((da_int)r());
    }

    boost::random::mt19937 rand_engine;
    rand_engine.seed(seed);

    if (classes == nullptr) {
        da_std::iota(shuffle_array, shuffle_array + m, 0);
        da_std::shuffle(shuffle_array, shuffle_array + m, rand_engine);
    } else {
        // If classes are floats/doubles must be transformed to da_int
        // with rounding up to some precision
        if constexpr (std::is_floating_point<T>::value) {
            std::vector<int64_t> int_classes;
            try {
                int_classes.resize(m);
            } catch (std::bad_alloc const &) {
                return da_status_memory_error; // LCOV_EXCL_LINE
            }
            da_status convert_status =
                convert_fp_classes(m, fp_precision, classes, int_classes);
            if (convert_status != da_status_success)
                return convert_status;

            const int64_t *int_classes_ptr = int_classes.data();
            status = stratified_shuffle(m, rand_engine, train_size, test_size,
                                        int_classes_ptr, shuffle_array);
        } else {
            status = stratified_shuffle(m, rand_engine, train_size, test_size, classes,
                                        shuffle_array);
        }
    }

    return status;
}

template <typename T>
da_status validate_parameters_train_test_split(da_order order, da_int m, da_int n,
                                               const T *X, da_int ldx, da_int train_size,
                                               da_int test_size, T *X_train,
                                               da_int ldx_train, T *X_test,
                                               da_int ldx_test) {

    if (X == nullptr || X_train == nullptr || X_test == nullptr) {
        return da_status_invalid_pointer;
    }

    if (m < 2 || n < 1) {
        return da_status_invalid_array_dimension;
    }

    if (order == row_major) {
        if (ldx < n || ldx_train < n || ldx_test < n) {
            return da_status_invalid_leading_dimension;
        }
    } else if (order == column_major) {
        if (ldx < m || ldx_train < train_size || ldx_test < test_size) {
            return da_status_invalid_leading_dimension;
        }
    }

    if (train_size < 1 || test_size < 1) {
        return da_status_invalid_input;
    }
    if ((train_size + test_size) > m) {
        return da_status_invalid_input;
    }

    return da_status_success;
}

template <typename T>
da_status train_test_split(da_order order, da_int m, da_int n, const T *X, da_int ldx,
                           da_int train_size, da_int test_size,
                           const da_int *shuffle_array, T *X_train, da_int ldx_train,
                           T *X_test, da_int ldx_test) {
    da_status status = validate_parameters_train_test_split(
        order, m, n, X, ldx, train_size, test_size, X_train, ldx_train, X_test, ldx_test);
    if (status != da_status_success) {
        return status;
    }

    // When train_size != test_size, one split will be larger than the other.
    // The larger split requires additional loop iterations to copy the remaining data.
    // Here we determine which split (train or test) gets the remainder and set up
    // the corresponding pointers and offsets for the extra copying.
    da_int ldx_remainder = 0;
    da_int m_X_addon_remainder = 0;
    da_int small_split = std::min(train_size, test_size);
    da_int big_split = std::max(train_size, test_size);
    T *X_remainder = nullptr;

    if (train_size > test_size) {
        X_remainder = X_train;
        ldx_remainder = ldx_train;
    } else if (test_size > train_size) {
        X_remainder = X_test;
        ldx_remainder = ldx_test;
        m_X_addon_remainder = train_size;
    }

    if (order == row_major) {
        if (shuffle_array != nullptr) {
            // Shuffle, Row major

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, n, train_size, test_size, X_train, ldx_train, X_test, ldx_test,  \
               shuffle_array, X, ldx)
            for (da_int i = 0; i < small_split; ++i) {
                da_int i_ldx_train = i * ldx_train;
                da_int i_ldx_test = i * ldx_test;
                da_int i_ldx_s = shuffle_array[i] * ldx;
                da_int i_ldx_st = shuffle_array[i + train_size] * ldx;

#pragma omp simd
                for (da_int j = 0; j < n; ++j) {
                    X_train[i_ldx_train + j] = X[i_ldx_s + j];
                    X_test[i_ldx_test + j] = X[i_ldx_st + j];
                }
            }

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, big_split, m_X_addon_remainder, n, X_remainder, ldx_remainder,   \
               shuffle_array, X, ldx)
            for (da_int i = small_split; i < big_split; ++i) {
                da_int i_ldx_s = shuffle_array[i + m_X_addon_remainder] * ldx;
                da_int i_ldx_remainder = i * ldx_remainder;

#pragma omp simd
                for (da_int j = 0; j < n; ++j) {
                    X_remainder[i_ldx_remainder + j] = X[i_ldx_s + j];
                }
            }

        } else {
            // No shuffle, Row major

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, n, train_size, test_size, X_train, ldx_train, X_test, ldx_test,  \
               X, ldx)
            for (da_int i = 0; i < small_split; ++i) {
                da_int i_ldx = i * ldx;
                da_int i_ldx_train = i * ldx_train;
                da_int i_ldx_test = i * ldx_test;
                da_int i_ldx_test_train = (i + train_size) * ldx;

#pragma omp simd
                for (da_int j = 0; j < n; ++j) {
                    X_train[i_ldx_train + j] = X[i_ldx + j];
                    X_test[i_ldx_test + j] = X[i_ldx_test_train + j];
                }
            }

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, m_X_addon_remainder, big_split, n, X_remainder, ldx_remainder,   \
               X, ldx)
            for (da_int i = small_split; i < big_split; ++i) {
                da_int i_ldx = (i + m_X_addon_remainder) * ldx;
                da_int i_ldx_remainder = i * ldx_remainder;

#pragma omp simd
                for (da_int j = 0; j < n; ++j) {
                    X_remainder[i_ldx_remainder + j] = X[i_ldx + j];
                }
            }
        }
    } else if (order == column_major) {
        if (shuffle_array != nullptr) {
            // Shuffle, Column major

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, n, train_size, test_size, X_train, ldx_train, X_test, ldx_test,  \
               shuffle_array, X, ldx)                                                    \
    num_threads(std::min((da_int)omp_get_max_threads(), n))
            for (da_int i = 0; i < n; ++i) {
                da_int i_ldx_train = i * ldx_train;
                da_int i_ldx_test = i * ldx_test;
                da_int i_ldx = i * ldx;

#pragma omp simd
                for (da_int j = 0; j < small_split; ++j) {
                    X_train[i_ldx_train + j] = X[i_ldx + shuffle_array[j]];
                    X_test[i_ldx_test + j] = X[i_ldx + shuffle_array[train_size + j]];
                }
            }

            if (small_split != big_split) {
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, big_split, m_X_addon_remainder, n, X_remainder, ldx_remainder,   \
               shuffle_array, X, ldx)                                                    \
    num_threads(std::min((da_int)omp_get_max_threads(), n))
                for (da_int i = 0; i < n; ++i) {
                    da_int i_ldx_remainder = i * ldx_remainder;
                    da_int i_ldx = i * ldx;

#pragma omp simd
                    for (da_int j = small_split; j < big_split; ++j) {
                        X_remainder[i_ldx_remainder + j] =
                            X[i_ldx + shuffle_array[j + m_X_addon_remainder]];
                    }
                }
            }
        } else {
            // No shuffle, Column major

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, n, train_size, X_train, ldx_train, X_test, ldx_test, X, ldx)     \
    num_threads(std::min((da_int)omp_get_max_threads(), n))
            for (da_int i = 0; i < n; ++i) {
                da_int i_ldx = i * ldx;
                da_int i_ldx_train = i * ldx_train;
                da_int i_ldx_test = i * ldx_test;
                da_int i_ldx_tr = i * ldx + train_size;

#pragma omp simd
                for (da_int j = 0; j < small_split; ++j) {
                    X_train[i_ldx_train + j] = X[i_ldx + j];
                    X_test[i_ldx_test + j] = X[i_ldx_tr + j];
                }
            }

            if (small_split != big_split) {
#pragma omp parallel for schedule(static) default(none)                                  \
    shared(small_split, big_split, m_X_addon_remainder, n, X_remainder, ldx_remainder,   \
               X, ldx) num_threads(std::min((da_int)omp_get_max_threads(), n))
                for (da_int i = 0; i < n; ++i) {
                    da_int i_ldx = i * ldx + m_X_addon_remainder;
                    da_int i_ldx_remainder = i * ldx_remainder;

#pragma omp simd
                    for (da_int j = small_split; j < big_split; ++j) {
                        X_remainder[i_ldx_remainder + j] = X[i_ldx + j];
                    }
                }
            }
        }
    }

    return da_status_success;
}

/*
Calling the function will do the following:
1. Point data_internal to the same data.
2. Argument checking on the data pointer and the size
3. Read the `check data` option and accordingly to check for NaNs.
*/
template <typename U>
da_status check_1D_array(bool check_data, da_errors::da_error_t *err, da_int n,
                         const U *data, const std::string &n_name,
                         const std::string &data_name, da_int n_min) {
    if (data == nullptr)
        return da_error(err, da_status_invalid_pointer,
                        "The array " + data_name + " is null.");

    // Check for illegal rows/columns arguments
    if (n < n_min)
        return da_error(err, da_status_invalid_array_dimension,
                        "The function was called with " + n_name + " = " +
                            std::to_string(n) + ". Constraint: " + n_name +
                            " >= " + std::to_string(n_min) + ".");

    if (check_data) {
        // Check for NaNs if supported and requested
        da_status status = ARCH::da_utils::check_data(column_major, n, 1, data, n);
        if (status == da_status_invalid_input)
            return da_error(err, da_status_invalid_input,
                            "The array " + data_name + " contains at least one NaN.");
    }

    return da_status_success;
}

/* Check if a data array contains categorical data encoded in [0, n-1].
 * returns n in n_categories if data is categorical, -1 otherwise.
 *
 * An array is considered categorical if all of its values are within tol of an integer
 * and are in the range [0, max_categories - 1].
 */
template <typename T>
da_status check_categorical_data(da_int n_data, const T *data, da_int &n_categories,
                                 da_int max_categories, T tol) {
    if (n_data < 1 || data == nullptr)
        return da_status_invalid_array_dimension;
    n_categories = -1;

    for (da_int i = 0; i < n_data; i++) {
        da_int val = std::round(data[i]);
        if (std::abs((T)val - data[i]) > tol || val < 0 || val + 1 > max_categories) {
            n_categories = -1;
            break;
        } else if (val + 1 > n_categories)
            n_categories = val + 1;
    }

    return da_status_success;
}

// Helper functions for converting between da_ and CBLAS_ enums
CBLAS_ORDER da_order_to_cblas_order(da_order order) {
    return (order == row_major) ? CblasRowMajor : CblasColMajor;
}

da_order cblas_order_to_da_order(CBLAS_ORDER order) {
    return (order == CblasRowMajor) ? row_major : column_major;
}

CBLAS_UPLO da_uplo_to_cblas_uplo(da_uplo uplo) {
    return (uplo == da_upper) ? CblasUpper : CblasLower;
}

da_uplo cblas_uplo_to_da_uplo(CBLAS_UPLO uplo) {
    return (uplo == CblasUpper) ? da_upper : da_lower;
}

CBLAS_TRANSPOSE da_transpose_to_cblas_transpose(da_transpose transpose) {
    switch (transpose) {
    case da_no_trans:
        return CblasNoTrans;
    case da_trans:
        return CblasTrans;
    case da_conj_trans:
        return CblasConjTrans;
    default:
        return CblasNoTrans;
    }
}

da_transpose cblas_transpose_to_da_transpose(CBLAS_TRANSPOSE transpose) {
    switch (transpose) {
    case CblasNoTrans:
        return da_no_trans;
    case CblasTrans:
        return da_trans;
    case CblasConjTrans:
        return da_conj_trans;
    default:
        return da_no_trans;
    }
}

template size_t hidden_settings_query<size_t>(const std::string &key,
                                              size_t default_value);
template unsigned int hidden_settings_query<unsigned int>(const std::string &key,
                                                          unsigned int default_value);
template da_int hidden_settings_query<da_int>(const std::string &key,
                                              da_int default_value);
template float hidden_settings_query<float>(const std::string &key, float default_value);
template double hidden_settings_query<double>(const std::string &key,
                                              double default_value);

template void copy_transpose_2D_array_row_to_column_major<float>(
    da_int n_rows, da_int n_cols, const float *A, da_int lda, float *B, da_int ldb);
template void copy_transpose_2D_array_column_to_row_major<float>(
    da_int n_rows, da_int n_cols, const float *A, da_int lda, float *B, da_int ldb);

template da_status check_data<float>(da_order order, da_int n_rows, da_int n_cols,
                                     const float *X, da_int ldx);

template da_status switch_order_copy<float>(da_order order, da_int n_rows, da_int n_cols,
                                            const float *X, da_int ldx, float *Y,
                                            da_int ldy);

template da_status switch_order_in_place<float>(da_order order_X_in, da_int n_rows,
                                                da_int n_cols, float *X, da_int ldx_in,
                                                da_int ldx_out);

template void copy_transpose_2D_array_row_to_column_major<double>(
    da_int n_rows, da_int n_cols, const double *A, da_int lda, double *B, da_int ldb);
template void copy_transpose_2D_array_column_to_row_major<double>(
    da_int n_rows, da_int n_cols, const double *A, da_int lda, double *B, da_int ldb);

template da_status check_data<double>(da_order order, da_int n_rows, da_int n_cols,
                                      const double *X, da_int ldx);

template da_status switch_order_copy<double>(da_order order, da_int n_rows, da_int n_cols,
                                             const double *X, da_int ldx, double *Y,
                                             da_int ldy);

template da_status switch_order_in_place<double>(da_order order_X_in, da_int n_rows,
                                                 da_int n_cols, double *X, da_int ldx_in,
                                                 da_int ldx_out);

template da_status validate_parameters_train_test_split<da_int>(
    da_order order, da_int m, da_int n, const da_int *X, da_int ldx, da_int train_size,
    da_int test_size, da_int *X_train, da_int ldx_train, da_int *X_test, da_int ldx_test);
template da_status validate_parameters_train_test_split<float>(
    da_order order, da_int m, da_int n, const float *X, da_int ldx, da_int train_size,
    da_int test_size, float *X_train, da_int ldx_train, float *X_test, da_int ldx_test);
template da_status validate_parameters_train_test_split<double>(
    da_order order, da_int m, da_int n, const double *X, da_int ldx, da_int train_size,
    da_int test_size, double *X_train, da_int ldx_train, double *X_test, da_int ldx_test);

template da_status stratified_shuffle<int32_t>(da_int m,
                                               boost::random::mt19937 &rand_engine,
                                               da_int train_size, da_int test_size,
                                               const int32_t *classes,
                                               da_int *shuffled_indices);
template da_status stratified_shuffle<int64_t>(da_int m,
                                               boost::random::mt19937 &rand_engine,
                                               da_int train_size, da_int test_size,
                                               const int64_t *classes,
                                               da_int *shuffled_indices);

template da_status convert_fp_classes<float>(da_int m, da_int precision,
                                             const float *classes,
                                             std::vector<int64_t> &int_classes);
template da_status convert_fp_classes<double>(da_int m, da_int precision,
                                              const double *classes,
                                              std::vector<int64_t> &int_classes);

template da_status get_shuffled_indices<da_int>(da_int m, da_int seed, da_int train_size,
                                                da_int test_size, da_int fp_precision,
                                                const da_int *classes,
                                                da_int *shuffle_array);
template da_status get_shuffled_indices<float>(da_int m, da_int seed, da_int train_size,
                                               da_int test_size, da_int fp_precision,
                                               const float *classes,
                                               da_int *shuffle_array);
template da_status get_shuffled_indices<double>(da_int m, da_int seed, da_int train_size,
                                                da_int test_size, da_int fp_precision,
                                                const double *classes,
                                                da_int *shuffle_array);

template da_status train_test_split<da_int>(da_order order, da_int m, da_int n,
                                            const da_int *X, da_int ldx,
                                            da_int train_size, da_int test_size,
                                            const da_int *shuffle_array, da_int *X_train,
                                            da_int ldx_train, da_int *X_test,
                                            da_int ldx_test);
template da_status train_test_split<float>(da_order order, da_int m, da_int n,
                                           const float *X, da_int ldx, da_int train_size,
                                           da_int test_size, const da_int *shuffle_array,
                                           float *X_train, da_int ldx_train,
                                           float *X_test, da_int ldx_test);
template da_status train_test_split<double>(da_order order, da_int m, da_int n,
                                            const double *X, da_int ldx,
                                            da_int train_size, da_int test_size,
                                            const da_int *shuffle_array, double *X_train,
                                            da_int ldx_train, double *X_test,
                                            da_int ldx_test);

template da_status check_1D_array<double>(bool check_data, da_errors::da_error_t *err,
                                          da_int n, const double *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min);
template da_status check_1D_array<float>(bool check_data, da_errors::da_error_t *err,
                                         da_int n, const float *data,
                                         const std::string &n_name,
                                         const std::string &data_name, da_int n_min);
template da_status check_1D_array<da_int>(bool check_data, da_errors::da_error_t *err,
                                          da_int n, const da_int *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min);

template da_status check_data<da_int>(da_order order, da_int n_rows, da_int n_cols,
                                      const da_int *X, da_int ldx);

template da_status check_categorical_data<float>(da_int n_data, const float *data,
                                                 da_int &n_categories,
                                                 da_int max_categories, float tol);
template da_status check_categorical_data<double>(da_int n_data, const double *data,
                                                  da_int &n_categories,
                                                  da_int max_categories, double tol);

} // namespace da_utils

} // namespace ARCH
