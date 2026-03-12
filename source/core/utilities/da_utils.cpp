/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <boost/sort/spreadsort/float_sort.hpp>
#include <cmath>
#include <map>
#include <random>
#include <type_traits>

// Conditional includes for parallel sorting
#if defined(__GLIBCXX__) && defined(_OPENMP)
#include <parallel/algorithm>
#elif defined(_MSC_VER) && defined(__cpp_lib_execution) && __cpp_lib_execution >= 201603L
#include <execution>
#endif

#define TRANSPOSE_BLOCK_SIZE da_int(64)

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
            static_assert(!std::is_same_v<T, T>,
                          "Unsupported type for hidden settings query");
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

#pragma omp parallel for collapse(2) schedule(static)
    for (da_int i_block = 0; i_block < n_rows; i_block += TRANSPOSE_BLOCK_SIZE) {
        for (da_int j_block = 0; j_block < n_cols; j_block += TRANSPOSE_BLOCK_SIZE) {
            const da_int i_end = std::min(i_block + TRANSPOSE_BLOCK_SIZE, n_rows);
            const da_int j_end = std::min(j_block + TRANSPOSE_BLOCK_SIZE, n_cols);
            for (da_int i = i_block; i < i_end; ++i) {
                for (da_int j = j_block; j < j_end; ++j) {
                    B[j * ldb + i] = A[i * lda + j];
                }
            }
        }
    }
}

template <typename T>
void copy_transpose_2D_array_column_to_row_major(da_int n_rows, da_int n_cols, const T *A,
                                                 da_int lda, T *B, da_int ldb) {

#pragma omp parallel for collapse(2) schedule(static)
    for (da_int j_block = 0; j_block < n_cols; j_block += TRANSPOSE_BLOCK_SIZE) {
        for (da_int i_block = 0; i_block < n_rows; i_block += TRANSPOSE_BLOCK_SIZE) {
            const da_int j_end = std::min(j_block + TRANSPOSE_BLOCK_SIZE, n_cols);
            const da_int i_end = std::min(i_block + TRANSPOSE_BLOCK_SIZE, n_rows);
            for (da_int j = j_block; j < j_end; ++j) {
                for (da_int i = i_block; i < i_end; ++i) {
                    B[j + i * ldb] = A[i + j * lda];
                }
            }
        }
    }
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

/*
Calling the function will do the following:
1. Argument checking on the data pointer and the size
2. Read the `check data` option and accordingly to check for NaNs.
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

/*
Calling the function will do the following:
1. Argument checking on the data pointer and the size and leading dimension
2. Read the `check data` option and accordingly to check for NaNs.
*/
template <typename T>
da_status check_2D_array(bool check_data, da_order order, da_errors::da_error_t *err,
                         da_int n_rows, da_int n_cols, const T *data, da_int lddata,
                         const std::string &n_rows_name, const std::string &n_cols_name,
                         const std::string &data_name, const std::string &lddata_name,
                         da_int n_rows_min, da_int n_cols_min) {

    da_status status = da_status_success;
    // Check for illegal rows/columns arguments
    if (n_rows < n_rows_min)
        return da_error(err, da_status_invalid_array_dimension,
                        "The function was called with " + n_rows_name + " = " +
                            std::to_string(n_rows) + ". Constraint: " + n_rows_name +
                            " >= " + std::to_string(n_rows_min) + ".");
    if (n_cols < n_cols_min)
        return da_error(err, da_status_invalid_array_dimension,
                        "The function was called with " + n_cols_name + " = " +
                            std::to_string(n_cols) + ". Constraint: " + n_cols_name +
                            " >= " + std::to_string(n_cols_min) + ".");

    if (data == nullptr)
        return da_error(err, da_status_invalid_pointer,
                        "The array " + data_name + " is null.");

    if (check_data) {
        status = ARCH::da_utils::check_data(order, n_rows, n_cols, data, lddata);
        if (status == da_status_invalid_input)
            return da_error(err, da_status_invalid_input,
                            "The array " + data_name + " contains at least one NaN.");
    }

    std::string wrong_order = "";

    switch (order) {
    case column_major:
        if (lddata < n_rows) {
            if (lddata >= n_cols) {
                wrong_order = "Column-major data was expected. Did you mean to set it to "
                              "row-major?";
            }
            return da_error(err, da_status_invalid_leading_dimension,
                            "The function was called with " + n_rows_name + " = " +
                                std::to_string(n_rows) + " and " + lddata_name + " = " +
                                std::to_string(lddata) + ". Constraint: " + lddata_name +
                                " >= " + n_rows_name + "." + wrong_order);
        }
        break;
    case row_major: {
        if (lddata < n_cols) {
            if (lddata >= n_rows) {
                wrong_order = "Row-major data was expected. Did you mean to set it to "
                              "column-major?";
            }
            return da_error(err, da_status_invalid_leading_dimension,
                            "The function was called with " + n_cols_name + " = " +
                                std::to_string(n_cols) + " and " + lddata_name + " = " +
                                std::to_string(lddata) + ". Constraint: " + lddata_name +
                                " >= " + n_cols_name + "." + wrong_order);
        }
        break;
    }
    default:
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected storage scheme was requested.");
        break;
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

template <typename T>
void parallel_argsort(std::vector<T> &values, std::vector<da_int> &indices) {
    // Perform argsort with automatic algorithm selection based on compiler and threading
    // Strategy:
    // SERIAL:
    // 1. Use boost spreadsort for single-threaded execution (always fast for indirect sorting)
    // PARALLEL:
    // 1. Use GNU parallel mode when libstdc++ is present (GCC/Clang on Linux)
    // 2. Use C++17 parallel execution on MSVC (MSVC has its own implementation of parallel algorithms)
    // 3. Fall back to boost spreadsort if nothing else is available
    auto comparator = [&](da_int i, da_int j) { return values[i] < values[j]; };

    auto rightshift = [&](const da_int &idx, const unsigned offset) {
        using sort_type =
            std::conditional_t<std::is_same<T, double>::value, int64_t, int32_t>;
        return boost::sort::spreadsort::float_mem_cast<T, sort_type>(values[idx]) >>
               offset;
    };

    // Check thread count to decide between parallel and serial sorting
    da_int num_threads = omp_get_max_threads();

    if (num_threads == 1) {
        // Single-threaded: always use boost spreadsort (optimized for indirect sorting)
        boost::sort::spreadsort::float_sort(indices.begin(), indices.end(), rightshift,
                                            comparator);
    } else {
        // Multi-threaded: dispatch based on compiler/library
#if defined(__GLIBCXX__) && defined(_OPENMP)
        // Use GNU parallel algorithms when libstdc++ is present (GCC/Clang on Linux) and OpenMP is available
        __gnu_parallel::sort(indices.begin(), indices.end(), comparator);

#elif defined(_MSC_VER) && defined(__cpp_lib_execution) && __cpp_lib_execution >= 201603L
        // Use C++17 parallel execution on MSVC
        std::sort(std::execution::par, indices.begin(), indices.end(), comparator);
#else
        // Fallback: use boost spreadsort
        boost::sort::spreadsort::float_sort(indices.begin(), indices.end(), rightshift,
                                            comparator);
#endif
    }
};

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

/*
 * Divide rows of matrix by their 2-norm
 *
 * For row_major: row_norms_work is not used
 * For column_major:
 *   - If row_norms_work is nullptr, memory will be allocated internally
 *   - If user supplies row_norms_work, array must be zeroed and length >= n_rows
 */
template <typename T>
da_status normalize_rows_inplace(da_order order, da_int n_rows, da_int n_cols, T *X,
                                 da_int ldx, T *row_norms_work) {
    if (order == row_major) {
        for (da_int i = 0; i < n_rows; i++) {
            T *row_ptr = X + i * ldx;
            T norm = da_blas::cblas_nrm2(n_cols, row_ptr, 1);
            T inv_norm = (norm == 0) ? T(1.0) : (T)1.0 / norm;
            da_blas::cblas_scal(n_cols, inv_norm, row_ptr, 1);
        }
    } else {
        // Column-major case: need work array for row norms
        std::vector<T> row_norms_alloc;
        T *row_norms = row_norms_work;

        if (row_norms == nullptr) {
            try {
                row_norms_alloc.resize(n_rows, (T)0.0);
            } catch (std::bad_alloc const &) {
                return da_status_memory_error;
            }
            row_norms = row_norms_alloc.data();
        }

        // Compute squared norms
        for (da_int j = 0; j < n_cols; j++) {
            T *col_ptr = X + j * ldx;
#pragma omp simd
            for (da_int i = 0; i < n_rows; i++) {
                row_norms[i] += col_ptr[i] * col_ptr[i];
            }
        }

        // Convert to inverse norms
        for (da_int i = 0; i < n_rows; i++) {
            T norm = std::sqrt(row_norms[i]);
            row_norms[i] = (norm == 0) ? T(1.0) : (T)1.0 / norm;
        }

        // Normalize in place
        for (da_int j = 0; j < n_cols; j++) {
            T *col_ptr = X + j * ldx;
#pragma omp simd
            for (da_int i = 0; i < n_rows; i++) {
                col_ptr[i] *= row_norms[i];
            }
        }
    }
    return da_status_success;
}

template <typename T>
da_status normalize_rows(da_order order, da_int n_rows, da_int n_cols, const T *X_in,
                         da_int ldx_in, T *X_out, da_int ldx_out, T *row_norms_work) {
    if (order == row_major) {
        for (da_int i = 0; i < n_rows; i++) {
            const T *row_in = X_in + i * ldx_in;
            T *row_out = X_out + i * ldx_out;
            T norm = da_blas::cblas_nrm2(n_cols, row_in, 1);
            T inv_norm = (norm == 0) ? T(1.0) : (T)1.0 / norm;
#pragma omp simd
            for (da_int j = 0; j < n_cols; j++) {
                row_out[j] = row_in[j] * inv_norm;
            }
        }
    } else {
        // Column-major case: need work array for row norms
        std::vector<T> row_norms_alloc;
        T *row_norms = row_norms_work;

        if (row_norms == nullptr) {
            try {
                row_norms_alloc.resize(n_rows, (T)0.0);
            } catch (std::bad_alloc const &) {
                return da_status_memory_error;
            }
            row_norms = row_norms_alloc.data();
        }

        // Compute squared norms
        for (da_int j = 0; j < n_cols; j++) {
            const T *col_in = X_in + j * ldx_in;
#pragma omp simd
            for (da_int i = 0; i < n_rows; i++) {
                row_norms[i] += col_in[i] * col_in[i];
            }
        }

        // Convert to inverse norms
        for (da_int i = 0; i < n_rows; i++) {
            T norm = std::sqrt(row_norms[i]);
            row_norms[i] = (norm == 0) ? T(1.0) : (T)1.0 / norm;
        }

        // Copy and normalize
        for (da_int j = 0; j < n_cols; j++) {
            const T *col_in = X_in + j * ldx_in;
            T *col_out = X_out + j * ldx_out;
#pragma omp simd
            for (da_int i = 0; i < n_rows; i++) {
                col_out[i] = col_in[i] * row_norms[i];
            }
        }
    }
    return da_status_success;
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

template da_status
check_2D_array<double>(bool check_data, da_order order, da_errors::da_error_t *err,
                       da_int n_rows, da_int n_cols, const double *data, da_int lddata,
                       const std::string &n_rows_name, const std::string &n_cols_name,
                       const std::string &data_name, const std::string &lddata_name,
                       da_int n_rows_min, da_int n_cols_min);

template da_status
check_2D_array<float>(bool check_data, da_order order, da_errors::da_error_t *err,
                      da_int n_rows, da_int n_cols, const float *data, da_int lddata,
                      const std::string &n_rows_name, const std::string &n_cols_name,
                      const std::string &data_name, const std::string &lddata_name,
                      da_int n_rows_min, da_int n_cols_min);

template da_status check_data<da_int>(da_order order, da_int n_rows, da_int n_cols,
                                      const da_int *X, da_int ldx);

template da_status check_categorical_data<float>(da_int n_data, const float *data,
                                                 da_int &n_categories,
                                                 da_int max_categories, float tol);
template da_status check_categorical_data<double>(da_int n_data, const double *data,
                                                  da_int &n_categories,
                                                  da_int max_categories, double tol);

template void parallel_argsort<float>(std::vector<float> &values,
                                      std::vector<da_int> &indices);
template void parallel_argsort<double>(std::vector<double> &values,
                                       std::vector<da_int> &indices);

template da_status normalize_rows_inplace<float>(da_order order, da_int n_rows,
                                                 da_int n_cols, float *X, da_int ldx,
                                                 float *row_norms_work);
template da_status normalize_rows_inplace<double>(da_order order, da_int n_rows,
                                                  da_int n_cols, double *X, da_int ldx,
                                                  double *row_norms_work);

template da_status normalize_rows<float>(da_order order, da_int n_rows, da_int n_cols,
                                         const float *X_in, da_int ldx_in, float *X_out,
                                         da_int ldx_out, float *row_norms_work);
template da_status normalize_rows<double>(da_order order, da_int n_rows, da_int n_cols,
                                          const double *X_in, da_int ldx_in,
                                          double *X_out, da_int ldx_out,
                                          double *row_norms_work);

} // namespace da_utils

} // namespace ARCH
