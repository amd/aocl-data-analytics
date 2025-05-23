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

#ifndef DATA_STORE_HPP
#define DATA_STORE_HPP

#include "aoclda.h"
#include "auto_detect_csv.hpp"
#include "csv_reader.hpp"
#include "interval.hpp"
#include "interval_map.hpp"
#include "interval_set.hpp"
#include "read_csv.hpp"
#include <ciso646> // Fixes an MSVC issue
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define DA_STRSZ 12
#define DA_STRINTERNAL "dainternal_"

namespace da_data {

using da_interval::interval;

bool check_internal_string(std::string &key);

enum block_type {
    block_none,
    block_string,
    block_int,
    block_real_s,
    block_real_d,
    block_char,
    block_str,
    block_bool // Primarily intended for uint8_t data obtained from true/false values in a CSV file
};

template <typename T> struct get_block_type {
    constexpr operator block_type() const { return block_none; }
};
template <> struct get_block_type<da_int> {
    constexpr operator block_type() const { return block_int; }
};
template <> struct get_block_type<da_int *> {
    constexpr operator block_type() const { return block_int; }
};
template <> struct get_block_type<float> {
    constexpr operator block_type() const { return block_real_s; }
};
template <> struct get_block_type<float *> {
    constexpr operator block_type() const { return block_real_s; }
};
template <> struct get_block_type<double> {
    constexpr operator block_type() const { return block_real_d; }
};
template <> struct get_block_type<double *> {
    constexpr operator block_type() const { return block_real_d; }
};
template <> struct get_block_type<std::string> {
    constexpr operator block_type() const { return block_string; }
};
template <> struct get_block_type<char *> {
    constexpr operator block_type() const { return block_char; }
};
template <> struct get_block_type<char **> {
    constexpr operator block_type() const { return block_str; }
};
template <> struct get_block_type<uint8_t> {
    constexpr operator block_type() const { return block_bool; }
};
template <> struct get_block_type<uint8_t *> {
    constexpr operator block_type() const { return block_bool; }
};

/* Missing values for each type:
 * - real numbers: missing if value = NaN
 * - integral types: missing if value >= max int
 * - all other types fall into non_missing_types which cannot always return not missing
 *
 * If missing values for other types are added, non_missing_types needs to be updated to not include
 * the new type.
 */
template <class T> struct non_missing_types {
    static constexpr bool value = !std::is_floating_point_v<T> && !std::is_integral_v<T>;
};
template <class T>
std::enable_if_t<std::is_floating_point_v<T>, bool> is_missing_value(T &val) {
    return std::isnan(val);
}
template <class T>
std::enable_if_t<std::is_integral_v<T>, bool> is_missing_value(T &val) {
    return val == std::numeric_limits<T>::max();
}
template <class T>
std::enable_if_t<non_missing_types<T>::value, bool>
is_missing_value([[maybe_unused]] T &val) {
    return false;
}

class block {
  public:
    da_int m, n;
    block_type btype = block_none;
    da_errors::da_error_t *err = nullptr;
    virtual ~block(){};
    /* mark in a boolean vector which rows contain missing
     * the vector is expected to be initialized, rows marked as containing missing values will be skipped
     * Input:
     * - idx_valid: index in the bool vector of the first row of the block.
     * - rows, cols: intervals of rows and columns in the block to be considered in the missing rows detection
     */
    virtual da_status missing_rows(std::vector<bool> &valid_row, da_int idx_valid,
                                   interval rows, interval cols) = 0;
};

template <class T> class block_base : public block {
  public:
    /* Get a column of the block
     * on output:
     * - col contains a pointer to the column idx of the block, data is not copied.
     * - "stride" contains the increment needed to get consecutive elements of the column
     * exit status:
     * - invalid input
     */
    virtual da_status get_col(da_int idx, T **col, da_int &stride) = 0;
    /* Copy a subset of a block into a dense preallocated memory block in "data"
     *
     * The subset is defined by the 2 index intervals:
     * cols = [lc, uc] and rows = [lr, ur]
     *
     * The output data is a dense matrix in column major ordering.
     * "data" can be bigger than the requested elements. the position where
     * the elements are defined in "data" is controlled by:
     * - ld_data: the leading dimension of the dense block data
     * - idx_start: the index where the first element of the slice should be placed in "data"
     * It is assumed enough memory has been allocated in "data", its dimensions are NOT checked.
     *  exit_status:
     * - invalid_input
     */
    virtual da_status copy_slice_dense(interval cols, interval rows, da_int idx_start,
                                       da_int ld_data, T *data) = 0;
};

/* Define a block of dense base data (int, float, double, string) */
template <class T> class block_dense : public block_base<T> {
    /* bl: dense data stored in a m*n vector
     * order: storage scheme used for the matrix in bl, row major or column major
     * own_data: mark if the memory is owned by the block
     * C_data: if true, bl needs to be deallocated with free instead of new.
     */
    T *bl = nullptr;
    da_order order;
    bool own_data = false;
    bool C_data = false;

  public:
    ~block_dense() {
        if (own_data) {
            C_data ? da_csv::free_data(&bl, this->m * this->n) : delete[] bl;
        }
    };

    /* Constructor can throw bad_alloc exception
     * It should be caught every time it is called
     */
    block_dense(da_int m, da_int n, T *data, da_errors::da_error_t &err,
                da_order order = row_major, bool copy_data = false, bool C_data = false) {
        if (m <= 0 || n <= 0 || data == nullptr)
            throw std::invalid_argument("");
        this->m = m;
        this->n = n;
        this->order = order;
        this->own_data = copy_data;
        this->err = &err;
        this->C_data = C_data;
        if (!copy_data)
            bl = data;
        else {
            bl = new T[m * n];
            for (da_int i = 0; i < m * n; i++)
                bl[i] = data[i];
        }
        this->btype = get_block_type<T>();
    };

    void set_own_data(bool own) { own_data = own; }

    da_status get_col(da_int idx, T **col, da_int &stride) {

        if (idx < 0 || idx >= this->n) {
            std::string buff = "idx = " + std::to_string(idx);
            buff += "idx must be between 0 and n = " + std::to_string(this->n);
            return da_error(this->err, da_status_invalid_input, buff);
        }
        switch (order) {
        case row_major:
            *col = &bl[idx];
            stride = this->n;
            break;

        case column_major:
            *col = &bl[this->m * idx];
            stride = 1;
            break;
        }

        return da_status_success;
    }

    da_status copy_slice_dense(interval cols, interval rows, da_int idx_start,
                               da_int ld_data, T *data) {

        if (!cols.is_valid_idx(this->n)) {
            std::string msg = "Column interval not valid. Input bounds: ";
            msg += "[" + std::to_string(cols.lower) + ", " + std::to_string(cols.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }
        if (!rows.is_valid_idx(this->m)) {
            std::string msg = "Row interval not valid. Input bounds: ";
            msg += "[" + std::to_string(rows.lower) + ", " + std::to_string(rows.upper) +
                   "].";
            msg += "Expected in: [0, " + std::to_string(this->m - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }

        da_int idx, ncols, nrows, idx_d;
        ncols = cols.upper - cols.lower + 1;
        nrows = rows.upper - rows.lower + 1;
        idx_d = 0;
        switch (order) {
        case column_major:
            idx = cols.lower * this->m;
            for (da_int j = 0; j < ncols; j++) {
                idx += rows.lower;
                idx_d += idx_start;
                for (da_int i = 0; i < nrows; i++) {
                    data[idx_d] = bl[idx];
                    idx++;
                    idx_d++;
                }
                idx += this->m - nrows - rows.lower;
                idx_d += ld_data - nrows - idx_start;
            }
            break;

        case row_major:
            idx = 0;
            for (da_int j = 0; j < ncols; j++) {
                idx = rows.lower * this->n + cols.lower + j;
                idx_d += idx_start;
                for (da_int i = 0; i < nrows; i++) {
                    data[idx_d] = bl[idx];
                    idx += this->n;
                    idx_d++;
                }
                idx_d += ld_data - nrows - idx_start;
            }
            break;
        }
        return da_status_success;
    }

    da_status missing_rows(std::vector<bool> &valid_row, da_int idx_valid, interval rows,
                           interval cols) {

        if (!cols.is_valid_idx(this->n)) {
            std::string msg = "Column interval not valid. Input bounds: ";
            msg += "[" + std::to_string(cols.lower) + ", " + std::to_string(cols.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }
        if (!rows.is_valid_idx(this->m)) {
            std::string msg = "Row interval not valid. Input bounds: ";
            msg += "[" + std::to_string(rows.lower) + ", " + std::to_string(rows.upper) +
                   "].";
            msg += "Expected in: [0, " + std::to_string(this->m - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }

        da_int ncols, nrows, idx;
        ncols = cols.upper - cols.lower + 1;
        nrows = rows.upper - rows.lower + 1;
        if (idx_valid + nrows > (da_int)valid_row.size() || idx_valid < 0)
            return da_error(this->err, da_status_invalid_input,
                            "mismatch between the size of the block and the size of the "
                            "boolean vector");

        if (non_missing_types<T>::value) {
            return da_status_success;
        }

        switch (order) {
        case row_major:
            for (da_int i = 0; i < nrows; i++) {
                idx = (rows.lower + i) * this->n + cols.lower;
                if (valid_row[idx_valid + i]) {
                    for (da_int j = 0; j < ncols; j++) {
                        if (is_missing_value<T>(bl[idx])) {
                            valid_row[idx_valid + i] = false;
                            break;
                        }
                        idx++;
                    }
                }
            }
            break;

        case column_major:
            idx = cols.lower * this->m + rows.lower;
            for (da_int j = 0; j < ncols; j++) {
                for (da_int i = 0; i < nrows; i++) {
                    if (valid_row[idx_valid + i]) {
                        if (is_missing_value<T>(bl[idx]))
                            valid_row[idx_valid + i] = false;
                    }
                    idx++;
                }
                idx += this->m - nrows;
            }
            break;
        }

        return da_status_success;
    }
};

/* Wrapper structure containing a pointer to a block and meta-data around the block */
class block_id {
  public:
    /* b: pointer to a generic block class
     * offset: column index of the first element in the block
     * next: pointer to another block_id containing the next rows
     * left_parent: pointer to the left most parent of the block_id; nullptr if it contains the first rows
     */
    block *b = nullptr;
    da_int offset = 0;
    std::shared_ptr<block_id> next = nullptr;
    std::shared_ptr<block_id> left_parent = nullptr;

    ~block_id() {
        if (b) {
            delete b;
            b = nullptr;
        }
    };
};

using namespace da_interval;
using columns_map = interval_map<std::shared_ptr<block_id>>;
using idx_slice = interval_map<da_int>;

struct coord_slice {
    std::unique_ptr<interval_set> col_slice;
    std::unique_ptr<interval_set> row_slice;
    coord_slice() {
        col_slice = std::make_unique<interval_set>();
        row_slice = std::make_unique<interval_set>();
    }
};
using selection_map = std::unordered_map<std::string, coord_slice>;

/* Main data store structure for heterogeneous data manipulation
 * supported features:
 * - Vertical and horizontal block concatenation
 * - slice extraction of homogeneous type in given rows and columns intervals
 * - rows and columns selection
 */
class data_store {
    /* m, n: total dimension of the data_store (aggregate of all the blocks)
     * cmap: interval map linking column indices to block IDs.
     *       [0, n-1] should have no missing element in cmap
     * */
    da_int m = 0, n = 0;
    columns_map cmap;

    /* Data to handle partially added rows
     * missing_block: if true, blocks the other functionality until the partial row block is finished
     * idx_start_missing: column index of the start of the missing block
     */
    bool missing_block = false;
    da_int idx_start_missing;

    /* Hash map of selections, using user defined labels (c strings) */
    selection_map selections;

    /* Bidirectional map linking column index to its name */
    using sz_t = std::vector<std::string const *>::size_type;
    std::unordered_map<std::string, da_int> name_to_index;
    std::vector<std::string const *> index_to_name;

    /* Error structure pointing to the main handle */
    da_errors::da_error_t *err;

  public:
    data_store(da_errors::da_error_t &err) { this->err = &err; }
    ~data_store() {
        for (auto it = cmap.begin(); it != cmap.end(); ++it) {
            std::shared_ptr<block_id> next = it->second, aux;
            while (next != nullptr) {
                aux = next->next;
                next->next = nullptr;
                next = aux;
            }
            it->second = nullptr;
        }
    }

    da_int get_num_rows() { return this->m; }
    da_int get_num_cols() { return this->n; }

    bool empty() { return m == 0 && n == 0 && cmap.empty(); }

    /* Concatenate dense blocks of columns or rows to the data_store.
     * takes user's dense data block of size m*n stored in *data and add it as a block to the datastore
     * Data is copied if copy_data is set to true, otherwise user's pointer is stored as is.
     * Data is NOT checked
     * Input:
     * - mc/mr: number of rows of the dense block to add
     * - nc/nr: number of columns of the dense block to add
     * - data: user data in a dense matrix to be added to the data store
     * - order: row or column major ordering for the data
     * - copy_data: controls if the user data is copied.
     *              if false, the user's pointer will be copied instead
     * - own_data: is the datastore the owner of the data in the block
     *             (e.g., copy_data is false but another function wants to transfer ownership of the pointer)
     * exit:
     * - invalid_input
     * - memory error
     */
    template <class T>
    da_status concatenate_columns(da_int mc, da_int nc, T *data, da_order order,
                                  bool copy_data = false, bool own_data = false,
                                  bool C_data = false) {

        // Cannot concatenate columns if the store is in the process of adding rows
        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot concatenate columns at this point");

        // mc must match m except if the initial data.frame is empty (new block is created)
        // mc*nc input data must not be empty
        if (mc <= 0)
            return da_error(err, da_status_invalid_input,
                            "Number of rows must be positive");
        if (nc <= 0)
            return da_error(err, da_status_invalid_input,
                            "Number of columns must be positive");
        if (m > 0 && m != mc)
            return da_error(err, da_status_invalid_input,
                            "Number of rows must match " + std::to_string(m) +
                                " (input: " + std::to_string(mc) + ")");

        // Create a dense block from the raw data
        std::shared_ptr<block_id> new_block;
        try {
            new_block = std::make_shared<block_id>(block_id());
            new_block->b =
                new block_dense<T>(mc, nc, data, *this->err, order, copy_data, C_data);
            new_block->offset = n;
            block_dense<T> *bd = static_cast<block_dense<T> *>(new_block->b);
            bd->set_own_data(own_data || copy_data);
            index_to_name.resize(n + nc, nullptr);
        } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }

        // Concatenate columns to the right, indices of new columns are [n, n+nc-1]
        cmap.insert(interval(n, n + nc - 1), new_block);

        // Update the size of the datastore if it was initially empty
        if (m == 0)
            m = mc;
        n += nc;

        return da_status_success;
    }

    template <class T>
    da_status concatenate_rows(da_int mr, da_int nr, T *data, da_order order,
                               bool copy_data = false, bool C_data = false) {
        da_status status = da_status_success;
        bool found;
        da_int lb, ub, idx_start;
        std::shared_ptr<block_id> new_block = nullptr;

        if (n <= 0) {
            // First block, columns need to be concatenated instead.
            status =
                this->concatenate_columns(mr, nr, data, order, copy_data, false, C_data);
        } else {
            bool cleanup = false;
            idx_start = 0;
            if (missing_block)
                idx_start = idx_start_missing;
            if (mr <= 0)
                return da_error(err, da_status_invalid_input,
                                "Number of rows must be positive");
            if (nr <= 0)
                return da_error(err, da_status_invalid_input,
                                "Number of columns must be positive");
            if (n > 0 && nr + idx_start > n)
                return da_error(err, da_status_invalid_input,
                                "Number of rows must match " + std::to_string(m) +
                                    " (input: " + std::to_string(nr + idx_start) + ")");

            // Create a dense block from the raw data
            try {
                new_block = std::make_shared<block_id>(block_id());
                new_block->b =
                    new block_dense<T>(mr, nr, data, *err, order, copy_data, C_data);
            } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
                return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Memory allocation error");
            }

            new_block->offset = idx_start;
            block_type btype = get_block_type<T>();

            if (!missing_block)
                m += mr;
            ub = idx_start - 1;
            std::shared_ptr<block_id> current_block;
            while (ub < idx_start + nr - 1) {
                found = cmap.find(ub + 1, current_block, lb, ub);
                if (!found)
                    break;
                if (btype != current_block->b->btype || ub > idx_start + nr - 1) {
                    cleanup = true;
                    status =
                        da_error(err, da_status_invalid_input,
                                 "cannot concatenate row(s), column types do not match.");
                    break;
                }
                while (current_block->next != nullptr)
                    current_block = current_block->next;
                current_block->next = new_block;
                if (new_block->left_parent == nullptr)
                    new_block->left_parent = current_block;
            }
            if (!cleanup) {
                if (idx_start + nr < n) {
                    missing_block = true;
                    idx_start_missing = idx_start + nr;
                } else {
                    missing_block = false;
                    idx_start_missing = 0;
                }
            } else {
                // Error occurred, clear the 'next' pointers from the existing blocks
                ub = idx_start - 1;
                std::shared_ptr<block_id> current_block;
                while (ub < idx_start + nr - 1) {
                    auto it = cmap.find(ub + 1);
                    if (it == cmap.end())
                        break;
                    current_block = it->second;
                    while (current_block->next != nullptr) {
                        if (current_block->next == new_block)
                            current_block->next = nullptr;
                        else
                            current_block = current_block->next;
                    }
                    ub = it->first.upper;
                    ++it;
                }
                m -= mr;
            }
        }
        return status; // Error message already loaded
    }

    /* concat the current datastore with the store passed on the interface horizontally
     * in output, 'this' will contain the concatenation and store will be destroyed.
     */
    da_status horizontal_concat(data_store &store) {
        if (m != store.m)
            return da_error(err, da_status_invalid_input,
                            "The number of rows in both stores must match (" +
                                std::to_string(m) + " vs " + std::to_string(store.m) +
                                ").");
        if (missing_block || store.missing_block)
            return da_error(
                err, da_status_invalid_input,
                "Cannot concatenate stores at this stage, some data is missing");

        columns_map::iterator it1 = store.cmap.begin(), it2;
        da_int n_orig = n;
        std::shared_ptr<block_id> store_block, next, current_block;
        while (it1 != store.cmap.end()) {
            da_int nc = it1->first.upper - it1->first.lower + 1;
            store_block = it1->second;
            store_block->offset += n_orig;
            cmap.insert(interval(n, n + nc - 1), store_block);
            current_block = store_block;
            next = store_block->next;
            while (next != nullptr) {
                if (next->left_parent == current_block)
                    next->offset = current_block->offset;
                current_block = next;
                next = next->next;
            }
            n += nc;
            it2 = it1;
            it1++;
            store.cmap.erase(it2);
        }
        store.m = 0;
        store.n = 0;

        return da_status_success;
    }

    /* Extract column idx into vector col
     * m: expected size of the column by the user; if wrong correct m size is returned
     * col: user allocated memory of size at least m
     */
    template <class T> da_status extract_column(da_int idx, da_int &m, T *col) {
        da_status status;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot extract columns at this point");

        if (m != this->m) {
            da_error(err, da_status_invalid_input,
                     "n_rows must match the size of the desired column. n_rows = " +
                         std::to_string(m) +
                         ", store n_rows = " + std::to_string(this->m));
            m = this->m;
            return da_status_invalid_input;
        }
        if (idx < 0 || idx >= this->n)
            return da_error(err, da_status_invalid_input,
                            "idx = " + std::to_string(idx) +
                                ". It must be set between 0 and " +
                                std::to_string(this->n - 1) + ".");

        T *c;
        da_int stride, nrows, lb, ub;
        da_int idxrow = 0;
        block_base<T> *bb;
        std::shared_ptr<block_id> id;
        cmap.find(idx, id, lb, ub);
        if (id->b->btype != get_block_type<T>())
            return da_error(
                err, da_status_invalid_input,
                "Incompatible types between the datastore and the input data");
        while (id != nullptr) {
            bb = static_cast<block_base<T> *>(id->b);
            status = bb->get_col(idx - id->offset, &c, stride);
            if (status != da_status_success)
                return da_error_trace( // LCOV_EXCL_LINE
                    err, da_status_internal_error,
                    "Unexpected error occurred. Possible memory corruption.");
            nrows = bb->m;
            for (da_int i = 0; i < nrows; i++)
                col[idxrow + i] = c[i * stride];
            idxrow += nrows;
            id = id->next;
        }

        return da_status_success;
    }

    /* Extract a slice in column major ordering from the data store
     * slice is defined by the 2 intervals rows and cols
     * On output, the data is stored in slice, starting in slice[first_idx], with the leading dimension ld_slice
     * dimensions of slice are NOT checked
     */
    template <class T>
    da_status extract_slice(interval rows, interval cols, da_int ld_slice,
                            da_int first_idx, T *slice) {

        if (!cols.is_valid_idx(this->n)) {
            std::string msg = "Column interval not valid. Input bounds: ";
            msg += "[" + std::to_string(cols.lower) + ", " + std::to_string(cols.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }
        if (!rows.is_valid_idx(this->m)) {
            std::string msg = "Row interval not valid. Input bounds: ";
            msg += "[" + std::to_string(rows.lower) + ", " + std::to_string(rows.upper) +
                   "].";
            msg += "Expected in: [0, " + std::to_string(this->m - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }
        if (ld_slice < rows.upper - rows.lower + 1)
            return da_error(err, da_status_invalid_input,
                            "The leading dimension must be lower than " +
                                std::to_string(rows.upper - rows.lower + 1));

        da_status status;

        da_int lcol = cols.lower;
        da_int ucol = cols.upper;
        da_int idx = first_idx;
        columns_map::iterator it;
        interval block_cols, block_rows;
        while (ucol - lcol >= 0) {
            it = cmap.find(lcol);
            std::shared_ptr<block_id> bid = it->second;
            if (bid->b->btype != get_block_type<T>())
                return da_error(err, da_status_invalid_input,
                                "Incompatible type in the slice");
            da_int uc = std::min(ucol, it->first.upper);
            da_int lrow = rows.lower;
            da_int lr = lrow;
            da_int urow = rows.upper;
            da_int idxr = idx;
            da_int first_row_idx = 0;
            while (urow - lr >= 0) {
                da_int ur = std::min(urow, first_row_idx + bid->b->m - 1);
                block_rows = {lr - first_row_idx, ur - first_row_idx};
                if (block_rows.upper >= block_rows.lower) {
                    block_cols = {lcol - bid->offset, uc - bid->offset};
                    block_base<T> *bb = static_cast<block_base<T> *>(bid->b);
                    status = bb->copy_slice_dense(block_cols, block_rows, idxr, ld_slice,
                                                  slice);
                    if (status != da_status_success)
                        return da_error( // LCOV_EXCL_LINE
                            err, da_status_internal_error,
                            "Unexpected error in copy_slice_dense");
                    idxr += ur - lr + 1;
                }
                lr = std::max(ur + 1, lrow);
                first_row_idx = ur + 1;
                bid = bid->next;
            }
            idx += ld_slice * (uc - lcol + 1);
            lcol = uc + 1;
        }

        return da_status_success;
    }

    void remove_selection(std::string key) { selections.erase(key); }

    /* select_[slice|columns|rows]; add an interval to the selection 'key' in the data_store
     * the indices have to be in the correct range
     * trying to add an index already in the selection will result in an error
     * exit status:
     * - da_status_invalid_input
     * - da_status_internal_error
     */
    da_status select_slice(std::string key, interval rows, interval cols) {
        da_status exit_status;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot select elements at this time");

        if (!cols.is_valid_idx(this->n)) {
            std::string msg = "Column interval not valid. Input bounds: ";
            msg += "[" + std::to_string(cols.lower) + ", " + std::to_string(cols.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }
        if (!rows.is_valid_idx(this->m)) {
            std::string msg = "Row interval not valid. Input bounds: ";
            msg += "[" + std::to_string(rows.lower) + ", " + std::to_string(rows.upper) +
                   "].";
            msg += "Expected in: [0, " + std::to_string(this->m - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }

        auto it = selections.find(key);
        if (it == selections.end()) {
            bool inserted;
            std::tie(it, inserted) =
                selections.insert(std::make_pair(key, coord_slice()));
            if (!inserted) {
                // LCOV_EXCL_START
                exit_status = da_status_internal_error;
                return da_error(err, exit_status,
                                "Unexpected failure in the creation of a new selection.");
                // LCOV_EXCL_STOP
            }
        }
        exit_status = it->second.row_slice->insert(rows);
        if (exit_status != da_status_success)
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpected failure in row selection.");
        exit_status = it->second.col_slice->insert(cols);
        if (exit_status != da_status_success) {
            it->second.row_slice->erase(rows);             // LCOV_EXCL_LINE
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpected failure in col selection.");
        }
        return da_status_success;
    }

    da_status select_columns(std::string key, interval cols) {
        da_status exit_status = da_status_success;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot select elements at this time");

        if (!cols.is_valid_idx(this->n)) {
            std::string msg = "Column interval not valid. Input bounds: ";
            msg += "[" + std::to_string(cols.lower) + ", " + std::to_string(cols.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }

        auto it = selections.find(key);
        if (it == selections.end()) {
            bool inserted;
            std::tie(it, inserted) =
                selections.insert(std::make_pair(key, coord_slice()));
            if (!inserted) {
                return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                                "Unexpected error in the interval insertion");
            }
        }
        exit_status = it->second.col_slice->insert(cols);

        return exit_status;
    }
    da_status select_rows(std::string key, interval rows) {
        da_status exit_status = da_status_success;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot select elements at this time");

        if (!rows.is_valid_idx(this->m)) {
            std::string msg = "Row interval not valid. Input bounds: ";
            msg += "[" + std::to_string(rows.lower) + ", " + std::to_string(rows.upper) +
                   "].";
            msg += "Expected in: [0, " + std::to_string(this->m - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }

        auto it = selections.find(key);
        if (it == selections.end()) {
            bool inserted;
            std::tie(it, inserted) =
                selections.insert(std::make_pair(key, coord_slice()));
            if (!inserted) {
                return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                                "Unexpected error. Possible memory corruption.");
            }
        }

        exit_status = it->second.row_slice->insert(rows);

        return exit_status;
    }

    da_status remove_columns_from_selection(std::string key, interval cols) {
        da_status exit_status = da_status_success;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot select elements at this time");

        if (!cols.is_valid_idx(this->n)) {
            std::string msg = "Column interval not valid. Input bounds: ";
            msg += "[" + std::to_string(cols.lower) + ", " + std::to_string(cols.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }

        auto it = selections.find(key);
        if (it == selections.end()) {
            std::string msg = key;
            msg += " is not a valid selection.";
            return da_warn(err, da_status_invalid_input, msg);
        }
        exit_status = it->second.col_slice->erase(cols);

        return exit_status;
    }
    da_status remove_rows_from_selection(std::string key, interval rows) {
        da_status exit_status = da_status_success;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot select elements at this time");

        if (!rows.is_valid_idx(this->m)) {
            std::string msg = "Row interval not valid. Input bounds: ";
            msg += "[" + std::to_string(rows.lower) + ", " + std::to_string(rows.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }

        auto it = selections.find(key);
        if (it == selections.end()) {
            std::string msg = key;
            msg += " is not a valid selection.";
            return da_warn(err, da_status_invalid_input, msg);
        }
        exit_status = it->second.row_slice->erase(rows);

        return exit_status;
    }

    template <class T>
    da_status extract_selection(std::string key, da_order order, da_int ld, T *data) {
        da_status exit_status = da_status_success, status;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot extract data at this point");

        auto it = selections.find(key);
        bool clear_selections = false, clear_cols = false, clear_rows = false;
        da_int idx = 0;
        da_int ncols = 0;
        std::string internal_key;

        if (selections.empty()) {
            // No selection defined create a temporary selection All
            internal_key = DA_STRINTERNAL;
            internal_key += "All";
            status = select_slice(internal_key, {0, m - 1}, {0, n - 1});
            if (status != da_status_success)
                return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                                "Internal error selecting a valid slice");
            it = selections.find(internal_key);
            clear_selections = true;
            exit_status = da_status_full_extraction;
            da_warn(err, exit_status,
                    "No selection was defined, the full store is being extracted");
        } else if (it == selections.end()) {
            exit_status = da_status_invalid_input;
            da_error(err, exit_status, "key was not found");
            goto exit;
        }

        if (it->second.row_slice->empty()) {
            // No rows in the current selection, create a temporary one containing all
            status = select_rows(key, {0, m - 1});
            if (status != da_status_success) {
                exit_status = da_status_internal_error; // LCOV_EXCL_LINE
                da_error_trace(                         // LCOV_EXCL_LINE
                    err, exit_status, "Internal error selecting a valid row slice");
                goto exit; // LCOV_EXCL_LINE
            }
            clear_rows = true;
        }

        if (it->second.col_slice->empty()) {
            // No cols in the current selection, create a temporary one containing all
            status = select_columns(key, {0, n - 1});
            if (status != da_status_success) {
                exit_status = da_status_internal_error; // LCOV_EXCL_LINE
                da_error_trace(                         // LCOV_EXCL_LINE
                    err, exit_status, "Internal error selecting a valid column slice");
                goto exit; // LCOV_EXCL_LINE
            }
            clear_cols = true;
        }

        if (order == column_major) {

            for (auto it_col = it->second.col_slice->begin();
                 it_col != it->second.col_slice->end(); ++it_col) {
                idx = ncols * ld;
                ncols += it_col->upper - it_col->lower + 1;
                for (auto it_row = it->second.row_slice->begin();
                     it_row != it->second.row_slice->end(); ++it_row) {
                    da_int nrows = it_row->upper - it_row->lower + 1;
                    status = extract_slice(*it_row, *it_col, ld, idx, data);
                    if (status != da_status_success) {
                        exit_status = status;
                        goto exit;
                    }
                    idx += nrows;
                }
            }

        } else {

            // For row-major ordering, we need to create a temporary array first
            da_int n_cols = 0, n_rows = 0, ldd = 0;
            for (auto it_col = it->second.col_slice->begin();
                 it_col != it->second.col_slice->end(); ++it_col) {
                n_cols += it_col->upper - it_col->lower + 1;
                for (auto it_row = it->second.row_slice->begin();
                     it_row != it->second.row_slice->end(); ++it_row) {
                    n_rows += it_row->upper - it_row->lower + 1;
                }
            }
            ldd = n_rows;
            // ncols and nrows are now correctly set
            std::vector<T> tmp_data;

            try {
                tmp_data.resize(n_cols * n_rows);
            } catch (std::bad_alloc const &) {
                exit_status = da_status_memory_error;
                da_error(err, exit_status, "Memory allocation error");
                goto exit;
            }

            for (auto it_col = it->second.col_slice->begin();
                 it_col != it->second.col_slice->end(); ++it_col) {
                idx = ncols * ld;
                ncols += it_col->upper - it_col->lower + 1;
                for (auto it_row = it->second.row_slice->begin();
                     it_row != it->second.row_slice->end(); ++it_row) {
                    da_int nrows = it_row->upper - it_row->lower + 1;
                    status = extract_slice(*it_row, *it_col, ldd, idx, tmp_data.data());
                    if (status != da_status_success) {
                        exit_status = status;
                        goto exit;
                    }
                    idx += nrows;
                }
            }

            // Convert tmp_data to row-major output array
            for (da_int i = 0; i < n_rows; i++) {
                for (da_int j = 0; j < n_cols; j++) {
                    data[j + i * ld] = tmp_data[i + j * ldd];
                }
            }
        }

    exit:
        if (clear_selections)
            selections.erase(internal_key);
        if (clear_rows)
            it->second.row_slice->clear();
        if (clear_cols)
            it->second.col_slice->clear();

        return exit_status;
    }

    /* From a given selection remove all rows that have missing data in it.
     * Input:
     * - key: name of the selection. If the key is not already present in the map, all rows will be considered
     * - full_rows: If true, the full columns in all the rows of the selection are checked.
     *              Otherwise, only the columns selected are.
     */
    da_status select_non_missing(std::string key, bool full_rows) {

        da_status status = da_status_success;
        da_int i;

        if (missing_block)
            return da_error(
                err, da_status_missing_block,
                "Row blocks are not complete, cannot select elements at this time");

        // Check if selection 'key' exists. If not, create one containing all rows and columns.
        // Ensure that the selection iterator 'it' always points to a valid selection.
        auto it = selections.find(key);
        bool clear_cols = false, clear_all_cols = false;
        if (it == selections.end()) {
            select_slice(key, {0, m - 1}, {0, n - 1});
            it = selections.find(key);
            clear_cols = true;
        }

        std::unique_ptr<interval_set> &col_slice = it->second.col_slice;
        std::unique_ptr<interval_set> &row_slice = it->second.row_slice;
        if (row_slice->empty()) {
            select_rows(key, {0, m - 1});
        }

        // Check ALL the rows
        std::vector<bool> valid_rows;
        try {
            valid_rows.resize(m, true);
        } catch (std::bad_alloc const &) {               // LCOV_EXCL_LINE
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }

        // Depending on the parameter full_rows, either all the columns are checked
        // for missing data or only the columns in the selection 'key'.
        // it_col and it_col_end are set as pointing to the start and the end of valid column
        // selections.
        interval_set::iterator it_col, it_col_end;
        std::string internal_key;
        if (full_rows) {
            internal_key = DA_STRINTERNAL;
            internal_key += "all cols";
            status = select_columns(internal_key, {0, n - 1});
            if (status != da_status_success) {
                status = da_error_trace( // LCOV_EXCL_LINE
                    err, da_status_internal_error,
                    "Could not create new tag with all columns");
                goto exit; // LCOV_EXCL_LINE
            }
            auto it_allcol = selections.find(internal_key);
            it_col = it_allcol->second.col_slice->begin();
            it_col_end = it_allcol->second.col_slice->end();
            clear_all_cols = true;
        } else {
            if (col_slice->begin() == col_slice->end()) {
                select_columns(key, {0, n - 1});
                clear_cols = true;
            }
            it_col = col_slice->begin();
            it_col_end = col_slice->end();
        }

        // Loop over the columns and rows of the selection to mark the rows with missing
        // data in valid_rows
        for (; it_col != it_col_end; ++it_col) {
            for (auto it_row = row_slice->begin(); it_row != row_slice->end(); ++it_row) {
                status = mark_missing_slice(*it_row, *it_col, valid_rows);
                if (status != da_status_success) {
                    status =
                        da_error_trace(err, da_status_internal_error, // LCOV_EXCL_LINE
                                       "Unexpected error. Possible memory corruption.");
                    goto exit; // LCOV_EXCL_LINE
                }
            }
        }

        // Remove the rows with missing data from the selection 'key'
        i = 0;
        while (i < m) {
            if (!valid_rows[i]) {
                da_int idx_stop = i;
                while (!valid_rows[idx_stop] && idx_stop < m) {
                    idx_stop++;
                }
                status = row_slice->erase({i, idx_stop - 1});
                if (status != da_status_success) {
                    // LCOV_EXCL_START
                    status = da_status_internal_error;
                    da_error_trace(err, status,
                                   "Failed to erase missing rows from the selection.");
                    goto exit;
                    // LCOV_EXCL_STOP
                }
                i = idx_stop - 1;
            }
            i++;
        }

    exit:
        if (clear_cols)
            it->second.col_slice->clear();
        if (clear_all_cols)
            remove_selection(internal_key);
        return status;
    }

    da_status mark_missing_slice(interval rows, interval cols,
                                 std::vector<bool> &valid_rows) {
        if (!cols.is_valid_idx(this->n)) {
            std::string msg = "Column interval not valid. Input bounds: ";
            msg += "[" + std::to_string(cols.lower) + ", " + std::to_string(cols.upper) +
                   "]. ";
            msg += "Expected in: [0, " + std::to_string(this->n - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }
        if (!rows.is_valid_idx(this->m)) {
            std::string msg = "Row interval not valid. Input bounds: ";
            msg += "[" + std::to_string(rows.lower) + ", " + std::to_string(rows.upper) +
                   "].";
            msg += "Expected in: [0, " + std::to_string(this->m - 1) + "]";
            return da_error(this->err, da_status_invalid_input, msg);
        }
        if ((da_int)valid_rows.size() != this->m)
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Wrong valid rows vector size: " +
                                std::to_string((da_int)valid_rows.size()) +
                                " != " + std::to_string(this->m));

        da_status status;
        da_int lcol = cols.lower;
        da_int ucol = cols.upper;
        da_int lrow = rows.lower;
        da_int urow = rows.upper;
        interval block_cols, block_rows;
        while (ucol - lcol >= 0) {
            auto it = cmap.find(lcol);
            std::shared_ptr<block_id> bid = it->second;
            da_int uc = std::min(ucol, it->first.upper);
            da_int lr = lrow;
            da_int first_row_idx = 0;
            while (urow - lr >= 0) {
                da_int ur = std::min(urow, first_row_idx + bid->b->m - 1);
                block_rows = {lr - first_row_idx, ur - first_row_idx};
                if (block_rows.upper >= block_rows.lower) {
                    block_cols = {lcol - bid->offset, uc - bid->offset};
                    status = bid->b->missing_rows(valid_rows, lr, block_rows, block_cols);
                    if (status != da_status_success)
                        return da_error_trace( // LCOV_EXCL_LINE
                            err, da_status_internal_error,
                            "Internal error. Possible  memory corruption.");
                }
                lr = ur + 1;
                first_row_idx = lr;
                bid = bid->next;
            }
            lcol = uc + 1;
        }
        return da_status_success;
    }

    /* get|set_element access and modify a single element in the data store
     * exit status:
     * - invalid_input
     * - internal_error
     */
    template <class T> da_status get_element(da_int i, da_int j, T &elem) {

        if (i < 0 || i >= this->m)
            return da_error(err, da_status_invalid_input,
                            "i = " + std::to_string(i) +
                                ". The row index must be between 0 and " +
                                std::to_string(this->m - 1) + ".");
        if (j < 0 || j >= this->n)
            return da_error(err, da_status_invalid_input,
                            "j = " + std::to_string(i) +
                                ". The column index must be between 0 and " +
                                std::to_string(this->n - 1) + ".");

        auto it = cmap.find(j);
        if (it == cmap.end())
            // Cannot happen, checks on i and j would have returned invalid input already
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Couldn't find the element");

        std::shared_ptr<block_id> bid = it->second;
        if (bid->b->btype != get_block_type<T>())
            return da_error(err, da_status_invalid_input, "Incompatible types");

        da_int offset = 0;
        while (i >= bid->b->m + offset) {
            offset += bid->b->m;
            bid = bid->next;
        }

        da_int rowidx = i - offset;
        da_int colidx = j - bid->offset;
        T *col = nullptr;
        da_int stride;
        block_base<T> *bb = static_cast<block_base<T> *>(bid->b);
        bb->get_col(colidx, &col, stride);
        elem = col[rowidx * stride];

        return da_status_success;
    }
    template <class T> da_status set_element(da_int i, da_int j, T elem) {

        if (i < 0 || i >= this->m)
            return da_error(err, da_status_invalid_input,
                            "i = " + std::to_string(i) +
                                ". The row index must be between 0 and " +
                                std::to_string(this->m - 1) + ".");
        if (j < 0 || j >= this->n)
            return da_error(err, da_status_invalid_input,
                            "j = " + std::to_string(i) +
                                ". The column index must be between 0 and " +
                                std::to_string(this->n - 1) + ".");

        auto it = cmap.find(j);
        if (it == cmap.end())
            // Cannot happen, checks on i and j would have returned invalid input already
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Couldn't find the element");

        std::shared_ptr<block_id> bid = it->second;
        if (bid->b->btype != get_block_type<T>())
            return da_error(err, da_status_invalid_input, "Incompatible types");

        da_int offset = 0;
        while (i >= bid->b->m + offset) {
            offset += bid->b->m;
            bid = bid->next;
        }

        da_int rowidx = i - offset;
        da_int colidx = j - bid->offset;
        T *col = nullptr;
        da_int stride;
        block_base<T> *bb = static_cast<block_base<T> *>(bid->b);
        bb->get_col(colidx, &col, stride);
        col[rowidx * stride] = elem;

        return da_status_success;
    }

    /* Column tags methods */
    da_status label_column(std::string label, da_int idx) {
        if (idx < 0 || idx >= n)
            return da_error(err, da_status_invalid_input,
                            "requested idx not in the range");
        if (index_to_name.size() != (sz_t)n)
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "maps and store size are out of sync");

        auto it = name_to_index.insert(std::make_pair(label, idx)).first;
        std::string const *pstr = &it->first;
        index_to_name[idx] = pstr;
        return da_status_success;
    }

    da_status get_idx_from_label(std::string label, da_int &idx) {
        auto it = name_to_index.find(label);
        if (it == name_to_index.end()) {
            return da_error(err, da_status_invalid_input,
                            "key " + label + " is not in the map");
        }
        idx = it->second;
        return da_status_success;
    }

    da_status get_col_label(da_int idx, std::string &name) {
        if (idx < 0 || idx >= n)
            return da_error(err, da_status_invalid_input,
                            "requested idx not in the range");

        if (index_to_name[idx] != nullptr)
            name = *index_to_name[idx];
        else
            name = "";
        return da_status_success;
    }

    /* Label all columns from an array of C strings
     * headings is NOT checked:
     * if set to nullptr, no columns will be tagged but no error will be returned
     *
     */
    da_status label_all_columns(char **headings) {
        da_status status = da_status_success;

        if (headings != nullptr) {
            std::string buf;
            for (da_int j = 0; j < n; j++) {
                buf = std::string(headings[j]);
                status = label_column(buf, j);
                if (status != da_status_success) {
                    // LCOV_EXCL_START
                    std::string err_msg = "Could not label column number: ";
                    err_msg += std::to_string(j);
                    return da_error_trace(err, da_status_internal_error, err_msg);
                    // LCOV_EXCL_STOP
                }
            }
        }
        return status;
    }

    /* The data is copied 3 times, which might be problematic for big CSVs:
     *     + read entire CSV in char** form
     *     + convert it into CSVColumnsType
     *     + create individual blocks and insert them into the datastore
     */

    /* Creates a new block based on a selection of columns from the CSVColumnsType which are of the same type */
    template <class T>
    da_status create_block_from_csv_columns(da_csv::csv_reader *csv,
                                            [[maybe_unused]] std::vector<T> dummy,
                                            da_auto_detect::CSVColumnsType &columns,
                                            da_int start_column, da_int end_column,
                                            da_int nrows) {
        da_status status;
        bool cleanup = false;
        da_int ncols;

        T *bl = nullptr;
        bool C_data;
        status = raw_ptr_from_csv_columns(csv, columns, start_column, end_column, nrows,
                                          &bl, C_data);
        if (status != da_status_success) {
            // LCOV_EXCL_START
            cleanup = true;
            status = da_error_trace(err, da_status_internal_error,
                                    "Unexpected error in creating raw pointers");
            goto exit;
            // LCOV_EXCL_STOP
        }

        ncols = end_column - start_column + 1;
        status = concatenate_cols_csv(nrows, ncols, bl, column_major, false, C_data);
        if (status != da_status_success) {
            // LCOV_EXCL_START
            status = da_error_trace(err, da_status_internal_error,
                                    "Unexpected error in concatenating columns");
            goto exit;
            // LCOV_EXCL_STOP
        }

    exit:
        if (cleanup) {
            if (bl != nullptr) {
                if (!C_data)
                    delete[] bl;
                else
                    free(bl);
            }
        }
        return status;
    }

    template <class T>
    da_status concatenate_cols_csv(da_int mc, da_int nc, T *data, da_order order,
                                   bool copy_data, bool C_data) {
        return concatenate_columns(mc, nc, data, order, copy_data, true, C_data);
    }
    template <class T>
    da_status raw_ptr_from_csv_columns([[maybe_unused]] da_csv::csv_reader *csv,
                                       da_auto_detect::CSVColumnsType &columns,
                                       da_int start_column, da_int end_column,
                                       da_int nrows, T **bl, bool &C_data) {
        da_int ncols = end_column - start_column + 1;
        C_data = false;
        try {
            *bl = new T[ncols * nrows];
        } catch (std::bad_alloc &) { // LCOV_EXCL_LINE
            return da_error(err, da_status_memory_error,
                            "Allocation error"); // LCOV_EXCL_LINE
        }

        for (da_int i = 0; i < ncols; i++) {
            if (std::vector<T> *T_col =
                    std::get_if<std::vector<T>>(&(columns[start_column + i]))) {
                for (da_int j = 0; j < nrows; j++) {
                    (*bl)[i * nrows + j] = (*T_col)[j];
                }
            } else {
                // This shouldn't be possible
                return da_error(err, da_status_internal_error,       // LCOV_EXCL_LINE
                                "wrong type detected unexpectedly"); // LCOV_EXCL_LINE
            }
        }

        return da_status_success;
    }

    /* Given a CSVColumns object create the relevant blocks for the datastore */
    da_status convert_csv_columns_to_blocks(da_csv::csv_reader *csv,
                                            da_auto_detect::CSVColumnsType &columns,
                                            da_int nrows, da_int ncols) {

        da_status status = da_status_success;

        // Active index denotes the type of the column we are currently looking at
        da_int active_index = (da_int)columns[0].index();
        da_int start_column = 0;

        for (da_int i = 1; i <= ncols; i++) {

            // Get current column index with a trick to make sure the last column is dealt with correctly
            da_int column_index =
                (da_int)((i < ncols) ? columns[i].index() : active_index + 1);

            if (column_index != active_index) {

                // Extract a dummy variable to enforce correct templating
                std::visit(
                    [&](const auto elem) {
                        status = create_block_from_csv_columns(
                            csv, elem, columns, start_column, i - 1, nrows);
                    },
                    columns[i - 1]);
                if (status != da_status_success)
                    return da_error_trace(err, da_status_internal_error, // LCOV_EXCL_LINE
                                          "Unexpected error");

                // Update the active index and start_column variables
                start_column = i;
                active_index = column_index;
            }
        }
        m = nrows;
        n = ncols;

        return da_status_success;
    }

    template <typename T>
    da_status load_from_csv_single_type(da_csv::csv_reader *csv, const char *filename) {

        using namespace da_csv;
        da_status status, exit_status;

        if (csv->datatype == csv_auto)
            return da_error(err, da_status_invalid_input,
                            "Single typed CSV reader cannot handle automatic detection.");

        T *data = nullptr;
        da_int get_headings = csv->first_row_header;
        char **headings = nullptr;
        da_int nrows = 0, ncols = 0;
        bool copy_data = false, own_data = true, C_data = true;

        exit_status = parse_and_process(csv, filename, &data, &nrows, &ncols,
                                        get_headings, &headings);
        // We need to take care checking for allowed error exits and warnings here to avoid memory leaks
        if (exit_status == da_status_parsing_error) {
            free_data(&headings, ncols);
            free_data(&data, 0);
            return da_error_trace(
                err, exit_status,
                "Parsing error, Consult error trace for further details");
        } else if (exit_status != da_status_success &&
                   exit_status != da_status_missing_data) {
            free_data(&headings, ncols);
            free_data(&data, 0);
            return da_error_trace(
                err, exit_status,
                "Parsing error, Consult error trace for further details");
        }
        status = concatenate_columns(nrows, ncols, data, csv->order, copy_data, own_data,
                                     C_data);
        if (status != da_status_success)
            return da_error_trace(err, da_status_internal_error, // LCOV_EXCL_LINE
                                  "Failed concatenation.");

        status = label_all_columns(headings);
        if (status != da_status_success)
            return da_error_trace(err, da_status_internal_error, // LCOV_EXCL_LINE
                                  "Unexpected error in column labeling.");
        if (headings != nullptr) {
            for (da_int j = 0; j < ncols; j++) {
                if (headings[j])
                    free(headings[j]);
            }
            free(headings);
        }

        return exit_status;
    }

    /* Get data from a CSV file and create blocks appropriately
     * Exit status:
     * - da_status_parsing_error
     * - da_status_ragged_csv
     */
    da_status load_from_csv(da_csv::csv_reader *csv, const char *filename) {
        da_status status, exit_status;

        if (!empty()) {
            return da_error(csv->err, da_status_parsing_error,
                            "CSV files can only be read into empty datastore objects.");
        }

        status = csv->read_options();
        if (status != da_status_success) {
            return da_error(err, da_status_internal_error,
                            "Error reading CSV options"); // LCOV_EXCL_LINE
        }

        da_int get_headings = csv->first_row_header;
        char **headings = nullptr;
        da_int nrows = 0, ncols = 0;
        char **data_char = nullptr;

        // User may have specified a single datatype or auto detection of multiple datatypes
        switch (csv->datatype) {
        case da_csv::csv_float: {
            exit_status = load_from_csv_single_type<float>(csv, filename);
            break;
        }
        case da_csv::csv_double: {
            exit_status = load_from_csv_single_type<double>(csv, filename);
            break;
        }
        case da_csv::csv_integer: {
            exit_status = load_from_csv_single_type<da_int>(csv, filename);
            break;
        }
        case da_csv::csv_char: {
            exit_status = load_from_csv_single_type<char *>(csv, filename);
            break;
        }
        case da_csv::csv_boolean: {
            exit_status = load_from_csv_single_type<uint8_t>(csv, filename);
            break;
        }
        case da_csv::csv_auto:
        default: {
            // Auto detection of datatype for each column
            exit_status = da_csv::parse_and_process(csv, filename, &data_char, &nrows,
                                                    &ncols, get_headings, &headings);
            if (exit_status == da_status_parsing_error) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_char, nrows * ncols);
                return da_error(err, da_status_parsing_error,
                                "Consult error trace for further details");
            } else if (exit_status != da_status_success &&
                       exit_status != da_status_missing_data) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_char, nrows * ncols);
                return da_error(err, exit_status,
                                "Consult error trace for further details");
            }

            da_auto_detect::CSVColumnsType columns;
            // Call routine to detect the datatype of each column
            da_auto_detect::detect_columns(csv, columns, data_char, nrows, ncols);
            // Convert columns into blocks for the datastore
            status = convert_csv_columns_to_blocks(csv, columns, nrows, ncols);
            if (status != da_status_success)
                return da_error_trace(err, status, // LCOV_EXCL_LINE
                                      "Could not concatenate the columns to the data "
                                      "store");

            da_csv::free_data(&data_char, nrows * ncols);

            status = label_all_columns(headings);
            if (headings != nullptr) {
                for (da_int j = 0; j < ncols; j++) {
                    if (headings[j])
                        free(headings[j]);
                }
                free(headings);
            }
            if (status != da_status_success)
                return da_error_trace( // LCOV_EXCL_LINE
                    err, da_status_internal_error, "Unexpected error in column labeling");
            break;
        }
        }

        return exit_status;
    }
}; // end of data_store

/* Template specialization declaration
 * This is needed for Windows builds
 */
template <>
da_status data_store::concatenate_cols_csv<char **>(da_int mc, da_int nc, char ***data,
                                                    da_order order, bool copy_data,
                                                    bool C_data);
template <>
da_status data_store::raw_ptr_from_csv_columns<char **>(
    [[maybe_unused]] da_csv::csv_reader *csv, da_auto_detect::CSVColumnsType &columns,
    da_int start_column, da_int end_column, da_int nrows, char ****bl, bool &C_data);
} // namespace da_data

#endif
