#ifndef DATA_STORE_HPP
#define DATA_STORE_HPP

#include "aoclda.h"
#include "auto_detect_csv.hpp"
#include "csv_reader.hpp"
#include "interval_map.hpp"
#include "read_csv.hpp"
#include <ciso646> // Fixes an MSVC issue
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace da_data {

using da_interval_map::interval;

bool validate_interval(interval p, da_int max_val);

enum block_type {
    block_none,
    block_string,
    block_int,
    block_real,
    block_char,
    block_str,
    block_bool //primarily intended for uint8_t data obtained from true/false values in a CSV file
};

template <typename T> struct get_block_type {};
template <> struct get_block_type<da_int> {
    constexpr operator block_type() const { return block_int; }
};
template <> struct get_block_type<da_int *> {
    constexpr operator block_type() const { return block_int; }
};
template <> struct get_block_type<float> {
    constexpr operator block_type() const { return block_real; }
};
template <> struct get_block_type<float *> {
    constexpr operator block_type() const { return block_real; }
};
template <> struct get_block_type<double> {
    constexpr operator block_type() const { return block_real; }
};
template <> struct get_block_type<double *> {
    constexpr operator block_type() const { return block_real; }
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

class block {
  public:
    da_int m, n;
    block_type btype = block_none;
    da_errors::da_error_t *err = nullptr;
    virtual ~block(){};
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
     */
    T *bl = nullptr;
    da_ordering order;
    bool own_data = false;

  public:
    ~block_dense() {
        if (own_data) {
            (get_block_type<T>() == block_char)
                ? da_csv::free_data(&bl, this->m * this->n)
                : delete[] bl;
        }
    };

    /* constructor can throw bad_alloc exception
     * it should be caught every time it is called
     */
    block_dense(da_int m, da_int n, T *data, da_errors::da_error_t &err,
                da_ordering order = row_major, bool copy_data = false) {
        if (m <= 0 || n <= 0 || data == nullptr)
            throw std::invalid_argument("");
        this->m = m;
        this->n = n;
        this->order = order;
        this->own_data = copy_data;
        this->err = &err;
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

        case col_major:
            *col = &bl[this->m * idx];
            stride = 1;
            break;
        }

        return da_status_success;
    }

    da_status copy_slice_dense(interval cols, interval rows, da_int idx_start,
                               da_int ld_data, T *data) {

        if (!validate_interval(cols, this->n) || !validate_interval(rows, this->m))
            return da_error(this->err, da_status_invalid_input,
                            "The input intervals are not valid");

        da_int idx, ncols, nrows, idx_d;
        ncols = cols.second - cols.first + 1;
        nrows = rows.second - rows.first + 1;
        idx_d = 0;
        switch (order) {
        case col_major:
            idx = cols.first * this->m;
            for (da_int j = 0; j < ncols; j++) {
                idx += rows.first;
                idx_d += idx_start;
                for (da_int i = 0; i < nrows; i++) {
                    data[idx_d] = bl[idx];
                    idx++;
                    idx_d++;
                }
                idx += this->m - nrows - rows.first;
                idx_d += ld_data - nrows - idx_start;
            }
            break;

        case row_major:
            idx = 0;
            for (da_int j = 0; j < ncols; j++) {
                idx = rows.first * this->n + cols.first + j;
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
};

/* wrapper structure containing a pointer to a block and meta-data around the block */
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

using namespace da_interval_map;
using columns_map = interval_map<std::shared_ptr<block_id>>;
using idx_slice = interval_map<da_int>;
using selection_map =
    std::unordered_map<std::string,
                       std::pair<std::unique_ptr<idx_slice>, std::unique_ptr<idx_slice>>>;

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

    /* To store column headings */
    char **col_headings = nullptr;

    /* Data to handle partially added rows 
     * missing_block: if true, blocks the other functionality until the partial row block is finished
     * idx_start_missing: column index of the start of the missing block
     */
    bool missing_block = false;
    da_int idx_start_missing;

    /* col|row_selection: interval maps containing the subset of rows and columns selected */
    idx_slice col_selection, row_selection;

    /* hash map of selections, using user defined labels (c strings)*/
    selection_map selections;

    /* Bidirectional map linking column index to its name */
    using sz_t = std::vector<std::string const *>::size_type;
    std::unordered_map<std::string, da_int> name_to_index;
    std::vector<std::string const *> index_to_name;

    /* error structure pointing to the main handle's*/
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

        if (col_headings) {
            for (da_int i = 0; i < n; i++) {
                if (col_headings[i])
                    free(col_headings[i]);
            }
            free(col_headings);
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
    da_status concatenate_columns(da_int mc, da_int nc, T *data, da_ordering order,
                                  bool copy_data = false, bool own_data = false) {

        // mc must match m except if the initial data.frame is empty (new block is created)
        // mc*nc input data must not be empty
        if (mc <= 0 || nc <= 0 || (m > 0 && m != mc))
            return da_error(err, da_status_invalid_input,
                            "Invalid dimensions in the provided data");

        // Create a dense block from the raw data
        std::shared_ptr<block_id> new_block;
        try {
            new_block = std::make_shared<block_id>(block_id());
            new_block->b = new block_dense<T>(mc, nc, data, *this->err, order, copy_data);
            new_block->offset = n;
            block_dense<T> *bd = static_cast<block_dense<T> *>(new_block->b);
            bd->set_own_data(own_data || copy_data);
        } catch (std::bad_alloc &) { // LCOV_EXCL_LINE
            return da_error(err, da_status_memory_error,
                            "Memory allocation error"); // LCOV_EXCL_LINE
        }

        // Concatenate columns to the right, indices of new columns are [n, n+nc-1]
        cmap.insert(interval(n, n + nc - 1), new_block);

        // update the size of the datastore if it was initially empty
        if (m == 0)
            m = mc;
        n += nc;

        return da_status_success;
    }

    template <class T>
    da_status concatenate_rows(da_int mr, da_int nr, T *data, da_ordering order,
                               bool copy_data = false) {
        da_status status = da_status_success;
        bool found;
        da_int lb, ub, idx_start;
        bool cleanup = false;
        std::shared_ptr<block_id> new_block = nullptr;

        if (n <= 0) {
            // First block, columns need to be concatenated instead.
            status = this->concatenate_columns(mr, nr, data, order, copy_data);
        } else {
            idx_start = 0;
            if (missing_block)
                idx_start = idx_start_missing;
            if (mr <= 0 || nr <= 0 || (n > 0 && nr + idx_start > n))
                return da_error(err, da_status_invalid_input,
                                "Invalid dimensions in the provided data");

            // Create a dense block from the raw data
            try {
                new_block = std::make_shared<block_id>(block_id());
                new_block->b = new block_dense<T>(mr, nr, data, *err, order, copy_data);
            } catch (std::bad_alloc &) { // LCOV_EXCL_LINE
                return da_error(err, da_status_memory_error,
                                "Memory allocation error"); // LCOV_EXCL_LINE
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
                    status = da_status_invalid_input;
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
            }
        }

        if (cleanup) {
            // error occured, clear the 'next' pointers from the existing blocks
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
                ub = it->first.second;
                ++it;
            }
            m -= mr;
        }
        return status;
    }

    /* concat the current datastore with the store passed on the interface horizontally 
     * in output, 'this' will contain the concatenation and store will be destroyed.
     */
    da_status horizontal_concat(data_store &store) {
        if (m != store.m)
            return da_error(err, da_status_invalid_input,
                            "Invalid dimensions in the provided data");
        if (missing_block || store.missing_block)
            return da_error(
                err, da_status_invalid_input,
                "Cannot concatenate stores at this stage, some data is missing");

        columns_map::iterator it1 = store.cmap.begin(), it2;
        da_int n_orig = n;
        std::shared_ptr<block_id> store_block, next, current_block;
        while (it1 != store.cmap.end()) {
            da_int nc = it1->first.second - it1->first.first + 1;
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

    da_status extract_headings(da_int n, char **headings) {
        if (n != this->n) {
            //m = this->m; TODO these would need to be pointers in the C interface
            return da_error(err, da_status_invalid_input,
                            "Invalid dimensions in the provided data");
        }

        if (this->col_headings != nullptr) {
            for (da_int i = 0; i < n; i++) {
                headings[i] = this->col_headings[i];
            }
        }

        return da_status_success;
    }

    /* Extract column idx into vector col
     * m: expected size of the column by the user; if wrong correct m size is returned
     * col: user allocated memory of size at least m
     */
    template <class T> da_status extract_column(da_int idx, da_int &m, T *col) {
        da_status status;

        if (m != this->m) {
            m = this->m;
            return da_error(err, da_status_invalid_input,
                            "Invalid dimensions in the provided data");
        }
        if (idx < 0 || idx >= this->n)
            return da_error(err, da_status_invalid_input, "Invalid idx");

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
                return da_error(err, da_status_internal_error,
                                "get_col failed unexpectedly:" + err->get_mesg());
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

        if (!validate_interval(rows, this->m) || !validate_interval(cols, this->n))
            return da_error(err, da_status_invalid_input, "Invalid intervals");
        if (ld_slice < rows.second - rows.first + 1)
            return da_error(err, da_status_invalid_input, "Invalid leading dimension");

        da_status status;

        da_int lcol = cols.first;
        da_int ucol = cols.second;
        da_int idx = first_idx;
        columns_map::iterator it;
        interval block_cols, block_rows;
        while (ucol - lcol >= 0) {
            it = cmap.find(lcol);
            std::shared_ptr<block_id> bid = it->second;
            if (bid->b->btype != get_block_type<T>())
                return da_error(err, da_status_invalid_input,
                                "Incompatible type in the slice");
            da_int uc = std::min(ucol, it->first.second);
            da_int lrow = rows.first;
            da_int lr = lrow;
            da_int urow = rows.second;
            da_int idxr = idx;
            da_int first_row_idx = 0;
            while (urow - lr >= 0) {
                da_int ur = std::min(urow, first_row_idx + bid->b->m - 1);
                block_rows = {lr - first_row_idx, ur - first_row_idx};
                if (block_rows.second >= block_rows.first) {
                    block_cols = {lcol - bid->offset, uc - bid->offset};
                    block_base<T> *bb = static_cast<block_base<T> *>(bid->b);
                    status = bb->copy_slice_dense(block_cols, block_rows, idxr, ld_slice,
                                                  slice);
                    if (status != da_status_success)
                        return da_error(
                            err, da_status_internal_error,
                            "Unexpected error in copy_slice_dense"); // LCOV_EXCL_LINE
                    idxr += ur - lr + 1;
                }
                lr = ur + 1;
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
        da_status exit_status = da_status_success;

        if (!validate_interval(cols, this->n) || !validate_interval(rows, this->m))
            return da_error(err, da_status_invalid_input,
                            "Invalid dimensions in the provided data");

        auto it = selections.find(key);
        if (it == selections.end()) {
            bool inserted;
            std::tie(it, inserted) = selections.insert(
                std::make_pair(key, std::make_pair(std::make_unique<idx_slice>(),
                                                   std::make_unique<idx_slice>())));
            if (!inserted) {
                exit_status = da_status_internal_error;
                goto exit;
            }
        }
        exit_status = it->second.first->insert(rows, rows.second - rows.first + 1);
        if (exit_status != da_status_success)
            goto exit;
        exit_status = it->second.second->insert(cols, cols.second - cols.first + 1);
        if (exit_status != da_status_success) {
            it->second.first->erase(rows.first);
            goto exit;
        }

    exit:
        return exit_status;
    }
    da_status select_columns(std::string key, interval cols) {
        da_status exit_status = da_status_success;

        if (!validate_interval(cols, this->n))
            return da_error(err, da_status_invalid_input, "Invalid intervals");

        auto it = selections.find(key);
        if (it == selections.end()) {
            bool inserted;
            std::tie(it, inserted) = selections.insert(
                std::make_pair(key, std::make_pair(std::make_unique<idx_slice>(),
                                                   std::make_unique<idx_slice>())));
            if (!inserted) {
                return da_error(err, da_status_invalid_input,
                                "Unexpected error in the interval insertion");
            }
        }
        exit_status = it->second.second->insert(cols, cols.second - cols.first + 1);

        return exit_status;
    }
    da_status select_rows(std::string key, interval rows) {
        da_status exit_status = da_status_success;

        if (!validate_interval(rows, this->m))
            return da_error(err, da_status_invalid_input, "Invalid interval");

        auto it = selections.find(key);
        if (it == selections.end()) {
            bool inserted;
            std::tie(it, inserted) = selections.insert(
                std::make_pair(key, std::make_pair(std::make_unique<idx_slice>(),
                                                   std::make_unique<idx_slice>())));
            if (!inserted) {
                return da_error(err, da_status_internal_error,
                                "Unexpected error in the interval insertion");
            }
        }

        exit_status = it->second.first->insert(rows, rows.second - rows.first + 1);

        return exit_status;
    }

    template <class T> da_status extract_selection(std::string key, da_int ld, T *data) {
        da_status exit_status = da_status_success, status;

        auto it = selections.find(key);
        bool clear_selections = false, clear_cols = false, clear_rows = false;
        da_int idx = 0;
        da_int ncols = 0;
        if (selections.empty()) {
            // no selection defined create a temporary selection all
            select_slice("All", {0, m - 1}, {0, n - 1});
            it = selections.find("All");
            clear_selections = true;
        } else if (it == selections.end()) {
            exit_status = da_status_invalid_input;
            goto exit;
        }

        if (it->second.first->empty()) {
            // no rows in the current selection, create a temporary one containing all
            status = select_rows(key, {0, m - 1});
            if (status != da_status_success) {
                exit_status = da_status_internal_error;
                goto exit;
            }
            clear_rows = true;
        }
        if (it->second.second->empty()) {
            // no cols in the current selection, create a temporary one containing all
            status = select_columns(key, {0, n - 1});
            if (status != da_status_success) {
                exit_status = da_status_internal_error;
                goto exit;
            }
            clear_cols = true;
        }

        for (auto it_col = it->second.second->begin(); it_col != it->second.second->end();
             ++it_col) {
            idx = ncols * ld;
            ncols += it_col->second;
            for (auto it_row = it->second.first->begin();
                 it_row != it->second.first->end(); ++it_row) {
                da_int nrows = it_row->second;
                exit_status = extract_slice(it_row->first, it_col->first, ld, idx, data);
                if (exit_status != da_status_success)
                    goto exit;
                idx += nrows;
            }
        }

    exit:
        if (clear_selections)
            selections.erase("All");
        if (clear_rows)
            it->second.first->erase(0);
        if (clear_cols)
            it->second.second->erase(0);

        return exit_status;
    }

    /* get|set_element access and modify a single element in the data store 
     * exit status:
     * - invalid_input
     * - internal_error
     */
    template <class T> da_status get_element(da_int i, da_int j, T &elem) {

        if (i < 0 || i >= this->m || j < 0 || j >= this->n)
            return da_error(err, da_status_invalid_input,
                            "Invalid dimensions in the provided data");

        auto it = cmap.find(j);
        if (it == cmap.end())
            // cannot happen, checks on i and j would have returned invalid input already
            return da_error(err, da_status_internal_error, "Couldn't find the element");

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

        if (i < 0 || i >= this->m || j < 0 || j >= this->n)
            return da_error(err, da_status_invalid_input,
                            "indices outside of the store dimensions");

        auto it = cmap.find(j);
        if (it == cmap.end())
            // cannot happen, checks on i and j would have returned invalid input already
            return da_error(err, da_status_internal_error, "Couldn't find the element");

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

    /* column tags methods */
    da_status tag_column(std::string name, da_int idx) {
        if (idx < 0 || idx >= n)
            return da_error(err, da_status_invalid_input,
                            "requested idx not in the range");
        if (index_to_name.size() != (sz_t)n)
            return da_error(err, da_status_internal_error,
                            "maps and store size are out of sync");

        auto it = name_to_index.insert(std::make_pair(name, idx)).first;
        std::string const *pstr = &it->first;
        index_to_name[idx] = pstr;
        return da_status_success;
    }

    da_status get_idx_from_tag(std::string key, da_int &idx) {
        auto it = name_to_index.find(key);
        if (it == name_to_index.end()) {
            return da_error(err, da_status_invalid_input, "key is not in the map");
        }
        idx = it->second;
        return da_status_success;
    }

    da_status get_col_name(da_int idx, std::string &name) {
        if (idx < 0 || idx >= n)
            return da_error(err, da_status_invalid_input,
                            "requested idx not in the range");

        name = *index_to_name[idx];
        return da_status_success;
    }

    /* FIXME
     * Ben review notes:
     * - if I understood correctly, the workflow for autodetection is as follows:
     *     + read entire CSV in char** form
     *     + convert it into CSVColumnsType
     *     + create individual blocks and insert them into the datastore
     *   the data is copied 3 times, which might be problematic for big CSVs. 
     *   Random ideas:
     *     + create an interface that reads directly into blocks assuming knowledge of the data types (auto or provided by users)
     *     + auto-detection works the same way but on a very small subset of rows to figure out the data types
     *     + that would probably mean that we would need a failsafe in the case that data is not correctly detected but we could get away with just returning not_implemented for now.
     * 
     * - same as in read_csv.hpp: handle should not be an argument at this level in the functionality. 
     * 
     * - in load_fro_csv, maybe part of the code could be factorized with templating? not sure of the best approach though
     */

    /* Creates a new block based on a selection of columns from the CSVColumnsType which are of the same type */
    template <class T>
    da_status create_block_from_csv_columns(da_csv::csv_reader *csv,
                                            [[maybe_unused]] std::vector<T> dummy,
                                            da_auto_detect::CSVColumnsType &columns,
                                            da_int start_column, da_int end_column,
                                            da_int nrows) {
        da_status status;

        T *bl = nullptr;
        status =
            raw_ptr_from_csv_columns(csv, columns, start_column, end_column, nrows, &bl);
        if (status != da_status_success)
            // FIXME: change to da_error_trace
            return (da_error(err, status, "unexpected"));

        da_int ncols = end_column - start_column + 1;
        status = concatenate_cols_csv(nrows, ncols, bl, col_major, false);
        if (status != da_status_success)
            // FIXME: change to da_error_trace
            return (da_error(err, status, "unexpected"));

        return da_status_success;
    }

    template <class T>
    da_status concatenate_cols_csv(da_int mc, da_int nc, T *data, da_ordering order,
                                   bool copy_data) {
        return concatenate_columns(mc, nc, data, order, copy_data, true);
    }

    template <class T>
    da_status raw_ptr_from_csv_columns([[maybe_unused]] da_csv::csv_reader *csv,
                                       da_auto_detect::CSVColumnsType &columns,
                                       da_int start_column, da_int end_column,
                                       da_int nrows, T **bl) {
        da_int ncols = end_column - start_column + 1;
        try {
            *bl = new T[ncols * nrows];
        } catch (std::bad_alloc &) {
            return da_error(err, da_status_memory_error, "Allocation error");
        }

        for (da_int i = 0; i < ncols; i++) {
            if (std::vector<T> *T_col =
                    std::get_if<std::vector<T>>(&(columns[start_column + i]))) {
                for (da_int j = 0; j < nrows; j++) {
                    (*bl)[i * nrows + j] = (*T_col)[j];
                }
            } else {
                // This shouldn't be possible
                return da_error(err, da_status_internal_error,
                                "wrong type detected unexpectedly");
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
        da_int active_index = columns[0].index();
        da_int start_column = 0;

        for (da_int i = 1; i <= ncols; i++) {

            // Get current column index with a trick to make sure the last column is dealt with correctly
            da_int column_index = (i < ncols) ? columns[i].index() : active_index + 1;

            if (column_index != active_index) {

                // Extract a dummy variable to enforce correct templating
                std::visit(
                    [&](const auto elem) {
                        status = create_block_from_csv_columns(
                            csv, elem, columns, start_column, i - 1, nrows);
                    },
                    columns[i - 1]);

                // Update the active index and start_column variables
                start_column = i;
                active_index = column_index;
            }
        }
        m = nrows;
        n = ncols;

        return da_status_success;
    }

    /* Get data from a CSV file and create blocks appropriately */
    da_status load_from_csv(da_csv::csv_reader *csv, const char *filename) {
        da_status error = da_status_success, tmp_error = da_status_success;

        if (!empty()) {
            return da_error(csv->err, da_status_parsing_error,
                            "CSV files can only be read into empty datastore objects.");
        }

        error = csv->read_options();
        if (error != da_status_success) {
            return da_error(err, da_status_internal_error, "Error reading CSV options");
        }

        da_int get_headings = csv->first_row_header;
        char **headings = nullptr;
        da_int nrows = 0, ncols = 0;
        float *data_f = nullptr;
        double *data_d = nullptr;
        da_int *data_int = nullptr;
        char **data_char = nullptr;
        uint8_t *data_bool = nullptr;

        // User may have specified a single datatype or auto detection of multiple datatypes
        switch (csv->datatype) {
        case da_csv::csv_float: {
            tmp_error = da_csv::parse_and_process(csv, filename, &data_f, &nrows, &ncols,
                                                  get_headings, &headings);
            // We need to take care checking for allowed error exits and warnings here to avoid memory leaks
            if (tmp_error == da_status_no_data) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_f, 0);
                return da_error(err, da_status_no_data, "No data");
            } else if (tmp_error != da_status_success &&
                       tmp_error != da_status_missing_data &&
                       tmp_error != da_status_bad_lines) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_f, 0);
                return da_error(err, tmp_error, "Error parsing CSV");
            }
            error = concatenate_columns(nrows, ncols, data_f, row_major, false);
            // We need to find the block and set copy_data to true so it knows it owns the memory
            block_dense<float> *tmp_bd =
                dynamic_cast<block_dense<float> *>((*cmap.begin()).second->b);
            if (tmp_bd) {
                tmp_bd->set_own_data(true);
            }
            break;
        }
        case da_csv::csv_double: {
            tmp_error = da_csv::parse_and_process(csv, filename, &data_d, &nrows, &ncols,
                                                  get_headings, &headings);
            if (tmp_error == da_status_no_data) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_d, 0);
                return da_error(err, da_status_no_data, "No data");
            } else if (tmp_error != da_status_success &&
                       tmp_error != da_status_missing_data &&
                       tmp_error != da_status_bad_lines) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_d, 0);
                return da_error(err, tmp_error, "Error parsing CSV");
            }
            error = concatenate_columns(nrows, ncols, data_d, row_major, false);
            block_dense<double> *tmp_bd =
                dynamic_cast<block_dense<double> *>((*cmap.begin()).second->b);
            if (tmp_bd) {
                tmp_bd->set_own_data(true);
            }
            break;
        }
        case da_csv::csv_integer: {
            tmp_error = da_csv::parse_and_process(csv, filename, &data_int, &nrows,
                                                  &ncols, get_headings, &headings);
            if (tmp_error == da_status_no_data) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_int, 0);
                return da_error(err, da_status_no_data, "No data");
            } else if (tmp_error != da_status_success &&
                       tmp_error != da_status_missing_data &&
                       tmp_error != da_status_bad_lines) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_int, 0);
                return da_error(err, tmp_error, "Error parsing CSV");
            }
            error = concatenate_columns(nrows, ncols, data_int, row_major, false);
            block_dense<da_int> *tmp_bd =
                dynamic_cast<block_dense<da_int> *>((*cmap.begin()).second->b);
            if (tmp_bd) {
                tmp_bd->set_own_data(true);
            }
            break;
        }
        case da_csv::csv_char: {
            tmp_error = da_csv::parse_and_process(csv, filename, &data_char, &nrows,
                                                  &ncols, get_headings, &headings);
            if (tmp_error == da_status_no_data) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_char, nrows * ncols);
                return da_error(err, da_status_no_data, "No data");
            } else if (tmp_error != da_status_success &&
                       tmp_error != da_status_missing_data &&
                       tmp_error != da_status_bad_lines) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_char, nrows * ncols);
                return da_error(err, tmp_error, "Error parsing CSV");
            }
            error = concatenate_columns(nrows, ncols, data_char, row_major, false);
            block_dense<char *> *tmp_bd =
                dynamic_cast<block_dense<char *> *>((*cmap.begin()).second->b);
            if (tmp_bd) {
                tmp_bd->set_own_data(true);
            }
            break;
        }
        case da_csv::csv_boolean: {
            tmp_error = da_csv::parse_and_process(csv, filename, &data_bool, &nrows,
                                                  &ncols, get_headings, &headings);
            if (tmp_error == da_status_no_data) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_bool, 0);
                return da_error(err, da_status_no_data, "No data");
            } else if (tmp_error != da_status_success &&
                       tmp_error != da_status_missing_data &&
                       tmp_error != da_status_bad_lines) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_bool, 0);
                return da_error(err, tmp_error, "Error parsing CSV");
            }
            error = concatenate_columns(nrows, ncols, data_bool, row_major, false);
            block_dense<uint8_t> *tmp_bd =
                dynamic_cast<block_dense<uint8_t> *>((*cmap.begin()).second->b);
            if (tmp_bd) {
                tmp_bd->set_own_data(true);
            }
            break;
        }
        case da_csv::csv_auto:
        default: {
            // Auto detection of datatype for each column
            tmp_error = da_csv::parse_and_process(csv, filename, &data_char, &nrows,
                                                  &ncols, get_headings, &headings);
            if (tmp_error == da_status_no_data) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_char, nrows * ncols);
                return da_error(err, da_status_no_data, "No data");
            } else if (tmp_error != da_status_success &&
                       tmp_error != da_status_missing_data &&
                       tmp_error != da_status_bad_lines) {
                da_csv::free_data(&headings, ncols);
                da_csv::free_data(&data_char, nrows * ncols);
                return da_error(err, tmp_error, "Error parsing CSV");
            }

            da_auto_detect::CSVColumnsType columns;
            // Call routine to detect the datatype of each column
            da_auto_detect::detect_columns(csv, columns, data_char, nrows, ncols);
            // Convert columns into blocks for the datastore
            error = convert_csv_columns_to_blocks(csv, columns, nrows, ncols);

            da_csv::free_data(&data_char, nrows * ncols);
            break;
        }
        }

        col_headings = headings;
        //std::string buf;
        //for (da_int j = 0; j < ncols; j++) {
        //    buf = std::string(headings[j]);
        //    tmp_error = tag_column(buf, j);
        //}
        if (error == da_status_success)
            error = tmp_error;
        return error;
    }
};

} // namespace da_data

#endif