#ifndef DATA_STORE_HPP
#define DATA_STORE_HPP

#include "aoclda.h"
#include "interval_map.hpp"
#include <ciso646> // Fixes an MSVC issue
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace da_data {

enum block_type {
    block_none,
    block_string,
    block_int,
    block_real,
};

template <typename T> struct get_block_type {};
template <> struct get_block_type<da_int> {
    constexpr operator block_type() const { return block_int; }
};
template <> struct get_block_type<float> {
    constexpr operator block_type() const { return block_real; }
};
template <> struct get_block_type<double> {
    constexpr operator block_type() const { return block_real; }
};
template <> struct get_block_type<std::string> {
    constexpr operator block_type() const { return block_string; }
};

class block {
  public:
    da_int m, n;
    block_type btype = block_none;
    virtual ~block(){};
};

template <class T> class block_base : public block {
  public:
    virtual da_status get_col(da_int idx, T **col, da_int &stride) = 0;
};

/* Define a block of dense base data (int, float, double, string) 
 * bl: dense data stored in a m*n vector
 */
template <class T> class block_dense : public block_base<T> {
    T *bl = nullptr;
    da_ordering order;
    bool copy_data = false;

  public:
    ~block_dense() {
        if (copy_data) {
            delete[] bl;
        }
    };

    block_dense(da_int m, da_int n, T *data, da_ordering order = row_major,
                bool copy_data = false) {
        if (m <= 0 || n <= 0 || data == nullptr)
            throw std::invalid_argument("");
        this->m = m;
        this->n = n;
        this->order = order;
        this->copy_data = copy_data;
        if (!copy_data)
            bl = data;
        else {
            bl = new T[m * n];
            for (da_int i = 0; i < m * n; i++)
                bl[i] = data[i];
        }
        this->btype = get_block_type<T>();
    };

    da_status get_col(da_int idx, T **col, da_int &stride) {

        if (idx < 0 || idx > this->n)
            return da_status_invalid_input;

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
};

class block_id {
  public:
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

class data_store {
    /* m, n: total dimension of the data_store (aggregate of all the blocks)*/
    da_int m = 0, n = 0;
    columns_map cmap;

    bool missing_block = false;
    da_int idx_start_missing;

  public:
    ~data_store() {}

    bool empty() { return m == 0 && n == 0 && cmap.empty(); }

    template <class T>
    da_status concatenate_columns(da_int mc, da_int nc, T *data, da_ordering order,
                                  bool copy_data = false) {

        if (mc <= 0 || nc <= 0 || (m > 0 and m != mc))
            return da_status_invalid_input;

        // Create a dense block from the raw data
        std::shared_ptr<block_id> new_block;
        try {
            new_block = std::make_shared<block_id>(block_id());
            new_block->b = new block_dense<T>(mc, nc, data, order, copy_data);
            new_block->offset = n;
        } catch (std::bad_alloc &) {
            return da_status_memory_error;
        }

        // Concatenate columns to the right, indices of new columns are [n, n+nc-1]
        cmap.insert(n, n + nc - 1, new_block);

        if (m == 0)
            m = mc;
        n += nc;

        return da_status_success;
    }

    template <class T>
    da_status concatenate_rows(da_int mr, da_int nr, T *data, da_ordering order,
                               bool copy_data = false) {
        da_status exit_status = da_status_success;
        bool found;
        da_int lb, ub, idx_start;
        bool cleanup = false;

        if (n <= 0) {
            // First block, columns need to be concatenated instead.
            exit_status = this->concatenate_columns(mr, nr, data, order, copy_data);
        } else {
            idx_start = 0;
            if (missing_block)
                idx_start = idx_start_missing;
            if (mr <= 0 || nr <= 0 || (n > 0 && nr + idx_start > n))
                return da_status_invalid_input;

            // Create a dense block from the raw data
            std::shared_ptr<block_id> new_block;
            try {
                new_block = std::make_shared<block_id>(block_id());
            } catch (std::bad_alloc &) {
                return da_status_memory_error;
            }
            new_block->b = new block_dense<T>(mr, nr, data, order, copy_data);

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
                    exit_status = da_status_invalid_input;
                    break;
                }
                while (current_block->next != nullptr)
                    current_block = current_block->next;
                current_block->next = new_block;
                if (new_block->left_parent == nullptr)
                    new_block->left_parent = current_block;
            }
            if (idx_start + nr < n) {
                missing_block = true;
                idx_start_missing = idx_start + nr;
            } else {
                missing_block = false;
                idx_start_missing = 0;
            }
        }

        if (cleanup) {
            // error occured, clear the 'next' pointers from the existing blocks
            ub = idx_start - 1;
            std::shared_ptr<block_id> current_block;
            while (ub < nr - 1) {
                found = cmap.find(ub + 1, current_block, lb, ub);
                if (!found)
                    break;
                current_block->next = nullptr;
            }
            m -= mr;
        }
        return exit_status;
    }

    /* Extract column idx into vector col
     * m: expected size of the column by the user; if wrong correct m size is returned
     * col: user allocated memory of size at least m
     */
    template <class T> da_status extract_column(da_int idx, da_int &m, T *col) {
        da_status status;

        if (m != this->m) {
            m = this->m;
            return da_status_invalid_input;
        }
        if (idx < 0 || idx >= this->n)
            return da_status_invalid_input;

        T *c;
        da_int stride, nrows, lb, ub;
        da_int idxrow = 0;
        block_dense<T> *bb;
        std::shared_ptr<block_id> id;
        cmap.find(idx, id, lb, ub);
        if (id->b->btype != get_block_type<T>())
            return da_status_invalid_input;
        while (id != nullptr) {
            bb = static_cast<block_dense<T> *>(id->b);
            status = bb->get_col(idx - id->offset, &c, stride);
            if (status != da_status_success)
                return da_status_internal_error;
            nrows = bb->m;
            for (da_int i = 0; i < nrows; i++)
                col[idxrow + i] = c[i * stride];
            idxrow += nrows;
            id = id->next;
        }

        return da_status_success;
    }

    /* concat the current datastore with the store passed on the interface horizontally 
     * in output, 'this' will contain the concatenation and store will be destroyed.
     */
    da_status horizontal_concat(data_store &store) {
        if (m != store.m)
            return da_status_invalid_input;
        if (missing_block || store.missing_block)
            return da_status_invalid_input;

        columns_map::iterator it1 = store.cmap.begin(), it2;
        da_int n_orig = n;
        std::shared_ptr<block_id> store_block, next, current_block;
        while (it1 != store.cmap.end()) {
            da_int nc = it1->first.second - it1->first.first + 1;
            store_block = it1->second;
            store_block->offset += n_orig;
            cmap.insert(n, n + nc - 1, store_block);
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
};

} // namespace da_data

class _da_datastore {
  public:
    da_data::data_store *store = nullptr;
};
#endif