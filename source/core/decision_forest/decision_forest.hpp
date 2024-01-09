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

#ifndef DF_HPP
#define DF_HPP

#include <functional>
#include <random>

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "options.hpp"

#include "decision_forest_aux.hpp"

#include "decision_forest_options.hpp"

#include "decision_forest_types.hpp"

namespace da_df {

template <typename T> struct Node {
    // data required for leaf nodes during prediction
    bool is_leaf = true;
    uint8_t y_pred = 0;
    // data required for leaf nodes during fitting
    da_int start_idx = 0;
    da_int n_obs = 0;
    // data required for root nodes during prediction
    da_int child_node_l_idx = 0;
    da_int child_node_r_idx = 0;
    da_int col_idx = 0;
    T x_threshold = 0.0;
    da_int level = 0;
};

// ---------------------------------------------
// declaration of decision_tree class
// ---------------------------------------------

template <typename T> class decision_tree : public basic_handle<T> {
  private:
    // pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // true if the training data has been set
    bool set_data_done = false;
    // true if the model has been successfully trained
    bool model_trained = false;

    // variables that get set inside set_training_data(...)
    da_int n_obs = 0;
    da_int n_features_to_select = 0;
    da_int d = 0;
    std::vector<T> x;
    std::vector<uint8_t> y;

    // variables that get set / modified inside fit() from options registry
    da_int seed_val;
    bool set_rng_state = true;
    da_int scoring_fun_id;
    std::string scoring_fun_str;
    da_int max_level = 0;

    // other variables that get set / modified inside fit()
    std::vector<Node<T>> model;
    std::mt19937_64 mt_gen;
    std::vector<da_int> feature_ind;

  public:
    da_options::OptionRegistry opts;
    decision_tree(da_errors::da_error_t &err) {
        // assumes that err is valid
        this->err = &err;
        register_df_options<T>(opts);
    }

    da_status set_training_data(da_int n_obs, da_int n_features, T *x, da_int ldx,
                                uint8_t *y);
    da_status sample_feature_ind(da_int n_samples, da_int *samples);
    da_status set_rng(std::mt19937_64 mt_gen) {
        da_status status = da_status_success;
        // this->opts.set("seed", 0, da_options::solver);
        this->set_rng_state = false;
        this->mt_gen = mt_gen;
        return status;
    };
    std::mt19937_64 get_rng() { return this->mt_gen; };
    da_status fit();
    da_status predict(da_int n_obs, da_int n_features, T *x, da_int ldx, uint8_t *y_pred);
    da_status score(da_int n_obs, da_int n_features, T *x, da_int ldx, uint8_t *y_test,
                    T *score);
    std::function<T(T, da_int, T, da_int)> score_fun;
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result) {

        return da_warn(err, da_status_unknown_query,
                       "There are no integer results available for this API.");
    };
};

// ---------------------------------------------
// declaration of decision_forest class
// ---------------------------------------------

template <typename T> class decision_forest : public basic_handle<T> {
  private:
    /* pointer to error trace */
    da_errors::da_error_t *err = nullptr;

    // true if the training data has been set
    bool set_data_done = false;
    // true if the model has been successfully trained
    bool model_trained = false;

    da_int n_obs = 0;
    da_int d = 0;
    std::vector<T> x;
    std::vector<uint8_t> y;

    da_int seed_val;
    std::mt19937_64 mt_gen;

    std::vector<decision_tree<T>> tree_vec;

  public:
    da_options::OptionRegistry opts;
    decision_forest(da_errors::da_error_t &err) {
        // assumes that err is valid
        this->err = &err;
        register_df_options<T>(opts);
    }

    da_status set_training_data(da_int n_obs, da_int n_features, T *x, da_int ldx,
                                uint8_t *y);

    da_status sample_feature_ind(da_int n_features, da_int n_samples,
                                 da_int *feature_ind);
    da_status sample_obs_ind(da_int n_obs, da_int n_samples, da_int *obs_ind);
    da_status fit_tree(decision_tree<T> *p_tree);
    da_status fit();
    da_status predict(da_int n_obs, da_int n_features, T *x, da_int ldx, uint8_t *y_pred);
    da_status score(da_int n_obs, da_int n_features, T *x, da_int ldx, uint8_t *y_test,
                    T *score);
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result) {

        return da_warn(err, da_status_unknown_query,
                       "There are no integer results available for this API.");
    };
};

// ---------------------------------------------
// member functions for decision_tree class
// ---------------------------------------------

template <typename T>
da_status decision_tree<T>::get_result(da_result query, da_int *dim, T *result) {
    da_status status = da_status_success;
    DA_PRINTF_DEBUG("Inside get_result \n");

    da_int rinfo_size = 3;

    if (result == nullptr || *dim <= 0) {
        return da_warn(err, da_status_invalid_array_dimension,
                       "The results array has not been allocated, or an unsuitable "
                       "dimension has been provided.");
    }

    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)seed_val;
        result[1] = (T)n_obs;
        result[2] = (T)d;
        break;
    default:
        return da_warn(err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return status;
}

template <typename T>
da_status decision_tree<T>::set_training_data(da_int n_obs, da_int n_features, T *x,
                                              da_int ldx, uint8_t *y) {

    da_status status = da_status_success;

    if (x == nullptr || y == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either x, or y are not valid pointers.");

    if (n_obs <= 0 || n_features <= 0) {
        DA_PRINTF_DEBUG("n_obs = %" DA_INT_FMT ", n_features = %" DA_INT_FMT " \n", n_obs,
                        n_features);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) +
                ", n_features = " + std::to_string(n_features) +
                ", the values of n_obs and n_features need to be greater than 0");
    }

    if (ldx < n_obs) {
        DA_PRINTF_DEBUG("ldx = %" DA_INT_FMT ", n_obs = %" DA_INT_FMT " \n", ldx, n_obs);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) + ", ldx = " + std::to_string(ldx) +
                ", the value of ldx needs to be at least as big as the value of n_obs");
    }

    this->n_obs = n_obs;
    this->d = n_features;

    // allocate memory foruser data
    try {
        this->x.resize(n_obs * n_features);
        this->y.resize(n_obs);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // copy user data (convert order from column-major to row-major)
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_obs; i++) {
            // ldx >= n_obs
            this->x[j + (i * n_features)] = x[i + (j * ldx)];
        }
    }

    for (da_int i = 0; i < n_obs; i++) {
        this->y[i] = y[i];
    }

    // record that data is initialized but fitting is not yet done
    set_data_done = true;
    model_trained = false;
    return status;
}

template <typename T> da_status decision_tree<T>::fit() {
    DA_PRINTF_DEBUG("Inside decision_tree<T>::fit \n");
#ifdef DA_LOGGING
    opts.print_options();
#endif

    da_status status = da_status_success;

    // check that set_training_data has been called
    if (set_data_done == false)
        return da_error(err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_df_set_training_data_s or da_df_set_training_data_d.");

    // set internal class data variables that directly correspond to options
    opts.get("seed", this->seed_val);
    opts.get("scoring function", scoring_fun_str, this->scoring_fun_id);
    opts.get("depth", this->max_level);
    opts.get("n_features_to_select", this->n_features_to_select);

    T diff_thres;
    opts.get("diff_thres", diff_thres);

    if (this->set_rng_state) {
        // set mt_gen (internal class data)
        if (this->seed_val == -1) {
            std::random_device r;
            this->seed_val = std::abs((da_int)r());
            mt_gen.seed(this->seed_val);
        } else {
            mt_gen.seed(this->seed_val);
        }
    }

    // initialize feature_ind (subset of indices and order in which features get selected)
    try {
        this->feature_ind.resize(n_features_to_select);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // set score_fun (std::function inside class)
    if (scoring_fun_id == scoring_fun::gini) {
        // Gini scoring function
        this->score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
            T score = 0.0;
            if (acc_l > 0) {
                T phat_l = acc_l / n_l;
                score += n_l * 2 * phat_l * (1 - phat_l);
            }
            if (acc_r > 0) {
                T phat_r = acc_r / n_r;
                score += n_r * 2 * phat_r * (1 - phat_r);
            }
            return score;
        };
    } else if (scoring_fun_id == scoring_fun::cross_entropy) {
        // Cross-entropy or deviance
        this->score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
            T phat_l = acc_l / n_l;
            T phat_r = acc_r / n_r;

            T score = 0.0;
            if (acc_l > 0) {
                score -= n_l * phat_l * log(phat_l);
            }
            if ((da_int)acc_l != n_l) {
                score -= n_l * (1 - phat_l) * log(1 - phat_l);
            }
            if (acc_r > 0) {
                score -= n_r * phat_r * log(phat_r);
            }
            if ((da_int)acc_r != n_r) {
                score -= n_r * (1 - phat_r) * log(1 - phat_r);
            }
            return score;
        };
    } else if (scoring_fun_id == scoring_fun::misclassification_error) {
        // Misclassification error
        this->score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
            da_int acc_l_argmax =
                (da_int)((acc_l > (n_l - acc_l)) ? acc_l : (n_l - acc_l));
            da_int acc_r_argmax =
                (da_int)((acc_r > (n_r - acc_r)) ? acc_r : (n_r - acc_r));

            da_int score = 0;
            if (acc_l_argmax > 0) {
                // T phat_l_argmax = (T)acc_l_argmax / n_l;
                // score += (da_int) (n_l * (1 - phat_l_argmax));
                score += (n_l - acc_l_argmax);
            }

            if (acc_r_argmax > 0) {
                // T phat_r_argmax = (T)acc_r_argmax / n_r;
                // score += (da_int) (n_r * (1 - phat_r_argmax));
                score += (n_r - acc_r_argmax);
            }
            return (T)score;
        };
    }

    for (da_int i = 0; i < 5; i++) {
        DA_PRINTF_DEBUG("%10.4f %10.4f %10.4f %10.4f \n", x[i * d], x[i * d + 1],
                        x[i * d + 2], x[i * d + 3]);
    }

    da_int n_nodes = 1;
    try {
        model.resize(n_nodes);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    model[0].is_leaf = true;
    model[0].start_idx = 0;
    model[0].n_obs = n_obs;
    model[0].level = 0;

    T score0 = no_split_score<T>(y.data(), n_obs, model[0].y_pred, score_fun);
    T score = score0;
    T min_score = score0;
    da_int split_node = 0;

    da_int n_nodes_m1 = n_nodes;
    da_int n_splits = 1;

    while (n_splits > 0) {
        n_nodes_m1 = n_nodes;
        n_splits = 0;
        for (da_int node_idx = 0; node_idx < n_nodes_m1; node_idx++) {
            da_int min_score_obs_idx, min_score_col_idx;
            // Node<T> &node = model[node_idx];
            da_int nn = model[node_idx].n_obs;
            da_int ii = model[node_idx].start_idx;

            bool level_flag = (max_level == -1) || (model[node_idx].level < max_level);
            if (model[node_idx].is_leaf && level_flag) {
                min_score =
                    no_split_score<T>(&y[ii], nn, model[node_idx].y_pred, score_fun);
                DA_PRINTF_DEBUG("min_score = %8.4f \n", min_score);
                score = min_score;
                split_node = 0;

                sample_feature_ind(n_features_to_select, feature_ind.data());

                for (da_int i = 0; i < n_features_to_select; i++) {
                    da_int col_idx = feature_ind[i];
                    da_int split_idx;

                    sort_1d_array(&y[ii], nn, &x[ii * d], d, col_idx);
                    sort_2d_array_by_col(&x[ii * d], nn, d, col_idx);

                    // split<T>(&y[ii], nn, split_idx, score, score_fun);
                    split<T>(&x[ii * d], d, col_idx, diff_thres, &y[ii], nn, split_idx,
                             score, score_fun);

                    if (score < min_score) {
                        min_score = score;
                        min_score_obs_idx = split_idx;
                        min_score_col_idx = col_idx;
                        split_node = 1;
                        n_splits += 1;
                    }
                }
                if (split_node) {
                    // add new leaves and populate with predicted values
                    Node<T> leaf1, leaf2;
                    model[node_idx].is_leaf = false;
                    leaf1.is_leaf = true;
                    leaf2.is_leaf = true;
                    leaf1.start_idx = model[node_idx].start_idx;
                    leaf1.n_obs = min_score_obs_idx;
                    leaf2.start_idx = model[node_idx].start_idx + min_score_obs_idx;
                    leaf2.n_obs = model[node_idx].n_obs - min_score_obs_idx;
                    leaf1.level = model[node_idx].level + 1;
                    leaf2.level = model[node_idx].level + 1;

                    sort_1d_array(&y[ii], nn, &x[ii * d], d, min_score_col_idx);
                    sort_2d_array_by_col(&x[ii * d], nn, d, min_score_col_idx);
#ifdef DA_LOGGING
                    T score_leaf1 = no_split_score<T>(&y[leaf1.start_idx], leaf1.n_obs,
                                                      leaf1.y_pred, score_fun);
                    T score_leaf2 = no_split_score<T>(&y[leaf2.start_idx], leaf2.n_obs,
                                                      leaf2.y_pred, score_fun);
#else
                    no_split_score<T>(&y[leaf1.start_idx], leaf1.n_obs, leaf1.y_pred,
                                      score_fun);
                    no_split_score<T>(&y[leaf2.start_idx], leaf2.n_obs, leaf2.y_pred,
                                      score_fun);
#endif

                    DA_PRINTF_DEBUG("ii = %" DA_INT_FMT ", nn = %" DA_INT_FMT " \n", ii,
                                    nn);
                    DA_PRINTF_DEBUG("min_score_obs_idx = %" DA_INT_FMT
                                    ", min_score_col_idx = %" DA_INT_FMT " \n",
                                    min_score_obs_idx, min_score_col_idx);
                    DA_PRINTF_DEBUG("leaf1.start_idx = %" DA_INT_FMT
                                    ", leaf1.n_obs = %" DA_INT_FMT
                                    ", leaf1.y_pred = %d \n",
                                    leaf1.start_idx, leaf1.n_obs, (int)leaf1.y_pred);
                    DA_PRINTF_DEBUG("leaf2.start_idx = %" DA_INT_FMT
                                    ", leaf2.n_obs = %" DA_INT_FMT
                                    ", leaf2.y_pred = %d \n",
                                    leaf2.start_idx, leaf2.n_obs, (int)leaf2.y_pred);
                    DA_PRINTF_DEBUG("min_score = %8.4f \n", min_score);
                    DA_PRINTF_DEBUG("score_leaf1 + score_leaf2 = %8.4f \n",
                                    score_leaf1 + score_leaf2);

                    model.push_back(leaf1);
                    model.push_back(leaf2);

                    // update global information about tree /fit
                    n_nodes += 2;

                    // save data required for predict method (column index, threshold)
                    T leaf1_xmax =
                        x[(leaf1.start_idx + leaf1.n_obs - 1) * d + min_score_col_idx];
                    T leaf2_xmin = x[leaf2.start_idx * d + min_score_col_idx];
                    T x_threshold = (leaf1_xmax + leaf2_xmin) / 2;

                    model[node_idx].child_node_l_idx = n_nodes - 2;
                    model[node_idx].child_node_r_idx = n_nodes - 1;
                    model[node_idx].col_idx = min_score_col_idx;
                    model[node_idx].x_threshold = x_threshold;
                }
            }
            DA_PRINTF_DEBUG("---------------------------------\n");
        }
    }
    this->model_trained = true;
    return status;
}

template <typename T>
da_status decision_tree<T>::predict(da_int n_obs, da_int n_features, T *x_in, da_int ldx,
                                    uint8_t *y_pred) {
    DA_PRINTF_DEBUG("Inside decision_tree<T>::predict \n");

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet");
    }

    if (x_in == nullptr || y_pred == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either x, or y are not valid pointers.");

    if (n_obs <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "n_obs = " + std::to_string(n_obs) +
                            ", the value of n_obs needs to be greater than 0");
    }

    if (n_features != this->d) {
        return da_error(this->err, da_status_invalid_input,
                        "n_features = " + std::to_string(n_features) +
                            ", expected n_features = " + std::to_string(this->d));
    }

    if (ldx < n_obs) {
        DA_PRINTF_DEBUG("ldx = %" DA_INT_FMT ", n_obs = %" DA_INT_FMT " \n", ldx, n_obs);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) + ", ldx = " + std::to_string(ldx) +
                ", the value of ldx needs to be at least as big as the value of n_obs");
    }

    // copy user data (convert order from column-major to row-major)
    std::vector<T> x(n_obs * this->d);
    for (da_int j = 0; j < this->d; j++) {
        for (da_int i = 0; i < n_obs; i++) {
            // ldx >= n_obs
            x[j + (i * this->d)] = x_in[i + (j * ldx)];
        }
    }

    // print parameters in fitted model
    for (da_int node_idx = 0; node_idx < (da_int)model.size(); node_idx++) {
        if (!model[node_idx].is_leaf) {
            DA_PRINTF_DEBUG("model[node_idx].col_idx = %" DA_INT_FMT " \n",
                            model[node_idx].col_idx);
            DA_PRINTF_DEBUG("model[node_idx].x_threshold = %6.2f \n",
                            model[node_idx].x_threshold);
        }
    }

    for (da_int i = 0; i < n_obs; i++) {
        T *xi = x.data() + (i * d);

        Node<T> node = model[0];
        while (node.is_leaf == false) {
            da_int child_idx = (xi[node.col_idx] > node.x_threshold)
                                   ? node.child_node_r_idx
                                   : node.child_node_l_idx;
            node = model[child_idx];
        }
        y_pred[i] = node.y_pred;
        if (i % 20 == 0) {
            DA_PRINTF_DEBUG("y_pred[i] = %hhu \n", y_pred[i]);
        }
    }

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::score(da_int n_obs, da_int n_features, T *x_in, da_int ldx,
                                  uint8_t *y_test, T *p_score) {
    DA_PRINTF_DEBUG("Inside decision_tree<T>::score \n");
    DA_PRINTF_DEBUG("model[0].col_idx = %" DA_INT_FMT " \n", model[0].col_idx);
    DA_PRINTF_DEBUG("model[0].x_threshold = %8.4f \n", model[0].x_threshold);

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet");
    }

    if (x_in == nullptr || y_test == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either x, or y are not valid pointers.");

    if (n_obs <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "n_obs = " + std::to_string(n_obs) +
                            ", the value of n_obs needs to be greater than 0");
    }

    if (n_features != this->d) {
        return da_error(this->err, da_status_invalid_input,
                        "n_features = " + std::to_string(n_features) +
                            ", expected n_features = " + std::to_string(this->d));
    }

    if (ldx < n_obs) {
        DA_PRINTF_DEBUG("ldx = %" DA_INT_FMT ", n_obs = %" DA_INT_FMT " \n", ldx, n_obs);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) + ", ldx = " + std::to_string(ldx) +
                ", the value of ldx needs to be at least as big as the value of n_obs");
    }

    // copy user data (convert order from column-major to row-major)
    try {
        std::vector<T> x(n_obs * this->d);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    for (da_int j = 0; j < this->d; j++) {
        for (da_int i = 0; i < n_obs; i++) {
            // ldx >= n_obs
            x[j + (i * this->d)] = x_in[i + (j * ldx)];
        }
    }

    uint8_t y_pred;
    T score = 0.0;

    for (da_int i = 0; i < n_obs; i++) {
        T *xi = x.data() + (i * d);

        Node<T> node = model[0];
        da_int child_idx;
        while (node.is_leaf == false) {
            child_idx = (xi[node.col_idx] > node.x_threshold) ? node.child_node_r_idx
                                                              : node.child_node_l_idx;
            node = model[child_idx];
        }
        y_pred = node.y_pred;
        score += (y_test[i] == y_pred) ? 1 : 0;
        if (i % 20 == 0) {
            DA_PRINTF_DEBUG("y_pred = %hhu \n", y_pred);
        }
    }
    score = score / n_obs;
    DA_PRINTF_DEBUG("score = %10.4f \n", score);
    (*p_score) = score;

    return da_status_success;
}

// ---------------------------------------------
// member functions for decision_forest class
// ---------------------------------------------

template <typename T>
da_status decision_forest<T>::get_result(da_result query, da_int *dim, T *result) {
    da_status status = da_status_success;
    DA_PRINTF_DEBUG("Inside get_result \n");

    da_int rinfo_size = 3;

    if (result == nullptr || *dim <= 0) {
        return da_warn(err, da_status_invalid_array_dimension,
                       "The results array has not been allocated, or an unsuitable "
                       "dimension has been provided.");
    }

    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)seed_val;
        result[1] = (T)n_obs;
        result[2] = (T)d;
        break;
    default:
        return da_warn(err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return status;
}

template <typename T>
da_status decision_forest<T>::set_training_data(da_int n_obs, da_int n_features, T *x,
                                                da_int ldx, uint8_t *y) {

    da_status status = da_status_success;

    if (x == nullptr || y == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either x, or y are not valid pointers.");

    if (n_obs <= 0 || n_features <= 0) {
        DA_PRINTF_DEBUG("n_obs = %" DA_INT_FMT ", n_features = %" DA_INT_FMT " \n", n_obs,
                        n_features);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) +
                ", n_features = " + std::to_string(n_features) +
                ", the values of n_obs and n_features need to be greater than 0");
    }

    if (ldx < n_obs) {
        DA_PRINTF_DEBUG("ldx = %" DA_INT_FMT ", n_obs = %" DA_INT_FMT " \n", ldx, n_obs);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) + ", ldx = " + std::to_string(ldx) +
                ", the value of ldx needs to be at least as big as the value of n_obs");
    }

    this->n_obs = n_obs;
    this->d = n_features;

    // allocate memory foruser data
    try {
        this->x.resize(n_obs * n_features);
        this->y.resize(n_obs);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // copy user data
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_obs; i++) {
            this->x[i + (j * n_obs)] = x[i + (j * ldx)];
        }
    }

    for (da_int i = 0; i < n_obs; i++) {
        this->y[i] = y[i];
    }

    // record that data is initialized but fitting is not yet done
    set_data_done = true;
    model_trained = false;

    return status;
}

template <typename T>
da_status decision_tree<T>::sample_feature_ind(da_int n_samples, da_int *samples) {
    DA_PRINTF_DEBUG("Inside decision_forest<T>::sample_features \n");
#ifdef DA_LOGGING
    opts.print_options();
#endif
    DA_PRINTF_DEBUG("\n");
    da_status status = da_status_success;

    da_int N = this->d; // n_features (set inside set_training_data)

    double top = N - n_samples;
    double Nreal = N;
    da_int idx0 = -1;
    std::vector<da_int> subsample(n_samples);

    da_int i = 0;
    da_int n = n_samples;
    double v, quot;
    da_int S;

    auto uniform_real_dist = std::uniform_real_distribution<double>(0.0, 1.0);

    while (n >= 2) {
        // sample v using Mersenne Twiser
        v = uniform_real_dist(mt_gen);
        // for debugging it can be useful to see what happens when v is fixed rather than pseudo-random
        // v = 0.1;
        S = 0;
        quot = top / Nreal;
        while (quot > v) {
            S += 1;
            top -= 1.0;
            Nreal -= 1.0; // update number of records remaining in dataset
            quot = (quot * top) / Nreal;
        };

        // if S=0 we take the following record from the previous iteration
        // if S>0 we skip S records
        subsample[i] = idx0 + S + 1;
        DA_PRINTF_DEBUG("idx0+S+1 = %d \n", (int)(idx0 + S + 1));

        // if S=0 we need to increment record index by 1   ahead of next iteration
        // if S>0 we need to increment record index by S+1 ahead of next iteration
        idx0 += S + 1;
        Nreal -= 1.0; // update number of records remaining, i.e., not yet sampled
        n -= 1;       // update number of records that still need to be selected
        i += 1;       // update index of subsample vector for next iteration
    }

    v = uniform_real_dist(mt_gen);
    S = (da_int)std::floor(std::nearbyint(Nreal) * v);
    subsample[i] = idx0 + S + 1;
    DA_PRINTF_DEBUG("idx0+S+1 = %d \n", (int)(idx0 + S + 1));

    DA_PRINTF_DEBUG("\n");
    DA_PRINTF_DEBUG("Sequential sample from unshuffled vector: \n");
    for (da_int i = 0; i < n_samples; i++) {
        DA_PRINTF_DEBUG("%2d, ", (int)subsample[i]);
    }
    DA_PRINTF_DEBUG("\n\n");

    for (da_int i = 0; i < n_samples; i++) {
        samples[i] = subsample[i];
    }

    // Shuffle algorithm
    std::uniform_int_distribution<da_int> uniform_dist(0, n_samples - 1);

    for (da_int i = 0; i < (n_samples - 2); i++) {
        uniform_dist = std::uniform_int_distribution<da_int>(i, n_samples - 1);
        da_int j = uniform_dist(mt_gen);
        DA_PRINTF_DEBUG("Randomly-chosen uniform int: %2d \n", (int)j);
        std::swap(samples[i], samples[j]);
    }

    DA_PRINTF_DEBUG("\n");
    DA_PRINTF_DEBUG("Shuffled sequential sample: \n");
    for (da_int i = 0; i < n_samples; i++) {
        DA_PRINTF_DEBUG("%2d, ", (int)samples[i]);
    }

    DA_PRINTF_DEBUG("\n");
    return status;
}

template <typename T>
da_status decision_forest<T>::sample_obs_ind(da_int n_obs, da_int n_samples,
                                             da_int *obs_ind) {
    DA_PRINTF_DEBUG("Inside decision_forest<T>::bootstrap_obs \n");
    da_status status = da_status_success;

    // sample n_samples time with replacement
    // from uniform integer distribution on range 0, (n_obs - 1)

    std::uniform_int_distribution<da_int> uniform_dist(0, n_obs - 1);

    for (da_int i = 0; i < n_samples; i++) {
        obs_ind[i] = uniform_dist(mt_gen);
    }

    return status;
}

template <typename T> da_status decision_forest<T>::fit_tree(decision_tree<T> *p_tree) {
    da_status status = da_status_success;

    da_int n_obs_per_tree, n_features_to_select;
    opts.get("n_obs_per_tree", n_obs_per_tree);
    opts.get("n_features_to_select", n_features_to_select);

    std::vector<da_int> obs_ind(n_obs_per_tree);
    std::vector<da_int> feature_ind(n_features_to_select);

    sample_obs_ind(n_obs, n_obs_per_tree, obs_ind.data());
    // sample_feature_ind(n_features_to_select, feature_ind.data());

    std::vector<T> xr(n_obs_per_tree * d);
    std::vector<uint8_t> yr(n_obs_per_tree);

    for (da_int i = 0; i < n_obs_per_tree; i++) {
        da_int ii = obs_ind[i];
        for (da_int j = 0; j < d; j++) {
            xr[i + j * (n_obs_per_tree)] = x[ii + j * (n_obs)];
        }
        yr[i] = y[ii];
    }

    // decision_tree<T> tree(*(this->err));
    // da_options::OptionRegistry & opts = this->opts;
    // da_options::OptionRegistry *opts = new da_options::OptionRegistry;
    // tree.opts = this->opts;

    // copy options from forest to tree
    p_tree->opts = this->opts;
    p_tree->set_rng(this->mt_gen);

    // copy data from forest into tree
    status = p_tree->set_training_data(n_obs_per_tree, d, xr.data(), n_obs_per_tree,
                                       yr.data());
    status = p_tree->fit();

    this->mt_gen = p_tree->get_rng();

    return status;
}

template <typename T> da_status decision_forest<T>::fit() {
    da_status status = da_status_success;

    // check that set_training_data has been called
    if (set_data_done == false)
        return da_error(err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_df_set_training_data_s or da_df_set_training_data_d.");

    da_int n_trees;
    opts.get("n_trees", n_trees);
    opts.get("seed", this->seed_val);

    // set mt_gen (internal class data)
    if (this->seed_val == -1) {
        std::random_device r;
        this->seed_val = std::abs((da_int)r());
        mt_gen.seed(this->seed_val);
    } else {
        mt_gen.seed(this->seed_val);
    }

    // allocate memory for tree objects in tree_vec using
    // decision_tree(da_errors::da_error_t &err) constructor
    try {
        tree_vec.resize(n_trees, decision_tree<T>(*(this->err)));
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    for (da_int k = 0; k < n_trees; k++) {
        status = fit_tree(&tree_vec[k]);
    }

    this->model_trained = true;
    return status;
}

template <typename T>
da_status decision_forest<T>::predict(da_int n_obs, da_int n_features, T *x, da_int ldx,
                                      uint8_t *y_pred) {
    da_status status = da_status_success;

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet");
    }

    if (x == nullptr || y_pred == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either x, or y are not valid pointers.");

    if (n_obs <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "n_obs = " + std::to_string(n_obs) +
                            ", the value of n_obs needs to be greater than 0");
    }

    if (n_features != this->d) {
        return da_error(this->err, da_status_invalid_input,
                        "n_features = " + std::to_string(n_features) +
                            ", expected n_features = " + std::to_string(this->d));
    }

    if (ldx < n_obs) {
        DA_PRINTF_DEBUG("ldx = %" DA_INT_FMT ", n_obs = %" DA_INT_FMT " \n", ldx, n_obs);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) + ", ldx = " + std::to_string(ldx) +
                ", the value of ldx needs to be at least as big as the value of n_obs");
    }

    da_int n_trees;
    opts.get("n_trees", n_trees);

    // predict class using majority vote over trees in ensemble
    std::vector<da_int> count(n_obs);
    for (da_int k = 0; k < n_trees; k++) {
        std::vector<uint8_t> y_pred_tree(n_obs);
        status = tree_vec[k].predict(n_obs, n_features, x, ldx, y_pred_tree.data());
        for (da_int i = 0; i < n_obs; i++) {
            count[i] += y_pred_tree[i];
        }
    }

    for (da_int i = 0; i < n_obs; i++) {
        if (2 * count[i] >= n_trees) {
            y_pred[i] = 1;
        } else {
            y_pred[i] = 0;
        }
    }

    return status;
}

template <typename T>
da_status decision_forest<T>::score(da_int n_obs, da_int n_features, T *x, da_int ldx,
                                    uint8_t *y_test, T *p_score) {
    da_status status = da_status_success;

    da_int n_trees;
    opts.get("n_trees", n_trees);

    DA_PRINTF_DEBUG("Inside decision_forest<T>::score \n");
    DA_PRINTF_DEBUG("model[0].col_idx = %" DA_INT_FMT " \n", model[0].col_idx);
    DA_PRINTF_DEBUG("model[0].x_threshold = %8.4f \n", model[0].x_threshold);

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet");
    }

    if (x == nullptr || y_test == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either x, or y are not valid pointers.");

    if (n_obs <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "n_obs = " + std::to_string(n_obs) +
                            ", the value of n_obs needs to be greater than 0");
    }

    if (n_features != this->d) {
        return da_error(this->err, da_status_invalid_input,
                        "n_features = " + std::to_string(n_features) +
                            ", expected n_features = " + std::to_string(this->d));
    }

    if (ldx < n_obs) {
        DA_PRINTF_DEBUG("ldx = %" DA_INT_FMT ", n_obs = %" DA_INT_FMT " \n", ldx, n_obs);
        return da_error(
            this->err, da_status_invalid_input,
            "n_obs = " + std::to_string(n_obs) + ", ldx = " + std::to_string(ldx) +
                ", the value of ldx needs to be at least as big as the value of n_obs");
    }

    // evaluate prediction counts across trees
    std::vector<da_int> count(n_obs);
    for (da_int k = 0; k < n_trees; k++) {
        std::vector<uint8_t> y_pred_tree(n_obs);
        status = tree_vec[k].predict(n_obs, n_features, x, ldx, y_pred_tree.data());
        for (da_int i = 0; i < n_obs; i++) {
            count[i] += y_pred_tree[i];
        }
    }

    // predict using majority vote, compare with y_test, and update score
    uint8_t y_pred;
    T score = 0.0;
    for (da_int i = 0; i < n_obs; i++) {
        if (2 * count[i] > n_trees) {
            y_pred = 1;
        } else {
            y_pred = 0;
        }
        score += (y_test[i] == y_pred) ? 1 : 0;
    }

    score = score / n_obs;
    DA_PRINTF_DEBUG("score = %10.4f \n", score);
    (*p_score) = score;

    return status;
}

} // namespace da_df

#endif // DF_HPP
