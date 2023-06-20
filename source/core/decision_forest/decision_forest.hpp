#ifndef DF_HPP
#define DF_HPP

#include <functional>
#include <random>

#include "aoclda.h"
#include "da_error.hpp"
#include "options.hpp"

#include "decision_forest_aux.hpp"

template <class T>
inline da_status register_df_options(da_options::OptionRegistry &opts) {
    da_int max_int = std::numeric_limits<da_int>::max();
    da_status status = da_status_success;

    try {
        std::shared_ptr<da_options::OptionString> os;
        std::shared_ptr<da_options::OptionNumeric<da_int>> oi;

        os = std::make_shared<da_options::OptionString>(da_options::OptionString(
            "scoring function", "Select scoring function to use",
            {{"gini", 0}, {"cross-entropy", 1}, {"misclassification-error", 2}}, "gini"));
        status = opts.register_opt(os);

        oi = std::make_shared<da_options::OptionNumeric<da_int>>(
            da_options::OptionNumeric<da_int>("depth", "set max depth of tree",
                                              -1, da_options::lbound_t::greaterequal,
                                              max_int, da_options::ubound_t::lessequal,
                                              -1));
        status = opts.register_opt(oi);

        oi = std::make_shared<da_options::OptionNumeric<da_int>>(
            da_options::OptionNumeric<da_int>("seed", "set random seed for Mersenne Twister (64-bit) PRNG",
                                              -1, da_options::lbound_t::greaterequal,
                                              max_int, da_options::ubound_t::lessequal,
                                              -1));
        status = opts.register_opt(oi);
    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    } catch (std::invalid_argument &e) {
        std::cerr << e.what() << std::endl; // LCOV_EXCL_LINE
        return da_status_internal_error;    // LCOV_EXCL_LINE
    } catch (...) {                         // LCOV_EXCL_LINE
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return status;
}

template <typename T> struct Node {
    // data required for leaf nodes during prediction
    bool is_leaf;
    uint8_t y_pred;
    // data required for leaf nodes during fitting
    da_int start_idx;
    da_int n_obs;
    // data required for root nodes during prediction
    da_int child_node_l_idx;
    da_int child_node_r_idx;
    da_int col_idx;
    T x_threshold;
    da_int level;
};

template <typename T> class decision_tree {
  private:
    /* pointer to error trace */
    da_errors::da_error_t *err = nullptr;

    /* true if the model has been successfully trained */
    bool model_trained = false;

    da_int n_obs = 0, n_features = 0;
    T *x = nullptr;
    uint8_t *y = nullptr;
    std::vector<Node<T>> model;

    std::mt19937_64 mt_gen;
    std::vector<da_int> shuff_vec;
  public:
    da_options::OptionRegistry opts;
    decision_tree()
    {
        register_df_options<T>(opts);
    }

    da_status set_training_data(da_int n_obs, da_int n_features, T *x, uint8_t *y);
    da_status fit();
    da_status predict(da_int n_obs, da_int n_features, T *x, uint8_t *y_pred);
    da_status score(da_int n_obs, da_int n_features, T *x, uint8_t *y_test, T &score);
    std::function<T(T, da_int, T, da_int)> score_fun;
};

template <typename T>
da_status decision_tree<T>::set_training_data(da_int n_obs, da_int n_features, T *x,
                                              uint8_t *y) {
    da_status status = da_status_success;
    if (n_obs <= 0 || n_features <= 0 || x == nullptr || y == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either n_obs, n_features, x, or y are not valid pointers.");

    model_trained = false;

    this->n_obs = n_obs;
    this->n_features = n_features;
    // copy user's feature pointers
    this->x = x;
    this->y = y;

    this->shuff_vec.resize(n_features);

    for (da_int j = 0; j < n_features; j++)
    {
        shuff_vec[j] = j;
    }

    return status;
}

template <typename T> da_status decision_tree<T>::fit() {
    DA_PRINTF_DEBUG("Inside decision_tree<T>::fit \n");
#ifdef DA_LOGGING
    opts.print_options();
#endif

    da_status status = da_status_success;

    da_int seed_val;
    opts.get("seed", seed_val);

    if (seed_val == -1)
    {
        std::random_device r;
        mt_gen.seed(r());
    }else{
        mt_gen.seed(seed_val);
    }

    da_int scoring_fun_id;
    std::string scoring_fun_str;
    opts.get("scoring function", scoring_fun_str, scoring_fun_id);

    da_int max_level;
    opts.get("depth", max_level);

    if (scoring_fun_str == "gini") {
        // Gini scoring function
        score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
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
    } else if (scoring_fun_str == "cross-entropy") {
        // Cross-entropy or deviance
        score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
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
    } else if (scoring_fun_str == "misclassification-error") {
        // Misclassification error
        score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
            da_int acc_l_argmax = (acc_l > (n_l - acc_l)) ? acc_l : (n_l - acc_l);
            da_int acc_r_argmax = (acc_r > (n_r - acc_r)) ? acc_r : (n_r - acc_r);

            T score = 0.0;
            if (acc_l_argmax > 0) {
                T phat_l_argmax = (T)acc_l_argmax / n_l;
                score += (1 - phat_l_argmax);
            }

            if (acc_r_argmax > 0) {
                T phat_r_argmax = (T)acc_r_argmax / n_r;
                score += (1 - phat_r_argmax);
            }
            return score;
        };
    }

    for (da_int i = 0; i < 5; i++) {
        DA_PRINTF_DEBUG("%10.4f %10.4f %10.4f %10.4f \n", x[i * n_features],
                        x[i * n_features + 1], x[i * n_features + 2],
                        x[i * n_features + 3]);
    }

    da_int n_nodes = 1;
    model.resize(n_nodes);
    model[0].is_leaf = true;
    model[0].start_idx = 0;
    model[0].n_obs = n_obs;
    model[0].level = 0;

    T score0 = no_split_score<T>(y, n_obs, model[0].y_pred, score_fun);
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

                for (int i=0; i < (n_features-2); i++)
                {
                    std::uniform_int_distribution<int> uniform_dist(i, n_features-1);
                    int j = uniform_dist(mt_gen);
                    DA_PRINTF_DEBUG("Randomly-chosen uniform int: " DA_INT_FMT " \n", j);
                    std::swap(shuff_vec[i], shuff_vec[j]);
                }

                for (da_int i=0; i < n_features; i++)
                {
                    da_int col_idx = shuff_vec[i];
                    da_int split_idx;

                    sort_1d_array(&y[ii], nn, &x[ii * n_features], n_features, col_idx);
                    sort_2d_array_by_col(&x[ii * n_features], nn, n_features, col_idx);

                    split<T>(&y[ii], nn, split_idx, score, score_fun);

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

                    sort_1d_array(&y[ii], nn, &x[ii * n_features], n_features,
                                  min_score_col_idx);
                    sort_2d_array_by_col(&x[ii * n_features], nn, n_features,
                                         min_score_col_idx);
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
                                    leaf1.start_idx, leaf1.n_obs, leaf1.y_pred);
                    DA_PRINTF_DEBUG("leaf2.start_idx = %" DA_INT_FMT
                                    ", leaf2.n_obs = %" DA_INT_FMT
                                    ", leaf2.y_pred = %d \n",
                                    leaf2.start_idx, leaf2.n_obs, leaf2.y_pred);
                    DA_PRINTF_DEBUG("min_score = %8.4f \n", min_score);
                    DA_PRINTF_DEBUG("score_leaf1 + score_leaf2 = %8.4f \n",
                                    score_leaf1 + score_leaf2);

                    model.push_back(leaf1);
                    model.push_back(leaf2);

                    // update global information about tree /fit
                    n_nodes += 2;

                    // save data required for predict method (column index, threshold)
                    T leaf1_xmax = x[(leaf1.start_idx + leaf1.n_obs - 1) * n_features +
                                     min_score_col_idx];
                    T leaf2_xmin = x[leaf2.start_idx * n_features + min_score_col_idx];
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

    return status;
}

template <typename T>
da_status decision_tree<T>::predict(da_int n_obs, da_int n_features, T *x,
                                    uint8_t *y_pred) {
    DA_PRINTF_DEBUG("Inside decision_tree<T>::predict \n");
    for (da_int node_idx = 0; node_idx < (da_int)model.size(); node_idx++) {
        if (!model[node_idx].is_leaf) {
            DA_PRINTF_DEBUG("model[node_idx].col_idx = %" DA_INT_FMT " \n",
                            model[node_idx].col_idx);
            DA_PRINTF_DEBUG("model[node_idx].x_threshold = %6.2f \n",
                            model[node_idx].x_threshold);
        }
    }

    for (da_int i = 0; i < n_obs; i++) {
        T *xi = x + (i * n_features);

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
da_status decision_tree<T>::score(da_int n_obs, da_int n_features, T *x, uint8_t *y_test,
                                  T &score) {
    DA_PRINTF_DEBUG("Inside decision_tree<T>::score \n");
    DA_PRINTF_DEBUG("model[0].col_idx = %" DA_INT_FMT " \n", model[0].col_idx);
    DA_PRINTF_DEBUG("model[0].x_threshold = %8.4f \n", model[0].x_threshold);

    uint8_t y_pred;
    score = 0;

    for (da_int i = 0; i < n_obs; i++) {
        T *xi = x + (i * n_features);

        Node<T> node = model[0];
        da_int child_idx;
        while (node.is_leaf == false) {
            child_idx = (xi[node.col_idx] > node.x_threshold) ? node.child_node_r_idx
                                                              : node.child_node_l_idx;
            node = model[child_idx];
        }
        y_pred = node.y_pred;
        score += y_test[i] == y_pred ? 1 : 0;
        if (i % 20 == 0) {
            DA_PRINTF_DEBUG("y_pred = %hhu \n", y_pred);
        }
    }
    score = score / n_obs;
    DA_PRINTF_DEBUG("score = %10.4f \n", score);

    return da_status_success;
}

#endif
