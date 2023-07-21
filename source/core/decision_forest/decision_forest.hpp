#ifndef DF_HPP
#define DF_HPP

#include <functional>

#include "aoclda.h"
#include "da_error.hpp"
#include "options.hpp"

template <class T>
inline da_status register_df_options(da_options::OptionRegistry &opts) {

    try {
        std::shared_ptr<da_options::OptionString> os;
        os = std::make_shared<da_options::OptionString>(
            da_options::OptionString("scoring function", "Select scoring function to use",
                         {{"gini", 0},
                         {"cross-entropy", 1},
                         {"misclassification-error", 2}},
                         "gini"));
        opts.register_opt(os);
    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    } catch (...) {                    // LCOV_EXCL_LINE
        // invalid use of the constructor, shouldn't happen (invalid_argument))
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

template <typename T>
struct Node
{
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
        /* type of the model, has to de set at initialization phase */
        df_model mod = decision_forest_undefined;

        /* pointer to error trace */
        da_errors::da_error_t *err = nullptr;

        /* true if the model has been successfully trained */
        bool model_trained = false;

        da_int n_obs = 0, n_features = 0;
        T *x = nullptr;
        uint8_t *y = nullptr;
        std::vector<Node<T>> model;
        da_int max_level = -1;
    public:
        da_options::OptionRegistry opts;
        decision_tree() {
            register_df_options<T>(opts);
        }

        da_status set_training_data(da_int n_obs, da_int n_features, T *x, uint8_t *y);
        da_status fit_tree();
        da_status fit();
        da_status predict_tree(da_int n_obs, da_int n_features, T *x, uint8_t *y_pred);
        da_status predict(da_int n_obs, da_int n_features, T *x, uint8_t *y_pred);
        da_status score(da_int n_obs, da_int n_features, T *x, uint8_t *y_test, T &score);
        da_status set_max_level(da_int max_level);
        std::function<T(T,da_int,T,da_int)> score_fun;
};

template <typename T>
da_status decision_tree<T>::set_training_data(da_int n_obs, da_int n_features, T *x, uint8_t *y) {
    if (n_obs <= 0 || n_features <= 0 || x == nullptr || y == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either n, m, x, or y are not valid pointers.");

    model_trained = false;

    this->n_obs = n_obs;
    this->n_features = n_features;
    // copy user's feature pointers
    this->x = x;
    this->y = y;

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::set_max_level(da_int max_level)
{
    if (max_level <= 0)
        return da_error(this->err, da_status_invalid_input,
                        "max_level value is not valid.");
    this->max_level = max_level;

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::fit() {

    fit_tree();

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict(da_int n_obs, da_int n_features, T *x, uint8_t *y_pred) {

    predict_tree(n_obs, n_features, x, y_pred);

    return da_status_success;
}

#endif

