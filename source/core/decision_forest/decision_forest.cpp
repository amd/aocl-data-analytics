#include "decision_forest.hpp"

// explicit instantiations
template class decision_tree<double>;
template class decision_tree<float>;

#include <vector>
#include <algorithm>
#include <cmath>

#include "decision_forest_aux.hpp"

template <typename T>
da_status decision_tree<T>::fit_tree()
{
    DA_PRINTF_DEBUG("Inside decision_tree<T>::fit_tree \n");

    da_int scoring_fun_id;
    std::string scoring_fun_str;
    opts.get("scoring function", scoring_fun_str, scoring_fun_id);

    if (scoring_fun_str == "gini"){
        // Gini scoring function
        score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
            T score = 0.0;
            if (acc_l > 0){
                T phat_l = acc_l / n_l;
                score += n_l * 2 * phat_l * (1 - phat_l);
            }
            if (acc_r > 0){
                T phat_r = acc_r / n_r;
                score += n_r * 2 * phat_r * (1 - phat_r);
            }
            return score;
        };
    }else if (scoring_fun_str == "cross-entropy"){
        // Cross-entropy or deviance
        score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
            T phat_l = acc_l / n_l;
            T phat_r = acc_r / n_r;

            T score = 0.0;
            if (acc_l > 0){
                score -= n_l * phat_l * log(phat_l);
            }
            if( (da_int)acc_l != n_l){
                score -= n_l * (1 - phat_l) * log(1 - phat_l);
            }
            if (acc_r > 0){
                score -= n_r * phat_r * log(phat_r);
            }
            if( (da_int)acc_r != n_r){
                score -= n_r * (1 - phat_r) * log(1 - phat_r);
            }
            return score;
        };
    }else if (scoring_fun_str == "misclassification-error"){
        // Misclassification error
        score_fun = [](T acc_l, da_int n_l, T acc_r, da_int n_r) {
            da_int acc_l_argmax = (acc_l > (n_l - acc_l)) ? acc_l : (n_l - acc_l);
            da_int acc_r_argmax = (acc_r > (n_r - acc_r)) ? acc_r : (n_r - acc_r);

            T score = 0.0;
            if (acc_l_argmax > 0){
                T phat_l_argmax = (T)acc_l_argmax / n_l;
                score += (1 - phat_l_argmax);
            }

            if (acc_r_argmax > 0){
                T phat_r_argmax = (T)acc_r_argmax / n_r;
                score += (1 - phat_r_argmax);
            }
            return score;
        };
    }

    for (da_int i = 0; i < 5; i++)
    {
        DA_PRINTF_DEBUG("%10.4f %10.4f %10.4f %10.4f \n",
            x[i*n_features], x[i*n_features + 1], x[i*n_features + 2], x[i*n_features + 3]);
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

    while (n_splits > 0)
    {
        n_nodes_m1 = n_nodes;
        n_splits = 0;
        for (da_int node_idx=0; node_idx < n_nodes_m1; node_idx++)
        {
            da_int min_score_obs_idx, min_score_col_idx;
            // Node<T> &node = model[node_idx];
            da_int nn = model[node_idx].n_obs;
            da_int ii = model[node_idx].start_idx;

            bool level_flag = (max_level == -1) || (model[node_idx].level < max_level);
            if (model[node_idx].is_leaf && level_flag)
            {
                min_score = no_split_score<T>(&y[ii], nn, model[node_idx].y_pred, score_fun);
                DA_PRINTF_DEBUG("min_score = %8.4f \n", min_score);
                score = min_score;
                split_node = 0;

                for (da_int col_idx=0; col_idx < n_features; col_idx++)
                {
                    da_int split_idx;

                    sort_1d_array(&y[ii], nn, &x[ii*n_features], n_features, col_idx);
                    sort_2d_array_by_col(&x[ii*n_features], nn, n_features, col_idx);

                    split<T>(&y[ii], nn, split_idx, score, score_fun);

                    if (score < min_score){
                        min_score = score;
                        min_score_obs_idx = split_idx;
                        min_score_col_idx = col_idx;
                        split_node = 1;
                        n_splits += 1;
                    }
                }
                if (split_node)
                {
                    // add new leaves and populate with predicted values
                    Node<T> leaf1, leaf2;
                    model[node_idx].is_leaf = false; leaf1.is_leaf = true; leaf2.is_leaf = true;
                    leaf1.start_idx = model[node_idx].start_idx; leaf1.n_obs = min_score_obs_idx;
                    leaf2.start_idx = model[node_idx].start_idx + min_score_obs_idx;
                    leaf2.n_obs = model[node_idx].n_obs - min_score_obs_idx;
                    leaf1.level = model[node_idx].level + 1; leaf2.level = model[node_idx].level + 1;

                    sort_1d_array(&y[ii], nn, &x[ii*n_features], n_features, min_score_col_idx);
                    sort_2d_array_by_col(&x[ii*n_features], nn,  n_features, min_score_col_idx);
#ifdef DA_LOGGING
                    T score_leaf1 = no_split_score<T>(&y[leaf1.start_idx], leaf1.n_obs, leaf1.y_pred, score_fun);
                    T score_leaf2 = no_split_score<T>(&y[leaf2.start_idx], leaf2.n_obs, leaf2.y_pred, score_fun);
#else
                    no_split_score<T>(&y[leaf1.start_idx], leaf1.n_obs, leaf1.y_pred, score_fun);
                    no_split_score<T>(&y[leaf2.start_idx], leaf2.n_obs, leaf2.y_pred, score_fun);
#endif

                    DA_PRINTF_DEBUG("ii = %" DA_INT_FMT ", nn = %" DA_INT_FMT " \n", ii, nn);
                    DA_PRINTF_DEBUG("min_score_obs_idx = %" DA_INT_FMT ", min_score_col_idx = %" DA_INT_FMT " \n", min_score_obs_idx, min_score_col_idx);
                    DA_PRINTF_DEBUG("leaf1.start_idx = %" DA_INT_FMT ", leaf1.n_obs = %" DA_INT_FMT ", leaf1.y_pred = %d \n",
                        leaf1.start_idx, leaf1.n_obs, leaf1.y_pred);
                    DA_PRINTF_DEBUG("leaf2.start_idx = %" DA_INT_FMT ", leaf2.n_obs = %" DA_INT_FMT ", leaf2.y_pred = %d \n",
                        leaf2.start_idx, leaf2.n_obs, leaf2.y_pred);
                    DA_PRINTF_DEBUG("min_score = %8.4f \n", min_score);
                    DA_PRINTF_DEBUG("score_leaf1 + score_leaf2 = %8.4f \n", score_leaf1 + score_leaf2);

                    model.push_back(leaf1);
                    model.push_back(leaf2);

                    // update global information about tree /fit
                    n_nodes += 2;

                    // save data required for predict method (column index, threshold)
                    T leaf1_xmax = x[(leaf1.start_idx + leaf1.n_obs -1) * n_features + min_score_col_idx];
                    T leaf2_xmin = x[ leaf2.start_idx *                   n_features + min_score_col_idx];
                    T x_threshold = (leaf1_xmax + leaf2_xmin) / 2;

                    model[node_idx].child_node_l_idx = n_nodes - 2; model[node_idx].child_node_r_idx = n_nodes - 1;
                    model[node_idx].col_idx = min_score_col_idx;
                    model[node_idx].x_threshold = x_threshold;
                }
            }
            DA_PRINTF_DEBUG("---------------------------------\n");
        }
    }

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict_tree(da_int n_obs, da_int n_features, T *x, uint8_t *y_pred)
{
    DA_PRINTF_DEBUG("Inside decision_tree<T>::predict_tree \n");
    for (da_int node_idx=0; node_idx < (da_int)model.size(); node_idx++)
    {
        if (!model[node_idx].is_leaf)
        {
            DA_PRINTF_DEBUG("model[node_idx].col_idx = %" DA_INT_FMT " \n", model[node_idx].col_idx);
            DA_PRINTF_DEBUG("model[node_idx].x_threshold = %6.2f \n", model[node_idx].x_threshold);
        }
    }

    for (da_int i = 0; i < n_obs; i++)
    {
        T *xi = x + (i*n_features);

        Node<T> node = model[0];
        while (node.is_leaf == false)
        {
            da_int child_idx = (xi[node.col_idx] > node.x_threshold) ?
                node.child_node_r_idx : node.child_node_l_idx;
            node = model[child_idx];
        }
        y_pred[i] = node.y_pred;
        if (i % 20 == 0)
        {
            DA_PRINTF_DEBUG("y_pred[i] = %hhu \n", y_pred[i]);
        }
    }

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::score(da_int n_obs, da_int n_features, T *x, uint8_t *y_test,
                                           T &score)
{
    DA_PRINTF_DEBUG("Inside decision_tree<T>::score \n");
    DA_PRINTF_DEBUG("model[0].col_idx = %" DA_INT_FMT " \n", model[0].col_idx);
    DA_PRINTF_DEBUG("model[0].x_threshold = %8.4f \n", model[0].x_threshold);

    uint8_t y_pred;
    score = 0;

    for (da_int i = 0; i < n_obs; i++)
    {
        T *xi = x + (i*n_features);

        Node<T> node = model[0];
        da_int child_idx;
        while (node.is_leaf == false)
        {
            child_idx = (xi[node.col_idx] > node.x_threshold) ?
                node.child_node_r_idx : node.child_node_l_idx;
            node = model[child_idx];
        }
        y_pred = node.y_pred;
        score += y_test[i] == y_pred ? 1 : 0;

        if (i % 20 == 0)
        {
            DA_PRINTF_DEBUG("y_pred = %hhu \n", y_pred);
        }
    }
    score = score / n_obs;
    DA_PRINTF_DEBUG("score = %10.4f \n", score);

    return da_status_success;
}
