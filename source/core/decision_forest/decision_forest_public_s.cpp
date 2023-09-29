#include "aoclda.h"
#include "da_handle.hpp"
#include "decision_forest.hpp"

da_status da_df_tree_set_training_data_s(da_handle handle, da_int n_obs,
                                    float *x, uint8_t *y) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->dt_s == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_s->set_training_data(n_obs, x, y);
}

da_status da_df_tree_fit_s(da_handle handle) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->dt_s == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_s->fit();
}

da_status da_df_tree_predict_s(da_handle handle, da_int n_obs,  float *x,
                          uint8_t *y_pred) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->dt_s == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_s->predict(n_obs, x, y_pred);
}

da_status da_df_tree_score_s(da_handle handle, da_int n_obs, float *x,
                        uint8_t *y_test, float *score) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->dt_s == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_s->score(n_obs, x, y_test, score);
}

// da_status da_df_sample_feature_ind_s(da_handle handle, da_int n) {
//     if (!handle)
//         return da_status_invalid_input;
//     if (handle->precision != da_single)
//         return da_status_wrong_type;
//     if (handle->df_s == nullptr)
//         return da_status_invalid_pointer;

//     return handle->df_s->sample_feature_ind(n);
// }

// da_status da_df_bootstrap_obs_s(da_handle handle, da_int n_trees, da_int d) {
//     if (!handle)
//         return da_status_invalid_input;
//     if (handle->precision != da_single)
//         return da_status_wrong_type;
//     if (handle->df_s == nullptr)
//         return da_status_invalid_pointer;

//     return handle->df_s->bootstrap_obs(n_trees, d);
// }

da_status da_df_set_training_data_s(da_handle handle, da_int n_obs,
                                    float *x, uint8_t *y) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->df_s == nullptr)
        return da_status_invalid_pointer;

    return handle->df_s->set_training_data(n_obs, x, y);
}

da_status da_df_fit_s(da_handle handle) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->df_s == nullptr)
        return da_status_invalid_pointer;

    return handle->df_s->fit();
}
