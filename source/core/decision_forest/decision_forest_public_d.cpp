#include "aoclda.h"
#include "da_handle.hpp"
#include "decision_forest.hpp"

da_status da_df_tree_set_training_data_d(da_handle handle, da_int n_obs,
                                    double *x, uint8_t *y) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->dt_d == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_d->set_training_data(n_obs, x, y);
}

da_status da_df_tree_fit_d(da_handle handle) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->dt_d == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_d->fit();
}

da_status da_df_tree_predict_d(da_handle handle, da_int n_obs, double *x,
                          uint8_t *y_pred) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->dt_d == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_d->predict(n_obs, x, y_pred);
}

da_status da_df_tree_score_d(da_handle handle, da_int n_obs, double *x,
                        uint8_t *y_test, double *score) {
    if (!handle)
        return da_status_invalid_input;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->dt_d == nullptr)
        return da_status_invalid_pointer;

    return handle->dt_d->score(n_obs, x, y_test, score);
}
