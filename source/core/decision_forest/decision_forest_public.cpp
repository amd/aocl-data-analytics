#include "aoclda.h"
#include "da_handle.hpp"
#include "decision_forest.hpp"

da_status da_df_set_max_level_d(da_handle handle, da_int max_level)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->set_max_level(max_level);
}

da_status da_df_set_max_level_s(da_handle handle, da_int max_level)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->df_s == nullptr)
        return da_status_invalid_pointer;

    return handle->df_s->set_max_level(max_level);
}

da_status da_df_set_training_data_d(da_handle handle,
    da_int n_obs, da_int n_features, double *x, uint8_t *y)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->set_training_data(n_obs, n_features, x, y);
}

da_status da_df_fit_d(da_handle handle)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->fit();
}

da_status da_df_predict_d(da_handle handle, da_int n_obs, da_int n_features, double *x, uint8_t *y_pred)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->predict(n_obs, n_features, x, y_pred);
}

da_status da_df_score_d(da_handle handle, da_int n_obs, da_int n_features, double *x, uint8_t *y_test,
                        double &score)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_double)
        return da_status_wrong_type;
    if (handle->df_d == nullptr)
        return da_status_invalid_pointer;

    return handle->df_d->score(n_obs, n_features, x, y_test, score);
}

da_status da_df_set_training_data_s(da_handle handle,
    da_int n_obs, da_int n_features, float *x, uint8_t *y)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->df_s == nullptr)
        return da_status_invalid_pointer;

    return handle->df_s->set_training_data(n_obs, n_features, x, y);
}

da_status da_df_fit_s(da_handle handle)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->df_s == nullptr)
        return da_status_invalid_pointer;

    return handle->df_s->fit();
}

da_status da_df_predict_s(da_handle handle, da_int n_obs, da_int n_features, float *x, uint8_t *y_pred)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->df_s == nullptr)
        return da_status_invalid_pointer;

    return handle->df_s->predict(n_obs, n_features, x, y_pred);
}

da_status da_df_score_s(da_handle handle, da_int n_obs, da_int n_features, float *x, uint8_t *y_test,
                        float &score)
{
    if (!handle)
        return da_status_memory_error;
    if (handle->precision != da_single)
        return da_status_wrong_type;
    if (handle->df_s == nullptr)
        return da_status_invalid_pointer;

    return handle->df_s->score(n_obs, n_features, x, y_test, score);
}