#ifndef AOCLDA_DF
#define AOCLDA_DF

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

typedef enum df_model_ {
    decision_forest_undefined = 0, ///< No model type set
} df_model;

da_status da_df_set_training_data_d(da_handle handle,
    da_int n_obs, da_int n_features, double *x, uint8_t *y);

da_status da_df_fit_d(da_handle handle);
da_status da_df_predict_d(da_handle handle, da_int n_obs, da_int n_features, double *x, uint8_t *y_pred);

da_status da_df_score_d(da_handle handle, da_int n_obs, da_int n_features, double *x, uint8_t *y_test,
                        double &score);

da_status da_df_set_training_data_s(da_handle handle,
    da_int n_obs, da_int n_features, float *x, uint8_t *y);

da_status da_df_fit_s(da_handle handle);
da_status da_df_predict_s(da_handle handle, da_int n_obs, da_int n_features, float *x, uint8_t *y_pred);

da_status da_df_score_s(da_handle handle, da_int n_obs, da_int n_features, float *x, uint8_t *y_test,
                        float &score);

da_status da_df_set_max_level_d(da_handle handle, da_int max_level);

da_status da_df_set_max_level_s(da_handle handle, da_int max_level);

#endif
