#ifndef AOCLDA_DF
#define AOCLDA_DF

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

da_status da_df_tree_set_training_data_d(da_handle handle, da_int n_obs,
                                    double *x, uint8_t *y);

da_status da_df_tree_fit_d(da_handle handle);
da_status da_df_tree_predict_d(da_handle handle, da_int n_obs,  double *x,
                          uint8_t *y_pred);

da_status da_df_tree_score_d(da_handle handle, da_int n_obs, double *x,
                        uint8_t *y_test, double *score);

da_status da_df_tree_set_training_data_s(da_handle handle, da_int n_obs,
                                    float *x, uint8_t *y);

da_status da_df_tree_fit_s(da_handle handle);
da_status da_df_tree_predict_s(da_handle handle, da_int n_obs,  float *x,
                          uint8_t *y_pred);

da_status da_df_tree_score_s(da_handle handle, da_int n_obs, float *x,
                        uint8_t *y_test, float *score);

da_status da_df_sample_features_s(da_handle handle, da_int d);

da_status da_df_bootstrap_obs_s(da_handle handle, da_int n_trees, da_int d);

#endif
