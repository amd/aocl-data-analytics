#include "aoclda.h"

// #include "../../source/core/decision_forest/decision_forest.hpp"

#include <random>

#include "gtest/gtest.h"

TEST(decision_forest, cpp_api_sample_features) {
    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    da_status status;
    status = da_handle_init_s(&df_handle, da_handle_decision_forest);

    status = da_options_set_int(df_handle, "n_features", 20);
    // status = da_options_set_int(df_handle, "seed", 1201);

    da_int n = 10;

    da_df_sample_features_s(df_handle, n);

}

TEST(decision_forest, cpp_api_generate_trees)
{
    da_int n_features          = 20;
    da_int n_features_per_tree = 5;

    da_int n_trees = 4;

    da_int n_samples;

    da_handle df_handle = nullptr;

    da_status status;
    status = da_handle_init_s(&df_handle, da_handle_decision_forest);

    da_df_bootstrap_obs_s(df_handle, n_trees, n_features_per_tree);

}
