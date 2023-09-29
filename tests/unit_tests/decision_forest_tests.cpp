#include "aoclda.h"

// #include "../../source/core/decision_forest/decision_forest.hpp"

#include <random>

#include "gtest/gtest.h"

// TEST(decision_forest, cpp_api_sample_features) {
//     // Initialize the decision forest class and fit model
//     da_handle df_handle = nullptr;
//     da_status status;
//     status = da_handle_init_s(&df_handle, da_handle_decision_forest);

//     status = da_options_set_int(df_handle, "n_features", 20);
//     // status = da_options_set_int(df_handle, "seed", 1201);

//     da_int n = 10;

//     da_df_sample_features_s(df_handle, n);

// }

// TEST(decision_forest, cpp_api_generate_trees)
// {
//     da_int n_features          = 20;
//     da_int n_features_per_tree = 5;

//     da_int n_trees = 4;

//     da_int n_samples;

//     da_handle df_handle = nullptr;

//     da_status status;
//     status = da_handle_init_s(&df_handle, da_handle_decision_forest);

//     da_df_bootstrap_obs_s(df_handle, n_trees, n_features_per_tree);

// }

TEST(decision_forest, cpp_api_sample_features) {

    da_datastore csv_handle;
    da_status status;

    // Read in training data
    csv_handle = nullptr;
    status = da_datastore_init(&csv_handle);

    char features_fp[256] = DATA_DIR;
    strcat(features_fp, "df_data/");
    strcat(features_fp, "training_features");
    strcat(features_fp, ".csv");

    char labels_fp[256] = DATA_DIR;
    strcat(labels_fp, "df_data/");
    strcat(labels_fp, "training_labels");
    strcat(labels_fp, ".csv");

    float *x = nullptr;
    uint8_t *y = nullptr;
    da_int n_obs = 0, d = 0, nrows_y = 0, ncols_y = 0;
    status = da_read_csv_s(csv_handle, features_fp, &x, &n_obs, &d, nullptr);
    status = da_read_csv_uint8(csv_handle, labels_fp, &y, &nrows_y, &ncols_y, nullptr);

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init_s(&df_handle, da_handle_decision_forest);

    status = da_options_set_int(df_handle, "n_features", d);
    status = da_options_set_int(df_handle, "n_obs_per_tree", 100);
    status = da_options_set_int(df_handle, "n_features_per_tree", 3);
    status = da_options_set_int(df_handle, "n_trees", 4);
    status = da_df_set_training_data_s(df_handle, n_obs, x, y);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Setup complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during training setup."
                  << std::endl;
    }

    // status = da_options_set_int(df_handle, "seed", 1201);

    da_df_fit_s(df_handle);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Fitting complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during fitting."
                  << std::endl;
    }

    if (x)
        free(x);

    if (y)
        free(y);

    da_datastore_destroy(&csv_handle);
    da_handle_destroy(&df_handle);

}
