#include "aoclda.h"

#include <random>

#include "gtest/gtest.h"

TEST(decision_tree, cpp_api_sample_features) {
    da_status status;

    std::vector<float> x = {0.0,};
    std::vector<uint8_t> y = {0,};
    da_int n_obs = 0, d = 0;

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init_s(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(status, da_status_success);

    // call set_training_data with invalid value
    status = da_df_tree_set_training_data_s(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_success);

    da_handle_destroy(&df_handle);

}
