#include "aoclda.h"
#include <iostream>
#include <vector>

int decision_tree_ex_d(std::string score_criteria) {
    int exit_code = 0;
    double tol = 1.0e-6;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Decision tree model (double precision)" << std::endl;
    std::cout << "Scoring Criteria: " << score_criteria << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_handle df_handle;
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

    double *x = nullptr;
    uint8_t *y = nullptr;
    da_int n_obs = 0, n_features = 0, nrows_y = 0, ncols_y = 0;
    status = da_read_csv_d(csv_handle, features_fp, &x, &n_obs, &n_features, nullptr);
    status = da_read_csv_uint8(csv_handle, labels_fp, &y, &nrows_y, &ncols_y, nullptr);

    // Initialize the decision tree class and fit model
    df_handle = nullptr;
    status = da_handle_init_d(&df_handle, da_handle_decision_tree);
    status = da_df_set_training_data_d(df_handle, n_obs, n_features, x, y);
    status = da_options_set_int(df_handle, "depth", 5);
    status = da_options_set_int(df_handle, "seed", 77);
    status = da_options_set_string(df_handle, "scoring function", score_criteria.data());
    status = da_df_fit_d(df_handle);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Fitting complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during training setup or fitting."
                  << std::endl;
    }

    // Read in data for making predictions
    char test_features_fp[256] = DATA_DIR;
    strcat(test_features_fp, "df_data/");
    strcat(test_features_fp, "test_features");
    strcat(test_features_fp, ".csv");

    char test_labels_fp[256] = DATA_DIR;
    strcat(test_labels_fp, "df_data/");
    strcat(test_labels_fp, "test_labels");
    strcat(test_labels_fp, ".csv");

    double *x_test = nullptr;
    uint8_t *y_test = nullptr;
    n_obs = 0;
    n_features = 0;
    nrows_y = 0;
    ncols_y = 0;

    status = da_read_csv_d(csv_handle, test_features_fp, &x_test, &n_obs, &n_features,
                           nullptr);
    status = da_read_csv_uint8(csv_handle, test_labels_fp, &y_test, &nrows_y, &ncols_y,
                               nullptr);

    // Make predictions with model and evaluate score
    std::vector<uint8_t> y_pred(n_obs);
    status = da_df_predict_d(df_handle, n_obs, n_features, x_test, y_pred.data());
    double score = 0.0;
    status = da_df_score_d(df_handle, n_obs, n_features, x_test, y_test, score);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Scoring complete." << std::endl;
        double score_exp;
        std::cout << "Score    = " << score << std::endl;
        if (score_criteria == "gini") {
            score_exp = 0.93250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "cross-entropy") {
            score_exp = 0.93250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "misclassification-error") {
            score_exp = 0.93500;
            std::cout << "Expected = " << score_exp << std::endl;
        }
        // Check result
        double err = std::abs(score - score_exp);
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        std::cout << "Something wrong happened during prediction setup or scoring."
                  << std::endl;
    }

    if (x_test)
        free(x_test);

    if (y_test)
        free(y_test);

    if (x)
        free(x);

    if (y)
        free(y);

    da_datastore_destroy(&csv_handle);
    da_handle_destroy(&df_handle);

    return exit_code;
}

int decision_tree_ex_s(std::string score_criteria) {
    int exit_code = 0;
    double tol = 1.0e-6;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Decision tree model (single precision)" << std::endl;
    std::cout << "Scoring Criteria: " << score_criteria << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    da_handle df_handle;
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
    da_int n_obs = 0, n_features = 0, nrows_y = 0, ncols_y = 0;
    status = da_read_csv_s(csv_handle, features_fp, &x, &n_obs, &n_features, nullptr);
    status = da_read_csv_uint8(csv_handle, labels_fp, &y, &nrows_y, &ncols_y, nullptr);

    // Initialize the decision tree class and fit model
    df_handle = nullptr;
    status = da_handle_init_s(&df_handle, da_handle_decision_tree);
    status = da_df_set_training_data_s(df_handle, n_obs, n_features, x, y);
    status = da_options_set_int(df_handle, "depth", 5);
    status = da_options_set_int(df_handle, "seed", 77);
    status = da_options_set_string(df_handle, "scoring function", score_criteria.data());
    status = da_df_fit_s(df_handle);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Fitting complete." << std::endl;
    } else {
        std::cout << "Something wrong happened during training setup or fitting."
                  << std::endl;
    }

    // Read in data for making predictions
    char test_features_fp[256] = DATA_DIR;
    strcat(test_features_fp, "df_data/");
    strcat(test_features_fp, "test_features");
    strcat(test_features_fp, ".csv");

    char test_labels_fp[256] = DATA_DIR;
    strcat(test_labels_fp, "df_data/");
    strcat(test_labels_fp, "test_labels");
    strcat(test_labels_fp, ".csv");

    float *x_test = nullptr;
    uint8_t *y_test = nullptr;
    n_obs = 0;
    n_features = 0;
    nrows_y = 0;
    ncols_y = 0;

    status = da_read_csv_s(csv_handle, test_features_fp, &x_test, &n_obs, &n_features,
                           nullptr);
    status = da_read_csv_uint8(csv_handle, test_labels_fp, &y_test, &nrows_y, &ncols_y,
                               nullptr);

    // Make predictions with model and evaluate score
    std::vector<uint8_t> y_pred(n_obs);
    status = da_df_predict_s(df_handle, n_obs, n_features, x_test, y_pred.data());
    float score = 0.0;
    status = da_df_score_s(df_handle, n_obs, n_features, x_test, y_test, score);

    std::cout << "----------------------------------------" << std::endl;
    if (status == da_status_success) {
        std::cout << "Scoring complete." << std::endl;
        double score_exp;
        std::cout << "Score    = " << score << std::endl;
        if (score_criteria == "gini") {
            score_exp = 0.93250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "cross-entropy") {
            score_exp = 0.93250;
            std::cout << "Expected = " << score_exp << std::endl;
        } else if (score_criteria == "misclassification-error") {
            score_exp = 0.93500;
            std::cout << "Expected = " << score_exp << std::endl;
        }
        // Check result
        double err = std::abs(score - score_exp);
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        std::cout << "Something wrong happened during prediction setup or scoring."
                  << std::endl;
    }

    if (x_test)
        free(x_test);

    if (y_test)
        free(y_test);

    if (x)
        free(x);

    if (y)
        free(y);

    da_datastore_destroy(&csv_handle);
    da_handle_destroy(&df_handle);

    return exit_code;
}

int main(int argc, char **argv) {
    std::string score_criteria;

    if (argc > 1) {
        score_criteria = argv[1];
    } else {
        score_criteria = "gini";
    }

    int exit_code = 0;
    exit_code = decision_tree_ex_d(score_criteria);
    exit_code = decision_tree_ex_s(score_criteria);

    return exit_code;
}
