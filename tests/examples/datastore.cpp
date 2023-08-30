#include "aoclda.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Load data from csv file" << std::endl;

    da_datastore store;
    std::string filename = std::string(DATA_DIR) + "/" + "datastore_ex.csv";
    da_status status;

    // Load data
    da_datastore_init(&store);
    da_datastore_options_set_string(store, "CSV datatype", "double");
    da_datastore_options_set_int(store, "CSV use header row", 1);
    status = da_data_load_from_csv(store, filename.c_str());
    if (status != da_status_success) {
        std::cout << "Data loading unsuccessful" << std::endl;
        return 1;
    }

    // Select the first 2 columns as the feature matrix and the last one as the response
    std::vector<double> features(10), rhs(5);
    da_data_select_columns(store, "features", 0, 1);
    da_data_select_columns(store, "rhs", 2, 2);
    da_data_extract_selection_real_d(store, "features", 5, features.data());
    da_data_extract_selection_real_d(store, "rhs", 5, rhs.data());

    // define the regression problem to solve
    da_handle handle;
    da_handle_init_d(&handle, da_handle_linmod);
    da_linmod_d_select_model(handle, linmod_model_mse);
    da_linmod_d_define_features(handle, 2, 5, features.data(), rhs.data());

    // solve the problem
    status = da_linmod_d_fit(handle);

    int exit = 0;
    if (status == da_status_success) {
        std::cout << "regression computed successfully!" << std::endl;
        da_int nx = 2;
        std::vector<double> x(2);
        da_handle_get_result_d(handle, da_result::da_linmod_coeff, &nx, x.data());
        std::cout << "Coefficients: " << x[0] << " " << x[1] << std::endl;
        std::cout << "(Expected   : " << 0.199256 << " " << 0.130354 << ")" << std::endl;
    } else
        exit = 1;

    da_datastore_destroy(&store);
    da_handle_destroy(&handle);

    return exit;
}