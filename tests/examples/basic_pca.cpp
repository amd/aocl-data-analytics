#include "aoclda.h"
#include <iostream>

int main() {

    // Initialize the pca object
    da_handle handle;
    da_status status;

    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << " Example to Use AOCLDA-PCA for double precision" << std::endl;
    std::cout << std::fixed << std::endl;

    // A: num_samples x num_features (n x p)
    da_int n = 3, p = 3;
    double A[3][3] = {{3, 2, 3}, {2, 3, 1}, {2, -2, 2}};
    //float refdoutput[16];
    da_int num_compnents = std::min(3, std::min(n, p));
    double doutput[24];

    da_handle_init_d(&handle, da_handle_pca);
    status = da_pca_d_init(handle, n, p, &A[0][0]);
    if (status == da_status_success) {
        std::cout << " PCA successfully initialized " << std::endl;
    } else {
        std::cout << " PCA Initialization Failed with status" << status << std::endl;
        da_handle_destroy(&handle);
        return 0;
    }

    //by default pca compute method is svd
    status = da_pca_set_method(handle, pca_method_svd);
    status = da_pca_set_num_components(handle, num_compnents);

    // compute pca
    status = da_pca_d_compute(handle);
    pca_results_flags flags = (pca_components);
    if (status == da_status_success) {
        std::cout << " PCA computed successfully " << std::endl;
        status = da_pca_d_get_results(handle, doutput, flags);
        if (status == da_status_success) {
            std::cout << " Successfully read the PCA results" << std::endl;
        } else {
            std::cout << " PCA get results failed with status" << status << std::endl;
        }
    } else {
        std::cout << " PCA computation Failed with status" << status << std::endl;
    }

    da_handle_destroy(&handle);
    std::cout << " PCA computed successfully for double precision" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << " Example to Use AOCLDA-PCA for single precision" << std::endl;
    std::cout << std::fixed << std::endl;

    // A: num_samples x num_features (n x p)
    da_int ns = 3, ps = 3;
    float As[3][3] = {{3, 2, 3}, {2, 3, 1}, {2, -2, 2}};
    //float refsoutput[16];
    float soutput[24];

    da_handle_init_s(&handle, da_handle_pca);
    status = da_pca_s_init(handle, ns, ps, &As[0][0]);
    if (status == da_status_success) {
        std::cout << " PCA successfully initialized " << std::endl;
    } else {
        std::cout << " PCA Initialization Failed with status" << status << std::endl;
        da_handle_destroy(&handle);
        return 0;
    }

    //by default pca compute method is svd
    status = da_pca_set_method(handle, pca_method_svd);
    num_compnents = 3;
    status = da_pca_set_num_components(handle, num_compnents);

    // compute pca
    status = da_pca_s_compute(handle);
    flags = (pca_components);
    if (status == da_status_success) {
        std::cout << " PCA computed successfully " << std::endl;
        status = da_pca_s_get_results(handle, soutput, flags);
        if (status == da_status_success) {
            std::cout << " Successfully read the PCA results" << std::endl;
        } else {
            std::cout << " PCA get results failed with status" << status << std::endl;
        }
    } else {
        std::cout << " PCA computation Failed with status" << status << std::endl;
    }

    da_handle_destroy(&handle);
    std::cout << " PCA computed successfully for single precision " << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    return 0;
}
