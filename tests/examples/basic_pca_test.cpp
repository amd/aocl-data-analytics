#include "aoclda.h"
#include <iostream>

int main() {

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "---------Example to Use AOCL-DA-PCA-----" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    // Problem dims and data
    int m = 3, n = 3;
    double A[3][3] = { {3, 2, 2}, {2, 3, -2}, {3, 1, 2} };
 
    // Initialize the pca object
    da_handle handle;
    da_status status;
    
    da_handle_init_d(&handle, da_handle_pca);

    status = da_pca_d_init(handle, m, n, &A[0][0]);
    if (status != da_status_success) {
        std::cout << " !!!!! PCA Init Failed !!!" << std:: endl;
        da_pca_destroy(handle);
        return 0;
    }

    printf("\nDone with da_pca_d_init()\n");

    //by default pca compute method is svd
    status = da_pca_set_method(handle, pca_method_svd);

    printf("Done with da_pca_set_method() \n");

    //Default num_compnents = 5;
    int num_compnents = 3;
    status = da_pca_set_num_components(handle, num_compnents);

    printf("\nDone with da_pca_set_num_components() \n");

    // compute pca
    status = da_pca_d_compute(handle);

    printf("\n Done with da_pca_d_compute() status: %d\n\n",status);

    if (status == da_status_success) {
        //status = da_pca_read_principal_components(handle, &nx, x);
    } else {
        std::cout << "Something wrong happened during PCA Computation. Terminating!"
                  << std::endl;
    }

    da_pca_destroy(handle);

    std::cout << "---------PCA Computed Successfully------" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}
