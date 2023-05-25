/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef AOCLDA_PCA
#define AOCLDA_PCA

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*TODO: Do we need the */
#define DA_BUFF_ALIGN_SIZE  64

typedef enum pca_comp_method_{
    pca_method_svd  = 0,
    pca_method_corr = 1
}pca_comp_method;

da_status da_pca_d_init(da_handle handle, da_int vectors, da_int features, double *dataX);
da_status da_pca_s_init(da_handle handle, da_int vectors, da_int features, float  *dataX);
void da_pca_destroy(da_handle handle);
da_status da_pca_set_method(da_handle, pca_comp_method_ method);
da_status da_pca_set_num_components(da_handle, da_int num_components);
da_status da_pca_d_compute(da_handle);
da_status da_pca_s_compute(da_handle);
da_status da_pca_d_get_results(da_handle);
da_status da_pca_s_get_results(da_handle);


#ifdef __cplusplus
}
#endif

#endif
