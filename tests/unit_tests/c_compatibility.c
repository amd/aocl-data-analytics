/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 */

/*
 * The purpose of this test is to check C compatibility of the aocl-da interfaces.
 * The test should be compiled using a C compiler and linked with a C++ compiler.
 * The contents of the test are largely irrelevant - the important check is that with a C
 * compiler we can successfully include aoclda.h.
 */

#include "aoclda.h"

int main() {

    int exit_code = 0;

    // problem data
    da_int m = 5, n = 2;
    double Al[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bl[5] = {1, 1, 1, 1, 1};

    // Initialize a linear regression
    da_handle handle = NULL;
    da_status status;
    da_handle_init_d(&handle, da_handle_linmod);
    da_linmod_d_select_model(handle, linmod_model_mse);
    da_linmod_d_define_features(handle, n, m, Al, bl);
    // Compute regression
    status = da_linmod_d_fit(handle);
    if (status != da_status_success) {
        exit_code = 1;
    }

    da_handle_destroy(&handle);

    return exit_code;
}