/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 */

#include "aoclda.h"
#include <iostream>
#include <vector>

/*
 * Basic t-SNE example
 *
 * This example computes a 2D t-SNE embedding for a small data matrix.
 */

int main() {
    da_handle handle = nullptr;
    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Basic t-SNE" << std::endl;
    std::cout << "t-SNE embedding for a small data matrix" << std::endl << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    int exit_code = 0;
    bool pass = true;

    double X[18] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.1, 1.9, 3.1,
                    4.1, 5.2, 5.8, 0.9, 2.2, 2.9, 4.2, 4.9, 6.2};
    da_int n_samples = 6, n_features = 3, ldx = 6;

    pass = pass && (da_handle_init_d(&handle, da_handle_tsne) == da_status_success);
    pass = pass && (da_options_set_int(handle, "n_components", 2) == da_status_success);
    pass =
        pass && (da_options_set_real_d(handle, "perplexity", 2.0) == da_status_success);
    pass = pass && (da_options_set_int(handle, "max_iter", 300) == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "theta", 0.0) == da_status_success);
    pass = pass && (da_options_set_int(handle, "seed", 42) == da_status_success);

    pass = pass && (da_tsne_set_data_d(handle, n_samples, n_features, X, ldx) ==
                    da_status_success);
    pass = pass && (da_tsne_compute_d(handle) == da_status_success);

    da_int n_components = 2;
    da_int emb_dim = n_samples * n_components;
    std::vector<double> embedding(static_cast<size_t>(emb_dim));
    pass = pass && (da_handle_get_result_d(handle, da_tsne_embedding, &emb_dim,
                                           embedding.data()) == da_status_success);

    if (pass) {
        std::cout << "t-SNE embedding computed successfully" << std::endl << std::endl;
        std::cout << "Embedding:" << std::endl;
        for (da_int i = 0; i < n_samples; ++i) {
            std::cout << embedding[i] << "  " << embedding[i + n_samples] << std::endl;
        }
    } else {
        exit_code = 1;
    }

    da_handle_destroy(&handle);
    std::cout << std::endl;
    std::cout << (pass ? "Example ran successfully." : "Example failed.") << std::endl;
    return exit_code;
}
