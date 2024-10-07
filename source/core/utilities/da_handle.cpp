/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "da_handle.hpp"
#include "aoclda.h"

/*
 * Get pointer to the option member of the currently active sub-handle, also
 * if refresh is true, the it calls the appropriate sub-handle's refresh()
 * member to indicate that substantial changes have occurred in the handle.
 * E.g. options changes that alter the model requiring re-training, etc...
 */
da_status _da_handle::get_current_opts(da_options::OptionRegistry **opts, bool refresh) {
    const std::string msg = "handle seems to be corrupted.";
    switch (handle_type) {
    case da_handle_linmod:
        switch (precision) {
        case da_double:
            if (linreg_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &linreg_d->opts;
            if (refresh)
                linreg_d->refresh();
            break;
        case da_single:
            if (linreg_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &linreg_s->opts;
            if (refresh)
                linreg_s->refresh();
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "handle precision ws not correctly set");
            break;
        }
        break;
    case da_handle_decision_tree:
        switch (precision) {
        case da_double:
            if (dectree_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &dectree_d->opts;
            if (refresh)
                dectree_d->refresh();
            break;
        case da_single:
            if (dectree_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &dectree_s->opts;
            if (refresh)
                dectree_s->refresh();
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "handle precision ws not correctly set");
            break;
        }
        break;
    case da_handle_decision_forest:
        switch (precision) {
        case da_double:
            if (forest_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &forest_d->opts;
            if (refresh)
                forest_d->refresh();
            break;
        case da_single:
            if (forest_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &forest_s->opts;
            if (refresh)
                forest_s->refresh();
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "handle precision ws not correctly set");
            break;
        }
        break;
    case da_handle_pca:
        switch (precision) {
        case da_double:
            if (pca_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &pca_d->opts;
            if (refresh)
                pca_d->refresh();
            break;
        case da_single:
            if (pca_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &pca_s->opts;
            if (refresh)
                pca_s->refresh();
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "handle precision ws not correctly set");
            break;
        }
        break;
    case da_handle_kmeans:
        switch (precision) {
        case da_double:
            if (kmeans_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &kmeans_d->opts;
            if (refresh)
                kmeans_d->refresh();
            break;
        case da_single:
            if (kmeans_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &kmeans_s->opts;
            if (refresh)
                kmeans_s->refresh();
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "handle precision ws not correctly set");
            break;
        }
        break;
    case da_handle_nlls:
        switch (precision) {
        case da_double:
            if (nlls_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &nlls_d->opt->opts;
            if (refresh)
                nlls_d->refresh();
            break;
        case da_single:
            if (nlls_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &nlls_s->opt->opts;
            if (refresh)
                nlls_s->refresh();
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "handle precision ws not correctly set");
            break;
        }
        break;
    case da_handle_knn:
        switch (precision) {
        case da_double:
            if (knn_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &knn_d->opts;
            if (refresh)
                knn_d->refresh();
            break;
        case da_single:
            if (knn_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &knn_s->opts;
            if (refresh)
                knn_s->refresh();
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "handle precision ws not correctly set");
            break;
        }
        break;
    default:
        return da_error(this->err, da_status_handle_not_initialized,
                        "handle has not been initialized.");
    }
    return da_status_success;
}
