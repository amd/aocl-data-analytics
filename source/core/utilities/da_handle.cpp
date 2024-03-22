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
 * if refresh is true, the it call's the appropiate sub-handle's refresh()
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
        }
        break;
    case da_handle_decision_tree:
        switch (precision) {
        case da_double:
            if (dt_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &dt_d->opts;
            if (refresh)
                dt_d->refresh();
            break;
        case da_single:
            if (dt_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &dt_s->opts;
            if (refresh)
                dt_s->refresh();
            break;
        }
        break;
    case da_handle_decision_forest:
        switch (precision) {
        case da_double:
            if (df_d == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &df_d->opts;
            if (refresh)
                df_d->refresh();
            break;
        case da_single:
            if (df_s == nullptr)
                return da_error(this->err, da_status_invalid_pointer, msg);
            *opts = &df_s->opts;
            if (refresh)
                df_s->refresh();
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
        }
        break;
    default:
        return da_error(this->err, da_status_handle_not_initialized,
                        "handle has not been initialized.");
    }
    return da_status_success;
}
