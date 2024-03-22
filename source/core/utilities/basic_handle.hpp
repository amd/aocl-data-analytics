/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef BASIC_HANDLE_HPP
#define BASIC_HANDLE_HPP
#include "aoclda_error.h"
#include "aoclda_result.h"

/*
 * Base handle class (basic_handle) that contains members that
 * are common for all specialized handles types, pca, linear
 * models, etc.
 *
 * This handle is inherited by all specialized (internal) handles.
 */
template <typename T> class basic_handle {
  public:
    virtual ~basic_handle(){};

    /*
     * Generic interface to extract all data stored
     * in handle via the da_get_result_X C API
     */
    virtual da_status get_result(da_result query, da_int *dim, T *result) = 0;
    virtual da_status get_result(da_result query, da_int *dim, da_int *result) = 0;

    /*
     * Function to inform that something related to the (sub)handle has
     * changed and to mark as update-required. E.g. options changed and potentially
     * the underlying model is different and a new call to fit is required.
     * Each (sub)handle is responsible to implement this function if required.
     */
    void refresh(){};
};
#endif