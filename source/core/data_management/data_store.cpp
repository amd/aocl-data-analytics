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

#include "data_store.hpp"

namespace da_data {

bool check_internal_string(std::string &key) {
    if (key.find(DA_STRINTERNAL, 0) != std::string::npos)
        return false;
    return true;
}

template <>
da_status data_store::concatenate_cols_csv<char **>(da_int mc, da_int nc, char ***data,
                                                    da_order order, bool copy_data,
                                                    bool C_data) {
    char **deref_data = *data;
    free(data);
    data = nullptr;
    da_status status =
        concatenate_columns(mc, nc, deref_data, order, copy_data, true, C_data);
    return status;
}

template <>
da_status data_store::raw_ptr_from_csv_columns<char **>(
    [[maybe_unused]] da_csv::csv_reader *csv, da_auto_detect::CSVColumnsType &columns,
    da_int start_column, da_int end_column, da_int nrows, char ****bl, bool &C_data) {

    parser_t *parser = csv->parser;
    int *maybe_int = nullptr;
    char *p_end = nullptr;

    da_int ncols = end_column - start_column + 1;
    // Because char_to_num below uses calloc, we need to use calloc here too rather than new so deallocing is well-defined
    *bl = (char ***)calloc(1, sizeof(char **));
    **bl = (char **)calloc(nrows * ncols, sizeof(char *));
    for (da_int i = 0; i < ncols; i++) {
        if (std::vector<char **> *char_col =
                std::get_if<std::vector<char **>>(&(columns[start_column + i]))) {
            for (da_int j = 0; j < nrows; j++) {
                da_status tmp_error =
                    da_csv::char_to_num(parser, *(*char_col)[j], &p_end,
                                        (char **)&(**bl)[i * nrows + j], maybe_int);
                if (tmp_error != da_status_success) {
                    // Only possible error from char_to_num when extracting char* in memory. Exclude this from coverage
                    // LCOV_EXCL_START
                    std::string buff;
                    buff = "Unable to parse data on line " + std::to_string(i) +
                           " entry " + std::to_string(j) + ".";
                    return da_error(err, tmp_error, buff);
                    // LCOV_EXCL_STOP
                }
            }
        } else {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Wrong data type detected unexpectedly");
        }
    }
    C_data = true;
    return da_status_success;
}
} // namespace da_data