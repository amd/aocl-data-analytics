#include "data_store.hpp"

using namespace da_data;

bool da_data::validate_interval(interval p, da_int max_val) {
    if (p.first > p.second)
        return false;
    if (p.first < 0 || p.second >= max_val)
        return false;
    return true;
}

bool da_data::check_internal_string(std::string &key) {
    if (key.find(DA_STRINTERNAL, 0, DA_STRSZ) != std::string::npos)
        return false;
    return true;
}

template <>
da_status da_data::data_store::concatenate_cols_csv<char **>(
    da_int mc, da_int nc, char ***data, da_ordering order, bool copy_data, bool C_data) {
    char **deref_data = *data;
    free(data);
    da_status status =
        concatenate_columns(mc, nc, deref_data, order, copy_data, true, C_data);
    return status;
}

template <>
da_status da_data::data_store::raw_ptr_from_csv_columns<char **>(
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
                if (tmp_error != da_status_success)
                    return da_error(err, tmp_error, "error in char_to_num");
            }
        } else {
            return da_error(err, da_status_internal_error,
                            "wrong type detected unexpectedly");
        }
    }
    C_data = true;
    return da_status_success;
}
