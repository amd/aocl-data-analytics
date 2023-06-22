#ifndef DA_HANDLE_HPP
#define DA_HANDLE_HPP

#include <new>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "csv_reader.hpp"
#include "da_error.hpp"
#include "linear_model.hpp"
#include "pca.hpp"

#define ERR_MSG_LEN 1024

struct _da_handle {
  public:
    da_csv::csv_reader *csv_parser = nullptr;
    linear_model<double> *linreg_d = nullptr;
    linear_model<float> *linreg_s = nullptr;
    da_handle_type handle_type = da_handle_uninitialized;
    // Pointer to error trace and related methods
    da_errors::da_error_t *err = nullptr;
    da_precision precision = da_double;
    da_pca<double> *pca_d = nullptr;
    da_pca<float> *pca_s = nullptr;

    da_status get_current_opts(da_options::OptionRegistry **opts);
};

#endif