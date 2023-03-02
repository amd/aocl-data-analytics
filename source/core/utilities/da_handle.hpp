#ifndef DA_HANDLE_HPP
#define DA_HANDLE_HPP

#include <new>
#include <stdio.h>
#include <string.h>

#include "aoclda.h"
#include "linear_model.hpp"
#include "tokenizer.h"

#define ERR_MSG_LEN 1024

class _da_handle {
  public:
    parser_t *parser = nullptr;
    linear_model<double> *linreg_d = nullptr;
    linear_model<float> *linreg_s = nullptr;
    da_handle_type handle_type = da_handle_uninitialized;
    char error_message[ERR_MSG_LEN] = "";
    da_precision precision = da_double;

    da_status get_current_opts(da_options::OptionRegistry **opts);
};

#endif