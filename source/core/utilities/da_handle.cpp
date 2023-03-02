#include "da_handle.hpp"
#include "aoclda.h"

da_status _da_handle::get_current_opts(da_options::OptionRegistry **opts) {

    switch (handle_type) {
    case da_handle_linmod:
        switch (precision) {
        case da_double:
            if (linreg_d == nullptr)
                return da_status_invalid_pointer;
            *opts = &linreg_d->opts;
            break;
        case da_single:
            if (linreg_s == nullptr)
                return da_status_invalid_pointer;
            *opts = &linreg_s->opts;
            break;
        }
        break;

    default:
        return da_status_handle_not_initialized;
    }

    return da_status_success;
}