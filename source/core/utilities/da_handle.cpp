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
    case da_handle_decision_tree:
        switch (precision) {
        case da_double:
            if (dt_d == nullptr)
                return da_status_invalid_pointer;
            *opts = &dt_d->opts;
            break;
        case da_single:
            if (dt_s == nullptr)
                return da_status_invalid_pointer;
            *opts = &dt_s->opts;
            break;
        }
        break;
    case da_handle_decision_forest:
        switch (precision) {
        case da_double:
            if (df_d == nullptr)
                return da_status_invalid_pointer;
            *opts = &df_d->opts;
            break;
        case da_single:
            if (df_s == nullptr)
                return da_status_invalid_pointer;
            *opts = &df_s->opts;
            break;
        }
        break;
    case da_handle_pca:
        switch (precision) {
        case da_double:
            if (pca_d == nullptr)
                return da_status_invalid_pointer;
            *opts = &pca_d->opts;
            break;
        case da_single:
            if (pca_s == nullptr)
                return da_status_invalid_pointer;
            *opts = &pca_s->opts;
            break;
        }
        break;
    default:
        return da_status_handle_not_initialized;
    }

    return da_status_success;
}
