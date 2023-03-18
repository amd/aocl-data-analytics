#ifndef AOCLDA_HANDLE
#define AOCLDA_HANDLE

#include "aoclda_error.h"
#include "aoclda_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum da_handle_type_ {
    da_handle_uninitialized,
    da_handle_csv_opts,
    da_handle_linmod,
} da_handle_type;

typedef struct _da_handle *da_handle;

/* Initialize da_handle struct with default values */
da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type);

da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type);

/* Print error information stored in the struct */
void da_handle_print_error_message(da_handle handle);

/* Check whether handle is of the correct type */
da_status da_check_handle_type(da_handle handle, da_handle_type expected_handle_type);

/* Destroy the da_handle struct */
void da_handle_destroy(da_handle *handle);

#ifdef __cplusplus
}
#endif

#endif