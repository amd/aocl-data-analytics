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
template <typename T>
class basic_handle {
  public:
    virtual ~basic_handle(){};

    /*
     * Generic interface to extract all data stored
     * in handle via the da_get_result_? C API
     */
    virtual da_status get_result(enum da_result query, da_int *dim, T *result) = 0;
    virtual da_status get_result(enum da_result query, da_int *dim, da_int *result) = 0;
};
#endif