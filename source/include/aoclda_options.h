/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef AOCLDA_OPTIONS_H
#define AOCLDA_OPTIONS_H

#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \file
 */

/** \{
 * \brief Member of the option setters group
 *
 * This function is part of the group of "option setters" and are all quite similar and differ on the last parameter \p value.
 * \param[in,out] handle The handle where to store the option setting,
 * \param[in] option Name of the option to set,
 * \param[in] value Value to set the option to.
 * \return Returns \ref da_status.
 *
 * These functions do not handle missing values.
 *
 * The function returns
 *
 * * \ref da_status_success          - option was successfully set,
 * * \ref da_status_option_not_found - option not found,
 * * \ref da_status_option_locked    - cannot change option at this point,
 * * \ref da_status_option_wrong_type - wrong option type passed,
 * * \ref da_status_option_invalid_bounds - option value is out-of-bounds,
 * * \ref da_status_option_invalid_value - cannot set option to an invalid value.
 *
 */
da_status da_options_set_int(da_handle handle, const char *option, da_int value);
da_status da_options_set_string(da_handle handle, const char *option, const char *value);
da_status da_options_set_s_real(da_handle handle, const char *option, float value);
da_status da_options_set_d_real(da_handle handle, const char *option, double value);
/** \} */

/** \{
 * \brief Member of the option getters group
 *
 * This function is part of the group of "option getters" and are all quite similar and differ
 * on the last parameters \p value and \p lvalue.
 * \param[in,out] handle The handle where to retrieve the option value,
 * \param[in] option Name of the option to get,
 * \param[in] value Value that contains the option setting.
 * \return Returns \ref da_status.
 * 
 * These functions do not handle missing values.
 * The function returns
 *
 * * \ref da_status_success          - option was successfully set,
 * * \ref da_status_option_not_found - option not found,
 * * \ref da_status_option_wrong_type - wrong option type passed.
 *
 */
da_status da_options_get_int(da_handle handle, const char *option, da_int *value);
da_status da_options_get_s_real(da_handle handle, const char *option, float *value);
da_status da_options_get_d_real(da_handle handle, const char *option, double *value);
/** \} */

/** \{
 * \brief Member of the option getters group
 *
 * This function is part of the group of "option getters" and are all quite similar and differ
 * on the last parameters \p value and \p lvalue.
 * \param[in,out] handle The handle where to retrieve the option value,
 * \param[in] option Name of the option to get,
 * \param[in] value Value that contains the option setting.
 * \param[in] lvalue length of the string \p value.
 * \return Returns \ref da_status.
 * 
 * These functions do not handle missing values.
 * 
 * The function returns
 *
 * * \ref da_status_success          - option was successfully set,
 * * \ref da_status_invalid_input    - length of string \p value, lvalue, is too small to containt the option value. Provide more space.
 * * \ref da_status_option_not_found - option not found,
 * * \ref da_status_option_wrong_type - wrong option type passed.
 *
 */
da_status da_options_get_string(da_handle handle, const char *option, char *value,
                                size_t *lvalue);

da_status da_datastore_options_set_int(da_datastore store, const char *option,
                                       da_int value);
da_status da_datastore_options_set_string(da_datastore store, const char *option,
                                          const char *value);
da_status da_datastore_options_set_s_real(da_datastore store, const char *option,
                                          float value);
da_status da_datastore_options_set_d_real(da_datastore store, const char *option,
                                          double value);
da_status da_datastore_options_get_int(da_datastore store, const char *option,
                                       da_int *value);
da_status da_datastore_options_get_s_real(da_datastore store, const char *option,
                                          float *value);
da_status da_datastore_options_get_d_real(da_datastore store, const char *option,
                                          double *value);
da_status da_datastore_options_get_string(da_datastore store, const char *option,
                                          char *value, size_t lvalue);
/** \} */

#endif //AOCLDA_OPTIONS_H
