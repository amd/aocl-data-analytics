/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#ifndef AOCLDA_OPTIONS_H
#define AOCLDA_OPTIONS_H

#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \file
 */

/** \{
 * \brief Set an option, to be stored inside the \p handle argument.
 * \param[inout] handle the \ref da_handle which will store the value of the option.
 * \param[in] option the name of the option to set.
 * \param[in] value the value to set the option to.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully set.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_locked - the option cannot be changed at this point.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_option_invalid_bounds - the option value is out of bounds.
 * - \ref da_status_option_invalid_value - cannot set option to an invalid value.
 * - \ref da_status_invalid_pointer - the \p handle has not been initialized.
 */
da_status da_options_set_int(da_handle handle, const char *option, da_int value);
da_status da_options_set_string(da_handle handle, const char *option, const char *value);
/** \} */

/** \{
 * \brief Set an option, to be stored inside the \p handle argument.
 * \param[inout] handle the \ref da_handle which will store the value of the option.
 * \param[in] option the name of the option to set.
 * \param[in] value the value to set the option to.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully set.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_locked - the option cannot be changed at this point.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_option_invalid_bounds - the option value is out of bounds.
 * - \ref da_status_option_invalid_value - cannot set option to an invalid value.
 * - \ref da_status_invalid_pointer - the \p handle has not been initialized.
 * - \ref da_status_wrong_type - the \p handle was initialized with a different floating-point precision from \p value.
 */
da_status da_options_set_real_s(da_handle handle, const char *option, float value);
da_status da_options_set_real_d(da_handle handle, const char *option, double value);
/** \} */

/**
 * \brief Get the current value of an option stored inside the \p handle argument.
 * \param[in] handle the \ref da_handle which stores the options.
 * \param[in] option the name of the option to get.
 * \param[out] value the value of the option obtained from the \p handle.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully returned in \p value.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_invalid_pointer - the \p handle has not been initialized.
 **/
da_status da_options_get_int(da_handle handle, const char *option, da_int *value);

/** \{
 * \brief Get the current value of an option stored inside the \p handle argument.
 * \param[in] handle the \ref da_handle which stores the options.
 * \param[in] option the name of the option to get.
 * \param[out] value the value of the option obtained from the \p handle.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully returned in \p value.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_invalid_pointer - the \p handle has not been initialized.
 * - \ref da_status_wrong_type - the \p handle was initialized with a different floating-point precision from \p value.
 */
da_status da_options_get_real_s(da_handle handle, const char *option, float *value);
da_status da_options_get_real_d(da_handle handle, const char *option, double *value);
/** \} */

/**
 * \brief Get the current value of an option stored inside the \p handle argument.
 * \param[in] handle the \ref da_handle which stores the options.
 * \param[in] option the name of the option to get.
 * \param[out] value the value of the option obtained from the \p handle.
 * \param[in] lvalue the length of the string \p value.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully returned in \p value.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_invalid_input - the length of the string \p value, lvalue, is too small to contain the option value. Please provide more space.
 * - \ref da_status_invalid_pointer - the \p handle has not been initialized.
 **/
da_status da_options_get_string(da_handle handle, const char *option, char *value,
                                da_int *lvalue);
/**
 * \brief Get the current value of an option stored inside the \p handle argument.
 * \param[in] handle the \ref da_handle which stores the options.
 * \param[in] option the name of the option to get.
 * \param[out] value the value of the option obtained from the \p handle.
 * \param[in] lvalue the length of the string \p value.
 * \param[out] key for the option string \p value. Some options have one or more
 *             aliases but share the same key.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully returned in \p value.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_invalid_input - the length of the string \p value, lvalue, is too small to contain the option value. Please provide more space.
 * - \ref da_status_invalid_pointer - the \p handle has not been initialized.
 **/
da_status da_options_get_string_key(da_handle handle, const char *option, char *value,
                                    da_int *lvalue, da_int *key);

/**
 * \brief Print options (key/value) stored inside the \p handle argument.
 * \param[in] handle the \ref da_handle which stores the options.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success
 * - \ref da_status_invalid_pointer - the \p handle has not been initialized.
 **/
da_status da_options_print(da_handle handle);

/** \{
 * \brief Set an option, to be stored inside the \p store argument.
 * \param[inout] store the \ref da_datastore which will store the value of the option.
 * \param[in] option the name of the option to set.
 * \param[in] value the value to set the option to.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully set.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_locked - the option cannot be changed at this point.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_option_invalid_bounds - the option value is out of bounds.
 * - \ref da_status_option_invalid_value - cannot set option to an invalid value.
 * - \ref da_status_invalid_pointer - the \p store has not been initialized.
 */
da_status da_datastore_options_set_int(da_datastore store, const char *option,
                                       da_int value);
da_status da_datastore_options_set_string(da_datastore store, const char *option,
                                          const char *value);
/** \} */

/** \{
 * \brief Set an option, to be stored inside the \p store argument.
 * \param[inout] store the \ref da_datastore which will store the value of the option.
 * \param[in] option the name of the option to set.
 * \param[in] value the value to set the option to.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully set.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_locked - the option cannot be changed at this point.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_option_invalid_bounds - the option value is out of bounds.
 * - \ref da_status_option_invalid_value - cannot set option to an invalid value.
 * - \ref da_status_invalid_pointer - the \p store has not been initialized.
 * - \ref da_status_wrong_type - the \p store was initialized with a different floating-point precision from \p value.
 */
da_status da_datastore_options_set_real_s(da_datastore store, const char *option,
                                          float value);
da_status da_datastore_options_set_real_d(da_datastore store, const char *option,
                                          double value);
/** \} */

/**
 * \brief Get the current value of an option stored inside the \p store argument.
 * \param[in] store the \ref da_datastore which stores the options.
 * \param[in] option the name of the option to get.
 * \param[out] value the value of the option obtained from the \p store.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully returned in \p value.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_invalid_pointer - the \p store has not been initialized.
 **/
da_status da_datastore_options_get_int(da_datastore store, const char *option,
                                       da_int *value);

/** \{
 * \brief Get the current value of an option stored inside the \p store argument.
 * \param[in] store the \ref da_datastore which stores the options.
 * \param[in] option the name of the option to get.
 * \param[out] value the value of the option obtained from the \p store.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully returned in \p value.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_invalid_pointer - the \p store has not been initialized.
 * - \ref da_status_wrong_type - the \p store was initialized with a different floating-point precision from \p value.
 */
da_status da_datastore_options_get_real_s(da_datastore store, const char *option,
                                          float *value);
da_status da_datastore_options_get_real_d(da_datastore store, const char *option,
                                          double *value);
/** \} */

/**
 * \brief Get the current value of an option stored inside the \p store argument.
 * \param[in] store the \ref da_datastore which stores the options.
 * \param[in] option the name of the option to get.
 * \param[out] value the value of the option obtained from the \p store.
 * \param[in] lvalue the length of the string \p value.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the option was successfully returned in \p value.
 * - \ref da_status_option_not_found - the option was not found. Check the value of \p option.
 * - \ref da_status_option_wrong_type - the wrong option type was passed.
 * - \ref da_status_invalid_input - the length of the string \p value, lvalue, is too small to contain the option value. Please provide more space.
 * - \ref da_status_invalid_pointer - the \p store has not been initialized.
 **/
da_status da_datastore_options_get_string(da_datastore store, const char *option,
                                          char *value, da_int lvalue);

/**
 * \brief Print options (key/value) stored inside the \p store argument.
 * \param[in] store the \ref da_datastore which stores the options.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success
 * - \ref da_status_invalid_pointer - the \p store has not been initialized.
 **/
da_status da_datastore_options_print(da_datastore store);
#endif //AOCLDA_OPTIONS_H
