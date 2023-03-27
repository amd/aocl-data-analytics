#ifndef AOCLDA_BASICSTATS
#define AOCLDA_BASICSTATS

#include "aoclda_error.h"
#include "aoclda_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * \anchor chapter_a
 * \file
 * \brief Chapter A - Basis statistical functions
 *
 * \todo Here are declared the basic statistical functions such as: mean, meadian,
 * mode, fivenum, min, max, etc.
 *
 * \section cha_intro Introduction
 * \section cha_io Input and output
 * \subsection cha_data Data
 */

/** \{
 * \brief Estimate the mean of a vector
 *
 * Calculate the average of elements in a dense array.
 * \param[in] n Size of array,
 * \param[in,out] *x Pointer to the array TODO this should have `const` keyword,
 * \param[in] incx Stride to use,
 * \param[out] *mean Pointer to scalar where to store mean.
 * \return Returns \ref da_status_.
 *
 * \section da_mean_nan Handling of missing values
 * These functions do not handle missing values. In the presence of NaN, the
 * result is also NaN. Need to add nan_handling option!
 *
 * \section da_mean_return Return
 * The function returns
 *
 * * \ref da_status_success - operation was successfully completed,
 * * da_status_        - on failure TODO complete.
 *
 * \section da_mean_ex Example
 * \todo Description and link to source.
 *
 * \section da_mean_detail Details
 * None
 *
 * \section da_mean_opt Options
 * None
 *
 * \section da_mean_ref References
 * None
 */

da_status da_mean_s(da_int n, float *x, da_int incx, float *mean);
da_status da_mean_d(da_int n, double *x, da_int incx, double *mean);

/**
 *\}
 */

#ifdef __cplusplus
}
#endif

#endif
