#ifndef AOCLDA_TYPES
#define AOCLDA_TYPES

#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>

/**
 * \file
 * \anchor apx_d
 * \brief Appendix D - Data types.
 *
 * \todo This header file defines data type precision to use
 * for floating point data type as well as the data type for
 * integers.
 */

/**
 * Precision enumeration used to set the floating points data type used.
 */
typedef enum da_precision_ {
    da_double = 0, ///< Use double precision floating point type
    da_single = 1, ///< Use single precison floating point type
} da_precision;

typedef enum da_ordering_ {
    row_major = 0,
    col_major,
} da_ordering;

/**
 * \def Build library for 64 bit integers
 */
#if defined(AOCLDA_ILP64)
/**
 * Use 64 bits integer type, to use 32 bits integer type set the CMake option <tt>BUILD_ILP64=OFF</tt>.
 */
typedef int64_t da_int;
#define DA_INT_MAX INT64_MAX
#define DA_INT_MIN INT64_MIN
#define DA_INT_FMT PRId64
#else
/**
 * Use 32 bits integer type (default), to use 64 bits integer type set the CMake option <tt>BUILD_ILP64=ON</tt>.
 */
typedef int32_t da_int;
#define DA_INT_MAX INT32_MAX
#define DA_INT_MIN INT32_MIN
#define DA_INT_FMT PRId32
#endif

#endif // AOCLDA_TYPES
