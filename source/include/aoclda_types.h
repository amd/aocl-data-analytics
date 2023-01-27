#ifndef AOCLDA_TYPES
#define AOCLDA_TYPES

#include <stddef.h>
#include <stdint.h>

typedef enum da_precision_ {
    da_double = 0,
    da_single = 1,
} da_precision;

#if defined(AOCLDA_ILP64)
typedef int64_t da_int;
#define DA_INT_MAX INT64_MAX
#define DA_INT_MIN INT64_MIN
#else
typedef int32_t da_int;
#define DA_INT_MAX INT32_MAX
#define DA_INT_MIN INT32_MIN
#endif

#endif // AOCLDA_TYPES