#include "stdio.h"
#include "string.h"

#ifndef DA_LOGGING
#define DA_PRINTF_DEBUG(...)
#else
#define __DA_PRINTF_DEBUG_FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define DA_PRINTF_DEBUG(FMT_STR, ...) \
    printf("[%s %s:%d] " FMT_STR, __FUNCTION__, __DA_PRINTF_DEBUG_FILENAME__, __LINE__ __VA_OPT__(,) __VA_ARGS__)
#endif
