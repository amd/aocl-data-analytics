#include "stdio.h"
#include "string.h"

#ifndef DA_LOGGING
#define DA_PRINTF_DEBUG(ARG1, ARG2)
#else
#define __DA_PRINTF_DEBUG_FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define DA_PRINTF_DEBUG(FMT_STR, VAL) \
    printf("[%s %s:%d] " FMT_STR, __FUNCTION__, __DA_PRINTF_DEBUG_FILENAME__, __LINE__, VAL)
#endif
