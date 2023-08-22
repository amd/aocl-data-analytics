/*
This file is based on code originally obtained from

https://github.com/pandas-dev/pandas/blob/d6608313e211be0a44608252a3a31cf5220963f4/pandas/_libs/src/parser/tokenizer.c
licensed under 3-clause BSD (see below)

Copyright (c) 2012, Lambda Foundry, Inc., except where noted

It incorporates components of WarrenWeckesser/textreader
(https://github.com/WarrenWeckesser/textreader), also licensed under 3-clause BSD:

Copyright 2012 Warren Weckesser

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce the
above copyright notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution. Neither the name of
the copyright holder nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission. THIS
SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

Modifications Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#ifndef CHAR_TO_NUM_HPP
#define CHAR_TO_NUM_HPP

#include "aoclda.h"
#include "tokenizer.h"
#include <cmath>
#include <inttypes.h>
#include <limits>
#include <string.h>

namespace da_csv {

inline void missing_data(int64_t *data) { *data = INT64_MAX; }

inline void missing_data(int32_t *data) { *data = INT32_MAX; }

inline void missing_data(uint8_t *data) { *data = UINT8_MAX; }

inline void missing_data(double *data) {
    *data = std::numeric_limits<double>::quiet_NaN();
}

inline void missing_data(float *data) { *data = std::numeric_limits<float>::quiet_NaN(); }

inline void missing_data([[maybe_unused]] char **data) { return; }

/* Routines for converting individual strings to floats or integers */

inline da_status char_to_num(parser_t *parser, const char *str, char **endptr,
                             double *number, int *maybe_int) {
    // get some data from the parser
    int skip_trailing = parser->skip_trailing;
    char decimal = parser->decimal;
    char sci = parser->sci;
    char tsep = parser->thousands;
    int exponent;
    da_status status = da_status_success;
    int negative;
    char *p = (char *)str;
    int num_digits;
    int num_decimals;
    int max_digits = 17;
    int n;

    if (maybe_int != NULL)
        *maybe_int = 1;
    // Cache powers of 10 in memory.
    static double e[] = {
        1.,    1e1,   1e2,   1e3,   1e4,   1e5,   1e6,   1e7,   1e8,   1e9,   1e10,
        1e11,  1e12,  1e13,  1e14,  1e15,  1e16,  1e17,  1e18,  1e19,  1e20,  1e21,
        1e22,  1e23,  1e24,  1e25,  1e26,  1e27,  1e28,  1e29,  1e30,  1e31,  1e32,
        1e33,  1e34,  1e35,  1e36,  1e37,  1e38,  1e39,  1e40,  1e41,  1e42,  1e43,
        1e44,  1e45,  1e46,  1e47,  1e48,  1e49,  1e50,  1e51,  1e52,  1e53,  1e54,
        1e55,  1e56,  1e57,  1e58,  1e59,  1e60,  1e61,  1e62,  1e63,  1e64,  1e65,
        1e66,  1e67,  1e68,  1e69,  1e70,  1e71,  1e72,  1e73,  1e74,  1e75,  1e76,
        1e77,  1e78,  1e79,  1e80,  1e81,  1e82,  1e83,  1e84,  1e85,  1e86,  1e87,
        1e88,  1e89,  1e90,  1e91,  1e92,  1e93,  1e94,  1e95,  1e96,  1e97,  1e98,
        1e99,  1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109,
        1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119, 1e120,
        1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129, 1e130, 1e131,
        1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139, 1e140, 1e141, 1e142,
        1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149, 1e150, 1e151, 1e152, 1e153,
        1e154, 1e155, 1e156, 1e157, 1e158, 1e159, 1e160, 1e161, 1e162, 1e163, 1e164,
        1e165, 1e166, 1e167, 1e168, 1e169, 1e170, 1e171, 1e172, 1e173, 1e174, 1e175,
        1e176, 1e177, 1e178, 1e179, 1e180, 1e181, 1e182, 1e183, 1e184, 1e185, 1e186,
        1e187, 1e188, 1e189, 1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197,
        1e198, 1e199, 1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208,
        1e209, 1e210, 1e211, 1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219,
        1e220, 1e221, 1e222, 1e223, 1e224, 1e225, 1e226, 1e227, 1e228, 1e229, 1e230,
        1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239, 1e240, 1e241,
        1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249, 1e250, 1e251, 1e252,
        1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259, 1e260, 1e261, 1e262, 1e263,
        1e264, 1e265, 1e266, 1e267, 1e268, 1e269, 1e270, 1e271, 1e272, 1e273, 1e274,
        1e275, 1e276, 1e277, 1e278, 1e279, 1e280, 1e281, 1e282, 1e283, 1e284, 1e285,
        1e286, 1e287, 1e288, 1e289, 1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296,
        1e297, 1e298, 1e299, 1e300, 1e301, 1e302, 1e303, 1e304, 1e305, 1e306, 1e307,
        1e308};

    // Skip leading whitespace.
    while (isspace_ascii(*p))
        p++;

    // Handle optional sign.
    negative = 0;
    switch (*p) {
    case '-':
        negative = 1; // Fall through to increment position.
    case '+':
        p++;
    }

    *number = 0.;
    exponent = 0;
    num_digits = 0;
    num_decimals = 0;

    // Process string of digits.
    while (isdigit_ascii(*p)) {
        if (num_digits < max_digits) {
            *number = *number * 10. + (*p - '0');
            num_digits++;
        } else {
            ++exponent;
        }

        p++;
        p += (tsep != '\0' && *p == tsep);
    }

    // Process decimal part
    if (*p == decimal) {
        if (maybe_int != NULL)
            *maybe_int = 0;
        p++;

        while (num_digits < max_digits && isdigit_ascii(*p)) {
            *number = *number * 10. + (*p - '0');
            p++;
            num_digits++;
            num_decimals++;
        }

        if (num_digits >= max_digits) // Consume extra decimal digits.
            while (isdigit_ascii(*p))
                ++p;

        exponent -= num_decimals;
    }

    if (num_digits == 0) {
        *number = 0.0;
        status = da_status_parsing_error;
        return status;
    }

    // Correct for sign.
    if (negative)
        *number = -*number;

    // Process an exponent string.
    if (toupper_ascii(*p) == toupper_ascii(sci)) {
        if (maybe_int != NULL)
            *maybe_int = 0;

        // Handle optional sign
        negative = 0;
        switch (*++p) {
        case '-':
            negative = 1; // Fall through to increment pos.
        case '+':
            p++;
        }

        // Process string of digits.
        num_digits = 0;
        n = 0;
        while (num_digits < max_digits && isdigit_ascii(*p)) {
            n = n * 10 + (*p - '0');
            num_digits++;
            p++;
        }

        if (negative)
            exponent -= n;
        else
            exponent += n;

        // If no digits after the 'e'/'E', un-consume it.
        if (num_digits == 0)
            p--;
    }

    if (exponent > 308) {
        *number = HUGE_VAL;
        status = da_status_parsing_error;
        return status;
    } else if (exponent > 0) {
        *number *= e[exponent];
    } else if (exponent < -308) { // Subnormal
        if (exponent < -616) {    // Prevent invalid array access.
            *number = 0.;
        } else {
            *number /= e[-308 - exponent];
            *number /= e[308];
        }

    } else {
        *number /= e[-exponent];
    }

    if (*number == HUGE_VAL || *number == -HUGE_VAL)
        status = da_status_parsing_error;

    if (skip_trailing) {
        // Skip trailing whitespace.
        while (isspace_ascii(*p))
            p++;
    }

    if (endptr)
        *endptr = p;
    return status;
}

inline da_status char_to_num(parser_t *parser, const char *str, char **endptr,
                             float *number, int *maybe_int) {
    // get some data from the parser
    int skip_trailing = parser->skip_trailing;
    char decimal = parser->decimal;
    char sci = parser->sci;
    char tsep = parser->thousands;

    da_status status = da_status_success;
    int exponent;
    int negative;
    char *p = (char *)str;
    int num_digits;
    int num_decimals;
    int max_digits = 9;
    int n;

    if (maybe_int != NULL)
        *maybe_int = 1;
    // Cache powers of 10 in memory.
    static float e[] = {1.f,   1e1f,  1e2f,  1e3f,  1e4f,  1e5f,  1e6f,  1e7f,
                        1e8f,  1e9f,  1e10f, 1e11f, 1e12f, 1e13f, 1e14f, 1e15f,
                        1e16f, 1e17f, 1e18f, 1e19f, 1e20f, 1e21f, 1e22f, 1e23f,
                        1e24f, 1e25f, 1e26f, 1e27f, 1e28f, 1e29f, 1e30f, 1e31f,
                        1e32f, 1e33f, 1e34f, 1e35f, 1e36f, 1e37f, 1e38f};

    // Skip leading whitespace.
    while (isspace_ascii(*p))
        p++;

    // Handle optional sign.
    negative = 0;
    switch (*p) {
    case '-':
        negative = 1; // Fall through to increment position.
        [[fallthrough]];
    case '+':
        p++;
    }

    *number = 0.;
    exponent = 0;
    num_digits = 0;
    num_decimals = 0;

    // Process string of digits.
    while (isdigit_ascii(*p)) {
        if (num_digits < max_digits) {
            *number = *number * 10.f + (*p - '0');
            num_digits++;
        } else {
            ++exponent;
        }

        p++;
        p += (tsep != '\0' && *p == tsep);
    }

    // Process decimal part
    if (*p == decimal) {
        if (maybe_int != NULL)
            *maybe_int = 0;
        p++;

        while (num_digits < max_digits && isdigit_ascii(*p)) {
            *number = *number * 10.f + (*p - '0');
            p++;
            num_digits++;
            num_decimals++;
        }

        if (num_digits >= max_digits) // Consume extra decimal digits.
            while (isdigit_ascii(*p))
                ++p;

        exponent -= num_decimals;
    }

    if (num_digits == 0) {
        *number = 0.0;
        status = da_status_parsing_error;
        return status;
    }

    // Correct for sign.
    if (negative)
        *number = -*number;

    // Process an exponent string.
    if (toupper_ascii(*p) == toupper_ascii(sci)) {
        if (maybe_int != NULL)
            *maybe_int = 0;

        // Handle optional sign
        negative = 0;
        switch (*++p) {
        case '-':
            negative = 1; // Fall through to increment pos.
        case '+':
            p++;
        }

        // Process string of digits.
        num_digits = 0;
        n = 0;
        while (num_digits < max_digits && isdigit_ascii(*p)) {
            n = n * 10 + (*p - '0');
            num_digits++;
            p++;
        }

        if (negative)
            exponent -= n;
        else
            exponent += n;

        // If no digits after the 'e'/'E', un-consume it.
        if (num_digits == 0)
            p--;
    }

    if (exponent > 38) {
        *number = HUGE_VALF;
        status = da_status_parsing_error;
        return status;
    } else if (exponent > 0) {
        *number *= e[exponent];
    } else if (exponent < -38) { // Subnormal
        if (exponent < -76) {    // Prevent invalid array access.
            *number = 0.;
        } else {
            *number /= e[-38 - exponent];
            *number /= e[38];
        }

    } else {
        *number /= e[-exponent];
    }

    if (*number == HUGE_VALF || *number == -HUGE_VALF)
        status = da_status_parsing_error;

    if (skip_trailing) {
        // Skip trailing whitespace.
        while (isspace_ascii(*p))
            p++;
    }

    if (endptr)
        *endptr = p;
    return status;
}

inline da_status char_to_num(parser_t *parser, const char *str, char **endptr,
                             int64_t *number, int *maybe_int) {
    // Get some information from the parser
    int64_t int_min = parser->int_min;
    int64_t int_max = parser->int_max;
    char tsep = parser->thousands;

    char *p = (char *)str;

    int isneg = 0;
    *number = 0;
    int d;
    da_status status = da_status_success;

    // Skip leading spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        isneg = 1;
        ++p;
    } else if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit_ascii(*p)) {
        // Error...
        status = da_status_parsing_error;
        return status;
    }

    if (isneg) {
        // If number is greater than pre_min, at least one more digit
        // can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;

        // Process the digits.
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    d = *++p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((*number > pre_min) ||
                    ((*number == pre_min) && (d - '0' <= dig_pre_min))) {
                    *number = *number * 10 - (d - '0');
                    d = *++p;
                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((*number > pre_min) ||
                    ((*number == pre_min) && (d - '0' <= dig_pre_min))) {
                    *number = *number * 10 - (d - '0');
                    d = *++p;
                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        }
    } else {
        // If number is less than pre_max, at least one more digit
        // can be processed without overflowing.
        int64_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;
        // Process the digits.
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    d = *++p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((*number < pre_max) ||
                    ((*number == pre_max) && (d - '0' <= dig_pre_max))) {
                    *number = *number * 10 + (d - '0');
                    d = *++p;

                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((*number < pre_max) ||
                    ((*number == pre_max) && (d - '0' <= dig_pre_max))) {
                    *number = *number * 10 + (d - '0');
                    d = *++p;

                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        }
    }

    // Skip trailing spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (*p) {
        status = da_status_parsing_error;
        return status;
    }

    if (maybe_int != NULL)
        *maybe_int = 1;
    if (endptr)
        *endptr = p;
    return status;
}

inline da_status char_to_num(parser_t *parser, const char *str, char **endptr,
                             int32_t *number, int *maybe_int) {

    int32_t const int_min = INT32_MIN;
    int32_t const int_max = INT32_MAX;
    // Get some information from the parser
    char tsep = parser->thousands;

    char *p = (char *)str;

    int isneg = 0;
    *number = 0;
    int d;
    da_status status = da_status_success;

    // Skip leading spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        isneg = 1;
        ++p;
    } else if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit_ascii(*p)) {
        // Error...
        status = da_status_parsing_error;
        return status;
    }

    if (isneg) {
        // If number is greater than pre_min, at least one more digit
        // can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;

        // Process the digits.
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    d = *++p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((*number > pre_min) ||
                    ((*number == pre_min) && (d - '0' <= dig_pre_min))) {
                    *number = *number * 10 - (d - '0');
                    d = *++p;
                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((*number > pre_min) ||
                    ((*number == pre_min) && (d - '0' <= dig_pre_min))) {
                    *number = *number * 10 - (d - '0');
                    d = *++p;
                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        }
    } else {
        // If number is less than pre_max, at least one more digit
        // can be processed without overflowing.
        int32_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;
        // Process the digits.
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    d = *++p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((*number < pre_max) ||
                    ((*number == pre_max) && (d - '0' <= dig_pre_max))) {
                    *number = *number * 10 + (d - '0');
                    d = *++p;

                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((*number < pre_max) ||
                    ((*number == pre_max) && (d - '0' <= dig_pre_max))) {
                    *number = *number * 10 + (d - '0');
                    d = *++p;

                } else {
                    status = da_status_parsing_error;
                    return status;
                }
            }
        }
    }

    // Skip trailing spaces.
    while (isspace_ascii(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (*p) {
        status = da_status_parsing_error;
        return status;
    }

    if (maybe_int != NULL)
        *maybe_int = 1;
    if (endptr)
        *endptr = p;
    return status;
}

/*
 * Function: to_boolean
 * --------------------
 *
 * Validate if item should be recognized as a boolean field.
 *
 * item: const char* representing parsed text
 * val : pointer to a uint8_t of boolean representation
 *
 * If item is determined to be boolean, this method will set
 * the appropriate value of val and return 0. A non-zero exit
 * status means that item was not inferred to be boolean, and
 * leaves the value of *val unmodified.
 */

inline da_status char_to_num([[maybe_unused]] parser_t *parser, const char *str,
                             [[maybe_unused]] char **endptr, uint8_t *number,
                             [[maybe_unused]] int *maybe_int) {

    da_status status = da_status_parsing_error;
    *number = 0;
    char *p = (char *)str;

    // Skip leading spaces
    while (isspace_ascii(*p)) {
        ++p;
    }

    if (strncasecmp(p, "TRUE", 4) == 0) {
        if (strlen(p) > 4 && !isspace_ascii(p[4])) {
            return status;
        } else {
            status = da_status_success;
            *number = 1;
        }
    } else if (strncasecmp(p, "FALSE", 5) == 0) {
        if (strlen(p) > 5 && !isspace_ascii(p[5])) {
            return status;
        } else {
            status = da_status_success;
            *number = 0;
        }
    } else {
        return status;
    }

    if (maybe_int != NULL)
        *maybe_int = 1;
    if (endptr)
        *endptr = p;

    return status;
}

/* Overloaded to enable storing a char from the parser's own char array, removing trailing whitespace and leading whitespace if option set*/
inline da_status char_to_num([[maybe_unused]] parser_t *parser, const char *str,
                             char **endptr, char **store,
                             [[maybe_unused]] int *maybe_int) {

    da_status status = da_status_success;

    char *p = (char *)str;
    size_t len = strlen(p);
    (*endptr) = p + len - 1;

    if (parser->skipinitialspace) {
        while (p < (*endptr) && isspace_ascii(*p))
            p++;
    }

    while ((*endptr) > p && isspace_ascii(*(*endptr)))
        (*endptr)--;

    len = (*endptr) - p + 1;

    *store = (char *)malloc(sizeof(char) * (1 + len));
    memset(*store, '\0', 1 + len);

    if (*store == NULL) {
        return da_status_memory_error;
    }

/* Most of the time MSVC compiler can automatically replace CRT functions with _s versions, but not this one */
#if defined(_MSC_VER)
    strncpy_s(*store, 1 + len, p, len);
#else
    strncpy(*store, p, len);
#endif

    return status;
}

} //namespace da_csv

#endif