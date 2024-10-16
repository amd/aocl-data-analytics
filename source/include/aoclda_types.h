/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 */

#ifndef AOCLDA_TYPES
#define AOCLDA_TYPES

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

/**
 * \file
 */

/**
 * \brief Enumeration used to set the floating point data type used.
 */
enum da_precision_ {
    da_unknown = 255, ///< Precision not set
    da_double = 0,    ///< Use double precision floating point type
    da_single = 1,    ///< Use single precision floating point type
};

/** @brief Alias for the \ref da_precision_ enum. */
typedef enum da_precision_ da_precision;

/**
 * \brief Enumeration used to set whether two-dimensional arrays are stored in column- or row-major order.
 */
enum da_order_ {
    row_major = 0, ///< Use row-major order
    column_major,  ///< Use column-major order
};

/** @brief Alias for the \ref da_order_ enum. */
typedef enum da_order_ da_order;

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
