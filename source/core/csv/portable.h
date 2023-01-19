// clang-format off
/*
This file is a modification of one originally obtained by from
https://github.com/pandas-dev/pandas/blob/b6736b449f3062cf05c224ad738c03cef62e248e/pandas/_libs/src/headers/portable.h
and originates from MUSL Libc, https://musl.libc.org/

It is licensed under the MIT license:

Copyright © 2005-2020 Rich Felker, et al.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Modifications Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#ifndef PORTABLE_H
#define PORTABLE_H

#if defined(_MSC_VER)
#define strcasecmp( s1, s2 ) _stricmp( s1, s2 )
#define strncasecmp( s1, s2, i1 ) _strnicmp( s1, s2, i1 )
#endif

#define isdigit_ascii(c) (((unsigned)(c) - '0') < 10u)
#define getdigit_ascii(c, default) (isdigit_ascii(c) ? ((int)((c) - '0')) : default)
#define isspace_ascii(c) (((c) == ' ') || (((unsigned)(c) - '\t') < 5))
#define toupper_ascii(c) ((((unsigned)(c) - 'a') < 26) ? ((c) & 0x5f) : (c))
#define tolower_ascii(c) ((((unsigned)(c) - 'A') < 26) ? ((c) | 0x20) : (c))

#endif