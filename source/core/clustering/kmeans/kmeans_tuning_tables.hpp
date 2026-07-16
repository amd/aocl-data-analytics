/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#ifndef KMEANS_TUNING_TABLES_HPP
#define KMEANS_TUNING_TABLES_HPP

#include "da_kernel_utils.hpp"

namespace da_kmeans {

// clang-format off

// ------ ELKAN UPDATE TUNING TABLE ---------------------------------------------------
constexpr TBL<KernelSelection>::type elkan_update = {{
  {generic,       tid<float>(),   {{{4, avx},{avx2}                             }}},
  {generic,       tid<double>(),  {{{avx2}                                      }}},
  {generic,       tid<_Float16>(),{{{scalar}                                    }}},
  {zen2,          tid<float>(),   {{{4, avx},{avx2}                             }}},
  {zen2,          tid<double>(),  {{{avx2}                                      }}},
  {zen2,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen3,          tid<float>(),   {{{4, avx},{avx2}                             }}},
  {zen3,          tid<double>(),  {{{avx2}                                      }}},
  {zen3,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen4,          tid<float>(),   {{{4, avx},{avx2}                             }}},
  {zen4,          tid<double>(),  {{{avx2}                                      }}},
  {zen4,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen5,          tid<float>(),   {{{4, avx},{avx2}                             }}},
  {zen5,          tid<double>(),  {{{6, avx},{15, avx2},{avx512}                }}},
  {zen5,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen6,          tid<float>(),   {{{4, avx},{avx2}                             }}},
  {zen6,          tid<double>(),  {{{6, avx},{15, avx2},{avx512}                }}},
  {zen6,          tid<_Float16>(),{{{8, scalar}, {16, avx}, {32, avx2}, {avx512}}}},
  {generic_avx512,tid<float>(),   {{{4, avx},{avx2}                             }}},
  {generic_avx512,tid<double>(),  {{{6, avx},{15, avx2},{avx512}                }}},
  {generic_avx512,tid<_Float16>(),{{{8, scalar}, {16, avx}, {32, avx2}, {avx512}}}}
}};

// ------ ELKAN REDUCE TUNING TABLE ---------------------------------------------------
constexpr TBL<KernelSelection>::type elkan_reduce = {{
  {generic,       tid<float>(),   {{{scalar}                                    }}},
  {generic,       tid<double>(),  {{{4, scalar}, {16, avx}, {avx2}              }}},
  {generic,       tid<_Float16>(),{{{scalar}                                    }}},
  {zen2,          tid<float>(),   {{{scalar}                                    }}},
  {zen2,          tid<double>(),  {{{4, scalar}, {16, avx}, {avx2}              }}},
  {zen2,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen3,          tid<float>(),   {{{scalar}                                    }}},
  {zen3,          tid<double>(),  {{{4, scalar}, {16, avx}, {avx2}              }}},
  {zen3,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen4,          tid<float>(),   {{{scalar}                                    }}},
  {zen4,          tid<double>(),  {{{4, scalar}, {16, avx}, {avx2}              }}},
  {zen4,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen5,          tid<float>(),   {{{8, scalar}, {avx}                          }}},
  {zen5,          tid<double>(),  {{{4, scalar}, {8, avx}, {15, avx2}, {avx512} }}},
  {zen5,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen6,          tid<float>(),   {{{8, scalar}, {avx}                          }}},
  {zen6,          tid<double>(),  {{{4, scalar}, {8, avx}, {15, avx2}, {avx512} }}},
  {zen6,          tid<_Float16>(),{{{8, scalar}, {16, avx}, {32, avx2}, {avx512}}}},
  {generic_avx512,tid<float>(),   {{{8, scalar}, {avx}                          }}},
  {generic_avx512,tid<double>(),  {{{4, scalar}, {8, avx}, {15, avx2}, {avx512} }}},
  {generic_avx512,tid<_Float16>(),{{{8, scalar}, {avx512}                       }}}
}};

// ------ LLOYD TUNING TABLE ---------------------------------------------------
constexpr TBL<KernelSelection>::type lloyd_tuning = {{
  {generic,       tid<float>(),   {{{2, scalar}, {16, avx}, {avx2}              }}},
  {generic,       tid<double>(),  {{{6, avx}, {avx2}                            }}},
  {generic,       tid<_Float16>(),{{{scalar}                                    }}},
  {zen2,          tid<float>(),   {{{2, scalar}, {16, avx}, {avx2}              }}},
  {zen2,          tid<double>(),  {{{6, avx}, {avx2}                            }}},
  {zen2,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen3,          tid<float>(),   {{{2, scalar}, {16, avx}, {avx2}              }}},
  {zen3,          tid<double>(),  {{{6, avx}, {avx2}                            }}},
  {zen3,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen4,          tid<float>(),   {{{2, scalar}, {16, avx}, {avx2}              }}},
  {zen4,          tid<double>(),  {{{6, avx}, {avx2}                            }}},
  {zen4,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen5,          tid<float>(),   {{{4, scalar}, {16, avx}, {avx2}              }}},
  {zen5,          tid<double>(),  {{{4, scalar}, { 6, avx}, {19, avx2}, {avx512}}}},
  {zen5,          tid<_Float16>(),{{{scalar}                                    }}},
  {zen6,          tid<float>(),   {{{4, scalar}, {16, avx}, {avx2}              }}},
  {zen6,          tid<double>(),  {{{4, scalar}, { 6, avx}, {19, avx2}, {avx512}}}},
  {zen6,          tid<_Float16>(),{{{8, scalar}, {16, avx}, {32, avx2}, {avx512}}}},
  {generic_avx512,tid<float>(),   {{{4, scalar}, {16, avx}, {avx2}              }}},
  {generic_avx512,tid<double>(),  {{{4, scalar}, { 6, avx}, {19, avx2}, {avx512}}}},
  {generic_avx512,tid<_Float16>(),{{{8, scalar}, {16, avx}, {32, avx2}, {avx512}}}}
}};

// clang-format on

} // namespace da_kmeans
#endif // KMEANS_TUNING_TABLES_HPP