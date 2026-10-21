# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
"""
Contains the functions to replace symbols from SciPy by the AOCL-DA patch
"""
# pylint: disable = possibly-used-before-assignment, duplicate-code

import scipy.stats as stats_scipy
import scipy.optimize as optimize_scipy
import scipy.interpolate as interpolate_scipy
from .._patching import register_patch, requested_patches
from ._stats import describe as describe_da
from ._stats import gmean as gmean_da
from ._stats import hmean as hmean_da
from ._stats import kurtosis as kurtosis_da
from ._stats import moment as moment_da
from ._stats import skew as skew_da
from ._stats import quantile as quantile_da
from ._optimize import least_squares as least_squares_da
from ._interpolate import CubicSpline as CubicSpline_da
from ._interpolate import CubicHermiteSpline as CubicHermiteSpline_da

# Now on a case-by-case basis, overwrite with AMD symbols where we have
# performant implementations

# Global map of the scipy symbols which have AMD implementations
# key: function name
# value: dict containing the scipy symbols and their replacements
#        pack - scipy subpackage
#        orig_sym - original scipy symbol to restore
#        da_sym - equivalent name in the DA scipy lib
AMD_SYMBOLS = {}
UNAVAILABLE_SYMBOLS = {}

register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'describe', stats_scipy, describe_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'gmean', stats_scipy, gmean_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'hmean', stats_scipy, hmean_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'kurtosis', stats_scipy, kurtosis_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'moment', stats_scipy, moment_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'skew', stats_scipy, skew_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'quantile', stats_scipy, quantile_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'least_squares', optimize_scipy, least_squares_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'CubicSpline', interpolate_scipy, CubicSpline_da)
register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
               'CubicHermiteSpline', interpolate_scipy,
               CubicHermiteSpline_da)


def sppatch(*args, print_patched=True):
    """
    Replace specified SciPy functions by their AOCL-DA equivalent
    """

    packages = requested_patches(args, AMD_SYMBOLS, UNAVAILABLE_SYMBOLS)

    # packages is a list of candidate function names to patch

    successfully_patched = []

    for package in packages:

        pack = AMD_SYMBOLS[package]['pack']
        sym = AMD_SYMBOLS[package]['da_sym']
        setattr(pack, package, sym)
        successfully_patched.append(package)

    if successfully_patched and print_patched:
        print(
            "AOCL Extension for SciPy enabled for the following functions:"
        )
        print(', '.join(successfully_patched))


def undo_sppatch(*args, print_patched=True):
    """
    Reinstate SciPy functions with their original symbols
    """

    packages = requested_patches(args, AMD_SYMBOLS, UNAVAILABLE_SYMBOLS)

    successfully_unpatched = []

    for package in packages:
        pack = AMD_SYMBOLS[package]['pack']
        sym = AMD_SYMBOLS[package]['orig_sym']
        setattr(pack, package, sym)
        successfully_unpatched.append(package)

    if successfully_unpatched and print_patched:
        print(
            "AOCL Extension for SciPy disabled for the following functions:"
        )
        print(', '.join(successfully_unpatched))
