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
# pylint: disable = possibly-used-before-assignment

import scipy.stats as stats_scipy
import scipy.optimize as optimize_scipy
import scipy.interpolate as interpolate_scipy
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
#        sk_sym - name of the scipy symbol to replace
#        da_sym - equivalent name in the DA scipy lib
AMD_SYMBOLS = {'describe': {'pack': stats_scipy,
                            'sk_sym': getattr(stats_scipy, "describe"),
                            'da_sym': describe_da},
               'gmean': {'pack': stats_scipy,
                         'sk_sym': getattr(stats_scipy, "gmean"),
                         'da_sym': gmean_da},
               'hmean': {'pack': stats_scipy,
                         'sk_sym': getattr(stats_scipy, "hmean"),
                         'da_sym': hmean_da},
               'kurtosis': {'pack': stats_scipy,
                            'sk_sym': getattr(stats_scipy, "kurtosis"),
                            'da_sym': kurtosis_da},
               'moment': {'pack': stats_scipy,
                          'sk_sym': getattr(stats_scipy, "moment"),
                          'da_sym': moment_da},
               'skew': {'pack': stats_scipy,
                        'sk_sym': getattr(stats_scipy, "skew"),
                        'da_sym': skew_da},
               'least_squares': {'pack': optimize_scipy,
                                 'sk_sym': getattr(optimize_scipy, "least_squares"),
                                 'da_sym': least_squares_da},
               'CubicSpline': {'pack': interpolate_scipy,
                               'sk_sym': getattr(interpolate_scipy, "CubicSpline"),
                               'da_sym': CubicSpline_da},
               'CubicHermiteSpline': {'pack': interpolate_scipy,
                                      'sk_sym': getattr(interpolate_scipy,
                                                        "CubicHermiteSpline"),
                                      'da_sym': CubicHermiteSpline_da},
               }

if hasattr(stats_scipy, "quantile"):
    AMD_SYMBOLS['quantile'] = {'pack': stats_scipy,
                               'sk_sym': stats_scipy.quantile,
                               'da_sym': quantile_da}


def sppatch(*args, print_patched=True):
    """
    Replace specified SciPy functions by their AOCL-DA equivalent
    """

    if not args:
        # No arguments specified, so patch everything possible
        packages = AMD_SYMBOLS.keys()
    elif isinstance(args[0], str):
        packages = [args[0]]
    elif isinstance(args[0], (list, tuple)):
        packages = args[0]
    else:
        raise TypeError("Unrecognized argument")

    # packages is a list of candidate function names to patch

    successfully_patched = []

    for package in packages:

        try:
            pack = AMD_SYMBOLS[package]['pack']
            sym = AMD_SYMBOLS[package]['da_sym']
            setattr(pack, package, sym)
            successfully_patched.append(package)
        except KeyError:
            print(f"The package {package} was not found.")

    if successfully_patched and print_patched:
        print(
            "AOCL Extension for SciPy enabled for the following functions:"
        )
        print(', '.join(successfully_patched))


def undo_sppatch(*args, print_patched=True):
    """
    Reinstate SciPy functions with their original symbols
    """

    if not args:
        packages = AMD_SYMBOLS.keys()
    elif isinstance(args[0], str):
        packages = [args[0]]
    elif isinstance(args[0], (list, tuple)):
        packages = args[0]
    else:
        raise TypeError("Unrecognized argument")

    successfully_unpatched = []

    for package in packages:
        try:
            pack = AMD_SYMBOLS[package]['pack']
            sym = AMD_SYMBOLS[package]['sk_sym']
            setattr(pack, package, sym)
            successfully_unpatched.append(package)
        except KeyError:
            print(f"The package {package} was not found.")

    if successfully_unpatched and print_patched:
        print(
            "AOCL Extension for SciPy disabled for the following functions:"
        )
        print(', '.join(successfully_unpatched))
