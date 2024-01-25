# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
Contains the functions to replace symbols from scikit-learn by the AOCL-DA monkey patch
"""

import sklearn.decomposition as decomp_sklearn
import sklearn.linear_model as linmod_sklearn
from ._pca import PCA as PCA_da
from ._linear_model import LinearRegression as LinearRegression_da
from ._linear_model import Ridge as Ridge_da
from ._linear_model import Lasso as Lasso_da

# Global map of the sklearn symbols to replace
# key: class name
# value: dict containing the sklearn symbols and their replacements
#        pack - sklearn subpackage
#        sk_sym - name of the sklearn symbol to replace
#        da_sym - equivalent name in the DA sklearn lib
SYMBOLS = {'PCA': {'pack': decomp_sklearn,
                   'sk_sym': getattr(decomp_sklearn, "PCA"),
                   'da_sym': PCA_da},
           'LinearRegression': {'pack': linmod_sklearn,
                                'sk_sym': getattr(linmod_sklearn, 'LinearRegression'),
                                'da_sym': LinearRegression_da},
           'Ridge': {'pack': linmod_sklearn,
                     'sk_sym': getattr(linmod_sklearn, 'Ridge'),
                     'da_sym': Ridge_da},
           'Lasso': {'pack': linmod_sklearn,
                     'sk_sym': getattr(linmod_sklearn, 'Lasso'),
                     'da_sym': Lasso_da}}


def skpatch():
    """
    Replace all sklearn packages listed in SYMBOLS by their DA equivalent
    """
    for method, sym in SYMBOLS.items():
        pack = sym['pack']
        da_sym = sym['da_sym']
        setattr(pack, method, da_sym)


def undo_skpatch():
    """
    Reinstore sklearn packages with their original symbols
    """
    for method, sym in SYMBOLS.items():
        pack = sym['pack']
        sk_sym = sym['sk_sym']
        setattr(pack, method, sk_sym)
