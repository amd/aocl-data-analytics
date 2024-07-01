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
Contains the functions to replace symbols from Scikit-learn by the AOCL-DA patch
"""
# pylint: disable = possibly-used-before-assignment

import os
import warnings
import contextlib
import sklearn.decomposition as decomp_sklearn
import sklearn.linear_model as linmod_sklearn
import sklearn.cluster as clustering_sklearn
import sklearn.tree as decision_tree_sklearn
import sklearn.ensemble as decision_forest_sklearn
import sklearn.metrics.pairwise as pairwise_sklearn
from ._pca import PCA as PCA_da
from ._kmeans import kmeans as kmeans_da
from ._linear_model import LinearRegression as LinearRegression_da
from ._linear_model import Ridge as Ridge_da
from ._linear_model import Lasso as Lasso_da
from ._linear_model import ElasticNet as ElasticNet_da
from ._linear_model import LogisticRegression as LogisticRegression_da
from ._decision_tree import DecisionTreeClassifier as DecisionTreeClassifier_da
from ._decision_forest import RandomForestClassifier as RandomForestClassifier_da
from ._metrics import pairwise_distances as pairwise_distances_da
# Check if we should be using Intel's Scikit-learn extension
try:
    USE_INTEL_SKLEARNEX = int(os.environ.get('USE_INTEL_SKLEARNEX'))
except (ValueError, TypeError):
    USE_INTEL_SKLEARNEX = 0

using_intel = False
if USE_INTEL_SKLEARNEX:
    try:
        from sklearnex import patch_sklearn, unpatch_sklearn
        using_intel = True
    except ImportError:
        warnings.warn(
            "Intel Extension for Scikit-learn not found", category=RuntimeWarning)

# Now on a case-by-case basis, overwrite with AMD symbols where we have performant implementations

# Global map of the sklearn symbols which have AMD implementations
# key: class name
# value: dict containing the sklearn symbols and their replacements
#        pack - sklearn subpackage
#        sk_sym - name of the sklearn symbol to replace
#        da_sym - equivalent name in the DA sklearn lib
AMD_SYMBOLS = {'PCA': {'pack': decomp_sklearn,
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
                         'da_sym': Lasso_da},
               'ElasticNet': {'pack': linmod_sklearn,
                         'sk_sym': getattr(linmod_sklearn, 'ElasticNet'),
                         'da_sym': ElasticNet_da},
               'LogisticRegression': {'pack': linmod_sklearn,
                                      'sk_sym': getattr(linmod_sklearn, 'LogisticRegression'),
                                      'da_sym': LogisticRegression_da},
               'KMeans': {'pack': clustering_sklearn,
                          'sk_sym': getattr(clustering_sklearn, "KMeans"),
                          'da_sym': kmeans_da},
               'DecisionTreeClassifier': {'pack': decision_tree_sklearn,
                                          'sk_sym': getattr(decision_tree_sklearn, 'DecisionTreeClassifier'),
                                          'da_sym': DecisionTreeClassifier_da},
               'RandomForestClassifier': {'pack': decision_forest_sklearn,
                                          'sk_sym': getattr(decision_forest_sklearn, 'RandomForestClassifier'),
                                          'da_sym': RandomForestClassifier_da},
                'pairwise_distances': {'pack': pairwise_sklearn,
                                'sk_sym': getattr(pairwise_sklearn, "pairwise_distances"),
                                'da_sym': pairwise_distances_da},
               }

# List of symbols where AMD is chosen over Intel
AMD_vs_INTEL = ['KMeans', 'LinearRegression', 'Ridge', 'PCA', 'DecisionTreeClassifier']


def skpatch(*args, print_patched=True):
    """
    Replace specified sklearn packages by their AOCL-DA equivalent
    """

    if not args:
        # No arguments specified, so patch everything possible
        if using_intel:
            # Patch everything in Intel, but suppress printing to screen
            with open(os.devnull, 'w', encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    patch_sklearn()

        packages = AMD_SYMBOLS.keys()
    elif isinstance(args[0], str):
        packages = [args[0]]
    elif isinstance(args[0], (list, tuple)):
        packages = args[0]
    else:
        raise TypeError("Unrecognized argument")

    # packages is a list of candidate package names to patch

    if using_intel:
        # Check if we should patch with Intel
        tmp_list = []
        intel_patches = []
        for package in packages:
            if package in AMD_vs_INTEL:
                tmp_list.append(package)
            else:
                intel_patches.append(package)

        packages = tmp_list
        for package in intel_patches:
            try:
                with open(os.devnull, 'w', encoding="utf-8") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        patch_sklearn(package)
            except ValueError:
                print(f"The package {package} was not found.")

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
            "AOCL Extension for Scikit-learn enabled for the following packages:"
        )
        print(', '.join(successfully_patched))


def undo_skpatch(*args, print_patched=True):
    """
    Reinstate sklearn packages with their original symbols
    """

    if using_intel:
        # Unpatch anything that might have been patched with Intel, but suppress printing to screen
        with open(os.devnull, 'w', encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                unpatch_sklearn()

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
            "AOCL Extension for Scikit-learn disabled for the following packages:"
        )
        print(', '.join(successfully_unpatched))
