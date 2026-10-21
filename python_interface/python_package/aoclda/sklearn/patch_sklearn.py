# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# pylint: disable = possibly-used-before-assignment, duplicate-code

import sklearn.covariance as cov_sklearn
import sklearn.decomposition as decomp_sklearn
import sklearn.linear_model as linmod_sklearn
import sklearn.cluster as clustering_sklearn
import sklearn.manifold as manifold_sklearn
import sklearn.tree as decision_tree_sklearn
import sklearn.ensemble as decision_forest_sklearn
import sklearn.metrics.pairwise as pairwise_sklearn
import sklearn.model_selection as model_selection_sklearn
import sklearn.neighbors as nearest_neighbors_sklearn
import sklearn.svm as svm_sklearn
from .._patching import register_patch, requested_patches
from ._empirical_covariance import EmpiricalCovariance as cov_da
from ._pca import PCA as PCA_da
from ._kernel_pca import KernelPCA as KernelPCA_da
from ._nmf import NMF as NMF_da
from ._kmeans import kmeans as kmeans_da
from ._dbscan import DBSCAN as DBSCAN_da
from ._linear_model import LinearRegression as LinearRegression_da
from ._linear_model import Ridge as Ridge_da
from ._linear_model import Lasso as Lasso_da
from ._linear_model import ElasticNet as ElasticNet_da
from ._linear_model import LogisticRegression as LogisticRegression_da
from ._decision_tree import DecisionTreeClassifier as DecisionTreeClassifier_da
from ._decision_forest import RandomForestClassifier as RandomForestClassifier_da
from ._metrics import pairwise_distances as pairwise_distances_da
from ._utils import train_test_split as train_test_split_da
from ._neighbors import KNeighborsClassifier as KNeighborsClassifier_da
from ._neighbors import KNeighborsRegressor as KNeighborsRegressor_da
from ._neighbors import NearestNeighbors as NearestNeighbors_da
from ._neighbors import RadiusNeighborsClassifier as RadiusNeighborsClassifier_da
from ._neighbors import RadiusNeighborsRegressor as RadiusNeighborsRegressor_da
from ._svm import SVC as SVC_da, SVR as SVR_da, NuSVC as NuSVC_da, NuSVR as NuSVR_da
from ._tsne import TSNE as TSNE_da

# Now on a case-by-case basis, overwrite with AMD symbols where we have
# performant implementations

# Global map of the sklearn symbols which have AMD implementations
# key: class name
# value: dict containing the sklearn symbols and their replacements
#        pack - sklearn subpackage
#        orig_sym - original sklearn symbol to restore
#        da_sym - equivalent name in the DA sklearn lib
AMD_SYMBOLS = {}
UNAVAILABLE_SYMBOLS = {}

_PATCHES = (
    ('EmpiricalCovariance', cov_sklearn, cov_da),
    ('PCA', decomp_sklearn, PCA_da),
    ('KernelPCA', decomp_sklearn, KernelPCA_da),
    ('NMF', decomp_sklearn, NMF_da),
    ('LinearRegression', linmod_sklearn, LinearRegression_da),
    ('Ridge', linmod_sklearn, Ridge_da),
    ('Lasso', linmod_sklearn, Lasso_da),
    ('ElasticNet', linmod_sklearn, ElasticNet_da),
    ('LogisticRegression', linmod_sklearn, LogisticRegression_da),
    ('KMeans', clustering_sklearn, kmeans_da),
    ('DBSCAN', clustering_sklearn, DBSCAN_da),
    ('TSNE', manifold_sklearn, TSNE_da),
    ('DecisionTreeClassifier', decision_tree_sklearn,
     DecisionTreeClassifier_da),
    ('RandomForestClassifier', decision_forest_sklearn,
     RandomForestClassifier_da),
    ('pairwise_distances', pairwise_sklearn, pairwise_distances_da),
    ('train_test_split', model_selection_sklearn, train_test_split_da),
    ('KNeighborsClassifier', nearest_neighbors_sklearn,
     KNeighborsClassifier_da),
    ('KNeighborsRegressor', nearest_neighbors_sklearn,
     KNeighborsRegressor_da),
    ('NearestNeighbors', nearest_neighbors_sklearn, NearestNeighbors_da),
    ('RadiusNeighborsClassifier', nearest_neighbors_sklearn,
     RadiusNeighborsClassifier_da),
    ('RadiusNeighborsRegressor', nearest_neighbors_sklearn,
     RadiusNeighborsRegressor_da),
    ('SVC', svm_sklearn, SVC_da),
    ('SVR', svm_sklearn, SVR_da),
    ('NuSVC', svm_sklearn, NuSVC_da),
    ('NuSVR', svm_sklearn, NuSVR_da),
)

for _name, _package, _da_symbol in _PATCHES:
    register_patch(AMD_SYMBOLS, UNAVAILABLE_SYMBOLS,
                   _name, _package, _da_symbol)


def skpatch(*args, print_patched=True):
    """
    Replace specified scikit-learn packages by their AOCL-DA equivalent
    """

    packages = requested_patches(args, AMD_SYMBOLS, UNAVAILABLE_SYMBOLS)

    # packages is a list of candidate package names to patch

    successfully_patched = []

    for package in packages:

        pack = AMD_SYMBOLS[package]['pack']
        sym = AMD_SYMBOLS[package]['da_sym']
        setattr(pack, package, sym)
        successfully_patched.append(package)

    if successfully_patched and print_patched:
        print(
            "AOCL Extension for scikit-learn enabled for the following packages:"
        )
        print(', '.join(successfully_patched))


def undo_skpatch(*args, print_patched=True):
    """
    Reinstate scikit-learn packages with their original symbols
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
            "AOCL Extension for scikit-learn disabled for the following packages:"
        )
        print(', '.join(successfully_unpatched))
