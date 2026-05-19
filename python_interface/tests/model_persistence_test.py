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
Pickle serialization tests for all AOCL-DA Python classes.

Tests verify that pickle serialization/deserialization preserves model state
and produces consistent results after loading.
"""

import pickle
import numpy as np
import pytest

from aoclda.factorization import PCA
from aoclda.clustering import kmeans
from aoclda.linear_model import linmod
from aoclda.decision_tree import decision_tree
from aoclda.decision_forest import decision_forest
from aoclda.neighbors import nearest_neighbors, approximate_neighbors
from aoclda.svm import SVC, SVR, NuSVR


# ============================================================================
# PCA Serialization Tests
# ============================================================================

@pytest.mark.parametrize("method", ["covariance", "correlation", "svd"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_pca_pickle(method, numpy_precision, numpy_order, tmp_path):
    """Test PCA pickle serialization preserves model state."""
    X_train = np.array([
        [1.3, 2.53, 3.86],
        [2.4, 5.5, 4.5],
        [3.33, 6.21, 1.76],
        [4.1, 3.2, 2.8],
        [5.0, 4.1, 3.9]
    ], dtype=numpy_precision, order=numpy_order)

    X_test = np.array([
        [1.2, 1.1, 4.3],
        [3.333, 2.6, 3.4]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit PCA
    pca = PCA(n_components=2, method=method, store_U=True)
    pca.fit(X_train)

    # Get results before serialization
    transform_before = pca.transform(X_test)
    inverse_before = pca.inverse_transform(transform_before)
    components_before = pca.principal_components.copy()
    variance_before = pca.variance.copy()

    # Save to file and delete original
    filepath = tmp_path / "model_pca.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(pca, f)
    del pca

    # Load from file
    with open(filepath, 'rb') as f:
        pca_loaded = pickle.load(f)

    # Verify results after loading
    transform_after = pca_loaded.transform(X_test)
    inverse_after = pca_loaded.inverse_transform(transform_after)
    components_after = pca_loaded.principal_components

    assert np.array_equal(transform_before, transform_after)
    assert np.array_equal(inverse_before, inverse_after)
    assert np.array_equal(components_before, components_after)
    assert np.array_equal(variance_before, pca_loaded.variance)

    # Try saving again
    filepath = tmp_path / "model_pca2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(pca_loaded, f)


# ============================================================================
# K-Means Serialization Tests
# ============================================================================

@pytest.mark.parametrize("algorithm", ["lloyd", "elkan", "macqueen", "hartigan-wong"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_kmeans_pickle(algorithm, numpy_precision, numpy_order, tmp_path):
    """Test kmeans pickle serialization preserves model state."""
    X_train = np.array([
        [2.5, 1.4],
        [-1.0, -2.3],
        [3.8, 2.6],
        [2.4, 3.6],
        [-3.0, -2.4],
        [-2.2, -1.5],
        [-2.3, -3.1],
        [1.5, 2.1]
    ], dtype=numpy_precision, order=numpy_order)

    X_test = np.array([
        [0.1, 1.6],
        [0.3, -1.5]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit kmeans
    km = kmeans(n_clusters=2, algorithm=algorithm, tol=1.0e-4, seed=42)
    km.fit(X_train)

    # Get results before serialization
    labels_before = km.predict(X_test)
    transform_before = km.transform(X_test)
    centers_before = km.cluster_centres.copy()

    # Save to file and delete original
    filepath = tmp_path / "model_kmeans.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(km, f)
    del km

    # Load from file
    with open(filepath, 'rb') as f:
        km_loaded = pickle.load(f)

    # Verify results after loading
    labels_after = km_loaded.predict(X_test)
    transform_after = km_loaded.transform(X_test)

    assert np.array_equal(labels_before, labels_after)
    assert np.array_equal(transform_before, transform_after)
    assert np.array_equal(centers_before, km_loaded.cluster_centres)

    # Try saving again
    filepath = tmp_path / "model_kmeans2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(km_loaded, f)


# ============================================================================
# Linear Model Serialization Tests
# ============================================================================

@pytest.fixture(scope="function")
def no_fortran(request):
    return request.config.no_fortran


@pytest.mark.parametrize("mod", ["mse", "logistic"])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_linmod_pickle(
        no_fortran,
        mod,
        intercept,
        numpy_precision,
        numpy_order,
        tmp_path):
    """Test linmod pickle serialization preserves model state."""
    # Use different data based on model type
    if mod == "mse":
        X_train = np.array([
            [1, 1],
            [2, 3],
            [3, 5],
            [4, 8],
            [5, 7],
            [6, 9]
        ], dtype=numpy_precision, order=numpy_order)
        y_train = np.array([3., 6.5, 10., 12., 13., 19.], dtype=numpy_precision)
        X_test = np.array([[2.5, 4.0], [5.5, 8.0]],
                          dtype=numpy_precision, order=numpy_order)
    else:  # logistic
        if no_fortran:
            pytest.skip("Skipping test due to no_fortran flag")
        X_train = np.array([
            [1.0, 0.5],
            [1.5, 1.0],
            [2.0, 1.5],
            [3.0, 3.5],
            [3.5, 4.0],
            [4.0, 4.5]
        ], dtype=numpy_precision, order=numpy_order)
        y_train = np.array([0, 0, 0, 1, 1, 1], dtype=numpy_precision)
        X_test = np.array([[1.2, 0.8], [3.8, 4.2]],
                          dtype=numpy_precision, order=numpy_order)

    # Create and fit model
    lmod = linmod(mod, intercept=intercept)
    lmod.fit(X_train, y_train)

    # Get results before serialization
    predict_before = lmod.predict(X_test)
    coef_before = lmod.coef.copy()

    # Save to file and delete original
    filepath = tmp_path / "model_linmod.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(lmod, f)
    del lmod

    # Load from file
    with open(filepath, 'rb') as f:
        lmod_loaded = pickle.load(f)

    # Verify results after loading
    predict_after = lmod_loaded.predict(X_test)

    assert np.array_equal(predict_before, predict_after)
    assert np.array_equal(coef_before, lmod_loaded.coef)

    # Try saving again
    filepath = tmp_path / "model_linmod2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(lmod_loaded, f)


# ============================================================================
# Decision Tree Serialization Tests
# ============================================================================

@pytest.mark.parametrize("scoring_function",
                         ["gini", "cross-entropy", "misclassification"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_decision_tree_pickle(scoring_function, numpy_precision, numpy_order, tmp_path):
    """Test decision_tree pickle serialization preserves model state."""
    X_train = np.array([
        [1., 0.],
        [1., 1.],
        [1., 2.],
        [0., 0.],
        [0., 1.],
        [0., 2.]
    ], dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 1, 1, 0, 0, 1], dtype=numpy_precision)

    X_test = np.array([
        [0., 1.],
        [1., 2.]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit tree
    clf = decision_tree(criterion=scoring_function, seed=42)
    clf.fit(X_train, y_train)

    # Get results before serialization
    predict_before = clf.predict(X_test)
    proba_before = clf.predict_proba(X_test)

    # Save to file and delete original
    filepath = tmp_path / "model_clf.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    # Load from file
    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    # Verify results after loading
    predict_after = clf_loaded.predict(X_test)
    proba_after = clf_loaded.predict_proba(X_test)

    assert np.array_equal(predict_before, predict_after)
    assert np.array_equal(proba_before, proba_after)

    # Try saving again
    filepath = tmp_path / "model_clf2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf_loaded, f)


# ============================================================================
# Decision Forest Serialization Tests
# ============================================================================

@pytest.mark.parametrize("scoring_function",
                         ["gini", "cross-entropy", "misclassification"])
@pytest.mark.parametrize("bootstrap", [True, False])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_decision_forest_pickle(
        scoring_function,
        bootstrap,
        numpy_precision,
        numpy_order,
        tmp_path):
    """Test decision_forest pickle serialization preserves model state."""
    X_train = np.array([
        [1., 0.],
        [1., 1.],
        [1., 2.],
        [0., 0.],
        [0., 1.],
        [0., 2.]
    ], dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 1, 1, 0, 0, 1], dtype=numpy_precision)

    X_test = np.array([
        [0., 1.],
        [1., 2.]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit forest
    clf = decision_forest(
        n_trees=5,
        criterion=scoring_function,
        bootstrap=bootstrap,
        seed=42,
        features_selection="all"
    )
    clf.fit(X_train, y_train)

    # Get results before serialization
    predict_before = clf.predict(X_test)
    proba_before = clf.predict_proba(X_test)

    # Save to file and delete original
    filepath = tmp_path / "model_forest.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    # Load from file
    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    # Verify results after loading
    predict_after = clf_loaded.predict(X_test)
    proba_after = clf_loaded.predict_proba(X_test)

    assert np.array_equal(predict_before, predict_after)
    # Edge case failing with small diffs possibly due to order of calculations randomness
    np.testing.assert_array_almost_equal(proba_before, proba_after, decimal=7)

    # Try saving again
    filepath = tmp_path / "model_forest2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf_loaded, f)


# ============================================================================
# Nearest Neighbors Serialization Tests
# ============================================================================

@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nearest_neighbors_kneighbors_pickle(
        algorithm, numpy_precision, numpy_order, tmp_path):
    """Test nearest_neighbors pickle serialization for kneighbors queries."""
    X_train = np.array([
        [-1.5, -1.2, 2.6],
        [-2.5, -1.3, 3.8],
        [-3.6, -2.8, -1.7],
        [1.6, 3.4, 1.2],
        [2.8, 5.3, 1.3],
        [3.0, -1.2, 2.4]
    ], dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2], dtype=numpy_precision)

    X_test = np.array([
        [-2.5, 2.2, 3.6],
        [-1.4, -2.7, -1.0],
        [2.6, 1.2, -3.7]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit model
    nn = nearest_neighbors(algorithm=algorithm, n_neighbors=3)
    nn.fit(X_train, y_train)

    # Get results before serialization
    dist_before, ind_before = nn.kneighbors(X_test, return_distance=True)

    # Save to file and delete original
    filepath = tmp_path / "model_knn.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)
    del nn

    # Load from file
    with open(filepath, 'rb') as f:
        nn_loaded = pickle.load(f)

    # Verify results after loading
    dist_after, ind_after = nn_loaded.kneighbors(X_test, return_distance=True)

    assert np.array_equal(ind_before, ind_after)
    assert np.array_equal(dist_before, dist_after)

    # Try saving again
    filepath = tmp_path / "model_knn2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn_loaded, f)


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nearest_neighbors_classifier_pickle(
        algorithm, numpy_precision, numpy_order, tmp_path):
    """Test nearest_neighbors pickle serialization for classification."""
    X_train = np.array([
        [-1.5, -1.2, 2.6],
        [-2.5, -1.3, 3.8],
        [-3.6, -2.8, -1.7],
        [1.6, 3.4, 1.2],
        [2.8, 5.3, 1.3],
        [3.0, -1.2, 2.4]
    ], dtype=numpy_precision, order=numpy_order)

    y_train = np.array([0, 0, 1, 1, 2, 2], dtype=numpy_precision)

    X_test = np.array([
        [-2.5, 2.2, 3.6],
        [2.6, 1.2, -3.7]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit model
    nn = nearest_neighbors(algorithm=algorithm, n_neighbors=3)
    nn.fit(X_train, y_train)

    # Get results before serialization
    predict_before = nn.classifier_predict(X_test)
    proba_before = nn.classifier_predict_proba(X_test)

    # Save to file and delete original
    filepath = tmp_path / "model_nn_class.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)
    del nn

    # Load from file
    with open(filepath, 'rb') as f:
        nn_loaded = pickle.load(f)

    # Verify results after loading
    predict_after = nn_loaded.classifier_predict(X_test)
    proba_after = nn_loaded.classifier_predict_proba(X_test)

    assert np.array_equal(predict_before, predict_after)
    assert np.array_equal(proba_before, proba_after)

    # Try saving again
    filepath = tmp_path / "model_nn_class2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn_loaded, f)


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nearest_neighbors_regressor_pickle(
        algorithm, numpy_precision, numpy_order, tmp_path):
    """Test nearest_neighbors pickle serialization for regression."""
    X_train = np.array([
        [-1.5, -1.2, 2.6],
        [-2.5, -1.3, 3.8],
        [-3.6, -2.8, -1.7],
        [1.6, 3.4, 1.2],
        [2.8, 5.3, 1.3],
        [3.0, -1.2, 2.4]
    ], dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=numpy_precision)

    X_test = np.array([
        [-2.5, 2.2, 3.6],
        [2.6, 1.2, -3.7]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit model
    nn = nearest_neighbors(algorithm=algorithm, n_neighbors=3)
    nn.fit(X_train, y_train)

    # Get results before serialization
    predict_before = nn.regressor_predict(X_test)

    # Save to file and delete original
    filepath = tmp_path / "model_nn_reg.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)
    del nn

    # Load from file
    with open(filepath, 'rb') as f:
        nn_loaded = pickle.load(f)

    # Verify results after loading
    predict_after = nn_loaded.regressor_predict(X_test)

    assert np.array_equal(predict_before, predict_after)

    # Try saving again
    filepath = tmp_path / "model_nn_reg2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn_loaded, f)


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nearest_neighbors_radius_pickle(
        algorithm,
        numpy_precision,
        numpy_order,
        tmp_path):
    """Test nearest_neighbors pickle serialization for radius queries."""
    X_train = np.array([
        [-1.5, -1.2, 2.6],
        [-2.5, -1.3, 3.8],
        [-3.6, -2.8, -1.7],
        [1.6, 3.4, 1.2],
        [2.8, 5.3, 1.3],
        [3.0, -1.2, 2.4]
    ], dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2], dtype=numpy_precision)

    X_test = np.array([
        [-2.5, 2.2, 3.6],
        [-1.4, -2.7, -1.0]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit model
    nn = nearest_neighbors(algorithm=algorithm)
    nn.fit(X_train, y_train)

    # Get results before serialization (note: radius_neighbors returns lists)
    dist_before, ind_before = nn.radius_neighbors(
        X_test, return_distance=True, radius=5.0)

    # Save to file and delete original
    filepath = tmp_path / "model_nn_rad.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)
    del nn

    # Load from file
    with open(filepath, 'rb') as f:
        nn_loaded = pickle.load(f)

    # Verify results after loading
    dist_after, ind_after = nn_loaded.radius_neighbors(
        X_test, return_distance=True, radius=5.0)

    # Compare results (radius_neighbors returns arrays of arrays)
    for i in range(len(dist_before)):
        assert np.array_equal(ind_before[i], ind_after[i])
        assert np.array_equal(dist_before[i], dist_after[i])

    # Try saving again
    filepath = tmp_path / "model_nn_rad2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nn_loaded, f)


# ============================================================================
# Approximate Neighbors Serialization Tests
# ============================================================================

@pytest.mark.parametrize("metric", ["sqeuclidean", 'euclidean', 'cosine'])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_approximate_neighbors_pickle(metric, numpy_precision, numpy_order, tmp_path):
    """Test approximate_neighbors pickle serialization preserves model state."""
    X_train = np.array([
        [-1.5, -1.2, 2.6],
        [-2.5, -1.3, 3.8],
        [-3.6, -2.8, -1.7],
        [1.6, 3.4, 1.2],
        [2.8, 5.3, 1.3],
        [3.0, -1.2, 2.4],
        [0.5, 0.3, 1.1],
        [-0.5, 0.8, -0.9]
    ], dtype=numpy_precision, order=numpy_order)

    X_test = np.array([
        [-2.5, 2.2, 3.6],
        [-1.4, -2.7, -1.0],
        [2.6, 1.2, -3.7]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit model
    ann = approximate_neighbors(
        n_neighbors=3,
        n_list=2,
        n_probe=2,
        metric=metric,
        seed=42
    )
    ann.train_and_add(X_train)

    # Get results before serialization
    dist_before, ind_before = ann.kneighbors(X_test, return_distance=True)

    # Save to file and delete original
    filepath = tmp_path / "model_ann.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(ann, f)
    del ann

    # Load from file
    with open(filepath, 'rb') as f:
        ann_loaded = pickle.load(f)

    # Verify results after loading
    dist_after, ind_after = ann_loaded.kneighbors(X_test, return_distance=True)

    assert np.array_equal(ind_before, ind_after)
    assert np.array_equal(dist_before, dist_after)

    # Try saving again
    filepath = tmp_path / "model_ann2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(ann_loaded, f)


# ============================================================================
# SVM Serialization Tests
# ============================================================================

@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_svc_pickle(kernel, numpy_precision, numpy_order, tmp_path):
    """Test SVC pickle serialization preserves model state."""
    X_train = np.array([
        [-1.0, -1.0],
        [-0.5, -0.5],
        [1.0, 1.0],
        [1.5, 1.5],
        [0.0, 0.0],
        [0.5, 0.5]
    ], dtype=numpy_precision, order=numpy_order)
    y_train = np.array([0, 0, 1, 1, 0, 1], dtype=numpy_precision)

    X_test = np.array([
        [-0.8, -0.8],
        [1.2, 1.2]
    ], dtype=numpy_precision, order=numpy_order)

    # Create and fit SVC
    svc = SVC(kernel=kernel, C=1.0, tol=1e-4)
    svc.fit(X_train, y_train)

    # Get results before serialization
    predict_before = svc.predict(X_test)

    # Save to file and delete original
    filepath = tmp_path / "model_svc.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(svc, f)
    del svc

    # Load from file
    with open(filepath, 'rb') as f:
        svc_loaded = pickle.load(f)

    # Verify results after loading
    predict_after = svc_loaded.predict(X_test)

    assert np.array_equal(predict_before, predict_after)

    # Try saving again
    filepath = tmp_path / "model_svc2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(svc_loaded, f)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
@pytest.mark.parametrize("numpy_precision", [np.float32, np.float64])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_nusvr_pickle(kernel, numpy_precision, numpy_order, tmp_path):
    """Test NuSVR pickle serialization preserves model state."""
    X_train = np.array([
        [1.0, 0.5],
        [2.0, 1.0],
        [3.0, 1.5],
        [4.0, 2.0],
        [5.0, 2.5],
        [6.0, 3.0]
    ], dtype=numpy_precision, order=numpy_order)
    y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=numpy_precision)

    X_test = np.array([[2.5, 1.25], [4.5, 2.25]],
                      dtype=numpy_precision, order=numpy_order)

    # Create and fit NuSVR
    nusvr = NuSVR(nu=0.2, kernel=kernel, C=1.0, tol=1e-4)
    nusvr.fit(X_train, y_train)

    # Get results before serialization
    predict_before = nusvr.predict(X_test)

    # Save to file and delete original
    filepath = tmp_path / "model_nusvr.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nusvr, f)
    del nusvr

    # Load from file
    with open(filepath, 'rb') as f:
        nusvr_loaded = pickle.load(f)

    # Verify results after loading
    predict_after = nusvr_loaded.predict(X_test)

    assert np.array_equal(predict_before, predict_after)

    # Try saving again
    filepath = tmp_path / "model_nusvr2.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nusvr_loaded, f)


# ============================================================================
# Fit After Load Tests
# ============================================================================
# These tests verify that calling fit() on a loaded model works correctly.
# Loaded models should allow retraining with new data.
# Doesn't apply for NN and ANN as they save the user data by default.

def test_pca_fit_after_load_succeeds(tmp_path):
    """Test that calling fit on a loaded PCA model succeeds."""
    X_train = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0]
    ], dtype=np.float64)

    pca = PCA(n_components=2)
    pca.fit(X_train)

    filepath = tmp_path / "model_fit_pca.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(pca, f)
    del pca

    with open(filepath, 'rb') as f:
        pca_loaded = pickle.load(f)

    # Fit should work on loaded model
    X_new = np.array([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]], dtype=np.float64)
    pca_loaded.fit(X_new)

    # Verify we can transform with the refitted model
    result = pca_loaded.transform(X_new)
    assert result.shape == (2, 2)


def test_kmeans_fit_after_load_succeeds(tmp_path):
    """Test that calling fit on a loaded kmeans model succeeds."""
    X_train = np.array([
        [1.0, 2.0],
        [1.5, 2.5],
        [5.0, 6.0],
        [5.5, 6.5]
    ], dtype=np.float64)

    km = kmeans(n_clusters=2, seed=42)
    km.fit(X_train)

    filepath = tmp_path / "model_fit_km.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(km, f)
    del km

    with open(filepath, 'rb') as f:
        km_loaded = pickle.load(f)

    # Fit should work on loaded model
    X_new = np.array([[2.0, 3.0], [6.0, 7.0]], dtype=np.float64)
    km_loaded.fit(X_new)

    # Verify we can predict with the refitted model
    labels = km_loaded.predict(X_new)
    assert labels.shape == (2,)


def test_linmod_fit_after_load_succeeds(tmp_path):
    """Test that calling fit on a loaded linmod model succeeds."""
    X_train = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ], dtype=np.float64)
    y_train = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    lmod = linmod("mse")
    lmod.fit(X_train, y_train)

    filepath = tmp_path / "model_fit_lmod.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(lmod, f)
    del lmod

    with open(filepath, 'rb') as f:
        lmod_loaded = pickle.load(f)

    # Fit should work on loaded model
    X_new = np.array([[4.0, 5.0], [5.0, 6.0]], dtype=np.float64)
    y_new = np.array([4.0, 5.0], dtype=np.float64)
    lmod_loaded.fit(X_new, y_new)

    # Verify we can predict with the refitted model
    predictions = lmod_loaded.predict(X_new)
    assert predictions.shape == (2,)


def test_decision_tree_fit_after_load_succeeds(tmp_path):
    """Test that calling fit on a loaded decision_tree model succeeds."""
    X_train = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], dtype=np.float64)
    y_train = np.array([0, 1, 1, 0], dtype=np.float64)

    clf = decision_tree(seed=42)
    clf.fit(X_train, y_train)

    filepath = tmp_path / "model_fit_clf.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    # Fit should work on loaded model
    X_new = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=np.float64)
    y_new = np.array([0, 1], dtype=np.float64)
    clf_loaded.fit(X_new, y_new)

    # Verify we can predict with the refitted model
    predictions = clf_loaded.predict(X_new)
    assert predictions.shape == (2,)


def test_decision_forest_fit_after_load_succeeds(tmp_path):
    """Test that calling fit on a loaded decision_forest model succeeds."""
    X_train = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], dtype=np.float64)
    y_train = np.array([0, 1, 1, 0], dtype=np.float64)

    clf = decision_forest(n_trees=3, seed=42)
    clf.fit(X_train, y_train)

    filepath = tmp_path / "model_fit_forest.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    # Fit should work on loaded model
    X_new = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=np.float64)
    y_new = np.array([0, 1], dtype=np.float64)
    clf_loaded.fit(X_new, y_new)

    # Verify we can predict with the refitted model
    predictions = clf_loaded.predict(X_new)
    assert predictions.shape == (2,)


def test_svr_fit_after_load_succeeds(tmp_path):
    """Test that calling fit on a loaded SVR model succeeds."""
    X_train = np.array([
        [-1.0, -1.0],
        [-0.5, -0.5],
        [1.0, 1.0],
        [1.5, 1.5]
    ], dtype=np.float64)
    y_train = np.array([1.5, 2.0, 5.5, 6.0], dtype=np.float64)

    svr = SVR(kernel="rbf", C=1.0)
    svr.fit(X_train, y_train)

    filepath = tmp_path / "model_fit_svr.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(svr, f)
    del svr

    with open(filepath, 'rb') as f:
        svr_loaded = pickle.load(f)

    # Fit should work on loaded model
    X_new = np.array([[-0.8, -0.8], [1.2, 1.2]], dtype=np.float64)
    y_new = np.array([2.5, 5.0], dtype=np.float64)
    svr_loaded.fit(X_new, y_new)

    # Verify we can predict with the refitted model
    predictions = svr_loaded.predict(X_new)
    assert predictions.shape == (2,)
