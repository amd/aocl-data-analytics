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
Pickle serialization tests for sklearn API with AOCL-DA patches.

Tests verify that models saved with patches can be loaded
and used correctly in different patching states.
"""

# pylint: disable = import-outside-toplevel, reimported, no-member

import pickle
import numpy as np
from aoclda.sklearn import skpatch, undo_skpatch


# ============================================================================
# PCA Pickle Tests
# ============================================================================

def test_pca_pickle_with_patch(tmp_path):
    """Test PCA: save with patch, load with patch."""
    skpatch("PCA", print_patched=False)

    from sklearn.decomposition import PCA

    X_train = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    X_test = np.array([[1.5, 2.5, 3.5]])

    pca = PCA(n_components=2)
    assert pca.aocl is True
    pca.fit(X_train)
    pred_before = pca.transform(X_test)

    filepath = tmp_path / "pca_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(pca, f)
    del pca

    with open(filepath, 'rb') as f:
        pca_loaded = pickle.load(f)

    assert pca_loaded.aocl is True
    pred_after = pca_loaded.transform(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("PCA", print_patched=False)


def test_pca_pickle_cross_patch(tmp_path):
    """Test PCA: save with patch, load without patch."""
    skpatch("PCA", print_patched=False)

    from sklearn.decomposition import PCA

    X_train = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    X_test = np.array([[1.5, 2.5, 3.5]])

    pca = PCA(n_components=2)
    assert pca.aocl is True
    pca.fit(X_train)
    pred_before = pca.transform(X_test)

    filepath = tmp_path / "pca_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(pca, f)
    del pca

    undo_skpatch("PCA", print_patched=False)

    # Load without patch - should still work as it's an AOCL-DA object
    with open(filepath, 'rb') as f:
        pca_loaded = pickle.load(f)

    assert pca_loaded.aocl is True
    pred_after = pca_loaded.transform(X_test)
    assert np.array_equal(pred_before, pred_after)

# ============================================================================
# KMeans Pickle Tests
# ============================================================================


def test_kmeans_pickle_with_patch(tmp_path):
    """Test KMeans: save with patch, load with patch."""
    skpatch("KMeans", print_patched=False)

    from sklearn.cluster import KMeans

    X_train = np.array([[1.0, 2.0], [1.5, 2.5], [5.0, 6.0], [5.5, 6.5]])
    X_test = np.array([[2.0, 3.0], [6.0, 7.0]])

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    assert km.aocl is True
    km.fit(X_train)
    pred_before = km.predict(X_test)

    filepath = tmp_path / "kmeans_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(km, f)
    del km

    with open(filepath, 'rb') as f:
        km_loaded = pickle.load(f)

    assert km_loaded.aocl is True
    pred_after = km_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("KMeans", print_patched=False)


def test_kmeans_pickle_cross_patch(tmp_path):
    """Test KMeans: save with patch, load without patch."""
    skpatch("KMeans", print_patched=False)

    from sklearn.cluster import KMeans

    X_train = np.array([[1.0, 2.0], [1.5, 2.5], [5.0, 6.0], [5.5, 6.5]])
    X_test = np.array([[2.0, 3.0], [6.0, 7.0]])

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    assert km.aocl is True
    km.fit(X_train)
    pred_before = km.predict(X_test)

    filepath = tmp_path / "kmeans_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(km, f)
    del km

    undo_skpatch("KMeans", print_patched=False)

    with open(filepath, 'rb') as f:
        km_loaded = pickle.load(f)

    assert km_loaded.aocl is True
    pred_after = km_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)


# ============================================================================
# LinearRegression Pickle Tests
# ============================================================================

def test_linear_regression_pickle_with_patch(tmp_path):
    """Test LinearRegression: save with patch, load with patch."""
    skpatch("LinearRegression", print_patched=False)

    from sklearn.linear_model import LinearRegression

    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_train = np.array([1.0, 2.0, 3.0])
    X_test = np.array([[4.0, 5.0]])

    lr = LinearRegression()
    assert lr.aocl is True
    lr.fit(X_train, y_train)
    pred_before = lr.predict(X_test)

    filepath = tmp_path / "linreg_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(lr, f)
    del lr

    with open(filepath, 'rb') as f:
        lr_loaded = pickle.load(f)

    assert lr_loaded.aocl is True
    pred_after = lr_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("LinearRegression", print_patched=False)


def test_linear_regression_pickle_cross_patch(tmp_path):
    """Test LinearRegression: save with patch, load without patch."""
    skpatch("LinearRegression", print_patched=False)

    from sklearn.linear_model import LinearRegression

    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_train = np.array([1.0, 2.0, 3.0])
    X_test = np.array([[4.0, 5.0]])

    lr = LinearRegression()
    assert lr.aocl is True
    lr.fit(X_train, y_train)
    pred_before = lr.predict(X_test)

    filepath = tmp_path / "linreg_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(lr, f)
    del lr

    undo_skpatch("LinearRegression", print_patched=False)

    with open(filepath, 'rb') as f:
        lr_loaded = pickle.load(f)

    assert lr_loaded.aocl is True
    pred_after = lr_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)


# ============================================================================
# DecisionTreeClassifier Pickle Tests
# ============================================================================

def test_decision_tree_pickle_with_patch(tmp_path):
    """Test DecisionTreeClassifier: save with patch, load with patch."""
    skpatch("DecisionTreeClassifier", print_patched=False)

    from sklearn.tree import DecisionTreeClassifier

    X_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[0.5, 0.5]])

    clf = DecisionTreeClassifier(random_state=42)
    assert clf.aocl is True
    clf.fit(X_train, y_train)
    pred_before = clf.predict(X_test)

    filepath = tmp_path / "dtree_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    assert clf_loaded.aocl is True
    pred_after = clf_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("DecisionTreeClassifier", print_patched=False)


def test_decision_tree_pickle_cross_patch(tmp_path):
    """Test DecisionTreeClassifier: save with patch, load without patch."""
    skpatch("DecisionTreeClassifier", print_patched=False)

    from sklearn.tree import DecisionTreeClassifier

    X_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[0.5, 0.5]])

    clf = DecisionTreeClassifier(random_state=42)
    assert clf.aocl is True
    clf.fit(X_train, y_train)
    pred_before = clf.predict(X_test)

    filepath = tmp_path / "dtree_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    undo_skpatch("DecisionTreeClassifier", print_patched=False)

    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    assert clf_loaded.aocl is True
    pred_after = clf_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)


# ============================================================================
# RandomForestClassifier Pickle Tests
# ============================================================================

def test_random_forest_pickle_with_patch(tmp_path):
    """Test RandomForestClassifier: save with patch, load with patch."""
    skpatch("RandomForestClassifier", print_patched=False)

    from sklearn.ensemble import RandomForestClassifier

    X_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[0.5, 0.5]])

    clf = RandomForestClassifier(n_estimators=3, random_state=42)
    assert clf.aocl is True
    clf.fit(X_train, y_train)
    pred_before = clf.predict(X_test)

    filepath = tmp_path / "forest_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    assert clf_loaded.aocl is True
    pred_after = clf_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("RandomForestClassifier", print_patched=False)


def test_random_forest_pickle_cross_patch(tmp_path):
    """Test RandomForestClassifier: save with patch, load without patch."""
    skpatch("RandomForestClassifier", print_patched=False)

    from sklearn.ensemble import RandomForestClassifier

    X_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[0.5, 0.5]])

    clf = RandomForestClassifier(n_estimators=3, random_state=42)
    assert clf.aocl is True
    clf.fit(X_train, y_train)
    pred_before = clf.predict(X_test)

    filepath = tmp_path / "forest_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    del clf

    undo_skpatch("RandomForestClassifier", print_patched=False)

    with open(filepath, 'rb') as f:
        clf_loaded = pickle.load(f)

    assert clf_loaded.aocl is True
    pred_after = clf_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)


# ============================================================================
# KNeighborsClassifier Pickle Tests
# ============================================================================

def test_kneighbors_classifier_pickle_with_patch(tmp_path):
    """Test KNeighborsClassifier: save with patch, load with patch."""
    skpatch("KNeighborsClassifier", print_patched=False)

    from sklearn.neighbors import KNeighborsClassifier

    X_train = np.array([[-1.0, -1.0], [-0.5, -0.5], [1.0, 1.0], [1.5, 1.5]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.0, 0.0]])

    knn = KNeighborsClassifier(n_neighbors=3)
    assert knn.aocl is True
    knn.fit(X_train, y_train)
    pred_before = knn.predict(X_test)

    filepath = tmp_path / "knn_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(knn, f)
    del knn

    with open(filepath, 'rb') as f:
        knn_loaded = pickle.load(f)

    assert knn_loaded.aocl is True
    pred_after = knn_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("KNeighborsClassifier", print_patched=False)


def test_kneighbors_classifier_pickle_cross_patch(tmp_path):
    """Test KNeighborsClassifier: save with patch, load without patch."""
    skpatch("KNeighborsClassifier", print_patched=False)

    from sklearn.neighbors import KNeighborsClassifier

    X_train = np.array([[-1.0, -1.0], [-0.5, -0.5], [1.0, 1.0], [1.5, 1.5]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.0, 0.0]])

    knn = KNeighborsClassifier(n_neighbors=3)
    assert knn.aocl is True
    knn.fit(X_train, y_train)
    pred_before = knn.predict(X_test)

    filepath = tmp_path / "knn_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(knn, f)
    del knn

    undo_skpatch("KNeighborsClassifier", print_patched=False)

    with open(filepath, 'rb') as f:
        knn_loaded = pickle.load(f)

    assert knn_loaded.aocl is True
    pred_after = knn_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)


# ============================================================================
# NuSVC Pickle Tests
# ============================================================================

def test_nusvc_pickle_with_patch(tmp_path):
    """Test NuSVC: save with patch, load with patch."""
    skpatch("NuSVC", print_patched=False)

    from sklearn.svm import NuSVC

    X_train = np.array([[-1.0, -1.0], [-0.5, -0.5], [1.0, 1.0], [1.5, 1.5]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.0, 0.0]])

    nusvc = NuSVC(nu=0.5, kernel='rbf')
    assert nusvc.aocl is True
    nusvc.fit(X_train, y_train)
    pred_before = nusvc.predict(X_test)

    filepath = tmp_path / "nusvc_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nusvc, f)
    del nusvc

    with open(filepath, 'rb') as f:
        nusvc_loaded = pickle.load(f)

    assert nusvc_loaded.aocl is True
    pred_after = nusvc_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("NuSVC", print_patched=False)


def test_nusvc_pickle_cross_patch(tmp_path):
    """Test NuSVC: save with patch, load without patch."""
    skpatch("NuSVC", print_patched=False)

    from sklearn.svm import NuSVC

    X_train = np.array([[-1.0, -1.0], [-0.5, -0.5], [1.0, 1.0], [1.5, 1.5]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.0, 0.0]])

    nusvc = NuSVC(nu=0.5, kernel='rbf')
    assert nusvc.aocl is True
    nusvc.fit(X_train, y_train)
    pred_before = nusvc.predict(X_test)

    filepath = tmp_path / "nusvc_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nusvc, f)
    del nusvc

    undo_skpatch("NuSVC", print_patched=False)

    with open(filepath, 'rb') as f:
        nusvc_loaded = pickle.load(f)

    assert nusvc_loaded.aocl is True
    pred_after = nusvc_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)


# ============================================================================
# SVR Pickle Tests
# ============================================================================

def test_svr_pickle_with_patch(tmp_path):
    """Test SVR: save with patch, load with patch."""
    skpatch("SVR", print_patched=False)

    from sklearn.svm import SVR

    X_train = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]])
    y_train = np.array([2.0, 4.0, 6.0, 8.0])
    X_test = np.array([[2.5, 1.25]])

    svr = SVR(kernel='linear', C=1.0)
    assert svr.aocl is True
    svr.fit(X_train, y_train)
    pred_before = svr.predict(X_test)

    filepath = tmp_path / "svr_sk_full.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(svr, f)
    del svr

    with open(filepath, 'rb') as f:
        svr_loaded = pickle.load(f)

    assert svr_loaded.aocl is True
    pred_after = svr_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)

    undo_skpatch("SVR", print_patched=False)


def test_svr_pickle_cross_patch(tmp_path):
    """Test SVR: save with patch, load without patch."""
    skpatch("SVR", print_patched=False)

    from sklearn.svm import SVR

    X_train = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]])
    y_train = np.array([2.0, 4.0, 6.0, 8.0])
    X_test = np.array([[2.5, 1.25]])

    svr = SVR(kernel='linear', C=1.0)
    assert svr.aocl is True
    svr.fit(X_train, y_train)
    pred_before = svr.predict(X_test)

    filepath = tmp_path / "svr_sk_cross.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(svr, f)
    del svr

    undo_skpatch("SVR", print_patched=False)

    with open(filepath, 'rb') as f:
        svr_loaded = pickle.load(f)

    assert svr_loaded.aocl is True
    pred_after = svr_loaded.predict(X_test)
    assert np.array_equal(pred_before, pred_after)
