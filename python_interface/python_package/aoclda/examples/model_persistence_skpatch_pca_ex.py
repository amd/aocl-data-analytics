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
Sklearn patching with model persistence example
"""

import sys
import os
import pickle
import numpy as np
from aoclda.sklearn import skpatch


def model_persistence_skpatch_pca_example():
    """
    Using sklearn patching with model persistence
    """

    # Patch sklearn to use AOCL-DA's PCA implementation
    print("Patching sklearn to use AOCL-DA's PCA implementation...")
    skpatch('PCA')

    from sklearn.decomposition import PCA

    # Generate sample data
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((1000, 10))

    # Fit the model using sklearn's API
    print("Training PCA model using sklearn API...")
    pca = PCA(n_components=5)
    pca.fit(X_train)

    # Transform some data
    X_transformed = pca.transform(X_train[:10])
    print(f"Transformed data shape: {X_transformed.shape}")

    # Verify AOCL-DA is being used
    if hasattr(pca, 'aocl') and pca.aocl:
        print("Using AOCL-DA implementation")
    else:
        print("Warning: Not using AOCL-DA implementation")

    # Save the trained model
    print("\nSaving model to 'pca_sklearn_model.pkl'...")
    with open('pca_sklearn_model.pkl', 'wb') as f:
        pickle.dump(pca, f)

    # Delete the original model
    del pca

    # Later, load and use the model
    print("Loading model from 'pca_sklearn_model.pkl'...")
    with open('pca_sklearn_model.pkl', 'rb') as f:
        pca_loaded = pickle.load(f)

    X_new_transformed = pca_loaded.transform(X_train[:10])

    # Verify the results match
    if np.array_equal(X_transformed, X_new_transformed):
        print("\nModel persistence verified - transformations match!")
    else:
        print("\nError: Transformations do not match")
        max_diff = np.max(np.abs(X_transformed - X_new_transformed))
        print(f"Maximum difference: {max_diff}")
        os.remove('pca_sklearn_model.pkl')
        sys.exit(1)

    # Clean up created files
    os.remove('pca_sklearn_model.pkl')


if __name__ == "__main__":
    try:
        model_persistence_skpatch_pca_example()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
