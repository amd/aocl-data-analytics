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
K-Means model persistence example
"""

import sys
import os
import pickle
import numpy as np
from aoclda.clustering import kmeans


def model_persistence_kmeans_ex():
    """
    K-Means model persistence using pickle
    """

    # Generate sample data and train a k-means model
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 5))

    print("Training k-means model...")
    # Fit the model
    km = kmeans(n_clusters=3, seed=42)
    km.fit(X)

    # Get predictions and centers before saving
    labels_before = km.predict(X[:10])
    centers_before = km.cluster_centres.copy()
    print(f"Cluster centers shape: {centers_before.shape}")
    print(f"First 10 predictions: {labels_before}")

    # Save the trained model to a file
    print("\nSaving model to 'kmeans_model.pkl'...")
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(km, f)

    # Delete the original model
    del km

    # Later, load the model from disk
    print("Loading model from 'kmeans_model.pkl'...")
    with open('kmeans_model.pkl', 'rb') as f:
        km_loaded = pickle.load(f)

    # Use the loaded model for predictions
    labels_after = km_loaded.predict(X[:10])
    centers_after = km_loaded.cluster_centres
    print(f"First 10 predictions after load: {labels_after}")

    # Verify results match
    labels_match = np.array_equal(labels_before, labels_after)
    centers_match = np.array_equal(centers_before, centers_after)

    if labels_match and centers_match:
        print("\nModel persistence verified - predictions and centers match!")
    else:
        print("\nError: Results do not match")
        if not labels_match:
            print("  - Labels differ")
        if not centers_match:
            print("  - Cluster centers differ")
        os.remove('kmeans_model.pkl')
        sys.exit(1)

    # Clean up created files
    os.remove('kmeans_model.pkl')


if __name__ == "__main__":
    try:
        model_persistence_kmeans_ex()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
