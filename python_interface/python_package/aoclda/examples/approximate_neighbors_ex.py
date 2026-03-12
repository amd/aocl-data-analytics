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
Approximate nearest neighbors example Python script
"""

import sys
import numpy as np
from aoclda.neighbors import approximate_neighbors


def approximate_neighbors_example():
    """
    Approximate Nearest Neighbors Example using IVFFlat algorithm
    """

    # Define training data (16 samples, 2 features)
    # Data forms 4 clusters
    x_train = np.array([[0.0, -0.1],
                        [1.1, 0.0],
                        [0.0, 1.1],
                        [1.0, 1.0],
                        [6.0, 0.0],
                        [7.2, 0.1],
                        [6.1, 1.0],
                        [7.0, 1.1],
                        [0.0, 10.0],
                        [1.0, 10.2],
                        [0.1, 11.0],
                        [1.1, 11.1],
                        [10.0, 10.0],
                        [11.1, 10.0],
                        [10.0, 11.2],
                        [11.0, 11.0]], dtype=np.float64)

    # Define query points
    x_test = np.array([[3.5, 0.4],
                       [0.4, 5.0],
                       [5.6, 5.1]], dtype=np.float64)

    print("\nApproximate Nearest Neighbors for a small data matrix\n")

    # n_list=4: create 4 clusters (centroids)
    # n_probe=2: search 2 nearest clusters at query time
    # seed=123: for reproducibility
    ann = approximate_neighbors(n_neighbors=3, n_list=4, n_probe=1,
                                metric='sqeuclidean', kmeans_iter=10, seed=123)

    # Train and add data to the index in one step
    ann.train_and_add(x_train)
    print("\nIndex trained and populated successfully.")

    # Query the k-nearest neighbors
    k_dist, k_ind = ann.kneighbors(x_test, return_distance=True)

    # Print results
    print("\nIndices of the approximate nearest neighbors:\n")
    print(k_ind)
    print("\nCorresponding squared Euclidean distances:\n")
    print(k_dist)

    # Print cluster information
    print("\nCluster centroids:\n")
    print(ann.cluster_centroids)
    print("\nNumber of vectors per cluster:")
    print(ann.list_sizes)
    print("\nTotal vectors in index:", ann.n_index)

    # Demonstrate the effect of n_probe on search accuracy
    # Lower n_probe = faster but potentially less accurate
    # Higher n_probe = slower but more accurate
    print("\n--- Comparing n_probe values ---\n")

    # Search with n_probe = 1 (set in constructor)
    k_dist_approx, k_ind_approx = ann.kneighbors(x_test, return_distance=True)
    print("n_probe = 1 (approximate):")
    print("  Indices:")
    print(k_ind_approx)
    print("  Distances:")
    print(k_dist_approx)

    # Search with n_probe = n_list (exact search over all clusters)
    ann.n_probe = ann.n_list
    k_dist_exact, k_ind_exact = ann.kneighbors(x_test, return_distance=True)
    print(f"\nn_probe = {ann.n_list} (exact, searches all clusters):")
    print("  Indices:")
    print(k_ind_exact)
    print("  Distances:")
    print(k_dist_exact)

    # Demonstrate the train then add workflow (alternative to train_and_add)
    print("\n--- Demonstrating train/add workflow ---\n")

    ann2 = approximate_neighbors(n_neighbors=3, n_list=4, n_probe=1,
                                 metric='sqeuclidean', seed=123)

    # Train only (computes centroids, does not add data)
    ann2.train(x_train)
    print("Index trained (centroids computed).")

    # Add data to the index
    ann2.add(x_train)
    print("Data added to the index.")

    # Query again
    k_dist2, k_ind2 = ann2.kneighbors(x_test, return_distance=True)

    # Verify results match the train_and_add workflow
    if np.array_equal(k_ind2, k_ind_approx):
        print("\nResults match the train_and_add workflow.")
    else:
        print("\nWarning: Results differ from train_and_add workflow.")
        sys.exit(1)

    print("\nApproximate nearest neighbors successfully computed\n")


if __name__ == "__main__":
    approximate_neighbors_example()
