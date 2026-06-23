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
Decision tree model persistence example
"""

import sys
import os
import pickle
import numpy as np
from aoclda.decision_tree import decision_tree


def model_persistence_decision_tree_example():
    """
    Decision tree model persistence using pickle
    """

    # Generate sample classification data
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 4))
    y_train = rng.integers(0, 3, 200)  # 3 classes

    print("Training decision tree classifier...")
    # Train a decision tree classifier
    tree = decision_tree(max_depth=5, criterion='gini')
    tree.fit(X_train, y_train)

    # Make predictions before saving
    X_test = rng.standard_normal((10, 4))
    predictions_before = tree.predict(X_test)
    print(f"Predictions before save: {predictions_before}")

    # Save the trained model
    print("\nSaving model to 'decision_tree_model.pkl'...")
    with open('decision_tree_model.pkl', 'wb') as f:
        pickle.dump(tree, f)

    # Delete the original model
    del tree

    # Load the model
    print("Loading model from 'decision_tree_model.pkl'...")
    with open('decision_tree_model.pkl', 'rb') as f:
        tree_loaded = pickle.load(f)

    # Make predictions with loaded model
    predictions_after = tree_loaded.predict(X_test)
    print(f"Predictions after load:  {predictions_after}")

    # Verify predictions match
    if np.array_equal(predictions_before, predictions_after):
        print("\nModel persistence verified - predictions match!")
    else:
        print("\nError: Predictions do not match")
        os.remove('decision_tree_model.pkl')
        sys.exit(1)

    # Clean up created files
    os.remove('decision_tree_model.pkl')


if __name__ == "__main__":
    try:
        model_persistence_decision_tree_example()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
