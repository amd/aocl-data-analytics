# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""
Utility functions.
"""

# pylint: disable=not-an-iterable,no-self-argument,no-member
# pylint: disable=import-error,invalid-name,too-many-arguments
# pylint: disable=missing-module-docstring,too-many-locals, anomalous-backslash-in-string

import math
from aoclda._internal_utils import check_convert_data
from ._aoclda.utils import (
    pybind_train_test_split, pybind_get_shuffled_indices, pybind_get_version, pybind_get_git_commit)


def train_test_split(
        *arrays, test_size=None, train_size=None, seed=-1,
        shuffle=True, stratify=None, fp_precision=1000):
    """
    Train Test Split.

    Divides data into training and test sets.

    Args:
        *arrays : array-like arrays/matrices with the same number of samples.

        test_size (float | int, optional): The size of the returned test array. If test_size is a
            float, then it should be between 0 and 1, and will represent the proportion of the
            dataset to include in the test data. If test_size is an int, then it will represent
            the number of rows to include in the test data. If it is None, its value is set to
            the complement of the train size, or if train_size = None, it is set to be 0.25.
            Default: None.

        train_size (float | int, optional): The size of the returned train array. If train_size
            is a float, then it should be between 0 and 1, and will represent the proportion of
            the dataset to include in the training data. If train_size is an int, then it will
            represent the number of rows to include in the training data. If it is None, its value
            is set to the complement of the test size. Default: None.

        seed (int, optional): Set the random seed for the random number generator.
            If the value is -1, a random seed is automatically generated. In this case
            the resulting classification will create non-reproducible results.
            Default: -1.

        shuffle (bool, optional): Specify whether shuffling of the data should be performed before
            the splitting. Default: True.

        stratify (array-like, optional): If not None, the array is used as class labels to
            perform a stratified shuffle. Stratified shuffling tries to preserve the
            proportions of the classes in the training and the test split, as they were in the
            original data.

        fp_precision (int, optional): an integer specifying the scaling factor applied to
            floating-point class labels to determine their precision. When stratify is not
            None and of type float or double, each class label is multiplied by fp_precision
            and then floored to obtain an integer class label for the stratified shuffling.
            Default: 1000.

    Returns:
        list, numpy.ndarray[float32 | float64 | da_int], length=2*len(arrays): A list of the split
        arrays containing the training and the test arrays [train_split, test_split, ...]
    """

    if stratify is not None:
        if shuffle is False:
            raise ValueError("shuffle cannot be False, when stratify is used")
        if len(stratify) != len(arrays[0]):
            raise ValueError(
                f"The stratify array must have the same number of samples (m={len(stratify)})"
                f" as the data arrays (m={len(arrays[0])}).")

        stratify, _, _ = check_convert_data(stratify)

    if isinstance(train_size, float) and (train_size >= 1 or train_size <= 0):
        raise ValueError(
            "train_size cannot be a float outside the range (0.0, 1.0)."
        )
    if isinstance(test_size, float) and (test_size >= 1 or test_size <= 0):
        raise ValueError(
            "test_size cannot be a float outside the range (0.0, 1.0)."
        )
    if isinstance(
            train_size, float) and isinstance(
            test_size, float) and (
            train_size + test_size) > 1:
        raise ValueError(
            f"train_size + test_size = {train_size + test_size}. Constraint: train_size + test_size<=1.")

    if train_size is None and test_size is None:
        train_size = 0.75
        test_size = 0.25

    # Train size is floored, test size is ceiled to prevent exceeding sample size.
    # This handles cases where splits result in fractional samples (.5).
    if train_size is not None and train_size < 1:
        train_size = int(math.floor(arrays[0].shape[0] * train_size))
    if test_size is not None and test_size < 1:
        test_size = int(math.ceil(arrays[0].shape[0] * test_size))

    if train_size is None:
        train_size = arrays[0].shape[0] - test_size
    elif test_size is None:
        test_size = arrays[0].shape[0] - train_size

    if shuffle:
        shuffled_indices = pybind_get_shuffled_indices(
            arrays[0].shape[0],
            seed, train_size, test_size, fp_precision, stratify)
    else:
        shuffled_indices = None

    response = []
    arrays = list(arrays)
    for i in range(len(arrays)):
        if (i > 0 and len(arrays[i]) != len(arrays[i - 1])):
            raise ValueError(
                "Input matrices must have the same number of samples. Arrays of" +
                f"length {len(arrays[i])} and {len(arrays[i - 1])} were encountered.")

        arrays[i], _, _ = check_convert_data(arrays[i])

        X_train, X_test = pybind_train_test_split(
            arrays[i],
            train_size=train_size, test_size=test_size,
            shuffled_indices=shuffled_indices)

        # Flatten dimension when the initial input was 1D array.
        if arrays[i].ndim == 1:
            response.extend([X_train.flatten(), X_test.flatten()])
        else:
            response.extend([X_train, X_test])

    return response

def get_version():
    """
    Get the version of the AOCL-DA library.

    Returns:
        str: The version string of the AOCL-DA library.
    """
    return pybind_get_version()

def get_git_commit():
    """
    Return the git tag or commit hash identifying the AOCL-DA build.

    Returns:
        str: Git tag or commit hash for the build.
    """
    return pybind_get_git_commit()