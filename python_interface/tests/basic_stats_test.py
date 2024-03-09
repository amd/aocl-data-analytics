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
# pylint: disable = missing-module-docstring
"""
Basic statistics Python test script
"""

import warnings
import numpy as np
import pytest
import aoclda.basic_stats as da_stats
from sklearn.preprocessing import scale
from scipy.stats import gmean, hmean, skew, kurtosis, moment

test_cases = [{
    'np_precision': np.float64,
    'np_order': "C"
}, {
    'np_precision': np.float64,
    'np_order': "F"
}, {
    'np_precision': np.float32,
    'np_order': "C"
}, {
    'np_precision': np.float32,
    'np_order': "F"
}]


@pytest.fixture(params=test_cases,
                ids=["doubleC", "doubleF", "floatC", "floatF"])
def get_data2D(request):
    """
    Pytest fixture that defines our 2D matrix input for tests
    """
    data = np.array([[12.1341, 8.523, 9.102, 5.841, 11.4572],
                     [43.524, 12.3654, 25.54361, 17.5943, 19.4932],
                     [6.13245, 34.123446, 18.54252, 50.412, 15.4324],
                     [6.2131, 61.213432, 15.14325, 3.14536, 7.14342]],
                    dtype=request.param['np_precision'],
                    order=request.param['np_order'])
    tol = np.sqrt(np.finfo(request.param['np_precision']).eps)
    return {"data": data, "tol": tol}


@pytest.fixture(params=test_cases,
                ids=["doubleC", "doubleF", "floatC", "floatF"])
def get_data1D(request):
    """
    Pytest fixture that defines our 1D matrix input for tests
    """
    data_1D = np.array([12.1341, 8.523, 9.102, 5.841, 11.4572],
                       dtype=request.param['np_precision'],
                       order=request.param['np_order'])

    data_2D = np.array([[12.1341, 8.523, 9.102, 5.841, 11.4572]],
                       dtype=request.param['np_precision'],
                       order=request.param['np_order'])
    tol = np.sqrt(np.finfo(request.param['np_precision']).eps)
    return {"data1D": data_1D, "data2D": data_2D, "tol": tol}


@pytest.mark.parametrize("da_func, other_func",
                         [(da_stats.mean, np.mean),
                          (da_stats.geometric_mean, gmean),
                          (da_stats.harmonic_mean, hmean),
                          (da_stats.kurtosis, kurtosis),
                          (da_stats.skewness, skew)],
                         ids=['mean', 'gmean', 'hmean', 'kurt', 'skew'])
@pytest.mark.parametrize("da_axis, np_axis", [("row", 1), ("col", 0),
                                              ("all", None)],
                         ids=['row', 'col', 'all'])
def test_general_functions_functionality2D(get_data2D, da_func, other_func,
                                           da_axis, np_axis):
    """
    Testing functionality of functions with similar APIs with 2D data
    """
    X = get_data2D["data"]
    tol = get_data2D["tol"]

    # Compute mean
    da_result = da_func(X, axis=da_axis)
    # check expected results
    ex_result = other_func(X, axis=np_axis)

    error = np.max(np.abs(da_result - ex_result))
    assert error < tol


@pytest.mark.parametrize("da_func, other_func",
                         [(da_stats.mean, np.mean),
                          (da_stats.geometric_mean, gmean),
                          (da_stats.harmonic_mean, hmean),
                          (da_stats.kurtosis, kurtosis),
                          (da_stats.skewness, skew)],
                         ids=['mean', 'gmean', 'hmean', 'kurt', 'skew'])
@pytest.mark.parametrize("da_axis, np_axis", [("row", None), ("all", None)],
                         ids=['row', 'all'])
def test_general_functions_functionality1D(get_data1D, da_func, other_func,
                                           da_axis, np_axis):
    """
    Testing functionality of functions with similar APIs with 1D data
    """
    X = get_data1D["data1D"]
    X2D = get_data1D["data2D"]
    tol = get_data1D["tol"]

    # Compute mean
    da_result = da_func(X, axis=da_axis)
    da_result2 = da_func(X2D, axis=da_axis)
    # check expected results
    ex_result = other_func(X, axis=np_axis)

    error = np.max(np.abs(da_result - ex_result))
    assert error < tol
    # Assert that 1D array passed as 2D array is also correct
    error = np.max(np.abs(da_result2 - ex_result))
    assert error < tol


@pytest.mark.parametrize("da_axis, np_axis", [("row", 1), ("col", 0),
                                              ("all", None)])
def test_variance_functionality(get_data2D, da_axis, np_axis):
    """
    Testing functionality of variance function
    """
    X = get_data2D["data"]
    tol = get_data2D["tol"]

    # Compute variance
    da_var = da_stats.variance(X, axis=da_axis, dof=-1)

    # check expected results
    ex_var = np.var(X, axis=np_axis)

    error = np.max(np.abs(da_var - ex_var))
    assert error < tol


def get_expected_moments_precomputed_mean(array, k, mean, axis):
    """
    Helper function to get expected output for moment function
    when using precalculated mean.
    """
    moments = []
    if axis == "row":
        for i in range(array.shape[0]):
            moments.append(moment(array[i, :], k, axis=None, center=mean[i]))
    if axis == "col":
        for i in range(array.shape[1]):
            moments.append(moment(array[:, i], k, axis=None, center=mean[i]))
    if axis == "all":
        np.ravel(array, order='A')
        return moment(array, k, axis=None, center=mean[0])
    return np.array(moments)


@pytest.mark.parametrize("da_axis, np_axis, shift",
                         [("row", 1, [4.1, 5, 10, 6]),
                          ("col", 0, [12, 10, 4, 1.2, 2.6]),
                          ("all", None, [10.4])],
                         ids=["row", "col", "all"])
def test_moment_functionality_2D(get_data2D, da_axis, np_axis, shift):
    """
    Testing functionality of moment function with 2D input
    """
    X = get_data2D["data"]
    # Increase the tolerance because expected test results for "all" case are of 1e12 magnitude
    # and cause some discrepancy after 3rd decimal point which is negligible at that size
    tol = get_data2D["tol"] * 2e6

    # Compute moments
    da_moment = da_stats.moment(X, 3, axis=da_axis)
    da_moment_with_precomputed_mean = da_stats.moment(X,
                                                      3,
                                                      mean=shift,
                                                      axis=da_axis)

    # check expected results
    ex_moment = moment(X, 3, axis=np_axis)
    ex_moment_with_precomputed_mean = get_expected_moments_precomputed_mean(
        X, 3, mean=shift, axis=da_axis)

    error = np.max(np.abs(da_moment - ex_moment))
    assert error < tol
    error = np.max(
        np.abs(da_moment_with_precomputed_mean -
               ex_moment_with_precomputed_mean))
    assert error < tol


@pytest.mark.parametrize("da_axis, np_axis, shift", [("row", None, [10.4])])
def test_moment_functionality_1D(get_data1D, da_axis, np_axis, shift):
    """
    Testing functionality of moment function with 1D input
    """
    X = get_data1D["data1D"]
    X2D = get_data1D["data2D"]
    tol = get_data1D["tol"]

    # Compute moments
    da_moment = da_stats.moment(X, 8, axis=da_axis)
    da_moment_with_precomputed_mean = da_stats.moment(X,
                                                      8,
                                                      mean=shift,
                                                      axis=da_axis)
    da_moment2 = da_stats.moment(X2D, 8, axis=da_axis)
    da_moment_with_precomputed_mean2 = da_stats.moment(X,
                                                       8,
                                                       mean=shift,
                                                       axis=da_axis)

    # check expected results
    ex_moment = moment(X, 8, axis=np_axis)
    ex_moment_with_precomputed_mean = get_expected_moments_precomputed_mean(
        X, 8, mean=shift, axis="all")

    error = np.max(np.abs(da_moment - ex_moment))
    assert error < tol
    error = np.max(
        np.abs(da_moment_with_precomputed_mean -
               ex_moment_with_precomputed_mean))
    assert error < tol
    # Assert that 1D array passed as 2D array is also correct
    error = np.max(np.abs(da_moment2 - ex_moment))
    assert error < tol
    error = np.max(
        np.abs(da_moment_with_precomputed_mean2 -
               ex_moment_with_precomputed_mean))
    assert error < tol


def closest_observation_func(data, q, axis):
    if axis == 'row':
        n = len(data[0, :])
        data = np.sort(data, axis=1)
        h = n * q - 1.5
        h = 0 if h < 0 else h
        return data[:, round(h)]
    if axis == 'col':
        n = len(data)
        data = np.sort(data, axis=0)
        h = n * q - 1.5
        h = 0 if h < 0 else h
        return data[round(h), :]
    else:
        data = np.sort(data, axis=None)
        n = len(np.ravel(data))
        h = n * q - 1.5
        h = 0 if h < 0 else h
        return np.ravel(data)[round(h)]


@pytest.mark.parametrize("calc_method", [
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
    "interpolated_inverted_cdf", "hazen", "weibull", "linear",
    "median_unbiased", "normal_unbiased"
])
@pytest.mark.parametrize("da_axis, np_axis", [("row", 1), ("col", 0),
                                              ("all", None)])
def test_quantile_functionality(get_data2D, calc_method, da_axis, np_axis):
    """
    Testing functionality of quantile function
    """
    X = get_data2D["data"]
    tol = get_data2D["tol"]

    # Compute quantile
    da_quantile1 = da_stats.quantile(X, 0.21, axis=da_axis, method=calc_method)
    da_quantile2 = da_stats.quantile(X, 0.88, axis=da_axis, method=calc_method)

    # check expected quantile
    ex_quantile1 = np.quantile(X, 0.21, axis=np_axis, method=calc_method)
    ex_quantile2 = np.quantile(X, 0.88, axis=np_axis, method=calc_method)

    # Take into account different definition of closest observation
    if calc_method == "closest_observation":
        ex_quantile1 = closest_observation_func(X, 0.21, axis=da_axis)
        ex_quantile2 = closest_observation_func(X, 0.88, axis=da_axis)

    error = np.max(np.abs(da_quantile1 - ex_quantile1))
    assert error < tol
    error = np.max(np.abs(da_quantile2 - ex_quantile2))
    assert error < tol


def fivesum(array, axis):
    """
    Helper function to achieve DA library-like output of five point summary function
    """
    result = [[], [], [], [], []]
    if axis == 0:
        for i in range(array.shape[1]):
            result[0].append(np.min(array[:, i]))
            result[1].append(np.percentile(array[:, i], 25, method='weibull'))
            result[2].append(np.percentile(array[:, i], 50, method='weibull'))
            result[3].append(np.percentile(array[:, i], 75, method='weibull'))
            result[4].append(np.max(array[:, i]))
    if axis == 1:
        for i in range(array.shape[0]):
            result[0].append(np.min(array[i, :]))
            result[1].append(np.percentile(array[i, :], 25, method='weibull'))
            result[2].append(np.percentile(array[i, :], 50, method='weibull'))
            result[3].append(np.percentile(array[i, :], 75, method='weibull'))
            result[4].append(np.max(array[i, :]))
    if axis == 2:
        np.ravel(array, order='A')
        result[0].append(np.min(array))
        result[1].append(np.percentile(array, 25, method='weibull'))
        result[2].append(np.percentile(array, 50, method='weibull'))
        result[3].append(np.percentile(array, 75, method='weibull'))
        result[4].append(np.max(array))
    return np.array(result)


@pytest.mark.parametrize("da_axis, np_axis", [("row", 1), ("col", 0),
                                              ("all", 2)])
def test_fps_functionality(get_data2D, da_axis, np_axis):
    """
    Testing functionality of five point summary function
    """
    X = get_data2D["data"]
    tol = get_data2D["tol"]

    # Compute five point summary
    min_da, lh_da, med_da, uh_da, max_da = da_stats.five_point_summary(
        X, axis=da_axis)
    results = np.array([min_da, lh_da, med_da, uh_da, max_da])

    # check expected results
    ex_results = fivesum(X, np_axis)

    error = np.max(np.abs(results - ex_results))
    assert error < tol


def get_standardized_with_shift_or_mean(array,
                                        axis,
                                        shift_a=None,
                                        scale_a=None):
    """
    Helper function to get expected output for standardize function
    when using shift or scale parameter.
    """
    X = np.copy(array)
    # Case when we want to do calculations over rows
    if axis == "row":
        if shift_a and scale_a:
            for i in range(array.shape[0]):
                X[i, :] = (array[i, :] - shift_a[i]) / scale_a[i]
        elif shift_a:
            for i in range(array.shape[0]):
                X[i, :] = array[i, :] - shift_a[i]
        elif scale_a:
            for i in range(array.shape[0]):
                X[i, :] = array[i, :] / scale_a[i]
        return X
    # Case when we want to do calculations over columns
    if axis == "col":
        if shift_a and scale_a:
            for i in range(array.shape[1]):
                X[:, i] = (array[:, i] - shift_a[i]) / scale_a[i]
        elif shift_a:
            for i in range(array.shape[1]):
                X[:, i] = array[:, i] - shift_a[i]
        elif scale_a:
            for i in range(array.shape[1]):
                X[:, i] = array[:, i] / scale_a[i]
        return X
    # Case when we want to do calculations over entire dataset
    if shift_a and scale_a:
        X = (array - shift_a[0]) / scale_a[0]
    elif shift_a:
        X = array - shift_a[0]
    elif scale_a:
        X = array / scale_a[0]
    return X


@pytest.mark.parametrize(
    "da_axis, np_axis, shift_a, scale_a",
    [("row", 1, [4.1, 5, 10, 6], [7, 3, 1.5, 2]),
     ("col", 0, [12, 10, 4, 1.2, 2.6], [1.2, 5, 1.76, 8, 4.7]),
     ("all", 2, [10.4], [4.9])],
    ids=['row', 'column', 'all'])
def test_standardize_functionality(get_data2D, da_axis, np_axis, shift_a,
                                   scale_a):
    """
    Testing functionality of standardization function
    """
    warnings.filterwarnings("ignore",module="sklearn")
    X = get_data2D["data"]
    tol = get_data2D["tol"]

    mean = da_stats.mean(X, axis=da_axis)
    sd = np.sqrt(da_stats.variance(X, axis=da_axis, dof=-1))

    # Compute standardized matrix without specified shift and scale
    standardized = da_stats.standardize(X, axis=da_axis, inplace=False, dof=-1)
    # Reverse that matrix
    reversed_standardized = da_stats.standardize(standardized,
                                                 axis=da_axis,
                                                 reverse=True,
                                                 shift=mean,
                                                 scale=sd,
                                                 inplace=False)
    # Compute standardized matrix with shift and without scale
    shifted = da_stats.standardize(X,
                                   axis=da_axis,
                                   inplace=False,
                                   shift=shift_a,
                                   dof=-1)
    reversed_shifted = da_stats.standardize(shifted,
                                            axis=da_axis,
                                            inplace=False,
                                            reverse=True,
                                            shift=shift_a,
                                            dof=-1)
    # Compute standardized matrix with scale and without shift
    scaled = da_stats.standardize(X,
                                  axis=da_axis,
                                  inplace=False,
                                  scale=scale_a,
                                  dof=-1)
    reversed_scaled = da_stats.standardize(scaled,
                                           axis=da_axis,
                                           inplace=False,
                                           reverse=True,
                                           scale=scale_a,
                                           dof=-1)
    # Compute standardized matrix with shift and scale
    shifted_scaled = da_stats.standardize(X,
                                          axis=da_axis,
                                          inplace=False,
                                          scale=scale_a,
                                          shift=shift_a,
                                          dof=-1)
    reversed_shifted_scaled = da_stats.standardize(shifted_scaled,
                                                   axis=da_axis,
                                                   inplace=False,
                                                   reverse=True,
                                                   scale=scale_a,
                                                   shift=shift_a,
                                                   dof=-1)

    # check expected results

    ex_shifted = get_standardized_with_shift_or_mean(X,
                                                     axis=da_axis,
                                                     shift_a=shift_a)
    # Compute standardized matrix with scale and without shift
    ex_scaled = get_standardized_with_shift_or_mean(X,
                                                    axis=da_axis,
                                                    scale_a=scale_a)
    # Compute standardized matrix with shift and scale
    ex_shifted_scaled = get_standardized_with_shift_or_mean(X,
                                                            axis=da_axis,
                                                            scale_a=scale_a,
                                                            shift_a=shift_a)

    if np_axis == 2:
        np_input = np.ravel(X)
        ex_standardized = scale(np_input)
        ex_standardized = ex_standardized.reshape((4, 5))
    else:
        ex_standardized = scale(X, axis=np_axis)

    error = np.max(np.abs(standardized - ex_standardized))
    assert error < tol
    error = np.max(np.abs(shifted - ex_shifted))
    assert error < tol
    error = np.max(np.abs(scaled - ex_scaled))
    assert error < tol
    error = np.max(np.abs(shifted_scaled - ex_shifted_scaled))
    assert error < tol
    error = np.max(np.abs(X - reversed_standardized))
    assert error < tol
    error = np.max(np.abs(X - reversed_shifted))
    assert error < tol
    error = np.max(np.abs(X - reversed_scaled))
    assert error < tol
    error = np.max(np.abs(X - reversed_shifted_scaled))
    assert error < tol


@pytest.mark.parametrize("da_biased, numpy_biased", [(-1, True), (0, False)])
def test_covariance_functionality(get_data2D, da_biased, numpy_biased):
    """
    Testing functionality of covariance function
    """
    X = get_data2D["data"]
    tol = get_data2D["tol"]

    # Compute covariance matrix
    covariance = da_stats.covariance_matrix(X, dof=da_biased)

    # check expected results
    covariance_ex = np.cov(X, bias=numpy_biased, rowvar=False)

    error = np.max(np.abs(covariance - covariance_ex))
    assert error < tol


def test_correlation_functionality(get_data2D):
    """
    Testing functionality of correlation function
    """
    X = get_data2D["data"]
    tol = get_data2D["tol"]

    # Compute correlation matrix
    correlation = da_stats.correlation_matrix(X)

    # check expected results
    correlation_ex = np.corrcoef(X, rowvar=False)

    error = np.max(np.abs(correlation - correlation_ex))
    assert error < tol


@pytest.mark.parametrize("func", [
    da_stats.mean, da_stats.geometric_mean, da_stats.harmonic_mean,
    da_stats.variance, da_stats.kurtosis, da_stats.skewness,
    da_stats.five_point_summary, da_stats.standardize
])
@pytest.mark.parametrize("da_axis", ["row", "col", "all"])
def test_error_exits_general(func, da_axis):
    """
    Testing error exits in basic statistics functions
    """
    # Check empty array input
    with pytest.raises(ValueError):
        func([], axis=da_axis)
    # Check wrong input type
    with pytest.raises(TypeError):
        func([[1, 2, 3], [4, 'a', 6]], axis=da_axis)
    # Check 3D input
    with pytest.raises(ValueError):
        func([[[]]], axis=da_axis)


@pytest.mark.parametrize("da_axis, wrong_shape_input", [("row", [1, 2, 3]),
                                                        ("col", [1, 2, 3, 4]),
                                                        ("all", [1, 2])])
def test_error_exits_moment(get_data2D, da_axis, wrong_shape_input):
    """
    Testing error exits in moment function
    """
    # Check wrong size for additional parameters
    with pytest.raises(ValueError):
        da_stats.moment(get_data2D['data'],
                        4,
                        axis=da_axis,
                        mean=wrong_shape_input)
    # Check empty array input
    with pytest.raises(ValueError):
        da_stats.moment([], k=2, axis=da_axis)
    # Check wrong input type
    with pytest.raises(TypeError):
        da_stats.moment([[1, 2, 3], [4, 'a', 6]], k=2, axis=da_axis)
    with pytest.raises(ValueError):
        da_stats.moment([[1, 2, 3], [4, 5, 6]], k=-1, axis=da_axis)
    with pytest.raises(TypeError):
        da_stats.moment([[1, 2, 3], [4, 5, 6]], k='a', axis=da_axis)
    # Check 3D input
    with pytest.raises(ValueError):
        da_stats.moment([[[]]], k=2, axis=da_axis)


@pytest.mark.parametrize("da_axis", ["row", "col", "all"])
def test_error_exits_quantile(da_axis):
    """
    Testing error exits in quantile function
    """
    X = np.array([[1, 2, 3], [4, -5, 6]])
    # Check invalid value for quantile
    with pytest.raises(ValueError):
        da_stats.quantile(X, 1.01, axis=da_axis)
    with pytest.raises(ValueError):
        da_stats.quantile(X, 0.12, axis=da_axis, method='lineear')
    # Check empty array input
    with pytest.raises(ValueError):
        da_stats.quantile([], 0.2, axis=da_axis)
    # Check wrong input type
    with pytest.raises(TypeError):
        da_stats.quantile([[1, 2, 3], [4, 'a', 6]], 0.2, axis=da_axis)
    with pytest.raises(ValueError):
        da_stats.quantile(X, q=-1.2, axis=da_axis)
    with pytest.raises(TypeError):
        da_stats.quantile(X, q='a', axis=da_axis)


@pytest.mark.parametrize("da_axis, wrong_shape_input", [("row", [1, 2, 3]),
                                                        ("col", [1, 2]),
                                                        ("all", [1, 2])])
def test_error_exits_standardize(da_axis, wrong_shape_input):
    """
    Testing error exits in standardize function
    """
    X = np.array([[1, 2, 3], [4, -5, 6]], dtype=np.float32, order='F')
    tol = np.sqrt(np.finfo(np.float32).eps)
    # Check that reverse standardization does not work without both shift and scale parameters
    with pytest.raises(ValueError):
        da_stats.standardize(X, reverse=True, axis=da_axis)
    # Check wrong shift/scale parameter size
    with pytest.raises(ValueError):
        da_stats.standardize(X, axis=da_axis, shift=wrong_shape_input)
    with pytest.raises(ValueError):
        da_stats.standardize(X, axis=da_axis, scale=wrong_shape_input)
    with pytest.raises(ValueError):
        da_stats.standardize(X,
                             axis=da_axis,
                             shift=wrong_shape_input,
                             scale=wrong_shape_input)
    # Check inplace functionality
    standardized = da_stats.standardize(X, inplace=True, axis=da_axis)
    error = np.max(np.abs(X - standardized))
    assert error < tol


@pytest.mark.parametrize(
    "func", [da_stats.covariance_matrix, da_stats.correlation_matrix])
def test_error_exits_cov_corr(func):
    """
    Testing error exits in covariance and correlation functions
    """
    # Check empty array input
    with pytest.raises(ValueError):
        func([])
    # Check wrong input type
    with pytest.raises(TypeError):
        func([[1, 2, 3], [4, 'a', 6]])
    # Check 3D input
    with pytest.raises(ValueError):
        func([[[]]])


@pytest.mark.parametrize("da_axis", ["row", "col", "all"])
def test_negative_geometric_mean(da_axis):
    """
    Testing negative entry to geometric mean function
    """
    X = np.array([[1, 2, 3], [4, -5, 6]])
    with pytest.raises(ValueError):
        da_stats.geometric_mean(X, axis=da_axis)
