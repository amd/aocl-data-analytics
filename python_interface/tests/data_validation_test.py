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
#

"""
Data validation Python test script.
"""

import numpy as np
import pandas as pd
import pytest
from aoclda._internal_utils import check_convert_data, get_int_info

int_type = "int" + get_int_info()

@pytest.mark.parametrize('force_float, da_dtype, np_dtype', [
    (True, 'float32', 'float32'), (True, 'float64', 'float64'),
    (False, 'float32', 'float32'), (False, 'float64', 'float64'),
    (False, 'float', 'float32'), (False, 'float', 'float64'),
    (False, 'da_int', int_type), (True, 'da_int', int_type)])
def test_no_casting_numpy(force_float, da_dtype, np_dtype):
    """
        Test data validation when no casting is needed.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np_dtype)
    X, _, _ = check_convert_data(X, order='A', dtype=da_dtype, force_dtype=force_float)

    assert X.dtype == np_dtype


@pytest.mark.parametrize('ini_dtype, da_conv_dtype, np_conv_dtype',
                         [('float16', 'float32', 'float32'),
                          ('float16', 'float64', 'float64'),
                          ('float16', 'float', 'float64'),
                          ('float16', 'da_int', int_type),
                          ('int16', 'float32', 'float32'),
                          ('int16', 'float64', 'float64'),
                          ('int16', 'float', 'float64'),
                          ('int16', 'da_int', int_type),
                          ('int32', 'float32', 'float32'),
                          ('int32', 'float64', 'float64'),
                          ('int32', 'float', 'float64'),
                          ('int64', 'float32', 'float32'),
                          ('int64', 'float64', 'float64'),
                          ('int64', 'float', 'float64'),
                          ('object', 'float32', 'float32'),
                          ('object', 'float64', 'float64'),
                          ('object', 'float', 'float64'),
                          ('object', 'da_int', int_type)])
def test_valid_numpy_type_casting_force(
        ini_dtype, da_conv_dtype, np_conv_dtype):
    """
        Test data validation on NumPy arrays when casting is needed and only floats are accepted.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)
    X, _, _ = check_convert_data(X, order='A', dtype=da_conv_dtype, force_dtype=True)
    assert X.dtype == np_conv_dtype


@pytest.mark.parametrize('ini_dtype, da_conv_dtype, np_conv_dtype',
                         [('float16', 'float32', 'float32'),
                          ('float16', 'float64', 'float64'),
                          ('float16', 'float', 'float64'),
                          ('float16', 'da_int', int_type),
                          ('int16', 'float32', int_type),
                          ('int16', 'float64', int_type),
                          ('int16', 'float', int_type),
                          ('int16', 'da_int', int_type),
                          ('object', 'da_int', int_type),
                          ('object', 'float32', 'float32'),
                          ('object', 'float64', 'float64'),
                          ('object', 'float', 'float64'),])
def test_valid_numpy_type_casting_not_force(
        ini_dtype, da_conv_dtype, np_conv_dtype):
    """
        Test data validation on NumPy arrays when casting is needed and integers are accepted.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)
    X, _, _ = check_convert_data(X, order='A', dtype=da_conv_dtype, force_dtype=False)
    assert X.dtype == np_conv_dtype


def test_invalid_numpy_type_casting():
    """
        Test data validation when invalid NumPy data is inputted.
    """
    X = np.array([[1, 1, 1], [2, 'a', 2], [3, 3, 3]])

    with pytest.raises(ValueError):
        X, _, _ = check_convert_data(X, order='A')


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float'])
def test_nan_numpy_validation(dtype):
    """
        Test data validation when data has NumPy nan in it.
    """
    X = np.array([[1, 1, 1], [2, np.nan, 2], [3, 3, 3]], dtype=dtype)

    X, _, _ = check_convert_data(X, order='A')
    assert X.dtype == dtype


@pytest.mark.parametrize('ini_dtype, da_conv_dtype, np_conv_dtype',
                         [('float16', 'float32', 'float32'),
                          ('float16', 'float64', 'float64'),
                          ('float16', 'float', 'float64'),
                          ('object', 'float32', 'float32'),
                          ('object', 'float64', 'float64'),
                          ('object', 'float', 'float64')])
def test_nan_numpy_casting(ini_dtype, da_conv_dtype, np_conv_dtype):
    """
        Test data validation on NumPy arrays when casting is needed and np.nan data is present.
    """
    X = np.array([[1, 1, 1], [2, np.nan, 2], [3, 3, 3]], dtype=ini_dtype)
    X, _, _ = check_convert_data(X, order='A', dtype=da_conv_dtype)
    assert np.isnan(X[1][1])
    assert X.dtype == np_conv_dtype


@pytest.mark.parametrize('force_float, da_dtype, np_dtype', [
    (True, 'float32', 'float32'), (True, 'float64', 'float64'),
    (False, 'float32', 'float32'), (False, 'float64', 'float64'),
    (False, 'da_int', int_type), (True, 'da_int', int_type)])
def test_no_casting_pandas(force_float, da_dtype, np_dtype):
    """
        Test data validation on Pandas when no casting is needed.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 2, 2]], dtype=np_dtype)
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(
            df, order='A', dtype=da_dtype, force_dtype=force_float)
    assert X.dtype == np_dtype


@pytest.mark.parametrize('ini_dtype, da_conv_dtype, np_conv_dtype',
                         [('float16', 'float32', 'float32'),
                          ('float16', 'float64', 'float64'),
                          ('float16', 'float', 'float64'),
                          ('float16', 'da_int', int_type),
                          ('int16', 'float32', 'float32'),
                          ('int16', 'float64', 'float64'),
                          ('int16', 'float', 'float64'),
                          ('int16', 'da_int', int_type),
                          ('int32', 'float32', 'float32'),
                          ('int32', 'float64', 'float64'),
                          ('int32', 'float', 'float64'),
                          ('int64', 'float32', 'float32'),
                          ('int64', 'float64', 'float64'),
                          ('int64', 'float', 'float64'),
                          ('object', 'float32', 'float32'),
                          ('object', 'float64', 'float64'),
                          ('object', 'float', 'float64'),
                          ('object', 'da_int', int_type)])
def test_valid_pandas_casting_force(ini_dtype, da_conv_dtype, np_conv_dtype):
    """
        Test data validation on Pandas when casting is needed when only floats are accepted.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(df, order='A', dtype=da_conv_dtype, force_dtype=True)
    assert X.dtype == np_conv_dtype


@pytest.mark.parametrize('ini_dtype, da_conv_dtype, np_conv_dtype',
                         [('float16', 'float32', 'float32'),
                          ('float16', 'float64', 'float64'),
                          ('float16', 'float', 'float64'),
                          ('float16', 'da_int', int_type),
                          ('int16', 'float32', int_type),
                          ('int16', 'float64', int_type),
                          ('int16', 'float', int_type),
                          ('int16', 'da_int', int_type),
                          ('object', 'da_int', int_type),
                          ('object', 'float32', 'float32'),
                          ('object', 'float64', 'float64'),
                          ('object', 'float', 'float64'),])
def test_valid_pandas_casting_not_force(
        ini_dtype, da_conv_dtype, np_conv_dtype):
    """
        Test data validation on Pandas when casting is needed when integers are also accepted.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(
            df, order='A', dtype=da_conv_dtype, force_dtype=False)
    assert X.dtype == np_conv_dtype


def test_invalid_pandas_type_casting():
    """
        Test data validation on Pandas when invalid data is present.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 'a', 2], [3, 3, 3]])
    with pytest.raises(ValueError), pytest.warns(UserWarning):
        X, _, _ = check_convert_data(df, order='A')


@pytest.mark.parametrize('ini_dtype, da_conv_dtype, np_conv_dtype',
                         [('float16', 'float32', 'float32'),
                          ('float16', 'float64', 'float64'),
                          ('float16', 'float', 'float64'),
                          ('object', 'float32', 'float32'),
                          ('object', 'float64', 'float64'),
                          ('object', 'float', 'float64')])
def test_nan_pandas_casting(ini_dtype, da_conv_dtype, np_conv_dtype):
    """
        Test data validation on Pandas when None, NA and nan are present.
    """
    # No data conversion as pd.NA cannot be converted to floats, etc.
    df = pd.DataFrame([[1, 1, 1], [2, pd.NA, 2], [3, 3, 3]])

    with pytest.raises(TypeError), pytest.warns(UserWarning):
        X, _, _ = check_convert_data(df, order='A', dtype=da_conv_dtype)

    df = pd.DataFrame([[1, 1, 1], [2, None, 2], [3, 3, 3]], dtype=ini_dtype)
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(df, order='A', dtype=da_conv_dtype)

    assert X.dtype == np_conv_dtype
    assert np.isnan(X[1][1])

    df = pd.DataFrame([[1, 1, 1], [2, np.nan, 2], [3, 3, 3]], dtype=ini_dtype)
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(df, order='A', dtype=da_conv_dtype)

    assert X.dtype == np_conv_dtype
    assert np.isnan(X[1][1])


@pytest.mark.parametrize('da_dtype, np_dtype',
                         [('float32', 'float32'),
                          ('float64', 'float64'),
                          ('float', 'float64'),
                          ('da_int', int_type)])
def test_list_casting(da_dtype, np_dtype):
    """
        Test data validation when input is a Python list.
    """
    x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(x, order='A', dtype=da_dtype, force_dtype=False)
    assert X.dtype == np_dtype

    x = [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(x, order='A', dtype=da_dtype, force_dtype=False)
    assert X.dtype == np_dtype


def test_list_invalid_input():
    """
        Test data validation when input is an invalid Python list.
    """
    x = [[1., 1, 1], [2, 'a', 2], [3, 3, 3]]

    with pytest.raises(ValueError), pytest.warns(UserWarning):
        X, _, _ = check_convert_data(x, order='A')


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float'])
def test_list_nan_casting(dtype):
    """
        Test data validation when input is a Python list with None.
    """
    x = [[1, 1, 1], [2, None, 2], [3, 3, 3]]
    with pytest.warns(UserWarning):
        X, _, _ = check_convert_data(x, order='A', dtype=dtype)
    assert np.isnan(X[1][1])
    assert X.dtype == dtype


def test_invalid_matrix_dimensions():
    """
        Test data validation when invalid matrix shape is inputted
    """

    x = [[1], [2, 2], [3, 3, 3]]

    with pytest.raises(ValueError), pytest.warns(UserWarning):
        X, _, _ = check_convert_data(x, order='A')


@pytest.mark.parametrize('dtype', ['float16', 'int16', 'bla', '1', 1, True])
def test_incorrect_parameter_force(dtype):
    """
        Test data validation when wrong dtype parameter is given.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    with pytest.raises(ValueError):
        X, _, _ = check_convert_data(X, order='A', dtype=dtype, force_dtype=True)


@pytest.mark.parametrize('order', ['F', 'C'])
def test_auto_order_deducing(order):
    """
        Test auto order deducing when validationg
    """

    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], order=order)

    X, deduced_order, _ = check_convert_data(X, order='A')

    assert order == deduced_order

    X = np.array([1, 1, 1], order=order)

    X, deduced_order, _ = check_convert_data(X, order='A')

    assert deduced_order == 'F'


@pytest.mark.parametrize('ini_order, fin_order', [('F', 'C'), ('C', 'F')])
def test_conversion_order_np(ini_order, fin_order):
    """
        Test conversion of orders for numpy arrays.
    """

    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], order=ini_order)

    with pytest.warns(UserWarning):
        X, order, _ = check_convert_data(X, order=fin_order)

    assert order == fin_order

    X = np.array([1, 1, 1], order=ini_order)

    X, order, _ = check_convert_data(X, order=fin_order)

    assert order == fin_order


@pytest.mark.parametrize('custom_order', ['C', 'F'])
def test_conversion_order_non_np(custom_order):
    """
        Test conversion of orders for non-numpy arrays.
    """

    X = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    with pytest.warns(UserWarning):
        Y, order, _ = check_convert_data(X, order='A')

    assert order == 'F'

    with pytest.warns(UserWarning):
        Y, order, _ = check_convert_data(X, order=custom_order)

    assert order == custom_order

    X = pd.DataFrame(X, columns=None)

    with pytest.warns(UserWarning):
        Y, order, _ = check_convert_data(X, order='A')

    assert order == 'F'

    with pytest.warns(UserWarning):
        Y, order, _ = check_convert_data(X, order=custom_order)

    assert order == custom_order


@pytest.mark.parametrize('order', ['F', 'C', 'A'])
def test_valid_parameter_order(order):
    """
        Test data validation when a valid order parameter is given.
    """

    if order in ['C', 'F']:
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], order=order)
    else:
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    X, _, _ = check_convert_data(X, order=order)


@pytest.mark.parametrize('order', ['col', 'row', 'c', 'f', 1, True, False])
def test_incorrect_parameter_order(order):
    """
        Test data validation when wrong order parameter is given.
    """

    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    with pytest.raises(ValueError):
        X, _, _ = check_convert_data(X, order=order)


@pytest.mark.parametrize('np_dtype, da_dtype',
                         [(int_type,
                           'da_int'),
                          ('float32',
                           'float32'),
                             ('float64',
                              'float64'),
                             ('float32',
                              'float'),
                             ('float32',
                              'float')])
def test_valid_return_dtype(np_dtype, da_dtype):
    """
        Test data validation returns the correct dtype based on input.
    """

    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np_dtype)

    X, _, return_dtype = check_convert_data(X, dtype=da_dtype, force_dtype=True)

    if da_dtype == 'float':
        assert return_dtype == np_dtype
    else:
        assert return_dtype == da_dtype
