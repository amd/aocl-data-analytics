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
aoclda._internal_utils module
"""

import warnings
import numpy as np
from ._aoclda._internal_utils import (
    pybind_get_int_info, pybind_debug_set, pybind_debug_get,
    pybind_debug_print_context_registry)


class debug():
    """
    A utility class for managing a context registry with debug information.
    """

    @staticmethod
    def set(dic):
        """ Setter for the context registry """
        if not isinstance(dic, dict):
            raise TypeError("set data ``dic'' must be a dictionary.")
        for key, value in dic.items():
            if value is None:
                value = ""
            pybind_debug_set(key, value)

    @staticmethod
    def get(key=None):
        """ Getter for the context registry """
        if key is None:
            pybind_debug_print_context_registry()
            return {}

        # check if key is a string or a list, tuple, or set of strings
        ok_list = isinstance(key, (list, tuple, set)) and all(
            isinstance(elem, str) for elem in key)
        ok_str = isinstance(key, str)
        ok = ok_list or ok_str
        if not ok:
            raise TypeError(
                "get data key ``key'' must contain at least one string.")

        keys = [key] if isinstance(key, str) else key

        dic = {}
        for k in keys:
            value = pybind_debug_get(k)
            dic[k] = value

        return dic


def get_int_info():
    """
    returns a string describing the integer size of da_int:
    "32" if da_int = int32_t
    "64" if da_int = int64_t
    "?"  otherwise
    """
    return pybind_get_int_info()


def get_order(X):
    """
    Determines the memory layout order ('C', 'F' or 'A') of an array-like input.

    For numpy arrays, checks the contiguity flags to determine if the array is
    C-contiguous (row-major) or F-contiguous (column-major). If both (1D arrays)
    it returns 'A' (Any). For non-numpy inputs, defaults to 'F' (column-major).

    Args:
        X (array-like): The data matrix to determine the order of.

    Returns:
        str: 'C' for row-major (C-contiguous), 'F' for column-major (F-contiguous),
            or 'A' (Any) if both 'C' and 'F' contiguous.
    """
    order = "F"

    if not isinstance(X, np.ndarray):
        return order

    c_contig = X.flags['C_CONTIGUOUS']
    f_contig = X.flags['F_CONTIGUOUS']

    if c_contig and not f_contig:
        order = "C"
    elif c_contig and f_contig:
        # For arrays that are both C and F contiguous, don't assign an arbitrary layout
        order = 'A'

    return order


def check_convert_data(X, order='A', dtype='float64', force_dtype=False):
    """
    Checks an array-like input X and returns a corresponding NumPy array. Data
    may be converted to a different type to match the list of accepted types
    [da_int, float32, float64] or can be transformed to a different order ['C', 'F'].

    If 'force_dtype' is not True, the original type of X is conserved as much
    as possible. In this case, the dtype argument is only used as a hint for the
    output data if no obvious type can be inferred from the input X.

    If 'force_dtype' is set to True, the output WILL contain dtype data (raising an exception
    if dtype is not supported).

    When 'order' is 'C' or 'F' ('Fortran') it will convert X to the specific memory layout.
    When 'order' is 'A', the function attempts to deduce the memory layout from X:
    - If X is both 'C' and 'F' contiguous (e.g., 1D arrays), return_order will be 'F'.
    - If X has a clear layout preference, that layout is preserved.
    - For non-numpy inputs, defaults to 'F' order.


    Args:
        X (array-like): The data matrix to validate and convert.

        order (str | optional): Memory layout for the output array. If 'A',
            deduces order from X. Accepted values are 'C' (row-major),
            'F' or 'Fortran' (column-major) and 'A' (Any). Default is 'A'.

        dtype (str, optional): Data type to be used on output. Valid values are
            [da_int, float, float32, float64]. If float and force_dtype are selected
            it will keep the input float type if it is float32 or float64. If
            da_int is selected it will convert to either int32 or int64 based
            on the library build. Default is 'float64'.

        force_dtype (bool, optional): Boolean to force data type conversion to dtype.
            Default is False.

    Returns:
        X (numpy.ndarray[float32 | float64 | da_int]): The validated data matrix.

        target_order (str): Memory layout of the matrix. 'C' or 'F'.

        return_dtype (str): dtype of the returned matrix X. 'float32', 'float64' or 'da_int'.
    """
    if order not in ['C', 'F', 'A', 'Fortran']:
        raise ValueError(
            f"{order} is not a valid value for order. Use 'C', 'F', 'Fortran' or 'A' instead.")

    x_order = get_order(X)

    target_order = order
    if order == 'A' and x_order == 'A':
        # For 1D arrays default to F
        target_order = 'F'
    elif order == 'A':
        target_order = x_order
    elif order == 'Fortran':
        target_order == 'F'

    supported_float_types = ['float32', 'float64']
    supported_int_type = "int" + get_int_info()

    if dtype not in supported_float_types and dtype not in ['da_int', 'float']:
        raise ValueError(
            f"{dtype} is not a valid value for dtype. Use one of float," +
            "float32, float64, da_int instead.")

    if dtype == "da_int":
        arg_dtype = supported_int_type
    else:
        arg_dtype = dtype

    target_dtype = 'float64'
    input_dtype = None
    if len(X) > 0 and hasattr(X[0], 'dtype'):
        input_dtype = X[0].dtype
    if force_dtype:
        # Force the output to match the dtype argument
        if arg_dtype in supported_float_types or arg_dtype == supported_int_type:
            target_dtype = arg_dtype
        elif arg_dtype == 'float' and input_dtype in supported_float_types:
            target_dtype = input_dtype
    else:
        # Try to keep the type as close as possible to the input
        if input_dtype is not None and np.isdtype(input_dtype, 'integral'):
            target_dtype = supported_int_type
        elif input_dtype in supported_float_types:
            target_dtype = input_dtype
        elif arg_dtype in supported_float_types or arg_dtype == supported_int_type:
            target_dtype = arg_dtype

    if target_dtype == supported_int_type:
        return_dtype = "da_int"
    else:
        return_dtype = str(np.dtype(target_dtype))

    if isinstance(
            X, np.ndarray) and input_dtype == target_dtype and target_order == x_order:
        return X, target_order, return_dtype

    try:
        if (target_order != x_order or not isinstance(
                X, np.ndarray)) and not x_order == 'A':
            warnings.warn(
                "The provided array had a different order from what was used " +
                f"to initialize the handle ({x_order} and {order}). For optimal " +
                "performance use consistent orders.",
                UserWarning)

        X = np.asarray(X, dtype=target_dtype, order=target_order)

    except AttributeError as exc:
        raise ValueError(
            "Internal error: array cannot be converted to a NumPy ndarray.") from exc

    return X, target_order, return_dtype
