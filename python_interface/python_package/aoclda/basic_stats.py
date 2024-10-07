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
# pylint: disable = missing-module-docstring, unused-import
"""
aoclda.basic_stats module
"""

from ._aoclda.basic_stats import (
    pybind_mean, pybind_harmonic_mean, pybind_geometric_mean,
    pybind_variance, pybind_skewness, pybind_kurtosis, pybind_moment,
    pybind_quantile, pybind_five_point_summary, pybind_standardize,
    pybind_covariance_matrix, pybind_correlation_matrix)


def mean(X, axis="col"):
    """
        Arithmetic mean of a data matrix along the specified axis.

        For a dataset :math:`\{x_1, ..., x_n\}`, the arithmetic mean, \
        :math:`\\bar{x}`, is defined as

        .. math::
            \\bar{x}=\\frac{1}{n}\sum_{i=1}^{n} x_i.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            axis (str, optional): axis over which means are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
                (n_features, ) or (1, ): Calculated means.
        """
    return pybind_mean(X, axis)


def harmonic_mean(X, axis="col"):
    """
        Harmonic mean of a data matrix along the specified axis.

        For a dataset :math:`\{x_1, ..., x_n\}`, the harmonic mean, \
        :math:`\\bar{x}_{harm}`, is defined as

        .. math::
            \\bar{x}_{harm}=\\frac{n}{\sum_{i=1}^{n} \\frac{1}{x_i}}.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            axis (str, optional): axis over which means are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
                (n_features, ) or (1, ): Calculated harmonic means.
        """
    return pybind_harmonic_mean(X, axis)


def geometric_mean(X, axis="col"):
    """
        Geometric mean of a data matrix along the specified axis.

        For a dataset :math:`\{x_1, ..., x_n\}`, the harmonic mean, \
        :math:`\\bar{x}_{geom}`, is defined as

        .. math::
            \\bar{x}_{geom} = \left(\prod_{i=1}^n x_i\\right)^{\\frac{1}{n}}
            \equiv \exp\left(\\frac{1}{n}\sum_{i=1}^n\ln x_i\\right).

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            axis (str, optional): axis over which means are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
                (n_features, ) or (1, ): Calculated geometric means.
        """
    return pybind_geometric_mean(X, axis)


def variance(X, dof=0, axis="col"):
    """
        Variance of a data matrix along the specified axis.

        For a dataset :math:`\{x_1, ..., x_n\}`, the variance, :math:`s^2`, is defined as

        .. math::
            s^2 = \\frac{1}{\\text{dof}}\sum_{i=1}^n(x_i-\\bar{x})^2,

        where dof is the number of degrees of freedom. Setting :math:`\\text{dof} = n` \
        gives the sample variance, whereas setting :math:`\\text{dof}=n-1` \
        gives an unbiased estimate of the population variance.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            dof (int, optional): number of degrees of freedom used to compute the variance

                - If ``dof`` < 0 - the degrees of freedom will be set to the number \
                of observations, where the number of observations is n_samples for \
                column-wise variances, n_features for row-wise variances and \
                n_samples :math:`\\times` n_features for the overall variance

                - If ``dof`` = 0 - the degrees of freedom will be set to the number \
                of observations - 1.

                - If ``dof`` > 0 - the degrees of freedom will be set to the specified value.
            axis (str, optional): The axis over which variances are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
                (n_features, ) or (1, ): Calculated variances.
        """
    return pybind_variance(X, dof, axis)


def skewness(X, axis="col"):
    """
        Skewness of a data matrix along the specified axis.

        The skewness is computed as the Fischer-Pearson coefficient of skewness \
        (that is, with the central moments scaled by the number of \
        observations, see :cite:t:`kozw2000`).

        For a dataset :math:`\{x_1, ..., x_n\}`, the skewness, :math:`g_1`, is defined as

        .. math::
            g_1 = \\frac{\\frac{1}{n}\sum_{i=1}^n(x_i-\\bar{x})^3}
            {\left[\\frac{1}{n}\sum_{i=1}^n(x_i-\\bar{x})^2\\right]^{3/2}}.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            axis (str, optional): axis over which skewnesses are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
                (n_features, ) or (1, ): Calculated skewnesses.
        """
    return pybind_skewness(X, axis)


def kurtosis(X, axis="col"):
    """
        Kurtosis of a data matrix along the specified axis.

        The kurtosis is computed using Fischer's coefficient of excess kurtosis \
        (that is, with the central moments scaled by the number of observations \
        and 3 subtracted to ensure normally distributed data gives a value of 0, \
        see :cite:t:`kozw2000`).

        For a dataset :math:`\{x_1, ..., x_n\}`, the kurtosis, :math:`g_2`, is defined as

        .. math::
            g_2 = \\frac{\\frac{1}{n}\sum_{i=1}^n(x_i-\\bar{x})^4}
            {\left[\\frac{1}{n}\sum_{i=1}^n(x_i-\\bar{x})^2\\right]^{2}}-3.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            axis (str, optional): axis over which kurtoses are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
                (n_features, ) or (1, ): Calculated kurtoses.
        """
    return pybind_kurtosis(X, axis)


def moment(X, k, mean=None, axis="col"):
    """
        Central moment of a data matrix along the specified axis.

        For a dataset :math:`\{x_1, ..., x_n\}`, the :math:`k`-th central moment, \
        :math:`m_k`, is defined as

        .. math::
            m_k=\\frac{1}{n}\sum_{i=1}^n(x_i-\\bar{x})^k.

        Here, the moments are scaled by the number of observations along the specified axis. \
        The function gives you the option of supplying precomputed means (via the argument \
        ``mean``) about which the moments are computed. Otherwise it will compute the means itself.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            k (int): the order of the moment to be computed.
            mean (numpy.ndarray, optional): 1D array with precomputed means
            axis (str, optional): axis over which moments are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
            (n_features, ) or (1, ): Calculated moments.
        """
    return pybind_moment(X, k, mean, axis)


def quantile(X, q, method="linear", axis="col"):
    """
        Selected quantile of a data matrix along the specified axis.

        Computes the q-th quantiles of a data matrix along the specified axis. \
        Note that there are multiple ways to define quantiles. The available quantile types \
        correspond to the 9 different quantile types commonly used (see :cite:t:`hyfa96` \
        for further details). These can specified using the ``method`` parameter. In each \
        case a number :math:`h` is computed, corresponding to the approximate location in the \
        data array of the required quantile ``q``.

        Note:

            - Methods ``'inverted_cdf'``, ``'averaged_inverted_cdf'`` and \
                ``'closest_observation'`` give discontinuous results.

            - Method ``'median_unbiased'`` is recommended if the sample distribution \
                function is unknown.

            - Method ``'normal_unbiased'`` is recommended if the sample distribution \
                function is known to be normal.

            - Method ``'closest_observation'`` in contrast to NumPy, R and SAS rounds \
                to nearest order statistic and NOT nearest even order statistic.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            q (float): the quantile required, must lie in the interval [0,1].
            method (str, optional): specifies the method used to compute the quantiles.

                - If ``method = 'inverted_cdf'`` :math:`h=n\\times q`, return \
                    :math:`\\texttt{x[i]}` where :math:`i = \lceil h \\rceil`.

                - If ``method = 'averaged_inverted_cdf'`` :math:`h=n\\times q + 0.5`,\
                    return :math:`(\\texttt{x[i]}+\\texttt{x[j]})/2` where \
                    :math:`i = \lceil h-1/2 \\rceil` and :math:`j = \lfloor h+1/2 \\rfloor`.

                - If ``method = 'closest_observation'`` :math:`h=n\\times q - 0.5`, return \
                    :math:`\\texttt{x[i]}` where :math:`i = \lfloor h \\rceil` is \
                    the nearest integer to :math:`h`.

                - If ``method = 'interpolated_inverted_cdf'`` :math:`h=n\\times q`, return \
                    :math:`\\texttt{x[i]} + (h-\lfloor h \\rfloor)(\\texttt{x[j]}-\\texttt{x[i]})` \
                    where :math:`i = \lfloor h\\rfloor` and :math:`j = \lceil h \\rceil`.

                - If ``method = 'hazen'`` :math:`h=n\\times q + 0.5`, return \
                    :math:`\\texttt{x[i]} + (h-\lfloor h \\rfloor)(\\texttt{x[j]}-\\texttt{x[i]})`\
                    where :math:`i = \lfloor h\\rfloor` and :math:`j = \lceil h \\rceil`.

                - If ``method = 'weibull'`` :math:`h=(n + 1)\\times q`, return \
                    :math:`\\texttt{x[i]} + (h-\lfloor h \\rfloor)(\\texttt{x[j]}-\\texttt{x[i]})`\
                    where :math:`i = \lfloor h\\rfloor` and :math:`j = \lceil h \\rceil`.

                - If ``method = 'linear'`` :math:`h=(n - 1)\\times q + 1`, return \
                    :math:`\\texttt{x[i]} + (h-\lfloor h \\rfloor)(\\texttt{x[j]}-\\texttt{x[i]})`\
                    where :math:`i = \lfloor h\\rfloor` and :math:`j = \lceil h \\rceil`.

                - If ``method = 'median_unbiased'`` :math:`h=(n + 1/3)\\times q + 1/3`, return \
                    :math:`\\texttt{x[i]} + (h-\lfloor h \\rfloor)(\\texttt{x[j]}-\\texttt{x[i]})` \
                    where :math:`i = \lfloor h\\rfloor` and :math:`j = \lceil h \\rceil`.

                - If ``method = 'normal_unbiased'`` :math:`h=(n + 1/4)\\times q + 3/8`, return \
                    :math:`\\texttt{x[i]} + (h-\lfloor h \\rfloor)(\\texttt{x[j]}-\\texttt{x[i]})`\
                    where :math:`i = \lfloor h\\rfloor` and :math:`j = \lceil h \\rceil`.

            axis (str, optional): The axis over which quantiles are calculated.

        Returns:
            numpy.ndarray. Depending on ``axis`` can have shape (n_samples, ), \
                (n_features, ) or (1, ): Calculated quantiles.
        """
    return pybind_quantile(X, q, method, axis)


def five_point_summary(X, axis="col"):
    """
        Summary statistics of a data matrix along the specified axis.

        Computes the maximum, minimum, median and upper/lower hinges of a data array along \
        the specified axis.

        Note:

            - On large datasets, this function is more efficient than calling ``quantile()`` five \
                times because it uses partly sorted arrays after each stage.

            - The ``'weibull'`` definition of quantiles is used to calculate the statistics.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            axis (str, optional): axis over which summary is calculated.

        Returns:
            tuple of numpy.ndarray. Depending on an ``axis`` numpy.ndarray can have shape \
                (n_samples, ), (n_features, ) or (1, ): Tuple with calculated minimum, lower \
                hinge, median, upper hinge and maximum, respectively.
        """
    return pybind_five_point_summary(X, axis)


def standardize(X,
                shift=None,
                scale=None,
                dof=0,
                reverse=False,
                inplace=False,
                axis="col"):
    """
        Standardize a data matrix along the specified axis.

        This function can be called in various different ways

            - If the arrays ``shift`` and ``scale`` are both null, then the mean and standard \
                deviations will be computed along the appropriate axis and will be \
                used to shift and scale the data.

            - If the arrays ``shift`` and ``scale`` are both supplied, then the data matrix ``X`` \
                will be shifted (by subtracting the values in ``shift``) then scaled (by dividing \
                by the values in ``scale``) along the selected axis.

            - If one of the arrays ``shift`` or ``scale`` is null then it will be ignored and only \
                the other will be used (so that the data is only shifted or only scaled).

            In each case, if a 0 scaling factor is encountered then it will not be used.

        An additional computational mode is available by setting ``reverse = True``. In this \
        case the standardization is reversed, so that the data matrix is multiplied by the \
        values in scale before adding the values in shift. This enables users to undo the \
        standardization after the data has been used in another computation.

        Note:

            - The ``inplace`` functionality will only work if the supplied ``X`` \
            array is F-contiguous

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            shift (numpy.ndarray, optional): 1D array of values used for shifting the data.
            scale (numpy.ndarray, optional): 1D array of values used for scaling the data.
            dof (int, optional): number of degrees of freedom used to compute standard deviations

                - If ``dof`` < 0 - the degrees of freedom will be set to the number \
                    of observations in specified axis.

                - If ``dof`` = 0 - the degrees of freedom will be set to the number \
                    of observations - 1.

                - If ``dof`` > 0 - the degrees of freedom will be set to the \
                    specified value.
            reverse (bool, optional): determines whether or not the standardization \
                proceeds in reverse

                - If ``reverse = false`` - the data matrix will be shifted (by subtracting\
                    the values in ``shift``) then scaled (by dividing by the values in ``scale``).

                - If ``reverse = true`` - the data matrix will be scaled (by multiplying \
                    by the values in ``scale``) then shifted (by adding the values in ``shift``).
            inplace (bool, optional): determines whether the standardization is done without a copy
            axis (str, optional): axis over which matrix is standardized.

        Returns:
            numpy.ndarray of shape (n_samples, n_features): Standardized matrix
        """
    return pybind_standardize(X, shift, scale, dof, reverse, inplace, axis)


def covariance_matrix(X, dof=0):
    """
        Covariance matrix of a data matrix, with the rows treated as observations \
        and the columns treated as variables.

        For a dataset :math:`X = [\\textbf{x}_1, \dots, \\textbf{x}_{n_{\\text{cols}}}]^T`\
        with column means :math:`\{\\bar{x}_1, \dots, \\bar{x}_{n_{\\text{cols}}}\}`\
        the :math:`(i,j)` element of the covariance matrix is given by covariance \
        between :math:`\\textbf{x}_i` and :math:`\\textbf{x}_j`:

        .. math::
            \\text{cov}(i,j) = \\frac{1}{\\text{dof}}(\\textbf{x}_i-
            \\bar{x}_i)\cdot(\\textbf{x}_j-\\bar{x}_j),

        where dof is the number of degrees of freedom. Setting \
        :math:`\\text{dof} = n_{\\text{cols}}` gives the sample covariances, \
        whereas setting :math:`\\text{dof} = n_{\\text{cols}} -1` gives unbiased estimates \
        of the population covariances. The argument ``dof`` is used to \
        specify the number of degrees of freedom.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).
            dof (int, optional): number of degrees of freedom used to compute covariances

                - If ``dof`` < 0 - the degrees of freedom will be set to the number of observations.

                - If ``dof`` = 0 - the degrees of freedom will be set to the number of \
                observations - 1.

                - If ``dof`` > 0 - the degrees of freedom will be set to the specified value.

        Returns:
            numpy.ndarray of shape (n_features, n_features): Covariance matrix
        """
    return pybind_covariance_matrix(X, dof)


def correlation_matrix(X):
    """
        Correlation matrix of a data matrix, with the rows treated as observations and the \
        columns treated as variables.

        For a dataset :math:`X = [\\textbf{x}_1, \dots, \\textbf{x}_{n_{\\text{cols}}}]^T` \
        with column means :math:`\{\\bar{x}_1, \dots, \\bar{x}_{n_{\\text{cols}}}\}` \
        and column standard deviations :math:`\{\sigma_1, \dots, \sigma_{n_{\\text{cols}}}\}` \
        the :math:`(i,j)` element of the correlation matrix is given by correlation \
        between :math:`\\textbf{x}_i` and :math:`\\textbf{x}_j`:

        .. math::
            \\text{corr}(i,j) = \\frac{\\text{cov}(i,j)}{\sigma_i\sigma_j}.

        Note that the values in the correlation matrix are independent of the number of degrees \
        of freedom used to compute the standard deviations and covariances.

        Args:
            X (numpy.ndarray): data matrix of shape (n_samples, n_features).

        Returns:
            numpy.ndarray of shape (n_features, n_features): Correlation matrix
        """
    return pybind_correlation_matrix(X)
