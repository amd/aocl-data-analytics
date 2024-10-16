..
    Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.



.. AUTO GENERATED. Do not hand edit this file! (see doc_test.cpp)

Supported Optional Parameters
******************************

.. note::
   This page lists optional parameters for **C APIs** only.

In all the following tables, :math:`\varepsilon`, refers to a *safe* machine precision (twice the actual machine precision) for the given floating point data type.

.. _opts_linearmodels:

Linear Models
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for Linear Models.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"

   "optim method", "string", ":math:`s=` `auto`", "Select optimization method to use.", ":math:`s=` `auto`, `bfgs`, `cg`, `chol`, `cholesky`, `coord`, `lbfgs`, `lbfgsb`, `qr`, `sparse_cg`, or `svd`."
   "scaling", "string", ":math:`s=` `auto`", "Scale or standardize feature matrix and response vector. Matrix is copied and then rescaled. Option key value auto indicates that rescaling type is chosen by the solver (this also includes no scaling).", ":math:`s=` `auto`, `centering`, `no`, `none`, `scale`, `scale only`, `standardise`, or `standardize`."
   "print options", "string", ":math:`s=` `no`", "Print options.", ":math:`s=` `no`, or `yes`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "print level", "integer", ":math:`i=0`", "Set level of verbosity for the solver.", ":math:`0 \le i \le 5`"
   "optim convergence tol", "real", ":math:`r=10/2\sqrt{2\,\varepsilon}`", "Tolerance to declare convergence for the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 < r < 1`"
   "intercept", "integer", ":math:`i=0`", "Add intercept variable to the model.", ":math:`0 \le i \le 1`"
   "optim iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform in the optimization phase. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`1 \le i`"
   "optim coord skip min", "integer", ":math:`i=2`", "Minimum times a coordinate change is smaller than coord skip tol to start skipping.", ":math:`2 \le i`"
   "optim coord skip max", "integer", ":math:`i=100`", "Maximum times a coordinate can be skipped, after this the coordinate is checked.", ":math:`10 \le i`"
   "debug", "integer", ":math:`i=0`", "Set debug level (internal use).", ":math:`0 \le i \le 3`"
   "logistic constraint", "string", ":math:`s=` `ssc`", "Affects only multinomial logistic regression. Type of constraint put on coefficients. This will affect number of coefficients returned. RSC - means we choose a reference category whose coefficients will be set to all 0. This results in K-1 class coefficients for problems with K classes. SSC - means the sum of coefficients class-wise for each feature is 0. It will result in K class coefficients for problems with K classes.", ":math:`s=` `reference category`, `rsc`, `ssc`, `symmetric`, or `symmetric side`."
   "optim time limit", "real", ":math:`r=10^6`", "Maximum time limit (in seconds). Solver will exit with a warning after this limit. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`0 < r`"
   "lambda", "real", ":math:`r=0`", "Penalty coefficient for the regularization terms: lambda( (1-alpha)/2 L2 + alpha L1 ).", ":math:`0 \le r`"
   "alpha", "real", ":math:`r=0`", "Coefficient of alpha in the regularization terms: lambda( (1-alpha)/2 L2 + alpha L1 ).", ":math:`0 \le r \le 1`"
   "optim progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "Factor used to detect convergence of the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 \le r`"


.. _opts_principalcomponentanalysis:

Principal Component Analysis
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for Principal Component Analysis.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"

   "degrees of freedom", "string", ":math:`s=` `unbiased`", "Whether to use biased or unbiased estimators for standard deviations and variances.", ":math:`s=` `biased`, or `unbiased`."
   "pca method", "string", ":math:`s=` `covariance`", "Compute PCA based on the covariance or correlation matrix.", ":math:`s=` `correlation`, `covariance`, or `svd`."
   "store u", "integer", ":math:`i=0`", "Whether or not to store the matrix U from the SVD.", ":math:`0 \le i \le 1`"
   "n_components", "integer", ":math:`i=1`", "Number of principal components to compute. If 0, then all components will be kept.", ":math:`0 \le i`"
   "svd solver", "string", ":math:`s=` `auto`", "Which LAPACK routine to use for the underlying singular value decomposition.", ":math:`s=` `auto`, `gesdd`, `gesvd`, `gesvdx`, or `syevd`."
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."


.. _opts_k-meansclustering:

k-means Clustering
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for k-means Clustering.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"

   "algorithm", "string", ":math:`s=` `lloyd`", "Choice of underlying k-means algorithm.", ":math:`s=` `elkan`, `hartigan-wong`, `lloyd`, or `macqueen`."
   "initialization method", "string", ":math:`s=` `random`", "How to determine the initial cluster centres.", ":math:`s=` `k-means++`, `random`, `random partitions`, or `supplied`."
   "convergence tolerance", "real", ":math:`r=10^{-4}`", "Convergence tolerance.", ":math:`0 \le r`"
   "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results.", ":math:`-1 \le i`"
   "max_iter", "integer", ":math:`i=300`", "Maximum number of iterations.", ":math:`1 \le i`"
   "n_clusters", "integer", ":math:`i=1`", "Number of clusters required.", ":math:`1 \le i`"
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "n_init", "integer", ":math:`i=10`", "Number of runs with different random seeds (ignored if you have specified initial cluster centres).", ":math:`1 \le i`"
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."


.. _opts_dbscanclustering:

DBSCAN clustering
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for DBSCAN clustering.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "power", "real", ":math:`r=2.0`", "The power of the Minkowski metric used (reserved for future use).", ":math:`0 \le r`"
   "metric", "string", ":math:`s=` `euclidean`", "Choice of metric used to compute pairwise distances (reserved for future use).", ":math:`s=` `euclidean`, `manhattan`, `minkowski`, or `sqeuclidean`."
   "algorithm", "string", ":math:`s=` `brute`", "Choice of algorithm (reserved for future use).", ":math:`s=` `auto`, `ball tree`, `brute`, `brute serial`, or `kd tree`."
   "leaf size", "integer", ":math:`i=30`", "Leaf size for KD tree or ball tree (reserved for future use).", ":math:`1 \le i`"
   "eps", "real", ":math:`r=10^{-4}`", "Maximum distance for two samples to be considered in each other's neighborhood.", ":math:`0 \le r`"
   "min samples", "integer", ":math:`i=5`", "Minimum number of neighborhood samples for a core point.", ":math:`1 \le i`"
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."


.. _opts_decisiontrees:

Decision Trees
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for Decision Trees.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "print timings", "string", ":math:`s=` `no`", "Print the timings of different parts of the fitting process.", ":math:`s=` `no`, or `yes`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
   "sorting method", "string", ":math:`s=` `boost`", "Select sorting method to use.", ":math:`s=` `boost`, or `stl`."
   "feature threshold", "real", ":math:`r=1e-06`", "Minimum difference in feature value required for splitting.", ":math:`0 \le r`"
   "tree building order", "string", ":math:`s=` `depth first`", "Select in which order to explore the nodes.", ":math:`s=` `breadth first`, or `depth first`."
   "node minimum samples", "integer", ":math:`i=2`", "The minimum number of samples required to split an internal node.", ":math:`2 \le i`"
   "predict probabilities", "integer", ":math:`i=1`", "evaluate class probabilities (in addition to class predictions).Needs to be 1 if calls to predict_proba or predict_log_probaare made after fit.", ":math:`0 \le i \le 1`"
   "scoring function", "string", ":math:`s=` `gini`", "Select scoring function to use.", ":math:`s=` `cross-entropy`, `entropy`, `gini`, `misclass`, `misclassification`, or `misclassification-error`."
   "maximum depth", "integer", ":math:`i=29`", "Set the maximum depth of trees.", ":math:`0 \le i \le 61`"
   "seed", "integer", ":math:`i=-1`", "Set the random seed for the random number generator. If the value is -1, a random seed is automatically generated. In this case the resulting classification will create non-reproducible results.", ":math:`-1 \le i`"
   "maximum features", "integer", ":math:`i=0`", "Set the number of features to consider when splitting a node. 0 means take all the features.", ":math:`0 \le i`"
   "minimum split score", "real", ":math:`r=0.03`", "Minimum score needed for a node to be considered for splitting.", ":math:`0 \le r \le 1`"
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "minimum split improvement", "real", ":math:`r=0.03`", "Minimum score improvement needed to consider a split from the parent node.", ":math:`0 \le r`"


.. _opts_decisionforests:

Decision Forests
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for Decision Forests.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "block size", "integer", ":math:`i=256`", "Set the size of the blocks for parallel computations.", ":math:`1 \le i \le 9223372036854775807`"
   "feature threshold", "real", ":math:`r=1e-06`", "Minimum difference in feature value required for splitting.", ":math:`0 \le r`"
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
   "minimum split improvement", "real", ":math:`r=0.03`", "Minimum score improvement needed to consider a split from the parent node.", ":math:`0 \le r`"
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "minimum split score", "real", ":math:`r=0.03`", "Minimum score needed for a node to be considered for splitting.", ":math:`0 \le r \le 1`"
   "maximum features", "integer", ":math:`i=0`", "Set the number of features to consider when splitting a node. 0 means take all the features.", ":math:`0 \le i`"
   "number of trees", "integer", ":math:`i=100`", "Set the number of trees to compute. ", ":math:`1 \le i`"
   "tree building order", "string", ":math:`s=` `depth first`", "Select in which order to explore the nodes.", ":math:`s=` `breadth first`, or `depth first`."
   "node minimum samples", "integer", ":math:`i=2`", "Minimum number of samples to consider a node for splitting.", ":math:`2 \le i`"
   "scoring function", "string", ":math:`s=` `gini`", "Select scoring function to use.", ":math:`s=` `cross-entropy`, `entropy`, `gini`, `misclass`, `misclassification`, or `misclassification-error`."
   "maximum depth", "integer", ":math:`i=29`", "Set the maximum depth of trees.", ":math:`0 \le i \le 61`"
   "seed", "integer", ":math:`i=-1`", "Set random seed for the random number generator. If the value is -1, a random seed is automatically generated. In this case the resulting classification will create non-reproducible results.", ":math:`-1 \le i`"
   "bootstrap", "string", ":math:`s=` `yes`", "Select whether to bootstrap the samples in the trees.", ":math:`s=` `no`, or `yes`."
   "sorting method", "string", ":math:`s=` `boost`", "Select sorting method to use.", ":math:`s=` `boost`, or `stl`."
   "bootstrap samples factor", "real", ":math:`r=0.8`", "Proportion of samples to draw from the data set to build each tree if 'bootstrap' was set to 'yes'.", ":math:`0 < r \le 1`"
   "features selection", "string", ":math:`s=` `sqrt`", "Select how many features to use for each split.", ":math:`s=` `all`, `custom`, `log2`, or `sqrt`."


.. _opts_nonlinearleastsquares:

Nonlinear Least Squares
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for Nonlinear Least Squares.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"

   "ralfit model", "string", ":math:`s=` `hybrid`", "NLLS model to solve.", ":math:`s=` `gauss-newton`, `hybrid`, `quasi-newton`, or `tensor-newton`."
   "print level", "integer", ":math:`i=1`", "Set level of verbosity for the solver: from 0, indicating no output, to 5, which is very verbose.", ":math:`0 \le i \le 5`"
   "derivative test tol", "real", ":math:`r=10^{-4}`", "Tolerance used to check user-provided derivatives by finite-differences. If <print level> is 1, then only the entries with larger discrepancy are reported, and if print level is greater than or equal to 2, then all entries are printed.", ":math:`0 < r \le 10`"
   "ralfit iteration limit", "integer", ":math:`i=100`", "Maximum number of iterations to perform.", ":math:`1 \le i`"
   "lbfgsb memory limit", "integer", ":math:`i=11`", "Number of vectors to use for approximating the Hessian.", ":math:`1 \le i \le 1000`"
   "lbfgsb iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform.", ":math:`1 \le i`"
   "coord iteration limit", "integer", ":math:`i=100000`", "Maximum number of iterations to perform.", ":math:`1 \le i`"
   "monitoring frequency", "integer", ":math:`i=0`", "How frequently to call the user-supplied monitor function.", ":math:`0 \le i`"
   "check derivatives", "string", ":math:`s=` `no`", "Check user-provided derivatives using finite-differences.", ":math:`s=` `no`, or `yes`."
   "ralfit nlls method", "string", ":math:`s=` `galahad`", "NLLS solver to use.", ":math:`s=` `aint`, `galahad`, `linear solver`, `more-sorensen`, or `powell-dogleg`."
   "optim method", "string", ":math:`s=` `lbfgsb`", "Select optimization solver to use.", ":math:`s=` `bfgs`, `coord`, `lbfgs`, `lbfgsb`, or `ralfit`."
   "ralfit convergence step size", "real", ":math:`r=\varepsilon/2`", "Absolute tolerance over the step size to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
   "coord restart", "integer", ":math:`i=\infty`", "Number of inner iterations to perform before requesting to perform a full evaluation of the step function.", ":math:`0 \le i`"
   "ralfit convergence rel tol grd", "real", ":math:`r=10/21\sqrt{2\,\varepsilon}`", "Relative tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
   "coord skip max", "integer", ":math:`i=100`", "Maximum times a coordinate can be skipped, after this the coordinate is checked.", ":math:`10 \le i`"
   "coord skip min", "integer", ":math:`i=2`", "Minimum times a coordinate change is smaller than coord skip tol to start skipping.", ":math:`2 \le i`"
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
   "ralfit globalization method", "string", ":math:`s=` `trust-region`", "Globalization method to use. This parameter makes use of the regularization term and power option values.", ":math:`s=` `reg`, `regularization`, `tr`, or `trust-region`."
   "ralfit convergence abs tol fun", "real", ":math:`r=10/21\sqrt{2\,\varepsilon}`", "Absolute tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
   "print options", "string", ":math:`s=` `no`", "Print options list.", ":math:`s=` `no`, or `yes`."
   "debug", "integer", ":math:`i=0`", "Set debug level (internal use).", ":math:`0 \le i \le 3`"
   "regularization term", "real", ":math:`r=0`", "Value of the regularization term. A value of 0 disables regularization.", ":math:`0 \le r`"
   "finite differences step", "real", ":math:`r=10\;\sqrt{2\,\varepsilon}`", "Size of step to use for estimating derivatives using finite-differences.", ":math:`0 < r < 10`"
   "lbfgsb convergence tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "Tolerance of the projected gradient infinity norm to declare convergence.", ":math:`0 < r < 1`"
   "lbfgsb progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "The iteration stops when (f^k - f{k+1})/max{abs(fk);abs(f{k+1});1} <= factr*epsmch where epsmch is the machine precision. Typical values for type double: 10e12 for low accuracy; 10e7 for moderate accuracy; 10 for extremely high accuracy.", ":math:`0 \le r`"
   "regularization power", "string", ":math:`s=` `quadratic`", "Value of the regularization power term.", ":math:`s=` `cubic`, or `quadratic`."
   "infinite bound size", "real", ":math:`r=10^{20}`", "Threshold value to take for +/- infinity.", ":math:`1000 < r`"
   "coord progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "The iteration stops when (fk - f{k+1})/max{abs(fk);abs(f{k+1});1} <= factr*epsmch where epsmch is the machine precision. Typical values for type double: 10e12 for low accuracy; 10e7 for moderate accuracy; 10 for extremely high accuracy.", ":math:`0 \le r`"
   "time limit", "real", ":math:`r=10^6`", "Maximum time allowed to run (in seconds).", ":math:`0 < r`"
   "coord convergence tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "Tolerance of the projected gradient infinity norm to declare convergence.", ":math:`0 < r < 1`"
   "ralfit convergence rel tol fun", "real", ":math:`r=10/21\sqrt{2\,\varepsilon}`", "Relative tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
   "coord skip tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "Coordinate skip tolerance, a given coordinate could be skipped if the change between two consecutive iterates is less than tolerance. Any negative value disables the skipping scheme.", ":math:`-1 \le r`"
   "ralfit convergence abs tol grd", "real", ":math:`r=500\;\sqrt{2\,\varepsilon}`", "Absolute tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"


.. _opts_k-nearestneighbors:

k-Nearest Neighbors
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for k-Nearest Neighbors.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"

   "weights", "string", ":math:`s=` `uniform`", "Weight function used to compute the k-nearest neighbors.", ":math:`s=` `distance`, or `uniform`."
   "metric", "string", ":math:`s=` `euclidean`", "Metric used to compute the pairwise distance matrix.", ":math:`s=` `cityblock`, `cosine`, `euclidean`, `l1`, `l2`, `manhattan`, `minkowski`, or `sqeuclidean`."
   "algorithm", "string", ":math:`s=` `brute`", "Algorithm used to compute the k-nearest neighbors.", ":math:`s=` `brute`."
   "minkowski parameter", "real", ":math:`r=2`", "Minkowski parameter for metric used for the computation of k-nearest neighbors.", ":math:`0 < r`"
   "number of neighbors", "integer", ":math:`i=5`", "Number of neighbors considered for k-nearest neighbors.", ":math:`1 \le i`"
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."


.. _opts_supportvectormachines:

Support Vector Machines
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of Options for Support Vector Machines.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "kernel", "string", ":math:`s=` `rbf`", "Kernel function to use for the calculations.", ":math:`s=` `linear`, `poly`, `polynomial`, `rbf`, or `sigmoid`."
   "coef0", "real", ":math:`r=0`", "Constant in 'polynomial' and 'sigmoid' kernels.", "There are no constraints on :math:`r`."
   "gamma", "real", ":math:`r=-1`", "Parameter for 'rbf', 'polynomial', and 'sigmoid' kernels. If the value is less than 0, it is set to 1/(n_features * Var(X)).", ":math:`-1 \le r`"
   "epsilon", "real", ":math:`r=0.1`", "Defines the tolerance for errors in predictions by creating an acceptable margin (tube) within which errors are not penalized. Applies to SVR", ":math:`0 \le r`"
   "tau", "real", ":math:`r=\varepsilon`", "Numerical stability parameter used in working set selection when kernel is not positive semi definite.", ":math:`0 \le r`"
   "tolerance", "real", ":math:`r=10^{-3}`", "Convergence tolerance.", ":math:`0 < r`"
   "nu", "real", ":math:`r=0.5`", "An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors. Applies to NuSVC and NuSVR.", ":math:`0 < r \le 1`"
   "max_iter", "integer", ":math:`i=0`", "Sets the maximum number of iterations. Use 0 to specify no limit.", ":math:`0 \le i`"
   "c", "real", ":math:`r=1`", "Regularization parameter. Controls the trade-off between maximizing the margin between classes and minimizing classification errors. A larger value means higher penalty to the loss function on misclassified observations. Applies to SVC, SVR and NuSVR.", ":math:`0 < r`"
   "degree", "integer", ":math:`i=3`", "Parameter for 'polynomial' kernel.", ":math:`1 \le i`"
   "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
   "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."


.. _opts_k-nearestneighbors:

k-nearest neighbors
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for k-nearest neighbors.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "weights", "string", ":math:`s=` `uniform`", "Weight function used to compute the k-nearest neighbors.", ":math:`s=` `distance`, or `uniform`."
   "metric", "string", ":math:`s=` `euclidean`", "Metric used to compute the pairwise distance matrix.", ":math:`s=` `euclidean`, or `sqeuclidean`."
   "algorithm", "string", ":math:`s=` `brute`", "Algorithm used to compute the k-nearest neighbors.", ":math:`s=` `brute`."
   "number of neighbors", "integer", ":math:`i=5`", "Number of neighbors considered for k-nearest neighbors.", ":math:`1 \le i`"


.. _opts_datastore:

Datastore handle :cpp:type:`da_datastore`
=============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for` :cpp:type:`da_datastore`.
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"

   "datastore precision", "string", ":math:`s=` `double`", "The precision used when reading floating point numbers using autodetection.", ":math:`s=` `double`, or `single`."
   "datatype", "string", ":math:`s=` `auto`", "If a CSV file is known to be of a single datatype, set this option to disable autodetection and make reading the file quicker.", ":math:`s=` `auto`, `boolean`, `double`, `float`, `integer`, or `string`."
   "use header row", "integer", ":math:`i=0`", "Whether or not to interpret the first row as a header.", ":math:`0 \le i \le 1`"
   "skip empty lines", "integer", ":math:`i=0`", "Whether or not to ignore empty lines in CSV files (note that caution should be used when using this in conjunction with options such as CSV skip rows since line numbers may no longer correspond to the original line numbers in the CSV file).", ":math:`0 \le i \le 1`"
   "delimiter", "string", ":math:`s=` `,`", "The delimiter used when reading CSV files.", ""
   "warn for missing data", "integer", ":math:`i=0`", "If set to 0, return error if missing data is encountered; if set to 1, issue a warning and store missing data as either a NaN (for floating point data) or the maximum value of the integer type being used.", ":math:`0 \le i \le 1`"
   "thousands", "string", "empty", "The character used to separate thousands when reading numeric values in CSV files.", ""
   "quote character", "string", ":math:`s=` `~"`", "The character used to denote quotations in CSV files.", ""
   "decimal", "string", ":math:`s=` `.`", "The character used to denote a decimal point in CSV files.", ""
   "scientific notation character", "string", ":math:`s=` `e`", "The character used to denote powers of 10 in floating point values in CSV files.", ""
   "skip footer", "integer", ":math:`i=0`", "Whether or not to ignore the last line when reading a CSV file.", ":math:`0 \le i \le 1`"
   "skip rows", "string", "empty", "A comma- or space-separated list of rows to ignore in CSV files.", ""
   "comment", "string", ":math:`s=` `#`", "The character used to denote comments in CSV files (note, if a line in a CSV file is to be interpreted as only containing a comment, the comment character should be the first character on the line).", ""
   "whitespace delimiter", "integer", ":math:`i=0`", "Whether or not to use whitespace as the delimiter when reading CSV files.", ":math:`0 \le i \le 1`"
   "escape character", "string", ":math:`s=` `\\`", "The escape character in CSV files.", ""
   "line terminator", "string", "empty", "The character used to denote line termination in CSV files (leave this empty to use the default).", ""
   "integers as floats", "integer", ":math:`i=0`", "Whether or not to interpret integers as floating point numbers when using autodetection.", ":math:`0 \le i \le 1`"
   "row start", "integer", ":math:`i=0`", "Ignore the specified number of lines from the top of the file (note that line numbers in CSV files start at 1).", ":math:`0 \le i`"
   "storage order", "string", ":math:`s=` `column-major`", "Whether to return data in row- or column-major format.", ":math:`s=` `column-major`, or `row-major`."
   "skip initial space", "integer", ":math:`i=0`", "Whether or not to ignore initial spaces in CSV file lines.", ":math:`0 \le i \le 1`"
   "double quote", "integer", ":math:`i=0`", "Whether or not to interpret two consecutive quotechar characters within a field as a single quotechar character.", ":math:`0 \le i \le 1`"

