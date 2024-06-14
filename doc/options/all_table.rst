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

In all the following tables, :math:`\varepsilon`, refers to a *safe* machine precision (twice the actual machine precion) for the given floating point data type.

.. _opts_linearmodel:

Linear Model
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for Linear Model.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "print level", "integer", ":math:`i=0`", "set level of verbosity for the solver", ":math:`0 \le i \le 5`"
   "optim convergence tol", "real", ":math:`r=10/2\sqrt{2\,\varepsilon}`", "tolerance to declare convergence for the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 < r < 1`"
   "intercept", "integer", ":math:`i=0`", "Add intercept variable to the model", ":math:`0 \le i \le 1`"
   "optim method", "string", ":math:`s=` `auto`", "Select optimization method to use.", ":math:`s=` `auto`, `bfgs`, `cg`, `chol`, `cholesky`, `coord`, `lbfgs`, `lbfgsb`, `qr`, `sparse_cg`, or `svd`."
   "optim iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform in the optimization phase. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`1 \le i`"
   "scaling", "string", ":math:`s=` `auto`", "Scale or standardize feature matrix and responce vector. Matrix is copied and then rescaled. Option key value auto indicates that rescaling type is choosen by the solver (this includes also no scaling).", ":math:`s=` `auto`, `centering`, `no`, `none`, `scale`, `scale only`, `standardise`, or `standardize`."
   "print options", "string", ":math:`s=` `no`", "Print options.", ":math:`s=` `no`, or `yes`."
   "optim coord skip min", "integer", ":math:`i=2`", "Minimum times a coordinate change is smaller than "coord skip tol" to start skipping", ":math:`2 \le i`"
   "optim coord skip max", "integer", ":math:`i=100`", "Maximum times a coordinate can be skipped, after this the coordinate is checked", ":math:`10 \le i`"
   "debug", "integer", ":math:`i=0`", "set debug level (internal use)", ":math:`0 \le i \le 3`"
   "optim time limit", "real", ":math:`r=10^6`", "Maximum time limit (in seconds). Solver will exit with a warning after this limit. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`0 < r`"
   "lambda", "real", ":math:`r=0`", "penalty coefficient for the regularization terms: lambda( (1-alpha)/2 L2 + alpha L1 )", ":math:`0 \le r`"
   "alpha", "real", ":math:`r=0`", "coefficient of alpha in the regularization terms: lambda( (1-alpha)/2 L2 + alpha L1 )", ":math:`0 \le r \le 1`"
   "optim progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "factor used to detect convergence of the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 \le r`"


.. _opts_pca:

PCA
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for PCA.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "svd solver", "string", ":math:`s=` `auto`", "Which LAPACK routine to use for the underlying singular value decomposition", ":math:`s=` `auto`, `gesdd`, `gesvd`, or `gesvdx`."
   "degrees of freedom", "string", ":math:`s=` `unbiased`", "Whether to use biased or unbiased estimators for standard deviations and variances", ":math:`s=` `biased`, or `unbiased`."
   "pca method", "string", ":math:`s=` `covariance`", "Compute PCA based on the covariance or correlation matrix", ":math:`s=` `correlation`, `covariance`, or `svd`."
   "n_components", "integer", ":math:`i=1`", "Number of principal components to compute. If 0, then all components will be kept.", ":math:`0 \le i`"


.. _opts_k-means:

k-means
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for k-means.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "algorithm", "string", ":math:`s=` `lloyd`", "Choice of underlying k-means algorithm", ":math:`s=` `elkan`, `hartigan-wong`, `lloyd`, or `macqueen`."
   "initialization method", "string", ":math:`s=` `random`", "How to determine the initial cluster centres", ":math:`s=` `k-means++`, `random`, `random partitions`, or `supplied`."
   "convergence tolerance", "real", ":math:`r=10^{-4}`", "Convergence tolerance", ":math:`0 \le r`"
   "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results", ":math:`-1 \le i`"
   "max_iter", "integer", ":math:`i=300`", "Maximum number of iterations", ":math:`1 \le i`"
   "n_init", "integer", ":math:`i=10`", "Number of runs with different random seeds (ignored if you have specified initial cluster centres)", ":math:`1 \le i`"
   "n_clusters", "integer", ":math:`i=1`", "Number of clusters required", ":math:`1 \le i`"


.. _opts_decisiontree:

Decision tree
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for Decision tree.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "minimum split improvement", "real", ":math:`r=0.03`", "Minimum score improvement needed to consider a split from the parent node.", ":math:`0 \le r`"
   "minimum split score", "real", ":math:`r=0.03`", "Minimum score needed for a node to be considered for splitting.", ":math:`0 \le r \le 1`"
   "tree building order", "string", ":math:`s=` `depth first`", "Select in which order to explore the nodes", ":math:`s=` `breadth first`, or `depth first`."
   "feature threshold", "real", ":math:`r=1e-06`", "Minimum difference in feature value required for splitting", ":math:`0 \le r`"
   "maximum features", "integer", ":math:`i=0`", "Set the number of features in consideration for splitting a node. 0 means take all the features.", ":math:`0 \le i`"
   "print timings", "string", ":math:`s=` `no`", "Print the timings of different part of the fitting process.", ":math:`s=` `no`, or `yes`."
   "seed", "integer", ":math:`i=-1`", "Set random seed for the random number generator. If the value is -1, a random seed is automatically generated.", ":math:`-1 \le i`"
   "maximum depth", "integer", ":math:`i=10`", "Set the maximum depth of trees.", ":math:`1 \le i \le 29`"
   "node minimum samples", "integer", ":math:`i=2`", "Minimum number of samples to consider a node for splitting", ":math:`2 \le i`"
   "scoring function", "string", ":math:`s=` `gini`", "Select scoring function to use", ":math:`s=` `cross-entropy`, `entropy`, `gini`, `misclass`, `misclassification`, or `misclassification-error`."


.. _opts_decisionforest:

Decision forest
==============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for Decision forest.`
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "minimum split improvement", "real", ":math:`r=0.03`", "Minimum score improvement needed to consider a split from the parent node.", ":math:`0 \le r`"
   "maximum features", "integer", ":math:`i=0`", "Set the number of features in consideration for splitting a node. 0 means take all the features.", ":math:`0 \le i`"
   "features selection", "string", ":math:`s=` `sqrt`", "Select how many features to use for each split", ":math:`s=` `all`, `custom`, `log2`, or `sqrt`."
   "bootstrap samples factor", "real", ":math:`r=0.8`", "Proportion of samples to draw from the data set to build each tree if 'bootstrap' was set to 'yes'.", ":math:`0 < r \le 1`"
   "minimum split score", "real", ":math:`r=0.03`", "Minimum score needed for a node to be considered for splitting.", ":math:`0 \le r \le 1`"
   "bootstrap", "string", ":math:`s=` `yes`", "Select wether to bootstrap the samples in the trees.", ":math:`s=` `no`, or `yes`."
   "feature threshold", "real", ":math:`r=1e-06`", "Minimum difference in feature value required for splitting", ":math:`0 \le r`"
   "tree building order", "string", ":math:`s=` `depth first`", "Select in which order to explore the nodes", ":math:`s=` `breadth first`, or `depth first`."
   "number of trees", "integer", ":math:`i=100`", "Set the number of trees to compute ", ":math:`1 \le i`"
   "seed", "integer", ":math:`i=-1`", "Set random seed for the random number generator. If the value is -1, a random seed is automatically generated.", ":math:`-1 \le i`"
   "maximum depth", "integer", ":math:`i=10`", "Set the maximum depth of trees.", ":math:`1 \le i \le 29`"
   "node minimum samples", "integer", ":math:`i=2`", "Minimum number of samples to consider a node for splitting", ":math:`2 \le i`"
   "scoring function", "string", ":math:`s=` `gini`", "Select scoring function to use", ":math:`s=` `cross-entropy`, `entropy`, `gini`, `misclass`, `misclassification`, or `misclassification-error`."


.. _opts_datastore:

Datastore handle :cpp:type:`da_datastore`
=============================================

The following options are supported.

.. csv-table:: :strong:`Table of options for` :cpp:type:`da_datastore`.
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "csv integers as floats", "integer", ":math:`i=0`", "Whether or not to interpret integers as floating point numbers when using autodetection", ":math:`0 \le i \le 1`"
   "csv datastore precision", "string", ":math:`s=` `double`", "The precision used when reading floating point numbers using autodetection", ":math:`s=` `double`, or `single`."
   "csv use header row", "integer", ":math:`i=0`", "Whether or not to interpret the first row as a header", ":math:`0 \le i \le 1`"
   "csv warn for missing data", "integer", ":math:`i=0`", "If set to 0, return error if missing data is encountered; if set to, 1 issue a warning and store missing data as either a NaN (for floating point data) or the maximum value of the integer type being used", ":math:`0 \le i \le 1`"
   "csv skip footer", "integer", ":math:`i=0`", "Whether or not to ignore the last line when reading a CSV file", ":math:`0 \le i \le 1`"
   "csv delimiter", "string", ":math:`s=` `,`", "The delimiter used when reading CSV files.", ""
   "csv whitespace delimiter", "integer", ":math:`i=0`", "Whether or not to use whitespace as the delimiter when reading CSV files", ":math:`0 \le i \le 1`"
   "csv decimal", "string", ":math:`s=` `.`", "The character used to denote a decimal point in CSV files", ""
   "csv skip initial space", "integer", ":math:`i=0`", "Whether or not to ignore initial spaces in CSV file lines", ":math:`0 \le i \le 1`"
   "csv line terminator", "string", "empty", "The character used to denote line termination in CSV files (leave this empty to use the default)", ""
   "csv row start", "integer", ":math:`i=0`", "Ignore the specified number of lines from the top of the file (note that line numbers in CSV files start at 1)", ":math:`0 \le i`"
   "csv comment", "string", ":math:`s=` `#`", "The character used to denote comments in CSV files (note, if a line in a CSV file is to be interpreted as only containing a comment, the comment character should be the first character on the line)", ""
   "csv quote character", "string", ":math:`s=` `~"`", "The character used to denote quotations in CSV files", ""
   "csv scientific notation character", "string", ":math:`s=` `e`", "The character used to denote powers of 10 in floating point values in CSV files", ""
   "csv escape character", "string", ":math:`s=` `\\`", "The escape character in CSV files", ""
   "csv thousands", "string", "empty", "The character used to separate thousands when reading numeric values in CSV files", ""
   "csv skip rows", "string", "empty", "A comma- or space-separated list of rows to ignore in CSV files", ""
   "csv datatype", "string", ":math:`s=` `auto`", "If a CSV file is known to be of a single datatype, set this option to disable autodetection and make reading the file quicker", ":math:`s=` `auto`, `boolean`, `double`, `float`, `integer`, or `string`."
   "csv data storage", "string", ":math:`s=` `column major`", "Whether to store data from CSV files in row or column major format", ":math:`s=` `column major`, or `row major`."
   "csv skip empty lines", "integer", ":math:`i=0`", "Whether or not to ignore empty lines in CSV files (note that caution should be used when using this in conjunction with options such as CSV skip rows since line numbers may no longer correspond to the original line numbers in the CSV file)", ":math:`0 \le i \le 1`"
   "csv double quote", "integer", ":math:`i=0`", "Whether or not to interpret two consecutive quotechar characters within a field as a single quotechar character", ":math:`0 \le i \le 1`"


.. only:: internal
   
   .. _opts_optimizationsolvers:
   
   Optimization Solvers
   ====================
   
   The following options are supported.
   
   .. csv-table:: :strong:`Table of options for optimization solvers.`
      :escape: ~
      :header: "Option name", "Type", "Default", "Description", "Constraints"
      
      "regularization power", "string", ":math:`s=` `quadratic`", "Value for the regularization power term.", ":math:`s=` `cubic`, or `quadratic`."
      "infinite bound size", "real", ":math:`r=10^{20}`", "threshold value to take for +/- infinity", ":math:`1000 < r`"
      "coord progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "the iteration stops when (fk - f{k+1})/max{abs(fk);abs(f{k+1});1} <= factr*epsmch where epsmch is the machine precision. Typical values for type double: 10e12 for low accuracy; 10e7 for moderate accuracy; 10 for extremely high accuracy.", ":math:`0 \le r`"
      "print level", "integer", ":math:`i=1`", "set level of verbosity for the solver 0 indicates no output while 5 is a very verbose printing", ":math:`0 \le i \le 5`"
      "monitoring frequency", "integer", ":math:`i=0`", "How frequent to call the user-supplied monitor function", ":math:`0 \le i`"
      "ralfit iteration limit", "integer", ":math:`i=100`", "Maximum number of iterations to perform.", ":math:`1 \le i`"
      "lbfgsb memory limit", "integer", ":math:`i=11`", "Number of vectors to use for approximating the Hessian", ":math:`1 \le i \le 1000`"
      "lbfgsb convergence tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "tolerance of the projected gradient infinity norm to declare convergence", ":math:`0 < r < 1`"
      "lbfgsb iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform", ":math:`1 \le i`"
      "coord iteration limit", "integer", ":math:`i=100000`", "Maximum number of iterations to perform", ":math:`1 \le i`"
      "lbfgsb progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "the iteration stops when (f^k - f{k+1})/max{abs(fk);abs(f{k+1});1} <= factr*epsmch where epsmch is the machine precision. Typical values for type double: 10e12 for low accuracy; 10e7 for moderate accuracy; 10 for extremely high accuracy.", ":math:`0 \le r`"
      "coord skip min", "integer", ":math:`i=2`", "Minimum times a coordinate change is smaller than "coord skip tol" to start skipping", ":math:`2 \le i`"
      "debug", "integer", ":math:`i=0`", "set debug level (internal use)", ":math:`0 \le i \le 3`"
      "regularization term", "real", ":math:`r=0`", "Value for the regularization term. A value of 0 disables regularization.", ":math:`0 \le r`"
      "time limit", "real", ":math:`r=10^6`", "maximum time allowed to run (in seconds)", ":math:`0 < r`"
      "coord convergence tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "tolerance of the projected gradient infinity norm to declare convergence", ":math:`0 < r < 1`"
      "ralfit convergence rel tol fun", "real", ":math:`r=10^{-8}`", "relative tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
      "coord skip tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "Coordinate skip tolerance, a given coordinate could be skipped if the change between two consecutive iterates is less than tolerance. Any negative value disables the skipping scheme", ":math:`-1 \le r`"
      "ralfit convergence abs tol grd", "real", ":math:`r=10^{-5}`", "absolute tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
      "coord skip max", "integer", ":math:`i=100`", "Maximum times a coordinate can be skipped, after this the coordinate is checked", ":math:`10 \le i`"
      "ralfit convergence rel tol grd", "real", ":math:`r=10^{-8}`", "relative tolerance on the gradient norm to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
      "ralfit nlls method", "string", ":math:`s=` `galahad`", "NLLS solver to use.", ":math:`s=` `aint`, `galahad`, `linear solver`, `more-sorensen`, or `powell-dogleg`."
      "ralfit convergence step size", "real", ":math:`r=\varepsilon/2`", "absolute tolerance over the step size to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
      "coord restart", "integer", ":math:`i=\infty`", "Number of inner inner iterations to perform before requesting to perform a full evaluation of the step function", ":math:`0 \le i`"
      "optim method", "string", ":math:`s=` `lbfgsb`", "Select optimization solver to use", ":math:`s=` `bfgs`, `coord`, `lbfgs`, `lbfgsb`, or `ralfit`."
      "ralfit model", "string", ":math:`s=` `hybrid`", "NLLS model to solve.", ":math:`s=` `gauss-newton`, `hybrid`, `quasi-newton`, or `tensor-newton`."
      "ralfit convergence abs tol fun", "real", ":math:`r=10^{-8}`", "absolute tolerance to declare convergence for the iterative optimization step. See details in optimization solver documentation.", ":math:`0 < r < 1`"
      "print options", "string", ":math:`s=` `no`", "Print options list", ":math:`s=` `no`, or `yes`."
      "ralfit globalization method", "string", ":math:`s=` `trust-region`", "Globalization method to use. This parameter makes use of the regularization term and power option values.", ":math:`s=` `reg`, `regularization`, `tr`, or `trust-region`."
      "storage scheme", "string", ":math:`s=` `c`", "Define the storage scheme used to store multi-dimensional arrays (Jacobian matrix, etc).", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."
   
