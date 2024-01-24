Supported Optional Parameters
**************************************

In all the following tables, :math:`\varepsilon`, refers to the machine precision for the given floating point data precision.

Options for Linear Model
==============================================

The following options are supported.

.. csv-table:: Options table
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "linmod optim method", "string", ":math:`s=` `auto`", "Select optimization method to use.", ":math:`s=` `auto`, `coord`, `lbfgs`, `lbfgsb`, or `qr`."
   "linmod optim progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "factor used to detect convergence of the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 \le r`"
   "linmod optim convergence tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "tolerance to declare convergence for the iterative optimization step. See option in the corresponding optimization solver documentation.", ":math:`0 < r < 1`"
   "print options", "string", ":math:`s=` `no`", "Print options.", ":math:`s=` `no`, or `yes`."
   "linmod lambda", "real", ":math:`r=0`", "penalty coefficient for the regularization terms: lambda( (1-alpha) L2 + alpha L1 )", ":math:`0 \le r`"
   "linmod alpha", "real", ":math:`r=0`", "coefficient of alpha in the regularization terms: lambda( (1-alpha) L2 + alpha L1 )", ":math:`0 \le r \le 1`"
   "linmod optim iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform in the optimization phase. Valid only for iterative solvers, e.g. L-BFGS-B, Coordinate Descent, etc.", ":math:`1 \le i`"
   "linmod intercept", "integer", ":math:`i=0`", "Add intercept variable to the model", ":math:`0 \le i \le 1`"
   "print level", "integer", ":math:`i=0`", "set level of verbosity for the solver", ":math:`0 \le i \le 5`"


Options for PCA
==============================================

The following options are supported.

.. csv-table:: Options table
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "svd solver", "string", ":math:`s=` `auto`", "Which LAPACK routine to use for the underlying singular value decomposition", ":math:`s=` `auto`, `gesdd`, `gesvd`, or `gesvdx`."
   "degrees of freedom", "string", ":math:`s=` `unbiased`", "Whether to use biased or unbiased estimators for standard deviations and variances", ":math:`s=` `biased`, or `unbiased`."
   "pca method", "string", ":math:`s=` `covariance`", "Compute PCA based on the covariance or correlation matrix", ":math:`s=` `correlation`, `covariance`, or `svd`."
   "n_components", "integer", ":math:`i=1`", "Number of principal components to compute", ":math:`1 \le i`"


Options for k-means
==============================================

The following options are supported.

.. csv-table:: Options table
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "algorithm", "string", ":math:`s=` `lloyd`", "Choice of underlying k-means algorithm", ":math:`s=` `elkan`, `hartigan-wong`, or `lloyd`."
   "initialization method", "string", ":math:`s=` `random`", "How to determine the initial cluster centres", ":math:`s=` `k-means++`, `random`, or `supplied`."
   "convergence tolerance", "real", ":math:`r=10^{-4}`", "Convergence tolerance", ":math:`0 < r`"
   "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results", ":math:`-1 \le i`"
   "max_iter", "integer", ":math:`i=300`", "Maximum number of iterations", ":math:`1 \le i`"
   "n_init", "integer", ":math:`i=10`", "Number of runs with different random seeds (ignored if you have specified initial cluster centres)", ":math:`1 \le i`"
   "n_clusters", "integer", ":math:`i=1`", "Number of clusters required", ":math:`1 \le i`"

.. _df_options:


Options for Decision Forest
==============================================

The following options are supported.

.. csv-table:: Options table
   :escape: ~
   :header: "Option name", "Type", "Default", "Description", "Constraints"
   
   "diff_thres", "real", ":math:`r=1e-06`", "minimum difference in feature value required for splitting", ":math:`0 < r`"
   "n_trees", "integer", ":math:`i=1`", "set number of trees", ":math:`0 < i`"
   "n_features_to_select", "integer", ":math:`i=1`", "set number of features in selection for splitting", ":math:`0 < i`"
   "seed", "integer", ":math:`i=-1`", "Set random seed for Mersenne Twister (64-bit) PRNG.  If the value is -1, the std::random_device function is used to generate a seed, otherwise the input value is used as a seed.", ":math:`-1 \le i`"
   "n_obs_per_tree", "integer", ":math:`i=1`", "set number of observations in each tree", ":math:`0 < i`"
   "depth", "integer", ":math:`i=-1`", "set max depth of tree", ":math:`-1 \le i`"
   "scoring function", "string", ":math:`s=` `gini`", "Select scoring function to use", ":math:`s=` `cross-entropy`, `gini`, or `misclassification-error`."


Options for datastore
=============================================

The following options are supported.

.. csv-table:: Options table
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
   "csv row start", "integer", ":math:`i=0`", "Ignore the specified number of lines from the top of the CSV file (note that line numbers in CSV files start at 1)", ":math:`0 \le i`"
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
   
   Optimization Solvers
   ====================
   
   The following options are supported.
   
   .. csv-table:: Options table
      :escape: ~
      :header: "Option name", "Type", "Default", "Description", "Constraints"
      
      "optim method", "string", ":math:`s=` `lbfgsb`", "Select optimization solver to use", ":math:`s=` `bfgs`, `coord`, `lbfgs`, or `lbfgsb`."
      "print options", "string", ":math:`s=` `no`", "Print options list", ":math:`s=` `no`, or `yes`."
      "coord skip tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "Coordinate skip tolerance", ":math:`0 < r`"
      "coord convergence tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "tolerance of the projected gradient infinity norm to declare convergence", ":math:`0 < r < 1`"
      "coord skip min", "integer", ":math:`i=5`", "Minimum times a coordinate change is smaller than "coord skip tol" to start skipping", ":math:`1 \le i`"
      "coord skip max", "integer", ":math:`i=8`", "Initial max times a coordinate can be skipped after this the coordinate is checked", ":math:`4 \le i`"
      "coord restart", "integer", ":math:`i=\infty`", "Number of inner iteration to perform before requesting to perform a full evaluation of the step function", ":math:`0 \le i`"
      "coord iteration limit", "integer", ":math:`i=100000`", "Maximum number of iterations to perform", ":math:`1 \le i`"
      "lbfgsb iteration limit", "integer", ":math:`i=10000`", "Maximum number of iterations to perform", ":math:`1 \le i`"
      "lbfgsb convergence tol", "real", ":math:`r=\sqrt{2\,\varepsilon}`", "tolerance of the projected gradient infinity norm to declare convergence", ":math:`0 < r < 1`"
      "lbfgsb memory limit", "integer", ":math:`i=11`", "Number of vectors to use for approximating the Hessian", ":math:`1 \le i \le 1000`"
      "debug", "integer", ":math:`i=0`", "set debug level (internal use)", ":math:`0 \le i \le 3`"
      "monitoring frequency", "integer", ":math:`i=0`", "How frequent to call the user-supplied monitor function", ":math:`0 \le i`"
      "print level", "integer", ":math:`i=1`", "set level of verbosity for the solver 0 indicates no output while 5 is a very verbose printing", ":math:`0 \le i \le 5`"
      "coord progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "the iteration stops when (fk - f{k+1})/max{abs(fk);abs(f{k+1});1} <= factr*epsmch where epsmch is the machine precision. Typical values for type double: 10e12 for low accuracy; 10e7 for moderate accuracy; 10 for extremely high accuracy.", ":math:`0 \le r`"
      "infinite bound size", "real", ":math:`r=10^{20}`", "threshold value to take for +/- infinity", ":math:`1000 < r`"
      "time limit", "real", ":math:`r=10^6`", "maximum time allowed to run", ":math:`0 < r`"
      "lbfgsb progress factor", "real", ":math:`r=\frac{10}{\sqrt{2\,\varepsilon}}`", "the iteration stops when (f^k - f{k+1})/max{abs(fk);abs(f{k+1});1} <= factr*epsmch where epsmch is the machine precision. Typical values for type double: 10e12 for low accuracy; 10e7 for moderate accuracy; 10 for extremely high accuracy.", ":math:`0 \le r`"
   
