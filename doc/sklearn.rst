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



.. _sklearn:

Extension for scikit-learn
****************************

In addition to the Python API, AOCL-DA offers an extension to enable existing users of scikit-learn
to extract better performance while making minimal changes to their code.

To use the extension, you must *patch* your existing code to replace the scikit-learn symbols with
AOCL-DA symbols. This can be done by inserting the following lines prior to your scikit-learn import
statement.

.. code-block::

   from aoclda.sklearn import skpatch, undo_skpatch
   skpatch()

You can switch back to standard scikit-learn using

.. code-block::

   undo_skpatch()

Note that after calling ``undo_skpatch``, you must reimport scikit-learn.

The ``skpatch`` and ``undo_skpatch`` functions can also be called with string or list arguments, specifying which scikit-learn package should be patched, for example:

.. code-block::

   skpatch("PCA")
   skpatch(["LinearRegression", "Ridge"])

Alternatively, you may wish to use the ``aoclda.sklearn`` module from the command line, without
making any changes to your own code:

.. code-block::

   python -m aoclda.sklearn your_python_script.py
   python -m aoclda.sklearn -m your_python_module

The following scikit-learn classes are currently available in the AOCL-DA extension.

.. list-table:: AOCL-DA Extension for scikit-learn
   :header-rows: 1

   * - scikit-learn class
     - Notes
   * - ``sklearn.cluster.KMeans``
     - ``fit``, ``transform``, ``predict``, ``fit_transform`` and ``fit_predict`` methods and various class attributes
   * - ``sklearn.decomposition.PCA``
     - ``fit``, ``transform``, ``inverse_transform`` and ``fit_transform`` methods and various class attributes
   * - ``sklearn.linear_model.LinearRegression``
     - ``fit``, ``predict`` and ``score`` methods and various class attributes
   * - ``sklearn.linear_model.Ridge``
     - ``fit`` and ``predict`` methods and various class attributes
   * - ``sklearn.linear_model.Lasso``
     - ``fit`` and ``predict`` methods and various class attributes
   * - ``sklearn.linear_model.ElasticNet``
     - ``fit`` and ``predict`` methods and various class attributes
   * - ``sklearn.linear_model.LogisticRegression``
     - ``fit`` and ``predict`` methods and various class attributes
   * - ``sklearn.tree.DecisionTreeClassifier``
     - ``fit``, ``predict``, ``score``, ``predict_proba`` and ``predict_log_proba`` methods and various class attributes
   * - ``sklearn.tree.RandomForestClassifier``
     - ``fit``, ``predict``, ``score``, ``predict_proba`` and ``predict_log_proba`` methods and various class attributes
   * - ``sklearn.metrics.pairwise``
     - ``pairwise_distances`` method with ``euclidean`` and ``sqeuclidean`` distances


Note that only a subset of the AOCL-DA functionality is available in this manner, and if, after
patching, you attempt to call class member functions which have not been implemented by AOCL-DA,
then a ``RuntimeError`` will be thrown. It is recommended that for the full benefit of using AOCL-DA
you use the Python APIs described on the subsequent pages of this manual.

The scikit-learn dispatcher
===========================
Since the AOCL-DA scikit-learn extension is not yet feature-complete, we also offer a dispatcher which can automatically select functions from Intel's Extension for scikit-learn where they are known to perform well.

To enable this dispatcher, set the environment variable ``USE_INTEL_SKLEARNEX``. The AOCL-DA scikit-learn extension will then detect your Intel installation and use it where appropriate.
