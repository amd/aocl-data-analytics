..
    Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

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



Decision tree and decision forest APIs
**************************************

This chapter contains two sets of APIs, one for classification using a single :ref:`decision tree <da_decision_trees_apis>` and another
one for the ensemble method :ref:`decision forests<da_decision_forests_apis>` (also known as random forests).

.. _da_decision_trees_apis:

Decision trees
==============

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      .. autoclass:: aoclda.decision_tree.decision_tree(criterion='gini', seed=-1, max_depth=10, max_features=0, min_samples_split=2, build_order='breadth first', min_impurity_decrease=0.0, min_split_score=0.0, feat_thresh=1.0e-06, check_data=false)
         :members:

   .. tab-item:: C
      :sync: C

      .. _da_tree_set_training_data:

      .. doxygenfunction:: da_tree_set_training_data_s
         :outline:
      .. doxygenfunction:: da_tree_set_training_data_d

      .. _da_tree_fit:

      .. doxygenfunction:: da_tree_fit_s
         :outline:
      .. doxygenfunction:: da_tree_fit_d

      .. _da_tree_predict:

      .. doxygenfunction:: da_tree_predict_s
         :outline:
      .. doxygenfunction:: da_tree_predict_d

      .. _da_tree_predict_proba:

      .. doxygenfunction:: da_tree_predict_proba_s
         :outline:
      .. doxygenfunction:: da_tree_predict_proba_d

      .. _da_tree_predict_log_proba:

      .. doxygenfunction:: da_tree_predict_log_proba_s
         :outline:
      .. doxygenfunction:: da_tree_predict_log_proba_d

      .. _da_tree_score:

      .. doxygenfunction:: da_tree_score_s
         :outline:
      .. doxygenfunction:: da_tree_score_d


.. _da_decision_forests_apis:

Decision forests
================

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      .. autoclass:: aoclda.decision_forest.decision_forest(criterion='gini', bootstrap=True, n_trees=100, features_selection='sqrt', max_features=0, seed=-1, max_depth=10, min_samples_split=2, build_order='breadth first', samples_factor=0.8, min_impurity_decrease=0.0, min_split_score=0.0, feat_thresh=1.0e-06, check_data=false)
         :members:

   .. tab-item:: C
      :sync: C

      .. _da_forest_set_training_data:

      .. doxygenfunction:: da_forest_set_training_data_s
         :outline:
      .. doxygenfunction:: da_forest_set_training_data_d

      .. _da_forest_fit:

      .. doxygenfunction:: da_forest_fit_s
         :outline:
      .. doxygenfunction:: da_forest_fit_d

      .. _da_forest_predict:

      .. doxygenfunction:: da_forest_predict_s
         :outline:
      .. doxygenfunction:: da_forest_predict_d

      .. _da_forest_predict_proba:

      .. doxygenfunction:: da_forest_predict_proba_s
         :outline:
      .. doxygenfunction:: da_forest_predict_proba_d

      .. _da_forest_predict_log_proba:

      .. doxygenfunction:: da_forest_predict_log_proba_s
         :outline:
      .. doxygenfunction:: da_forest_predict_log_proba_d

      .. _da_forest_score:

      .. doxygenfunction:: da_forest_score_s
         :outline:
      .. doxygenfunction:: da_forest_score_d
