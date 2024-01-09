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

:orphan:

Decision Forests
****************

Decision Forests (also referred to as Random Forests) are ensembles of Decision Trees.  Decision Trees split the feature
domain into rectangular regions and make predictions based on the label values in each region.

A node in the Decision Tree is terminal if it does not have any child nodes.  Let :math:`m` be the index of a terminal
node in the tree.  If we are solving a classification problem with two classes, the proportion of observations in the
second class for node `m` is defined as,

.. math::

   \hat{p}^0_m = \frac{1}{N_m} \sum_{x_i \in R_m} I(y_i = 1)

where :math:`R_m` is the rectangular region corresponding to the :math:`m` th node, :math:`x_i` is a (multi-dimensional)
observation, :math:`y_i` is a label, and :math:`N_m` is the number of observations in :math:`R_m`.

The node impurity is defined as,

.. math::

   C^{0}_m = Q(\hat{p}^0_m)

where :math:`Q(\hat{p}^0_m)` is a function that quantifies the error in the classification fit, with the property that
if all observations in :math:`R_m` are in the same class, then :math:`Q(\hat{p}^0_m)=0` and we say that the node is
pure.

AOCL-DA supports the following choices of :math:`Q(p)`,

.. math::

   \mathrm{Misclassification\ error: }   & \ 1 - \max(p, 1-p) \\
   \mathrm{Gini\ index: }                & \ 1 - 2 p (1-p)    \\
   \operatorname{Cross-entropy or deviance: } & \ -p \log(p) - (1-p) \log(1-p)

Fitting
--------

Decision Tree fitting is done by growing a tree recursively, starting with a single base node.  Each node has a depth
associated with it, which corresponds to the number of splits required to get to that node from the base node.  If the
depth of the node is less than the maximum depth and the node is not pure, we minimize the sum of the node impurities
over each possible split,

.. math::

   C_m(j^*, s^*) = \min_{j, s} \bigg( Q(\hat{p}^1_m(j,s)) + Q(\hat{p}^2_m(j,s)) \bigg)

where

.. math::

   \hat{p}^1_m(j,s) &= \sum_{ \{i \ : x_i(j) \leq s \} } I(y_i = 1) \\
   \hat{p}^2_m(j,s) &= \sum_{ \{i \ : x_i(j) > s \} } I(y_i = 1)

and :math:`j` is the feature index, so that :math:`x_i(j)` is the :math:`j` th feature of the :math:`i` th observation.

Then node :math:`m` stops being a terminal node, and two new terminal nodes are created as children of node :math:`m`.
The domains for the new child nodes are defined by splitting :math:`R_m` into two rectangles using the variable index
:math:`j^*` and the threshold :math:`s^*`.

This recursive splitting continues until all terminal nodes are either pure, or have reached the maximum depth.  The
maximum depth is a user-defined parameter and, for Decision Trees, needs to be selected in such a way that avoids
over-fitting.

Decision Forest fitting is done by bootstrapping observations from the training data for each Decision Tree in the
ensemble.  Then fitting an ensemble of Decision Trees.  In contrast to standard Decision Tree fitting, the candidate set
of features for each split is a randomly sampled subset.  The subset selection is repeated at each split so a different
subset of candidate features is used on each split.  In both Decision Trees and Decision Forests, the order in which
features are selected is randomised.  This means that if two different splits have the same value of :math:`C_m(j, s)`,
then whichever value of :math:`j` is sampled first will be the split variable.

Prediction
------------

Decision Tree prediction is done by using the set of split feature indices and thresholds to determine which terminal
node a new observations belongs to.  Once the terminal node has been identified, the prediction is determined by the
proportion of observations with label :math:`1` in the training set: if an observation :math:`x` is associated with a
terminal node :math:`m`, and :math:`\hat{p}^0_m > 0.5`, the predicted label is :math:`1`, otherwise it is :math:`0`.

Decision Forest prediction is done by producing an ensemble of Decision Tree predictions and using a majority vote to
determine the prediction of the Decision Forest.

Typical workflow for Decision Random Forests
---------------------------------------------

The following workflow can be used to fit a Decision Tree or a Decision Forest model and use it make predictions,

1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_decision_tree`` /
   ``da_handle_decision_forest``.
2. Pass data to the handle using either :cpp:func:`da_df_set_training_data_s` or :cpp:func:`da_df_set_training_data_d`.
3. Set optional parameters, such as maximum depth, using :cpp:func:`da_options_set_int` and
   :cpp:func:`da_options_set_string`  (see :ref:`options section <df_options>`).
4. Fit the model using :cpp:func:`da_df_fit_s` or :cpp:func:`da_df_fit_d`.
5. Evaluate prediction accuracy on test data using :cpp:func:`da_df_score_s` or :cpp:func:`da_df_score_d`.
6. Make predictions using the fitted model using :cpp:func:`da_df_predict_s` or :cpp:func:`da_df_predict_d`.

Options
-------

For details of optional parameters see the :ref:`options section <df_options>`.

Examples
--------

See ``examples/decision_tree_ex.cpp`` for an example of how to use these functions.

Further Reading
----------------

An introduction to Decision Trees and to Random Forests can be found in Chapters 9 and 15 of :cite:t:`hastie`.

.. toctree::
    :maxdepth: 1
    :hidden:

    decision_tree_api
