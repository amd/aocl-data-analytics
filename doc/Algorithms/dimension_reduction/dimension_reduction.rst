..
    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

Dimension Reduction
*******************

This chapter contains algorithms for reducing the dimensionality of data, typically for
visualization or as a preprocessing step for downstream analysis.

.. _tsne_intro:

*t*-SNE
=======

*t*-distributed Stochastic Neighbor Embedding (*t*-SNE) is a nonlinear dimensionality
reduction technique for embedding high-dimensional data into a low-dimensional space
(typically two or three dimensions) suitable for visualization :cite:p:`da_vandermaaten2008`.

For each pair of data points in the high-dimensional space, *t*-SNE defines a conditional
probability that reflects how likely one point would pick the other as its neighbor under
a Gaussian kernel centered on that point. The bandwidths of the Gaussian kernels are
chosen so that each point's conditional distribution has a fixed perplexity (an effective
number of neighbors). These conditional probabilities are symmetrized to form a joint
distribution :math:`P` over pairs. In the low-dimensional embedding, a similar joint
distribution :math:`Q` is constructed using a Student's :math:`t`-distribution with one
degree of freedom (a Cauchy distribution), which has heavier tails than a Gaussian and
alleviates the crowding problem that arises when mapping many dimensions down to few.
The embedding coordinates are then optimized by minimizing the Kullback-Leibler divergence
:math:`\mathrm{KL}(P \| Q)` via gradient descent.

AOCL-DA supports both exact *t*-SNE and the Barnes-Hut approximation
:cite:p:`da_vandermaaten2014`. In the exact method, pairwise interactions between all
points are computed at each iteration. The Barnes-Hut variant uses a space-partitioning
tree to approximate repulsive forces between distant points, reducing the per-iteration
cost from :math:`O(n^2)` to :math:`O(n \log n)`. The trade-off between speed and accuracy
is controlled by the :math:`\theta \in [0, 1]` parameter: :math:`\theta = 0` gives the exact
algorithm, while larger values increase the approximation but improve performance.

Mathematical formulation
------------------------

Given :math:`n` data points :math:`x_1, \dots, x_n` in :math:`\mathbb{R}^d`, *t*-SNE first
computes conditional probabilities :math:`p_{j|i}` representing the similarity of
:math:`x_j` to :math:`x_i`:

.. math::

   p_{j|i} = \frac{\exp\!\bigl(-\lVert x_i - x_j \rVert^2 / 2\sigma_i^2\bigr)}
                   {\sum_{k \neq i} \exp\!\bigl(-\lVert x_i - x_k \rVert^2 / 2\sigma_i^2\bigr)},

where :math:`\sigma_i` is chosen so that the perplexity of the conditional distribution
equals a user-specified target. The symmetrized joint distribution is

.. math::

   P_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}.

In the low-dimensional embedding :math:`y_1, \dots, y_n`, pairwise affinities use a
Student's :math:`t`-distribution with one degree of freedom:

.. math::

   Q_{ij} = \frac{\bigl(1 + \lVert y_i - y_j \rVert^2\bigr)^{-1}}
                  {\sum_{k \neq l} \bigl(1 + \lVert y_k - y_l \rVert^2\bigr)^{-1}}.

The embedding is found by minimizing the Kullback--Leibler divergence:

.. math::

   \mathrm{KL}(P \| Q) = \sum_{i \neq j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}.

Implementation notes
--------------------

The implementation uses ``theta`` to select the algorithmic path:

- ``theta = 0`` uses the exact method.
- ``theta > 0`` uses the Barnes-Hut approximation.

The early exaggeration factor is applied for the first
:math:`\min(250,` ``max_iter`` :math:`)` iterations, then set to 1 for the remaining
iterations.

The embedding is updated with gradient descent using momentum and per-coordinate adaptive
gains. Each coordinate maintains a velocity :math:`v_i` that accumulates gradient
information across iterations. At each iteration the velocity and embedding are updated as

.. math::

   v_i^{(t+1)} &= \alpha \, v_i^{(t)}
                 - \eta \, g_i^{(t)} \, \frac{\partial \mathrm{KL}}{\partial y_i^{(t)}}, \\
   y_i^{(t+1)} &= y_i^{(t)} + v_i^{(t+1)},

where :math:`\eta` is the learning rate, :math:`g_i^{(t)}` is an adaptive per-coordinate
gain, and :math:`\alpha` is the momentum coefficient. Momentum carries forward a fraction
of the previous update, which smooths the trajectory and helps avoid poor local optima.
:math:`\alpha` starts at 0.5 and switches to 0.8 at iteration 250. When early exaggeration
ends, gains and momentum are reset to their initial values.

Convergence checks are performed every 50 iterations and at the final iteration. The
algorithm can terminate early when either:

- the gain-weighted gradient norm falls below ``min grad norm``; or
- there is no KL-divergence improvement for ``n_iter_without_progress`` iterations after
  the early exaggeration phase.

Setting ``min grad norm`` to 0 disables the gradient norm criterion. Setting
``n_iter_without_progress`` to 0 disables stagnation detection.

In the Barnes-Hut approximation, :math:`k = \min(n - 1,\; \lfloor 3 \times \text{perplexity} + 1 \rfloor)`
nearest neighbors are used to compute the sparse affinity matrix.

Per-row conditional probabilities are calibrated by binary search. In numerically
degenerate cases (for example, underflow during calibration), the corresponding row falls
back to a uniform distribution over its neighbors.

Outputs from *t*-SNE
--------------------

After a *t*-SNE computation the following results are stored:

- **embedding** - the low-dimensional coordinates of shape (n_samples, n_components).
- **KL divergence** - the final KL divergence value.
- **n_iter** - the number of iterations actually performed.

Typical workflow for *t*-SNE
----------------------------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      1. Initialize a :func:`aoclda.dimension_reduction.tsne` object with options set in the class constructor.
      2. Compute the embedding using :func:`aoclda.dimension_reduction.tsne.fit` or :func:`aoclda.dimension_reduction.tsne.fit_transform`.
      3. Extract results from the :func:`aoclda.dimension_reduction.tsne` object via its class attributes.

   .. tab-item:: C
      :sync: C

      1. Initialize a :cpp:type:`da_handle` with :cpp:type:`da_handle_type` ``da_handle_tsne``.
      2. Set options using :ref:`da_options_set_? <da_options_set>` (see :ref:`below <tsne_options>`).
      3. Pass data to the handle using :cpp:func:`da_tsne_set_data_?<da_tsne_set_data_s>`.
      4. Compute the embedding using :cpp:func:`da_tsne_compute_?<da_tsne_compute_s>`.
      5. Extract results using :ref:`da_handle_get_result_? <da_handle_get_result>`.

.. _tsne_options:

Options
-------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The available Python options are detailed in the :func:`aoclda.dimension_reduction.tsne` class
      constructor.

   .. tab-item:: C
      :sync: C

      The following options can be set using :ref:`da_options_set_? <da_options_set>`:

      .. update options using table _opts_t-sne

      .. csv-table:: *t*-SNE options
         :header: "Option name", "Type", "Default", "Description", "Constraints"

         "n_components", "integer", ":math:`i=2`", "Number of embedding dimensions.", ":math:`1 \le i \le 3`"
         "perplexity", "real", ":math:`r=30`", "Target perplexity for conditional probabilities.", ":math:`1 \le r`"
         "learning rate", "real", ":math:`r=-1`", "Gradient descent learning rate. Use any non-positive value for auto: max(N / early_exaggeration / 4, 50).", "There are no constraints on :math:`r`."
         "max_iter", "integer", ":math:`i=1000`", "Maximum number of gradient descent iterations.", ":math:`1 \le i`"
         "n_iter_without_progress", "integer", ":math:`i=300`", "Stop if no progress is made for this many iterations.", ":math:`0 \le i`"
         "min_grad_norm", "real", ":math:`r=1e-07`", "Stop if the gradient norm is below this threshold.", ":math:`0 \le r`"
         "early exaggeration", "real", ":math:`r=12`", "Exaggeration factor for early iterations.", ":math:`1 \le r`"
         "theta", "real", ":math:`r=0.5`", "Barnes-Hut approximation parameter (0 for exact).", ":math:`0 \le r \le 1`"
         "init", "string", ":math:`s=` `pca`", "Initialization method for the embedding.", ":math:`s=` `pca`, `random`, or `supplied`."
         "seed", "integer", ":math:`i=0`", "Seed for random number generation; set to -1 for non-deterministic results.", ":math:`-1 \le i`"
         "mixed precision", "string", ":math:`s=` `no`", "Whether to use mixed precision iterative refinement, in which lower precision arithmetic is used before switching to the working precision for the final iterations.", ":math:`s=` `no`, or `yes`."
         "low precision max_iter", "integer", ":math:`i=200`", "If mixed precision iterative refinement is enabled, maximum number of iterations for the low precision phase.", ":math:`1 \le i`"
         "low precision min_grad_norm", "real", ":math:`r=0.0001`", "If mixed precision iterative refinement is enabled, gradient norm convergence threshold for the low precision phase.", ":math:`0 \le r`"
         "check data", "string", ":math:`s=` `no`", "Check input data for NaNs prior to performing computation.", ":math:`s=` `no`, or `yes`."
         "storage order", "string", ":math:`s=` `column-major`", "Whether data is supplied and returned in row- or column-major order.", ":math:`s=` `c`, `column-major`, `f`, `fortran`, or `row-major`."

      If ``init`` is set to ``supplied``, an explicit initial embedding must be provided
      via :ref:`da_tsne_set_init_embedding_? <da_tsne_set_init_embedding>` before calling
      :ref:`da_tsne_compute_? <da_tsne_compute>`.

      If ``learning rate`` is set to any non-positive value (the default is ``-1``), the learning rate is
      automatically set to :math:`\max(n_{\mathrm{samples}} /` ``early exaggeration`` :math:`/ 4,\; 50)`.

      For options with data-dependent bounds, the valid maxima are determined after data
      is set:

      - ``n_components`` must satisfy :math:`n_{\mathrm{components}} \le \min(3, n_{\mathrm{features}})`.
      - ``perplexity`` must satisfy :math:`\text{perplexity} \le n_{\mathrm{samples}} - 1`.

      If a user-provided value exceeds these bounds, it is reduced to the nearest valid
      value and the API returns an incompatible-options warning status.

      The option ``mixed precision`` switches on an experimental mode in which an initial embedding is found in lower precision before refining the result in the working precision.
      The option ``low precision max_iter`` sets the maximum number of iterations for the low-precision phase, and ``low precision min_grad_norm`` sets the convergence tolerance for the low-precision phase.


Examples
--------

.. tab-set::

   .. tab-item:: Python
      :sync: Python

      The code below is supplied with your installation (see :ref:`Python examples <python_examples>`).

      .. dropdown:: t-SNE Example

          .. literalinclude:: ../../../python_interface/python_package/aoclda/examples/tsne_ex.py
              :language: Python
              :linenos:

   .. tab-item:: C
      :sync: C

      The code below can be found in ``tsne.cpp`` in the ``examples`` folder of your installation.

      .. dropdown:: t-SNE Example

          .. literalinclude:: ../../../tests/examples/tsne.cpp
              :language: C++
              :linenos:


Further reading
---------------

The original *t*-SNE algorithm is described in :cite:t:`da_vandermaaten2008`. The Barnes-Hut
acceleration is introduced in :cite:t:`da_vandermaaten2014`.

Dimension Reduction APIs
========================

*t*-SNE
-------

.. tab-set::

   .. tab-item:: Python

      .. autoclass:: aoclda.dimension_reduction.tsne(n_components=2, perplexity=30.0, learning_rate=-1.0, max_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-7, early_exaggeration=12.0, theta=0.5, init='pca', seed=0, mixed_precision=False, low_precision_max_iter=None, low_precision_min_grad_norm=None, check_data=False)
         :members:

   .. tab-item:: C

      .. _da_tsne_set_data:

      .. doxygenfunction:: da_tsne_set_data_s
         :project: da
         :outline:
      .. doxygenfunction:: da_tsne_set_data_d
         :project: da

      .. _da_tsne_set_init_embedding:

      .. doxygenfunction:: da_tsne_set_init_embedding_s
         :project: da
         :outline:
      .. doxygenfunction:: da_tsne_set_init_embedding_d
         :project: da

      .. _da_tsne_compute:

      .. doxygenfunction:: da_tsne_compute_s
         :project: da
         :outline:
      .. doxygenfunction:: da_tsne_compute_d
         :project: da
