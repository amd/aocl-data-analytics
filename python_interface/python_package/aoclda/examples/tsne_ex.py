# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
t-SNE embedding example Python script
"""

import sys
import numpy as np
from aoclda.dimension_reduction import tsne


def tsne_example():
    """
    t-SNE embedding example
    """
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [1.1, 1.9, 3.1],
                  [4.1, 5.2, 5.8],
                  [0.9, 2.2, 2.9],
                  [4.2, 4.9, 6.2]])

    print("\nt-SNE embedding for a small data matrix\n")
    try:
        emb = tsne(n_components=2, perplexity=2.0, max_iter=300, theta=0.0, seed=42)
        y = emb.fit_transform(x)
    except RuntimeError:
        sys.exit(1)

    print("\nEmbedding:\n")
    print(y)
    print("\nKL divergence:", emb.kl_divergence)


if __name__ == "__main__":
    tsne_example()
