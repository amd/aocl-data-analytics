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
SVM example Python script
"""

import sys
import numpy as np
from aoclda.svm import SVC, SVR, NuSVC, NuSVR


def svm_examples():
    """
    Demonstrate basic usage of SVC, NuSVC, SVR, and NuSVR with small datasets.
    """

    # --------------------------
    # SVC example (classification)
    # --------------------------
    X_cls = np.array([[-2.99, 0.04], [-0.15, 2.52], [-0.09, 0.91], [0.45, 1.12],
                     [-1.03, 0.3], [-0.02, -0.9], [1.59, 1.88], [0.34, -0.15]])
    y_cls = np.array([0, 0, 0, 1, 0, 0, 1, 1])

    print("\nSVC classification example\n")
    try:
        svc_model = SVC(kernel="rbf", C=1.0)
        svc_model.fit(X_cls, y_cls)
        y_pred_svc = svc_model.predict(X_cls)
    except RuntimeError:
        sys.exit(1)
    print(f"SVC predictions: {y_pred_svc}")

    # --------------------------
    # NuSVC example (classification)
    # --------------------------
    print("\nNuSVC classification example\n")
    try:
        nusvc_model = NuSVC(nu=0.5, kernel="rbf")
        nusvc_model.fit(X_cls, y_cls)
        y_pred_nusvc = nusvc_model.predict(X_cls)
    except RuntimeError:
        sys.exit(1)
    print(f"NuSVC predictions: {y_pred_nusvc}")

    # --------------------------
    # SVR example (regression)
    # --------------------------
    X_reg = np.array([[-0.46, -0.47], [0.5, -0.14], [-1.72, -0.56], [0.07, -1.42],
                     [-0.91, -1.41], [-1.01, 0.31], [1.58, 0.77], [1.47, -0.23]])
    y_reg = np.array([-36.2, 27.76, -114.51, -20.17, -
                     79.45, -56.15, 109.22, 85.06])

    print("\nSVR regression example\n")
    try:
        svr_model = SVR(kernel="rbf", C=1.0)
        svr_model.fit(X_reg, y_reg)
        y_pred_svr = svr_model.predict(X_reg)
    except RuntimeError:
        sys.exit(1)
    print(f"SVR predictions: {y_pred_svr}")

    # --------------------------
    # NuSVR example (regression)
    # --------------------------
    print("\nNuSVR regression example\n")
    try:
        nusvr_model = NuSVR(nu=0.4, kernel="rbf")
        nusvr_model.fit(X_reg, y_reg)
        y_pred_nusvr = nusvr_model.predict(X_reg)
    except RuntimeError:
        sys.exit(1)
    print(f"NuSVR predictions: {y_pred_nusvr}")


if __name__ == "__main__":
    svm_examples()
