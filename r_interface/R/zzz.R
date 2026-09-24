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

#' @import R6
#' @import Rcpp
#' @useDynLib aoclda, .registration = TRUE
NULL

.onLoad <- function(libname, pkgname) {
    # Load the Rcpp modules containing the C++ classes
    Rcpp::loadModule("decision_tree_module", what = TRUE)
    Rcpp::loadModule("decision_forest_module", what = TRUE)
    Rcpp::loadModule("linmod_module", what = TRUE)
    Rcpp::loadModule("knn_module", what = TRUE)
    Rcpp::loadModule("dbscan_module", what = TRUE)
    Rcpp::loadModule("kmeans_module", what = TRUE)
    Rcpp::loadModule("interpolation_module", what = TRUE)
    Rcpp::loadModule("pca_module", what = TRUE)
    Rcpp::loadModule("kernel_pca_module", what = TRUE)
    Rcpp::loadModule("nmf_module", what = TRUE)
    Rcpp::loadModule("tsne_module", what = TRUE)
    Rcpp::loadModule("svm_module", what = TRUE)
    Rcpp::loadModule("utils_module", what = TRUE)
    Rcpp::loadModule("nlls_module", what = TRUE)
    Rcpp::loadModule("approx_nn_module", what = TRUE)
    Rcpp::loadModule("metrics_module", what = TRUE)
    Rcpp::loadModule("basic_stats_module", what = TRUE)
    Rcpp::loadModule("kernel_module", what = TRUE)
}

.onAttach <- function(libname, pkgname) {
    packageStartupMessage(
        "AOCL-DA R interface loaded (EXPERIMENTAL -- APIs may change without notice)")
}
