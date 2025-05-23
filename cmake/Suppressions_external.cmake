# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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


# LBFGSB source code
set_source_files_properties(
  lbfgsb.F90 linpack.F90
  PROPERTIES
    COMPILE_FLAGS
    "-Wno-unused-variable -Wno-compare-reals -Wno-unused-dummy-argument -Wno-maybe-uninitialized")

# RALFit source code
set_source_files_properties(
  ral_nlls_workspaces.F90 ral_nlls_fd.F90 ral_nlls_ciface.F90
  ral_nlls_internal.F90 ral_nlls_workspaces.F90
  ral_nlls_dtrs.F90 ral_nlls_bounds.F90 ral_nlls_fd.F90
  PROPERTIES
    COMPILE_FLAGS
    "-Wno-compare-reals -Wno-unused-dummy-argument -Wno-unused-function -Wno-unused-label -Wno-character-truncation -Wno-unused-variable -Wno-maybe-uninitialized"
)
