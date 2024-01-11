# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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


# For this to work, need to export PYTHONPATH=/path/to/install/python/aoclda

import os

if os.name == 'nt':
    #AOCL_ROOT = os.environ['AOCL_ROOT']
    #AOCLDA_ROOT = os.environ['AOCLDA_ROOT']
    INTEL_FCOMPILER = os.environ['INTEL_FCOMPILER']

    #try:
    #    os.add_dll_directory(AOCLDA_ROOT + r'\bin\LP64')
    #except OSError:
    #    os.add_dll_directory(AOCL_ROOT + r'\amd-da\lib\LP64')

    #os.add_dll_directory(AOCL_ROOT + r'\amd-libflame\lib\LP64')
    #os.add_dll_directory(AOCL_ROOT + r'\amd-blis\lib\LP64')
    #os.add_dll_directory(AOCL_ROOT + r'\amd-utils\lib')
    os.add_dll_directory(INTEL_FCOMPILER + r'\redist\intel64_win\compiler')
    current_dir = os.path.dirname(__file__)
    os.add_dll_directory(current_dir)

import numpy as np

from ._aoclda import single, double
from .basic_stats import *
from .factorization import *
from .linear_model import *
