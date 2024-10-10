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

# pylint: disable = missing-module-docstring, missing-class-docstring, attribute-defined-outside-init

import os
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel
from packaging.tags import sys_tags

# Create a specific bdist_wheel to signal to setup.py that the wheel is not pure python

class spec_bdist_wheel(bdist_wheel):
    def get_tag(self):
        python_tag = 'py3'
        abi_tag, platform_tag = None, None

        for tag in sys_tags():
            if abi_tag is None:
                abi_tag = str(tag.abi)
            if platform_tag is None:
                platform_tag = str(tag.platform)
            if abi_tag is not None and platform_tag is not None:
                break
        abi_tag = 'none'
        if 'manylinux' in str(platform_tag):
            platform_tag = 'linux_x86_64'


        return python_tag, abi_tag, platform_tag

    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

# List of all dependent libraries that were copied in the python_package install
lib_extensions = ['.so', '.dll', '.lib', '.pyd']
dep_libs = []
for lib in os.listdir('aoclda'):
    if any(ext for ext in lib_extensions if ext in lib):
        dep_libs.append(lib)

setup(
    name="aoclda",
    cmdclass={'bdist_wheel': spec_bdist_wheel},
    long_description="AOCL-DA Python Interfaces",
    version="5.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={'aoclda': dep_libs},
    install_requires=['numpy<2.0', 'wheel'],
)
