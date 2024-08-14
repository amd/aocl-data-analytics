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


# pylint: disable = missing-module-docstring, import-outside-toplevel

import sys
from .patch_sklearn import skpatch, undo_skpatch

__all__ = ["skpatch", "undo_skpatch"]


def main():
    '''
    Load the scikit-learn patch then execute the user's script
    '''

    import argparse
    import runpy

    parser = argparse.ArgumentParser(
        description="AOCL-DA Extension for scikit-learn")

    parser.add_argument(
        "-m", action="store_true", dest="is_module")
    parser.add_argument("name", help="Your Python script or module name")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="Command line arguments for your Python script")

    args = parser.parse_args()

    # Call patch to replace scikit-learn symbols with AOCL-DA
    skpatch()

    sys.argv = [args.name] + args.args

    if args.is_module:
        runpy.run_module(args.name, run_name="__main__")
    else:
        runpy.run_path(args.name, run_name="__main__")

if __name__ == "__main__":
    sys.exit(main())
