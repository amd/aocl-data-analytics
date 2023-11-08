# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import glob
import sys
import os
import tempfile

total_APIs = 0
single_APIs = 0
double_APIs = 0
string_APIs = 0
int_APIs = 0
misc_APIs = 0


def process_file(filename):

    print(f"Processing {filename} ... ", end="")

    with open(filename, "r") as file:
        contents = file.readlines()

    global total_APIs, single_APIs, double_APIs, int_APIs, string_APIs, misc_APIs

    for line in contents:
        if (line.startswith("da_status da_") or line.startswith("void da_")):
            total_APIs += 1
            if (line.find("_s(") != -1):
                single_APIs += 1
            elif (line.find("_d(") != -1):
                double_APIs += 1
            elif (line.find("_int(") != -1 or line.find("_int32(") != -1 or
                  line.find("_int64(") != -1 or line.find("_uint8(") != -1 or
                    line.find("_uint32(") != -1 or line.find("_uint64(") != -1):
                int_APIs += 1
            elif (line.find("_char(") != -1 or line.find("_string(") != -1 or
                  line.find("_str(") != -1):
                string_APIs += 1
            else:
                misc_APIs += 1

    print("DONE")


def main():

    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print(
            "Usage: python count_APIs.py [files in which to count AOCL-DA APIs e.g. a list of files or wildcards such as *.cpp].")
        print("To count the total number of APIs in the library use: python count_APIs.py /path/to/include/* .")
        sys.exit(1)

    # Get command line arguments
    file_patterns = sys.argv[1:]

    # Split arguments into individual files and process each one
    for file_pattern in file_patterns:
        for filename in glob.glob(file_pattern):
            if os.path.isfile(filename):
                process_file(filename)

    # Print results
    print(f"\n Total number of APIs found: {total_APIs}")
    print("\n Breakdown:")
    print(f"\n     {single_APIs} single precision floating point APIs")
    print(f"\n     {double_APIs} double precision floating point APIs")
    print(f"\n     {int_APIs} integer APIs")
    print(f"\n     {string_APIs} string or char APIs")
    print(f"\n     {misc_APIs} other APIs\n")


if __name__ == "__main__":
    main()
