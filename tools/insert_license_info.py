#! /usr/bin/python3
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

import argparse
import glob
import sys
import os
import tempfile

license_header = """Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

"""

star_extensions = [".c", ".cc", ".h", ".hh", ".c++", ".cpp", ".h++", ".hpp", ".cxx", ".hxx"]
# .txt for CMakeLists.txt files
hash_extensions = [".py", ".sh", ".txt", ".cmake"]
# Fortran extension require explicit lower/upper case names to be added
exclamation_extensions = [".f90", ".i90", ".FPP", ".F90", ".I90"]
f77_extensions = [".f", ".fpp", ".for", ".ftn", ".i", ".F", ".FOR", ".FTN", ".I"]
rst_extensions = [".rst"]

def get_comment_character(extension):
    pre_comment = ""
    post_comment = ""
    if extension.lower() in star_extensions:
        comment = " * "
        pre_comment = "/*"
        post_comment = " */"
    elif extension.lower() in hash_extensions:
        comment = "# "
    elif extension in exclamation_extensions:
        comment = "! "
    elif extension in f77_extensions:
        comment = "c    "
    elif extension.lower() in rst_extensions:
        pre_comment = ".."
        comment = "    "
    else:
        comment = -1

    post_comment += "\n"

    return [pre_comment, post_comment, comment]

def process_file(filename):

    print(f"Processing {filename} ... ", end ="")

    # Check file extension to see what the comment character should be
    extension = os.path.splitext(filename)[1]
    pre_comment, post_comment, comment = get_comment_character(extension)
    if comment == -1:
        print("ignoring unknown file type.")
        return

    with open(filename, "r") as file:
        contents = file.readlines()

    # Check for existing copyright info
    for line in contents:
        if (line.lower().find("Copyright (C)".lower()) != -1 and line.lower().find("Advanced Micro Devices".lower()) != -1):
            print("copyright info already present.")
            return

    # Prepend the license and copyright banner to the file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, 'w') as temp_file_obj:
        if pre_comment != "":
            temp_file_obj.write(pre_comment+"\n")
        for line in license_header.splitlines():
            temp_file_obj.write(comment+line+"\n")
        if post_comment != "":
            temp_file_obj.write(post_comment+"\n")
        for line in contents:
            temp_file_obj.write(line)

    os.replace(temp_file.name, filename)

    print("DONE")

def main():

    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("Usage: python insert_license_info.py [files to process e.g. a list of files or wildcards such as *.cpp]")
        sys.exit(1)

    # Get command line arguments
    file_patterns = sys.argv[1:]

    # Split arguments into individual files and process each one
    for file_pattern in file_patterns:
        for filename in glob.glob(file_pattern):
            if os.path.isfile(filename):
                process_file(filename)

if __name__ == "__main__":
    main()

