#!/bin/bash
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

# Attempt to fix copyright year if not matching the current year.
# It is assumed that the copyright label should on the top of the files
# in the form: ' * Copyright (C) 2020-2023' or ' * Copyright (C) 2023'

print_help() {
  echo $0 [file1 [file2 ...]]
  echo Attempt to fix copyright year if not matching the current year.
  echo If no file is given, all staged files are checked.
}

if [ "$1" == "--help" -o "$1" == "-h" ] ; then
  print_help
  exit 0
fi

if [ "$#" -eq 0 ] ; then
  print_help
  echo ""
  echo "Working on all staged files"
  git status --porcelain | grep '^[AM]' | cut -c4- | xargs --no-run-if-empty bash $0
  exit
fi

current_year=$(date '+%Y')

for f in "$@"; do

  head -3 "$f" | grep --silent "Copyright (C) \([0-9]\{4\}-\)\?$current_year"
  if [ "$?" -ne 0 ]; then
    sed -i "1,3s/Copyright (C) \([0-9]\{4\}\)-\([0-9]\{4\}\) Advanced Micro Devices/Copyright (C) \1-$current_year Advanced Micro Devices/; 1,3s/Copyright (C) \([0-9]\{4\}\) Advanced Micro Devices/Copyright (C) \1-$current_year Advanced Micro Devices/" "$f"
    # Check if fix was successful (it might fail in case that the copyright line
    # doesn't follow the exact pattern)
    head -3 "$f" | grep --silent "Copyright (C) \([0-9]\{4\}-\)\?$current_year"
    if [ "$?" -ne 0 ]; then
      echo "$f" WAS NOT fixed
    else
      echo "$f" is fixed
    fi
  fi

done


