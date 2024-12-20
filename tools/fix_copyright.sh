#!/bin/bash
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
############################################################
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


