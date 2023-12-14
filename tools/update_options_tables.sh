#!/bin/bash
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
# Awk inchantation to find and update sprinkled options in the documentation.
# Usage $0 file1.rst [file2.rst [...]], optionaly it is possible to point
# to the database file using
# $0 --db=path/to/all_table.rst file1.rst [...]
# --db=file MUST be the first argument
# The script is not very robust and relies on the ReSt to be tidy

if [[ "$1" =~ --db=.* ]] ; then
  # update location
  DB="${1#--db=}"
  echo Setting all_table.rst database path to \"$DB\"
  shift
fi

if [ ! -f ${DB:=options/all_table.rst} ] ; then
  echo "Error: Options database file all_table.rst not found in $DB, set the correct path using --db=path/to/all_table.rst"
  exit 1
fi

DBBASE="`basename $DB`"

if [ $# -lt 1 ] ; then
  echo Error: no files to process?
  exit 2
fi

for f in $* ; do

  if [ "`basename $f`" == "$DBBASE" ] ; then
    echo "$f" is the batabase, skipping.
    continue
  fi

  if [ ! -f "$f" ] ; then
    echo "$f" is not a regular file, skipping.
    continue
  fi
  if [ ! "${f: -4}" == ".rst" ] ; then
    echo "$f" does not have .rst extension, skipping.
    continue
  fi

  echo processing "$f"...

# awk -v DB="$DB" -- '
awk -i inplace -v DB="$DB" -- '
BEGIN {
        intable = 0
        firstblank = 0
}

tolower($0) ~ /^[[:blank:]]*:header: "option name", "type", "default", "description", "constraints"/ {
        # print("found table")
        # remember the indentation
        match($0, "^[[:blank:]]*")
        space = substr($0, 0, RLENGTH)
        # option table found
        intable = 1
        firstblank = 0
        print
        next
}

/^[[:blank:]]*$/ {
        if (intable == 1 && firstblank == 1) {
                # table end, reset flags
                intable = 0
                firstblank = 0
        # print("2nd blank")
        }
        if (intable == 1 && firstblank == 0) {
                # options starts
                firstblank = 1
        # print("1st blank")
        }
        print
        next
}

{
        if (intable == 1 && firstblank == 1) {
                old = $0
                split($0, tok, "\"")
                name = tok[2]
                query = "grep -m 1 -i \"" name "\" " DB
                if ((query | getline new) != 1) {
                        print "Error: option:", name, "could not be queried?"
                        exit 2
                }
                close(query)
                # Use target indentation
                sub("^[[:blank:]]*", space, new)
                print (new)
        } else {
                print
        }
}
' "$f"

status=$?

if [ $status -ne 0 ] ; then
  echo "Error running awk on $f!"
  exit 4
fi

done
