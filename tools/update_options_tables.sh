#!/bin/bash
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# Awk incantation to find and update sprinkled options in the documentation.
# Usage $0 file1.rst [file2.rst [...]], optionally it is possible to point
# to the database file using
# $0 --db=path/to/all_table.rst file1.rst [...]
# --db=file MUST be the first argument
# The script is not very robust and relies on the ReSt to be tidy

TOOLS=$(dirname "$0")
FINDTBL="${TOOLS}/findtbl.sh"

if [ ! -x "$FINDTBL" ] ; then
  echo "Error: auxiliary script findtbl.sh not found in ${TOOLS}..."
  exit 1
fi


if [[ "$1" =~ --db=.* ]] ; then
  # Update location
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

if [ -t ] ; then
    # Interactive terminal
    COLERR="\033[31m\033[1m"
    COLWAR="\033[33m\033[1m"
    COLRES="\033[0m"
else
    COLERR=""
    COLWAR=""
    COLRES=""
fi

for f in $* ; do

  if [ "`basename $f`" == "$DBBASE" ] ; then
    echo "$f" is the database, skipping.
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

INPLACE="-i inplace"
awk $INPLACE -v DB="$DB" -v C="$COLERR" -v W="$COLWAR" -v R="$COLRES" -v findtbl="$FINDTBL" -- '
BEGIN {
        intable = 0
        firstblank = 0
        warn = 0
        tabletag = "UNKNOWN"
        ln = 0
}

{ ln++ }

/^[[:blank:]]*.. update options using table.*$/ {
        if (match($6, "_[[:alpha:]]+_[[:alpha:]]+") > 0){
           tabletag = $6
        } else {
            tabletag = "UNKNOWN"
        }
}

tolower($0) ~ /^[[:blank:]]*:header: "option name", "type", "default", "description", "constraints"/ {
        # print("found table")
        if (tabletag == "UNKNOWN") {
            print C "ERROR: table does not specify which tabletag to use!" R > "/dev/stderr"
            print C "       make sure the comment right above the table specifies" R > "/dev/stderr"
            print C "       from which table in the database (all_table.rst) to use" R > "/dev/stderr"
            print C "       for an example, see Internal Help > Documentation Utilities > Section Adding an Options Table" R > "/dev/stderr"
            print C "\n       Try adding a tag banner above the .. csv: line close to " ln "." R > "/dev/stderr"
        }
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
                tabletag = "UNKNOWN"
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

                # query = "grep -m 1 -i  \"\\\\\\\""name"\\\\\\\"\"" " " DB
                query = findtbl " \""name"\" \""tabletag"\" " DB
                # print W "about to call query=" query R > "/dev/stderr"
                if ((query | getline new) != 1) {
                        err = "ERROR: option \"" name "\" not found in database DB"
                        print C err R > "/dev/stderr"
                        found = match(old, "^[[:blank:]]*\"NOT FOUND~")
                        if (found == 0){
                            sub("^[[:blank:]]*", "", old)
                            new = "\"NOT FOUND~" old
                        } else {
                            new = old
                        }
                        warn++
                }
                close(query)
                # Use target indentation
                sub("^[[:blank:]]*", space, new)
                print new
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
