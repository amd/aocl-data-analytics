#!/bin/bash
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Awk incantation to find and update sprinkled options in the documentation.
# Usage $0 file1.rst [file2.h [...]], optionally it is possible to point
# to the database file using
# $0 --db=path/to/all_table.rst file1.rst [...]
# --db=file MUST be the first argument
# The script is not very robust and relies on the ReSt to be tidy

TOOLS=$(dirname "$0")
FINDTBL="${TOOLS}/findtbl.sh"

PROG="$(basename "$0")"

usage() {
  cat <<EOF
Usage: $PROG [--db=path/to/all_table.rst] file1.rst [file2.h ...]
       $PROG -h | --help

Find and update the sprinkled options tables in the documentation .rst or
.h and .hpp files, refreshing each option's "type", "default", "description",
"constraints" and other columns from the central options database (all_table.rst).

For every input .rst, .h, or .hpp file the script scans for blocks introduced by a
  .. update options using table <_tabletag_>
comment followed by a CSV table whose header contains:
  :header: "option name", "type", "default", "description", "constraints"
Each option row is then looked up in the database (via findtbl.sh) using the
option name and the table tag, and rewritten in place. Options that cannot be
found in the database are flagged with a "NOT FOUND~" marker and reported on
stderr.

Arguments:
  file1.rst [file2.h ...]    One or more .rst, .h, or .hpp files to update in place. Files
                             that are the database itself, are not regular
                             files, or have another extension are skipped.

Options:
  --db=PATH   Path to the options database file all_table.rst.
              If given, it MUST be the first argument.
              Defaults to: C/options/all_table.rst
  -h, --help  Print this help message and exit.

Notes:
  * Files are edited in place (awk -i inplace); commit or back up first.
  * Requires the auxiliary script findtbl.sh next to this script.
  * The parser is not very robust and relies on tidy ReST formatting.

Examples:
  $PROG C/lin_reg/lin_reg.rst
  $PROG --db=doc/C/options/all_table.rst C/lin_reg/*.rst

  To update all the ReST files
  find . -name '*.rst' -exec $PROG --db=doc/C/options/all_table.rst {} \;
EOF
}

for arg in "$@" ; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

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

if [ ! -f ${DB:=C/options/all_table.rst} ] ; then
  echo "Error: Options database file all_table.rst not found in $DB, set the correct path using --db=path/to/all_table.rst"
  exit 1
fi

DBBASE="`basename $DB`"

if [ $# -lt 1 ] ; then
  usage
  echo
  echo Input error: no files to process?
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
  case "$f" in
    *.rst|*.h|*.hpp) ;;
    *)
      echo "$f" does not have .rst, .h, or .hpp extension, skipping.
      continue
      ;;
  esac

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

/^[[:blank:]]*($|\\endrst)/ {
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
