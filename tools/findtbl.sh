#!/bin/bash
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# find in database all_table.rst the correct table (using table tags)
# and inside that table look for a given option (name)
# Usage findtbl.sh optionname tabletag all_table.rst

if [ $# -ge 3 ] ; then
        OPTIONNAME="$1"
        TABLETAG=$2
        DB="$3"
else
  echo "Usage: optionname tabletag path/to/all_table.rst" > /dev/stderr
  exit 1
fi

if [ ! -f "$DB" ] ; then
  echo "Error: Options database file all_table.rst not found in $DB, set the correct path and try again" > /dev/stderr
  exit 2
fi

awk -v optionname="${OPTIONNAME}" -v tabletag=${TABLETAG} '
BEGIN {
        foundtag = 0
        tablepresent = 0
        name = optionname
        tag = tabletag ":"
        # print "findtbl: " optionname " in " tabletag > "/dev/stderr"
}

/^[[:blank:]]*\.\..*_opts.*$/ {
    if (foundtag != 0) {
        # table was found but option name not. Error.
        print "ERROR: option \"" name "\" in table with tag: \"" tabletag "\" NOT FOUND in database." > "/dev/stderr"
        exit 3
    }
    qtag = $2
    sub("^[[:blank:]]*", "", qtag)
    if (qtag == tag) {
        foundtag = 1
        tablepresent = 1
    } else {
        foundtag = 0
    }
    next
}

/^[[:blank:]]*".+"/ {
        if (foundtag != 0) {
            split($0, tok, "\"")
            oname = tok[2]
            if (oname == name) {
                print $0
                # Job done. Exit
                exit 0
            }
        }
        next
}

END {
        if (tablepresent==0){
            print "ERROR: table with tag: \"" tabletag "\" NOT FOUND in database." > "/dev/stderr"
            exit 4
        }
}
' "$DB"