#!/bin/bash
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
        if (oname != name) {
            print "ERROR: option \"" name "\" NOT FOUND in table with tag: \"" tabletag "\" in database." > "/dev/stderr"
            exit 5
        }
}
' "$DB"