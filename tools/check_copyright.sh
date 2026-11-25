#!/bin/bash
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Check if files have
# * the correct (current) copyright (ending) year
# * the correct verbatim copyright body
# and return number of non-conforming files.
#
# It is assumed that the copyright label is on the top 3 lines of the file
# in the form: 'Copyright (c) 2020-2023' or 'Copyright (c) 2023'
#
# Usage
#   tools/check_copyright.sh [-p] [file1 [file2 ...]]
#   -p   print the copyright header or a message indicating incorrect copyright found
#
# Notes
#
# 1. If no files are specified, only staged files are checked and that the year is the current one
#
# 2. To white list files add the pattern to tools/copyright_suppression.txt

INFO=""
FAIL=""
OK=""
RESET=""
SKIP=""

if [ -t 1 ] ; then
  if [ "`tput colors`" -gt 0 ] ; then
    INFO='\e[0;36m'
    FAIL='\e[0;31m'
    OK='\e[0;32m'
    RESET='\e[0m'
    SKIP='\e[0;37m'
  fi
fi

suppression=tools/copyright_suppression.txt
if [ ! -f "$suppression" ] ; then
  echo "Error: file $suppression NOT found!"
  echo "execute this script from the project root directory!"
  exit 5
fi
# Whitelist (shell pattern) of files which are considered always OK
whitelist=`cat $suppression`

print_help() {
  echo "Usage: $0 [-p] [file1 [file2 ...]]"
  echo
  echo "Checks if files have"
  echo "  * the correct (current) copyright (ending) year"
  echo "  * the correct verbatim copyright body"
  echo "and returns the number of non-conforming files."
  echo
  echo "The copyright label is expected on the top lines of the file in the"
  echo "form: 'Copyright (c) 2020-2023' or 'Copyright (c) 2023'"
  echo
  echo "  -p   print the copyright header (for ANY year) or a message indicating"
  echo "       incorrect copyright found. The flag NEEDS to be the first argument."
  echo "       Can also be overridden by the AOCLDA_CHECK_COPYRIGHT environment variable."
  echo
  echo "Notes"
  echo "  1. If no files are specified, only staged files are checked and that"
  echo "     the year is the current one."
  echo "  2. Only regular text files are checked. To whitelist files add the"
  echo "     pattern to $suppression."
  echo
}

# Allow for CI environments to override the PRINT variable
PRINT=${AOCLDA_CHECK_COPYRIGHT:-}

# Expected license body with the comment marker removed from each line
# (the space following the marker is kept and trailing spaces are not allowed).
expected_body='
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

'

if [ "$#" -gt 0 ]; then
  if [ "$1" = "-p" ]; then
    PRINT='-p'
    shift
  fi
fi

if [ "$#" -eq 0 ]; then
  print_help
  echo ""
  echo "Checking staged files:"
  # Check what is staged, not the working tree (which may hold unstaged
  # edits): export the staged content into a scratch directory with the same
  # layout and run this script there on the same relative paths.
  mapfile -d '' -t staged < <(git diff --cached --name-only --diff-filter=ACMR -z)
  if [ "${#staged[@]}" -eq 0 ]; then
    echo "No staged files to check."
    exit 0
  fi
  script=$(realpath "$0")
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT
  git checkout-index --prefix="$tmpdir/" -- "${staged[@]}" || exit 5
  # The staged suppression file wins if it is part of the commit.
  mkdir -p "$tmpdir/tools" && cp -n "$suppression" "$tmpdir/tools/"
  (cd "$tmpdir" && bash "$script" $PRINT "${staged[@]}")
  exit
fi

nfailed=0

if [ "$PRINT" == "-p" ]; then
  current_year='\(20[1-4][0-9]\)'
  message=" does NOT include copyright banner"
  function prn () { echo -en ${OK}'[  OK  ] '
                    # printf '%s: "%s"\n' "$1" "$2"
                    printf '%s\n' "$1"
                    echo -en ${RESET}
                  }
else
  current_year=$(date '+%Y')
  message=" does NOT include $current_year in the copyright"
  function prn () { echo -e ${OK}"[  OK  ]" "$1" is ok.${RESET} ; }
fi

for f in "$@"; do

  if [ ! -f "$f" ]; then
    echo -e ${SKIP}[ SKIP ] "$f" is not a regular file, skipping
    continue
  fi

  MIME=`file -bi "$f" | grep -q text`
  if [ "$?" -ne 0 ] ; then
    echo -e ${SKIP}[ SKIP ] "$f" is not a text file, skipping${RESET}
    continue
  fi

  # Check if the first argument matches any of the whitelisted names
  wlisted=0
  for regex in ${whitelist[@]}; do
    regex=`echo $regex | tr -d \'`
    if [[ "$f" =~ $regex ]] ; then
      wlisted=1
      break
    fi
  done
  if [ "$wlisted" = 1 ]; then
    echo -e ${INFO}"[ WHITE]" "$f" is whitelisted and not checked${RESET}
    continue
  fi

  # Check copyright header
  CP=`head -3 "$f" | grep "Copyright ([cC]) \(20[1-4][0-9]-\)\?$current_year Advanced Micro Devices, Inc. All rights reserved."`
  if [ "$?" -ne 0 ]; then
    # Notify problem
    echo -e ${FAIL}[FAILED] "$f" $message${RESET}
    let nfailed++
  else
    # print OK
    prn "$f" "$CP"
  fi

  # Check copyright body text (check for AI hallucinations)
  # The license body must start on the line right after the AMD copyright line
  # and use the same comment prefix (whatever precedes 'Copyright' on that
  # line, e.g. ' * ', '# ', '! '). Each expected line is rebuilt as that prefix
  # followed by the license text, blank lines being the prefix with trailing
  # spaces removed, and compared byte for byte (whitespace and line endings
  # included) against the file.
  cline=`head -3 "$f" | grep -n -m1 "Copyright ([cC]) .*Advanced Micro Devices, Inc. All rights reserved."`
  if [ -z "$cline" ]; then
    body_ok=1
    diff_out="         no AMD copyright line found to anchor the license body"
  else
    lnum=${cline%%:*}
    prefix=${cline#*:}
    prefix=${prefix%%Copyright*}
    blank="${prefix%"${prefix##*[![:space:]]}"}"
    expected=()
    while IFS= read -r line; do
      if [ -z "$line" ]; then
        expected+=("$blank")
      else
        expected+=("$prefix${line:1}")
      fi
    done <<< "$(printf '%s' "$expected_body")"
    diff_out=`diff <(printf '%s\n' "${expected[@]}") \
                   <(tail -n +$((lnum + 1)) "$f" | head -n ${#expected[@]}) \
              | sed 's/^/         /'`
    body_ok=0
    [ -n "$diff_out" ] && body_ok=1
  fi
  
  if [ "$body_ok" -ne 0 ]; then
    echo -e ${FAIL}[FAILED] "$f" does NOT include the correct license body${RESET}
    if [ $VERBOSE ]; then
      # '<' is what is expected, '>' is what the file actually has.
      echo "$diff_out"
    fi
    let nfailed++
  fi

done

exit $(( nfailed < 100 ? nfailed : 100 ))

