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


#!/bin/bash

if [ $# -lt 1 ] ; then
  echo "Build C++ program to generate an enum MD table of the form"
  echo "enum_name | enum_value | Doxygen description comment"
  echo "Doxygen comment must only use ///< text and have a single"
  echo "enum per line."
  echo "usage $0 file.h[pp]"
  exit 0
fi

awk -v FILE=$1 ' BEGIN { s=0 }
/typedef[[:blank:]]+/ { s=1;
print("#include <iostream>\n#include \"" FILE "\"\nint main(void) {")
                        print("std::cout << \"Table for enum `" $3 "`\\n\\n\";")
                        print("std::cout << \"| Enum | Value | Description |\\n\";")
                        print("std::cout << \"|:-----|:-----:|:------------|\\n\";")
                        next
                      }
/[[:blank:]]*[[:print:]]+,/ { enum = $1
                              str = $0
                              sub(",", "", enum)
                              p = index(str, "///<");
                              if ( p == 0 ) { desc = "FIXME" } else { desc = substr(str, p); sub("///<", "", desc)}
                              print("std::cout << \"| `" enum "` | \" << ", enum, " << \"| ", desc, " |\\n\";")
                              next
                            }
/\}/ { if (s==1) { print("return 0;\n}\n")
                   exit
                 }
     }
' $1
