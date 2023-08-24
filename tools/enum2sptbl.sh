#!/bin/bash

if [ $# -lt 1 ] ; then
  echo "Build C++ program to generate an enum Sphinx table of the form"
  echo "enum_name | enum_value | Doxygen description comment"
  echo "Doxygen comment must only use ///< text and have a single"
  echo "enum per line."
  echo "usage $0 file.h[pp]"
  exit 0
fi

awk -v FILE=$1 ' BEGIN { s=0 }
/typedef[[:blank:]]+/ { s=1;
print("#include <iostream>\n#include \"" FILE "\"\nint main(void) {")
                        print("std::cout << \".. csv-table:: enum ``" $3 "``\\n\";")
                        print("std::cout << \"    :header: \\\"Enum\\\", \\\"Value\\\", \\\"Description\\\" \\n\";")
                        print("std::cout << \"    :widths: 30, 10, 60\\n\\n\";")
                        next
                      }
/[[:blank:]]*[[:print:]]+,/ { enum = $1
                              str = $0
                              sub(",", "", enum)
                              p = index(str, "///<");
                              if ( p == 0 ) { desc = "FIXME" } else { desc = substr(str, p); sub("///<", "", desc)}
                              print("std::cout << \"    ``" enum "``, \" << ", enum, " << \", \\\""desc"\\\"\\n\";")
                              next
                            }
/\}/ { if (s==1) { print("return 0;\n}\n")
                   exit
                 }
     }
' $1
