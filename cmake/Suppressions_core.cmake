# csv
set_source_files_properties(core/csv/read_csv_public.cpp core/csv/read_csv.cpp core/csv/tokenizer.c
                            PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-unused-parameter -Wno-implicit-fallthrough -Wno-format -Wno-old-style-declaration -Wno-sign-compare -Wno-type-limits")
