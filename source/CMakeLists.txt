set(NLN_REG_SRC core/linear_model/linreg_public.cpp)
set(BASIC_STATISTICS_SRC core/basic_statistics/basic_statistics_public.cpp)
#set(DRIVERS core/drivers/lbfgsb_driver.cpp)
set(CSV core/csv/tokenizer.c core/csv/read_csv_public.cpp)
set(DA_HANDLE core/utilities/da_handle_public.cpp)

set(DA_CORE_SRC ${NLN_REG_SRC} ${BASIC_STATISTICS_SRC} ${CSV} ${DA_HANDLE})

add_library(da_core ${DA_CORE_SRC})
target_include_directories(da_core PUBLIC include)
target_include_directories(da_core PRIVATE core/drivers/ core/optimization/ core/utilities core/csv core/linear_model)
target_link_libraries(da_core lbfgsb lbfgsb_s ${BLIS} ${FLAME} -lgomp)

include(Suppressions_core)