#include "aoclda.h"
#include <iostream>

int main()
{
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Unit test for DA_PRINTF_DEBUG" << std::endl;
    std::cout << "If DA_LOGGING is defined output should be" << std::endl;
    std::cout << std::endl;
    std::cout << "[main printf_debug.cpp:24] Inside main " << std::endl;
    std::cout << "[main printf_debug.cpp:26] x =   2.90 " << std::endl;
    std::cout << "[main printf_debug.cpp:28] x =   2.90, y = 15 " << std::endl;
    std::cout << std::endl;
    std::cout << "Otherwise DA_PRINTF_DEBUG should be silent" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::endl;

    double x = 1.45;
    x = x * 2;
    da_int y = 12;
    y = y + 3;

    // check macro works when we are only printing a string with no variables
    DA_PRINTF_DEBUG("Inside main \n");
    // check macro works when we are printing a single variable
    DA_PRINTF_DEBUG("x = %6.2f\n", x);
    // check macro works when we are printing multiple variables including a variable of integer type
    DA_PRINTF_DEBUG("x = %6.2f, y = %" DA_INT_FMT "\n", x, y);

    return 0;
}
