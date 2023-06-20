#include "aoclda.h"
#include <iostream>

int main()
{
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Unit test for DA_PRINTF_DEBUG" << std::endl;
    std::cout << "If DA_LOGGING is defined output should be" << std::endl;;
    std::cout << "[main printf_debug.cpp:15] x =     2.90" << std::endl;
    std::cout << "Otherwise DA_PRINTF_DEBUG should be silent" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    double x = 1.45;
    x = x * 2;
    DA_PRINTF_DEBUG("x = %8.2f\n", x);

    return 0;
}
