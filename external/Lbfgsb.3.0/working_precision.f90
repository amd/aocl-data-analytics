Module working_precision
#ifndef SINGLE_PREC
    Integer, parameter :: wp = selected_real_kind(15)
#else
    Integer, parameter :: wp = selected_real_kind(6)
#endif
End Module