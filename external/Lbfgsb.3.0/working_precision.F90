Module working_precision
#ifdef SINGLE_PREC
    Integer, parameter :: wp = selected_real_kind(6)
#else
    Integer, parameter :: wp = selected_real_kind(15)
#endif
End Module