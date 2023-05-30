#ifdef SINGLE_PREC
#define PREF s
#else
#define PREF d
#endif

#define CONCAT_(a) a
#define CONCAT(a,b) CONCAT_(a)CONCAT_(b)

! Renaming macro to build subroutine names based on the
! requested precision, copy -> dcopy or scopy
! This macro is called on every routine call and definition
#define PREC(f) CONCAT(PREF,f)
