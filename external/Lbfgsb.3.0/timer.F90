!
!  L-BFGS-B is released under the “New BSD License” (aka “Modified BSD License”
!  or “3-clause license”)
!  Please read attached file License.txt
!
#include "preprocessor.fpp"
    Subroutine PREC(timer)(ttime)
#ifdef SINGLE_PREC
   Use working_precision, Only: wp => sp
#else
   Use working_precision, Only: wp => dp
#endif
      Real (Kind=wp) :: ttime
!
      Real (Kind=wp) :: temp
!
!     This routine computes cpu time in double precision; it makes use of
!     the intrinsic f90 cpu_time therefore a conversion type is
!     needed.
!
!           J.L Morales  Departamento de Matematicas,
!                        Instituto Tecnologico Autonomo de Mexico
!                        Mexico D.F.
!
!           J.L Nocedal  Department of Electrical Engineering and
!                        Computer Science.
!                        Northwestern University. Evanston, IL. USA
!
!                        January 21, 2011
!
      temp = real(ttime, kind=wp)
      Call cpu_time(temp)
      ttime = real(temp, kind=wp)

      Return

    End Subroutine
