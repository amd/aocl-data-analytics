! Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
! 
! Redistribution and use in source and binary forms, with or without modification,
! are permitted provided that the following conditions are met:
! 1. Redistributions of source code must retain the above copyright notice,
!    this list of conditions and the following disclaimer.
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
! 3. Neither the name of the copyright holder nor the names of its contributors
!    may be used to endorse or promote products derived from this software without
!    specific prior written permission.
! 
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
! ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
! WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
! IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
! INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
! OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
! WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
! ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
! 


#include "preprocessor.FPP"
Subroutine PREC(lbfgsb_solver)(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, itask, &
   iprint, lsavei, isave, dsave)
#ifdef SINGLE_PREC
   Use sworking_precision, Only: wp
#else
   Use dworking_precision, Only: wp
#endif
   Implicit None
!  Arguments
   Integer :: lsavei(4)
   Integer :: n, m, iprint, itask, nbd(n), iwa(3*n), isave(44)
   Real (Kind=wp) :: f, factr, pgtol, x(n), l(n), u(n), g(n), &
      wa(2*m*n+5*n+11*m*m+8*m), dsave(29)
!  Local variables
   Logical :: lsave(4)
   Character(Len=60), Dimension(28), Parameter  :: tasks = (/ Character(Len=60) :: &
      'NEW_X',& ! 1  action: monitor
      'START', &! 2  action: 1st iteration
      'STOP', & ! 3  action: abort
      'FG',    &! 4  action: evaluage f+g
      'ABNORMAL_TERMINATION_IN_LNSRCH', &! 5 out:warn
      'CONVERGENCE', &! 6 out:sucess
      'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL', &! 7 out:sucess
      'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', &! 8 out:sucess
      'RESTART_FROM_LNSRCH', &! 9 out:warn
      'ERROR: FTOL .LT. ZERO', &! 10 out:error:invalid input
      'ERROR: GTOL .LT. ZERO', &! 11 out:error:invalid input
      'ERROR: INITIAL G .GE. ZERO', &! 12  out:error:invalid input gradient rubbish no search direction perturbate point
      'ERROR: INVALID NBD', &! 13 out:error:invalid input
      'ERROR: N .LE. 0', &! 14 out:error:empty search space
      'ERROR: NO FEASIBLE SOLUTION', &! 15 out error: problem infeasible
      'ERROR: STP .GT. STPMAX', &! 16 out warn: num difficulties
      'ERROR: STP .LT. STPMIN', &! 17 out warn: num difficulties
      'ERROR: STPMAX .LT. STPMIN', &! 18 out warn: num difficulties
      'ERROR: STPMIN .LT. ZERO', &! 19 out warn: num difficulties
      'FG_LNSRCH', &! 20 action: evaluage f+g
      'FG_START', &! 21 action: evaluage f+g
      'ERROR: XTOL .LT. ZERO', &! 22 internal use only -> NEW_X
      'WARNING: ROUNDING ERRORS PREVENT PROGRESS', &! 23 out warn: num difficulties
      'WARNING: STP = STPMAX', &! 24 out warn: num difficulties
      'WARNING: STP = STPMIN', &! 25 out warn: num difficulties
      'WARNING: XTOL TEST SATISFIED', &! 26 out warn: suboptimal?
      'ERROR: FACTR .LT. 0', &! 27 out:error:invalid input
      'ERROR: M .LE. 0' &! 28 out:error:invalid input
      /)

   Character(Len=60) :: task, csave
   Integer :: i
!  bridge routine with friendly interface to c and tranlates L-BFGS-B tasks, etc.
!  c <-> setulb
!  Bridging elements to translate
!  Character(60) task  <-> Integer :: itask
   If (itask < 1 .Or. itask > 28) Then
!     invalid task number
      itask = 0
      Go To 100
   End If
   task = tasks(itask)
!  Logical :: lsave(4) <-> Integer :: lsavei(4)
   If (task /= 'START') Then
      Do i = 1, 4
         lsave(i) = merge(.True., .False., lsavei(i)/=0)
      End Do
   End If

!  Call setulb(...)
   Call PREC(setulb)(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, task, &
      iprint, csave, lsave, isave, dsave)

!  Bridge back elements
!  Logical :: lsave(4) <-> Integer :: lsavei(4)
   Do i = 1, 4
      lsavei(i) = Merge(1, 0, lsave(i))
   End Do
!  Character(60) task  <-> Integer :: itask
   Select Case (task)
    Case ('NEW_X')
      itask = 1
    Case ('START')
      itask = 2
    Case ('STOP')
      itask = 3
    Case ('FG')
      itask = 4
    Case ('ABNORMAL_TERMINATION_IN_LNSRCH')
      itask = 5
    Case ('CONVERGENCE')
      itask = 6
    Case ('CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL')
      itask = 7
    Case ('CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH')
      itask = 8
    Case ('RESTART_FROM_LNSRCH')
      itask = 9
    Case ('ERROR: FTOL .LT. ZERO')
      itask = 10
    Case ('ERROR: GTOL .LT. ZERO')
      itask = 11
    Case ('ERROR: INITIAL G .GE. ZERO')
      itask = 12
    Case ('ERROR: INVALID NBD')
      itask = 13
    Case ('ERROR: N .LE. 0')
      itask = 14
    Case ('ERROR: NO FEASIBLE SOLUTION')
      itask = 15
    Case ('ERROR: STP .GT. STPMAX')
      itask = 16
    Case ('ERROR: STP .LT. STPMIN')
      itask = 17
    Case ('ERROR: STPMAX .LT. STPMIN')
      itask = 18
    Case ('ERROR: STPMIN .LT. ZERO')
      itask = 19
    Case ('FG_LNSRCH')
      itask = 20
    Case ('FG_START')
      itask = 21
    Case ('ERROR: XTOL .LT. ZERO')
      itask = 22
    Case ('WARNING: ROUNDING ERRORS PREVENT PROGRESS')
      itask = 23
    Case ('WARNING: STP = STPMAX')
      itask = 24
    Case ('WARNING: STP = STPMIN')
      itask = 25
    Case ('WARNING: XTOL TEST SATISFIED')
      itask = 26
    Case ('ERROR: FACTR .LT. 0')
      itask = 27
    Case ('ERROR: M .LE. 0')
      itask = 28
    Case Default
      ! Could not find task! Maybe new task?
      itask = 29
   End Select

100 Continue
   Return
End Subroutine
