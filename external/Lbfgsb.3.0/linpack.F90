!
!  L-BFGS-B is released under the “New BSD License” (aka “Modified BSD License”
!  or “3-clause license”)
!  Please read attached file License.txt
!
#include "preprocessor.fpp"
    Subroutine PREC(pofa)(a, lda, n, info)
      Use working_precision, Only: wp
      Integer :: lda, n, info
      Real (Kind=wp) :: a(lda, *)
!
!     dpofa factors a double precision symmetric positive definite
!     matrix.
!
!     dpofa is usually called by dpoco, but it can be called
!     directly with a saving in time if  rcond  is not needed.
!     (time for dpoco) = (1 + 18/n)*(time for dpofa) .
!
!     on entry
!
!        a       double precision(lda, n)
!                the symmetric matrix to be factored.  only the
!                diagonal and upper triangle are used.
!
!        lda     integer
!                the leading dimension of the array  a .
!
!        n       integer
!                the order of the matrix  a .
!
!     on return
!
!        a       an upper triangular matrix  r  so that  a = trans(r)*r
!                where  trans(r)  is the transpose.
!                the strict lower triangle is unaltered.
!                if  info .ne. 0 , the factorization is not complete.
!
!        info    integer
!                = 0  for normal return.
!                = k  signals an error condition.  the leading minor
!                     of order  k  is not positive definite.
!
!     linpack.  this version dated 08/14/78 .
!     cleve moler, university of new mexico, argonne national lab.
!
!     subroutines and functions
!
!     blas ddot
!     fortran sqrt
!
!     internal variables
!
      Real (Kind=wp) :: PREC(dot), t
      Real (Kind=wp) :: s
      Integer :: j, jm1, k
!     begin block with ...exits to 40
!
!
      Do j = 1, n
        info = j
        s = 0.0E0_wp
        jm1 = j - 1
        If (jm1<1) Go To 100
        Do k = 1, jm1
          t = a(k, j) - PREC(dot)(k-1, a(1,k), 1, a(1,j), 1)
          t = t/a(k, k)
          a(k, j) = t
          s = s + t*t
        End Do
100     Continue
        s = a(j, j) - s
!     ......exit
        If (s<=0.0E0_wp) Go To 110
        a(j, j) = sqrt(s)
      End Do
      info = 0
110   Continue
      Return
    End Subroutine

!====================== The end of dpofa ===============================

    Subroutine PREC(trsl)(t, ldt, n, b, job, info)
      Use working_precision, Only: wp
      Integer :: ldt, n, job, info
      Real (Kind=wp) :: t(ldt, *), b(*)
!
!
!     dtrsl solves systems of the form
!
!                   t * x = b
!     or
!                   trans(t) * x = b
!
!     where t is a triangular matrix of order n. here trans(t)
!     denotes the transpose of the matrix t.
!
!     on entry
!
!         t         double precision(ldt,n)
!                   t contains the matrix of the system. the zero
!                   elements of the matrix are not referenced, and
!                   the corresponding elements of the array can be
!                   used to store other information.
!
!         ldt       integer
!                   ldt is the leading dimension of the array t.
!
!         n         integer
!                   n is the order of the system.
!
!         b         double precision(n).
!                   b contains the right hand side of the system.
!
!         job       integer
!                   job specifies what kind of system is to be solved.
!                   if job is
!
!                        00   solve t*x=b, t lower triangular,
!                        01   solve t*x=b, t upper triangular,
!                        10   solve trans(t)*x=b, t lower triangular,
!                        11   solve trans(t)*x=b, t upper triangular.
!
!     on return
!
!         b         b contains the solution, if info .eq. 0.
!                   otherwise b is unaltered.
!
!         info      integer
!                   info contains zero if the system is nonsingular.
!                   otherwise info contains the index of
!                   the first zero diagonal element of t.
!
!     linpack. this version dated 08/14/78 .
!     g. w. stewart, university of maryland, argonne national lab.
!
!     subroutines and functions
!
!     blas daxpy,ddot
!     fortran mod
!
!     internal variables
!
      Real (Kind=wp) :: PREC(dot), temp
      Integer :: case, j, jj
!
!     begin block permitting ...exits to 150
!
!        check for zero diagonal elements.
!
      Do info = 1, n
!     ......exit
        If (t(info,info)==0.0E0_wp) Go To 190
      End Do
      info = 0
!
!        determine the task and go to it.
!
      case = 1
      If (mod(job,10)/=0) case = 2
      If (mod(job,100)/10/=0) case = case + 2
      Go To (100, 120, 140, 160) case
!
!        solve t*x=b for t lower triangular
!
100   Continue
      b(1) = b(1)/t(1, 1)
      If (n<2) Go To 110
      Do j = 2, n
        temp = -b(j-1)
        Call PREC(axpy)(n-j+1, temp, t(j,j-1), 1, b(j), 1)
        b(j) = b(j)/t(j, j)
      End Do
110   Continue
      Go To 180
!
!        solve t*x=b for t upper triangular.
!
120   Continue
      b(n) = b(n)/t(n, n)
      If (n<2) Go To 130
      Do jj = 2, n
        j = n - jj + 1
        temp = -b(j+1)
        Call PREC(axpy)(j, temp, t(1,j+1), 1, b(1), 1)
        b(j) = b(j)/t(j, j)
      End Do
130   Continue
      Go To 180
!
!        solve trans(t)*x=b for t lower triangular.
!
140   Continue
      b(n) = b(n)/t(n, n)
      If (n<2) Go To 150
      Do jj = 2, n
        j = n - jj + 1
        b(j) = b(j) - PREC(dot)(jj-1, t(j+1,j), 1, b(j+1), 1)
        b(j) = b(j)/t(j, j)
      End Do
150   Continue
      Go To 180
!
!        solve trans(t)*x=b for t upper triangular.
!
160   Continue
      b(1) = b(1)/t(1, 1)
      If (n<2) Go To 170
      Do j = 2, n
        b(j) = b(j) - PREC(dot)(j-1, t(1,j), 1, b(1), 1)
        b(j) = b(j)/t(j, j)
      End Do
170   Continue
180   Continue
190   Continue
      Return
    End Subroutine
