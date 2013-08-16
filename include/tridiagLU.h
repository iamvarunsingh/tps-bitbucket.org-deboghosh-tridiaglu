/* 

  Parallel direct solver for tridiagonal systems 

  tridiagLU  (a,b,c,x,n,r,comm) (Parallel tridiagonal solver)
  tridiagLURD(a,b,c,x,n,r,comm) (Parallel tridiagonal solver
                                 based on the recursive-
                                 doubling algorithm)

  Arguments:-
    a   [0,n-1] double*         subdiagonal entries
    b   [0,n-1] double*         diagonal entries
    c   [0,n-1] double*         superdiagonal entries
    x   [0,n-1] double*         right-hand side (solution)
    n           int             local size of the system
    r           TridiagLUTime*  structure containing the runtimes
                                  total_time
                                  stage1_time
                                  stage2_time
                                  stage3_time
                                  stage4_time
                                *Note that these are process-specific. 
                                 Calling function needs to do something 
                                 to add/average them to get some global 
                                 value.
    comm        MPI_COMM*       MPI Communicator (NULL for serial)

  Return value (int) -> 0 (successful solve), -1 (singular system)

  Note:-
    x contains the final solution (right-hand side is replaced)
    a,b,c are not preserved
    On rank=0,        a[0] has to be zero.
    On rank=nproc-1,  c[n-1] has to be zero.

  For a serial tridiagonal solver, compile with the flag "-Dserial"
  or call with rank = 0 and nproc = 1.

*/

typedef struct _tridiagLUruntimes_ {
  double  total_time;
  double  stage1_time;
  double  stage2_time;
  double  stage3_time;
  double  stage4_time;
} TridiagLUTime;

int tridiagLU  (double*,double*,double*,double*,int,void*,void*);
int tridiagLURD(double*,double*,double*,double*,int,void*,void*);
