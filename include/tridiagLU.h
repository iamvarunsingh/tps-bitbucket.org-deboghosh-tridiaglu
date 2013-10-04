/* 

  Parallel direct solver for tridiagonal systems 

  tridiagLU  (a,b,c,x,n,ns,r,m) (Parallel tridiagonal solver)
  tridiagLURD(a,b,c,x,n,ns,r,m) (Parallel tridiagonal solver
                                 based on the recursive-
                                 doubling algorithm)

  Arguments:-
    a   [0,ns-1]x[0,n-1] double**         subdiagonal entries
    b   [0,ns-1]x[0,n-1] double**         diagonal entries
    c   [0,ns-1]x[0,n-1] double**         superdiagonal entries
    x   [0,ns-1]x[0,n-1] double**         right-hand side (solution)
    n                    int              local size of the system
    ns                   int              number of systems to solve
    r                    TridiagLUTime*   structure containing the runtimes
                                            total_time
                                            stage1_time
                                            stage2_time
                                            stage3_time
                                            stage4_time
                        ** Note that these are process-specific. Calling 
                           function needs to do something to add/average 
                           them to get some global value.
                        ** Can be NULL if runtimes are not needed.
    m                   MPIContext*       structure containing the MPI
                                          context
                                          **See below

  Return value (int) -> 0 (successful solve), -1 (singular system)

  Note:-
    x contains the final solution (right-hand side is replaced)
    a,b,c are not preserved
    On rank=0,        a[0] has to be zero.
    On rank=nproc-1,  c[n-1] has to be zero.

  ** Compile with either of the following flags:
  "-Dgather_and_solve"  : Reduced systems are gathered on one processor 
                          and solved
  "-Drecursive_doubling": Reduced systems are solved on all processors 
                          using the recursive doubling algorithm

  For a serial tridiagonal solver, compile with the flag "-Dserial"
  or send NULL as the argument for the MPI communicator.

*/


/* Data structure containing the stage runtimes */
typedef struct _tridiagLUruntimes_ {
  double  total_time;
  double  stage1_time;
  double  stage2_time;
  double  stage3_time;
  double  stage4_time;
} TridiagLUTime;


/* 
  Data structure for the MPI context

  rank      rank of this process with respect to the processes parti-
            cipating in the tridiagonal solve
  nproc     number of processes participating in the tridiagonal solve
  comm      MPI communicator
  proc      an array of size nproc containing the actual rank in comm
            for each rank 0,...,nproc
*/

typedef struct _mpi_context_ {
  int   rank;
  int   nproc;
  void* comm;
  int*  proc;
} MPIContext;

int tridiagLU  (double**,double**,double**,double**,int,int,void*,void*);
int tridiagLURD(double**,double**,double**,double**,int,int,void*,void*);
