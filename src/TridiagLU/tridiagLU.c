#include <stdio.h>
#ifndef serial
#include <mpi.h>
#endif

/* Parallel direct solver for tridiagonal systems */
/*
  Arguments:-
    a   [0,n-1] double  subdiagonal entries
    b   [0,n-1] double  diagonal entries
    c   [0,n-1] double  superdiagonal entries
    x   [0,n-1] double  right-hand side
    n           int     local size of the system
    rank        int     rank of this process
    nproc       int     total number of processes

  Output:-
    x will contain the solution at the end of this function

  Return value:-
    0   -> successful
    -1  -> singular system

  Note:-
    a,b,c are not preserved
    On rank=0,        a[0] has to be zero.
    On rank=nproc-1,  c[n-1] has to be zero.

  For a serial tridiagonal solver, compile with the flag "-Dserial"
  or call with rank = 0 and nproc = 1.
*/


int tridiagLU(double *a,double *b,double *c,double *x,int n,int rank,int nproc)
{
  int         i,ierr = 0;
  int         istart,iend;
  double      sendbuf[4],recvbuf[4];
#ifndef serial
  MPI_Request request[2];
  MPI_Status  status [2];
#endif

  /* Stage 1 - Parallel elimination of subdiagonal entries */
  istart  = (rank == 0 ? 1 : 2);
  iend    = n;
  for (i = istart; i < iend; i++) {
    if (b[i-1] == 0)  return(-1);
    double factor = a[i] / b[i-1];
    b[i] -=  factor * c[i-1];
    a[i]  = -factor * a[i-1];
    x[i] -=  factor * x[i-1];
    if (rank) {
      double factor = c[0] / b[i-1];
      c[0]  = -factor * c[i-1];
      x[0] -=  factor * x[i-1];
    }
  }

  /* Stage 2 - Eliminate the first sub- & super-diagonal entries */
  /* This needs the last (a,b,c) from the previous process       */
  sendbuf[0] = a[n-1]; sendbuf[1] = b[n-1]; sendbuf[2] = c[n-1]; sendbuf[4] = x[n-1];
#ifndef serial
  if (nproc > 1) {
    if (rank != nproc-1)  MPI_Isend(&sendbuf[0],4,MPI_DOUBLE,rank+1,1436,MPI_COMM_WORLD,&request[0]);
    if (rank)             MPI_Irecv(&recvbuf[0],4,MPI_DOUBLE,rank-1,1436,MPI_COMM_WORLD,&request[1]);
    MPI_Waitall(2,request,status);
  }
#endif
  double am1, bm1, cm1, xm1;
  am1 = recvbuf[0]; bm1 = recvbuf[1]; cm1 = recvbuf[2]; xm1 = recvbuf[3];
  /* The first process sits this one out */
  if (rank) {
    double factor;
    if (bm1 == 0) return(-1);
    factor =  a[0] / bm1;
    b[0]  -=  factor * cm1;
    a[0]   = -factor * am1;
    x[0]  -=  factor * xm1;
    if (b[n-1] == 0) return(-1);
    factor =  c[0] / b[n-1];
    b[0]  -=  factor * a[n-1];
    c[0]   = -factor * c[n-1];
    x[0]  -=  factor * x[n-1];
  }


  /* Stage 3 - Solve the reduced (nproc-1) X (nproc-1) tridiagonal system */


  /* Stage 4 - Parallel back-substitution to get the solution  */
  double xp1 = 0.0;
  sendbuf[0] = x[0];
#ifndef serial
  if (nproc > 1) {
    if (rank)             MPI_Isend(&sendbuf[0],1,MPI_DOUBLE,rank-1,1538,MPI_COMM_WORLD,&request[0]);
    if (rank != nproc-1)  MPI_Irecv(&xp1       ,1,MPI_DOUBLE,rank+1,1538,MPI_COMM_WORLD,&request[1]);
    MPI_Waitall(2,&request[0],status);
  }
#endif
  istart = n-1;
  iend   = (rank == 0 ? 0 : 1);
  if (b[istart] == 0) return(-1);
  x[istart] = (x[istart]-a[istart]*x[0]-c[istart]*xp1) / b[istart];
  for (i = istart-1; i > iend-1; i--) {
    if (b[i] == 0) return(-1);
    x[i] = (x[i]-c[i]*x[i+1]-a[i]*x[0]) / b[i];
  }

  return(0);
}
