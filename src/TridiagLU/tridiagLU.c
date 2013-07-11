#include <stdlib.h>
#ifndef serial
#include <mpi.h>
#endif

/* Parallel direct solver for tridiagonal systems */

/*
  Arguments:-
    a   [0,n-1] double*   subdiagonal entries
    b   [0,n-1] double*   diagonal entries
    c   [0,n-1] double*   superdiagonal entries
    x   [0,n-1] double*   right-hand side
    n           int       local size of the system
    rank        int       rank of this process
    nproc       int       total number of processes

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
  MPI_Request *request;
  MPI_Status  *status;
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
      b[0] -=  factor * a[i-1];
      x[0] -=  factor * x[i-1];
    }
  }

  /* Stage 2 - Eliminate the first sub- & super-diagonal entries */
  /* This needs the last (a,b,c) from the previous process       */
  sendbuf[0] = a[n-1]; sendbuf[1] = b[n-1]; sendbuf[2] = c[n-1]; sendbuf[3] = x[n-1];
#ifndef serial
  if (nproc > 1) {
    int nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
    request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
    status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
    if (rank != nproc-1)  MPI_Isend(&sendbuf[0],4,MPI_DOUBLE,rank+1,1436,MPI_COMM_WORLD,&request[0]);
    if (rank == nproc-1 ) MPI_Irecv(&recvbuf[0],4,MPI_DOUBLE,rank-1,1436,MPI_COMM_WORLD,&request[0]);
    else if (rank)        MPI_Irecv(&recvbuf[0],4,MPI_DOUBLE,rank-1,1436,MPI_COMM_WORLD,&request[1]);
    MPI_Waitall(nreq,&request[0],&status[0]);
    free(request);
    free(status);
  }
#endif
  /* The first process sits this one out */
  if (rank) {
    double am1, bm1, cm1, xm1;
    am1 = recvbuf[0]; bm1 = recvbuf[1]; cm1 = recvbuf[2]; xm1 = recvbuf[3];
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


  /* Stage 3 - Solve the reduced (nproc-1) X (nproc-1) tridiagonal system   */
  /**** BAD IMPLEMENTATION - solving this on ALL processes except the first */
  double xp1 = 0.0; /* solution for the first element on the next process   */
  if (nproc > 1) {
    double *ra,*rb,*rc,*rx; /* arrays for the reduced tridiagonal system      */

    /* allocate the arrays */
    ra = (double*) calloc (nproc-1, sizeof(double));
    rb = (double*) calloc (nproc-1, sizeof(double));
    rc = (double*) calloc (nproc-1, sizeof(double));
    rx = (double*) calloc (nproc-1, sizeof(double));

    /* set this process' element of these arrays and rest to zero */
    for (i = 0; i < nproc-1; i++) ra[i] = rb[i] = rc[i] = rx[i] = 0.0;
    if (rank) {
      ra[rank-1] = a[0]; 
      rb[rank-1] = b[0];
      rc[rank-1] = c[0];
      rx[rank-1] = x[0];
    }

    /* assemble the complete arrays across all processes */
#ifndef serial
    if (nproc > 1) MPI_Allreduce(ra,ra,nproc-1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if (nproc > 1) MPI_Allreduce(rb,rb,nproc-1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if (nproc > 1) MPI_Allreduce(rc,rc,nproc-1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if (nproc > 1) MPI_Allreduce(rx,rx,nproc-1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif

    /* solve the system independently on all the process */
    if (nproc-1 > 1) {
      ierr = tridiagLU(ra,rb,rc,rx,nproc-1,0,1);
      if (ierr) return(ierr);
    } else rx[0] /= rb[0];

    /* save the solution */
    if (rank) x[0] = rx[rank-1];
    if (rank != nproc-1)  xp1 = rx[rank];

    /* clean up */
    free(ra);
    free(rb);
    free(rc);
    free(rx);
  }

  /* Stage 4 - Parallel back-substitution to get the solution  */
  istart = n-1;
  iend   = (rank == 0 ? 0 : 1);
  if (b[istart] == 0) return(-1);
  x[istart] = (x[istart]-a[istart]*x[0]-c[istart]*xp1) / b[istart];
  for (i = istart-1; i > iend-1; i--) {
    if (b[i] == 0) return(-1);
    x[i] = (x[i]-c[i]*x[i+1]-a[i]*x[0]) / b[i];
  }

  /* Done - now x contains the solution */
  return(0);
}




  /*
  sendbuf[0] = x[0];
#ifndef serial
  if (nproc > 1) {
    int nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
    request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
    status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
    if (rank)                  MPI_Isend(&sendbuf[0],1,MPI_DOUBLE,rank-1,1538,MPI_COMM_WORLD,&request[0]);
    if (!rank)                 MPI_Irecv(&xp1       ,1,MPI_DOUBLE,rank+1,1538,MPI_COMM_WORLD,&request[0]);
    else if (rank != nproc-1)  MPI_Irecv(&xp1       ,1,MPI_DOUBLE,rank+1,1538,MPI_COMM_WORLD,&request[1]);
    MPI_Waitall(nreq,&request[0],&status[0]);
    free(request);
    free(status);
  }
#endif
  */
