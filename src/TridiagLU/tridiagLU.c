#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>

int tridiagLU(double *a,double *b,double *c,double *x,int n,void *r,void *comnctr)
{
  int             i,istart,iend,ierr = 0;
  int             rank,nproc;
  double          sendbuf[4],recvbuf[4];
  TridiagLUTime   *runtimes = (TridiagLUTime*) r;
  struct timeval  start,stage1,stage2,stage3,stage4;
#ifndef serial
  MPI_Comm        *comm = (MPI_Comm*) comnctr;
  MPI_Request     *request;
  MPI_Status      *status;
#endif

  /* start */
  gettimeofday(&start,NULL);

#ifdef serial
  rank  = 0;
  nproc = 1;
#else
  if (comm) {
    MPI_Comm_size(*comm,&nproc);
    MPI_Comm_rank(*comm,&rank );
  } else {
    rank  = 0;
    nproc = 1;
  }
#endif

  /* Stage 1 - Parallel elimination of subdiagonal entries */
  istart  = (rank == 0 ? 1 : 2);
  iend    = n;
  for (i = istart; i < iend; i++) {
    if (b[i-1] == 0) return(-1);
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

  /* end of stage 1 */
  gettimeofday(&stage1,NULL);

  /* Stage 2 - Eliminate the first sub- & super-diagonal entries */
  /* This needs the last (a,b,c) from the previous process       */
  sendbuf[0] = a[n-1]; sendbuf[1] = b[n-1]; sendbuf[2] = c[n-1]; sendbuf[3] = x[n-1];
#ifndef serial
  if (nproc > 1) {
    int nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
    request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
    status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
    if (rank != nproc-1)  MPI_Isend(&sendbuf[0],4,MPI_DOUBLE,rank+1,1436,*comm,&request[0]);
    if (rank == nproc-1 ) MPI_Irecv(&recvbuf[0],4,MPI_DOUBLE,rank-1,1436,*comm,&request[0]);
    else if (rank)        MPI_Irecv(&recvbuf[0],4,MPI_DOUBLE,rank-1,1436,*comm,&request[1]);
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

  /* end of stage 2 */
  gettimeofday(&stage2,NULL);


  /* Stage 3 - Solve the reduced (nproc-1) X (nproc-1) tridiagonal system   */
  double xp1 = 0.0; /* solution for the first element on the next process   */
#ifndef serial
  if (nproc > 1) {
#if defined(gather_and_solve)
    /* Gathering reduced system on root and solving */
    /* allocate the arrays for the reduced tridiagonal system on root */
    double *ra,*rb,*rc,*rx; 
    if (!rank) {
      ra = (double*) calloc (nproc, sizeof(double));
      rb = (double*) calloc (nproc, sizeof(double));
      rc = (double*) calloc (nproc, sizeof(double));
      rx = (double*) calloc (nproc, sizeof(double));
      for (i = 0; i < nproc; i++) ra[i] = rb[i] = rc[i] = rx[i] = 0.0;
    }

    /* allocate send and receive buffers and form the send packet of data */
    double *sendbuf, *recvbuf;
    sendbuf = (double*) calloc (4,sizeof(double));
    if (!rank) recvbuf = (double*) calloc (4*nproc,sizeof(double));
    sendbuf[0] = (rank ? a[0] : 0.0);
    sendbuf[1] = (rank ? b[0] : 1.0);
    sendbuf[2] = (rank ? c[0] : 0.0);
    sendbuf[3] = (rank ? x[0] : 0.0);

    /* gather the reduced system on root process */
    MPI_Gather(sendbuf,4,MPI_DOUBLE,recvbuf,4,MPI_DOUBLE,0,*comm);

    /* extract the data from the recvbuf and solve on root (serial) */
    if (!rank)  {
      int n;
      for (n = 0; n < nproc; n++) {
        ra[n] = recvbuf[4*n+0];
        rb[n] = recvbuf[4*n+1];
        rc[n] = recvbuf[4*n+2];
        rx[n] = recvbuf[4*n+3];
      }
      ierr = tridiagLU(ra,rb,rc,rx,nproc,NULL,NULL);
      if (ierr) return(ierr);
    }

    /* scatter the solution back */
    double x0;
    MPI_Scatter(rx,1,MPI_DOUBLE,&x0,1,MPI_DOUBLE,0,*comm);
    if (rank) x[0] = x0;

    /* clean up */
    if (!rank) {
      free(ra);
      free(rb);
      free(rc);
      free(rx);
      free(recvbuf);
    }
    free(sendbuf);
#elif defined(recursive_doubling)
    /* Solving the reduced system in parallel by recursive-doubling algorithm */
    double zero = 0.0, one = 1.0;
    /* all process except 0 call the recursive-doubling tridiagonal solver */
    if (rank) ierr = tridiagLURD(&a[0],&b[0],&c[0],&x[0],1,NULL,comm);
    else      ierr = tridiagLURD(&zero,&one ,&zero,&zero,1,NULL,comm);
    if (ierr) return(ierr);
#endif /* type of solution for reduced system */
    /* Each process, get the first x of the next process */
    MPI_Status  rcvsts;
    MPI_Request sndreq;
    if (rank)           MPI_Isend(&x[0],1,MPI_DOUBLE,rank-1,1323,*comm,&sndreq);
    if (rank+1 < nproc) MPI_Recv (&xp1 ,1,MPI_DOUBLE,rank+1,1323,*comm,&rcvsts);
  }
#endif /* if not serial                       */
  /* end of stage 3 */
  gettimeofday(&stage3,NULL);

  /* Stage 4 - Parallel back-substitution to get the solution  */
  istart = n-1;
  iend   = (rank == 0 ? 0 : 1);
  if (b[istart] == 0) return(-1);
  x[istart] = (x[istart]-a[istart]*x[0]-c[istart]*xp1) / b[istart];
  for (i = istart-1; i > iend-1; i--) {
    if (b[i] == 0) return(-1);
    x[i] = (x[i]-c[i]*x[i+1]-a[i]*x[0]) / b[i];
  }

  /* end of stage 4 */
  gettimeofday(&stage4,NULL);

  /* Done - now x contains the solution */

  /* save runtimes if needed */
  if (runtimes) {
    long long walltime;
    walltime = ((stage1.tv_sec * 1000000 + stage1.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    runtimes->stage1_time = (double) walltime / 1000000.0;
    walltime = ((stage2.tv_sec * 1000000 + stage2.tv_usec) - (stage1.tv_sec * 1000000 + stage1.tv_usec));
    runtimes->stage2_time = (double) walltime / 1000000.0;
    walltime = ((stage3.tv_sec * 1000000 + stage3.tv_usec) - (stage2.tv_sec * 1000000 + stage2.tv_usec));
    runtimes->stage3_time = (double) walltime / 1000000.0;
    walltime = ((stage4.tv_sec * 1000000 + stage4.tv_usec) - (stage3.tv_sec * 1000000 + stage3.tv_usec));
    runtimes->stage4_time = (double) walltime / 1000000.0;
    walltime = ((stage4.tv_sec * 1000000 + stage4.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    runtimes->total_time = (double) walltime / 1000000.0;
  }

  return(0);
}
