#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>

int tridiagLU(double **a,double **b,double **c,double **x,
              int n,int ns,void *r,void *comnctr)
{
  int             d,i,istart,iend;
  int             rank,nproc;
  TridiagLUTime   *runtimes = (TridiagLUTime*) r;
  struct timeval  start,stage1,stage2,stage3,stage4;
#ifndef serial
  int             ierr = 0;
  const int       nvar = 4;
  double          *sendbuf,*recvbuf;
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

  if ((ns == 0) || (n == 0)) return(0);
  /* some allocations */
  double *xs1,*xp1; /* to exchange the first element on the next process   */
  xp1 = (double*) calloc(ns,sizeof(double)); for (d=0; d<ns; d++) xp1[d] = 0.0;
  xs1 = (double*) calloc(ns,sizeof(double)); for (d=0; d<ns; d++) xs1[d] = 0.0;

  /* Stage 1 - Parallel elimination of subdiagonal entries */
  istart  = (rank == 0 ? 1 : 2);
  iend    = n;
  for (d = 0; d < ns; d++) {
    for (i = istart; i < iend; i++) {
      if (b[d][i-1] == 0) return(-1);
      double factor = a[d][i] / b[d][i-1];
      b[d][i] -=  factor * c[d][i-1];
      a[d][i]  = -factor * a[d][i-1];
      x[d][i] -=  factor * x[d][i-1];
      if (rank) {
        double factor = c[d][0] / b[d][i-1];
        c[d][0]  = -factor * c[d][i-1];
        b[d][0] -=  factor * a[d][i-1];
        x[d][0] -=  factor * x[d][i-1];
      }
    }
  }

  /* end of stage 1 */
  gettimeofday(&stage1,NULL);

  /* Stage 2 - Eliminate the first sub- & super-diagonal entries */
  /* This needs the last (a,b,c,x) from the previous process     */
#ifndef serial
  sendbuf = (double*) calloc (ns*nvar,sizeof(double*));
  recvbuf = (double*) calloc (ns*nvar,sizeof(double*));
  for (d=0; d<ns; d++) {
    sendbuf[d*nvar+0] = a[d][n-1]; 
    sendbuf[d*nvar+1] = b[d][n-1]; 
    sendbuf[d*nvar+2] = c[d][n-1]; 
    sendbuf[d*nvar+3] = x[d][n-1];
  }
  if (nproc > 1) {
    int nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
    request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
    status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
    if (rank != nproc-1)  MPI_Isend(sendbuf,nvar*ns,MPI_DOUBLE,rank+1,1436,*comm,&request[0]);
    if (rank == nproc-1 ) MPI_Irecv(recvbuf,nvar*ns,MPI_DOUBLE,rank-1,1436,*comm,&request[0]);
    else if (rank)        MPI_Irecv(recvbuf,nvar*ns,MPI_DOUBLE,rank-1,1436,*comm,&request[1]);
    MPI_Waitall(nreq,&request[0],&status[0]);
    free(request);
    free(status);
  }
  /* The first process sits this one out */
  if (rank) {
    for (d = 0; d < ns; d++) {
      double am1, bm1, cm1, xm1;
      am1 = recvbuf[d*nvar+0]; 
      bm1 = recvbuf[d*nvar+1]; 
      cm1 = recvbuf[d*nvar+2]; 
      xm1 = recvbuf[d*nvar+3];
      double factor;
      if (bm1 == 0) return(-1);
      factor =  a[d][0] / bm1;
      b[d][0]  -=  factor * cm1;
      a[d][0]   = -factor * am1;
      x[d][0]  -=  factor * xm1;
      if (b[d][n-1] == 0) return(-1);
      factor =  c[d][0] / b[d][n-1];
      b[d][0]  -=  factor * a[d][n-1];
      c[d][0]   = -factor * c[d][n-1];
      x[d][0]  -=  factor * x[d][n-1];
    }
  }
  free(sendbuf); free(recvbuf);

  /* end of stage 2 */
  gettimeofday(&stage2,NULL);

  /* Stage 3 - Solve the reduced (nproc-1) X (nproc-1) tridiagonal system   */
  if (nproc > 1) {
#if defined(gather_and_solve)
    /* Gathering reduced systems and solving                  */
    /* allocate the arrays for the reduced tridiagonal system */
//    double **ra,**rb,**rc,**rx; 
//    ra = (double**) calloc (ns_local,sizeof(double));
//    rb = (double**) calloc (ns_local,sizeof(double));
//    rc = (double**) calloc (ns_local,sizeof(double));
//    rx = (double**) calloc (ns_local,sizeof(double));
//    for (d = 0; d < ns_local; d++) {
//      ra[d] = (double*) calloc (nproc, sizeof(double));
//      rb[d] = (double*) calloc (nproc, sizeof(double));
//      rc[d] = (double*) calloc (nproc, sizeof(double));
//      rx[d] = (double*) calloc (nproc, sizeof(double));
//      for (i = 0; i < nproc; i++) 
//        ra[d][i] = rb[d][i] = rc[d][i] = rx[d][i] = 0.0;
//    }

    /* allocate send and receive buffers and form the send packet of data */
//    sendbuf = (double*) calloc (nvar,sizeof(double));
//    if (!rank) recvbuf = (double*) calloc (nvar*nproc,sizeof(double));
//    sendbuf[0] = (rank ? a[0] : 0.0);
//    sendbuf[1] = (rank ? b[0] : 1.0);
//    sendbuf[2] = (rank ? c[0] : 0.0);
//    sendbuf[3] = (rank ? x[0] : 0.0);

    /* gather the reduced system on root process */
//    MPI_Gather(sendbuf,nvar,MPI_DOUBLE,recvbuf,nvar,MPI_DOUBLE,0,*comm);

    /* extract the data from the recvbuf and solve */
//    for (d = 0; d < ns_local; d++) {
//      for (i = 0; i < nproc; i++) {
//        ra[d][i] = recvbuf[d][nvar*i+0];
//        rb[d][i] = recvbuf[d][nvar*i+1];
//        rc[d][i] = recvbuf[d][nvar*i+2];
//        rx[d][i] = recvbuf[d][nvar*i+3];
//      }
//    }
//    ierr = tridiagLU(ns_local,ra,rb,rc,rx,nproc,NULL,NULL);
//    if (ierr) return(ierr);

    /* scatter the solution back */
//    double x0;
//    MPI_Scatter(rx,1,MPI_DOUBLE,&x0,1,MPI_DOUBLE,0,*comm);
//    if (rank) x[0] = x0;

    /* clean up */
//    if (ns_local > 0) {
//      for (d = 0; d < ns_local; d++) {
//        free(ra[d]);
//        free(rb[d]);
//        free(rc[d]);
//        free(rx[d]);
//      }
//      free(ra);
//      free(rb);
//      free(rc);
//      free(rd);
//    }
#elif defined(recursive_doubling)
    /* Solving the reduced system in parallel by recursive-doubling algorithm */
    double **zero, **one;
    zero    = (double**) calloc (ns,sizeof(double*));
    one     = (double**) calloc (ns,sizeof(double*));
    for (d=0; d<ns; d++) {
      zero[d] = (double* ) calloc (1,sizeof(double )); zero[d][0] = 0.0;
      one [d] = (double* ) calloc (1,sizeof(double )); one [d][0] = 1.0;
    }
    if (rank) ierr = tridiagLURD(a,b,c,x,1,ns,NULL,comm);
    else      ierr = tridiagLURD(zero,one,zero,zero,1,ns,NULL,comm);
    if (ierr) return(ierr);
    for (d=0; d<ns; d++) free(zero[d]); free(zero);
    for (d=0; d<ns; d++) free(one [d]); free(one );
#endif /* type of solution for reduced system */

    /* Each process, get the first x of the next process */
    MPI_Status  rcvsts;
    MPI_Request sndreq;
    for (d=0; d<ns; d++)  xs1[d] = x[d][0];
    if (rank)           MPI_Isend(xs1,ns,MPI_DOUBLE,rank-1,1323,*comm,&sndreq);
    if (rank+1 < nproc) MPI_Recv (xp1,ns,MPI_DOUBLE,rank+1,1323,*comm,&rcvsts);
  }
#endif /* if not serial */
  /* end of stage 3 */
  gettimeofday(&stage3,NULL);

  /* Stage 4 - Parallel back-substitution to get the solution  */
  istart = n-1;
  iend   = (rank == 0 ? 0 : 1);
  for (d = 0; d < ns; d++) {
    if (b[d][istart] == 0) return(-1);
    x[d][istart] = (x[d][istart]-a[d][istart]*x[d][0]-c[d][istart]*xp1[d]) / b[d][istart];
    for (i = istart-1; i > iend-1; i--) {
      if (b[d][i] == 0) return(-1);
      x[d][i] = (x[d][i]-c[d][i]*x[d][i+1]-a[d][i]*x[d][0]) / b[d][i];
    }
  }

  /* end of stage 4 */
  gettimeofday(&stage4,NULL);

  /* Done - now x contains the solution */
  free(xp1);
  free(xs1);

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
