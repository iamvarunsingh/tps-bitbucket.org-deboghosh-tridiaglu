#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>

#ifndef serial
static int RecursiveDoublingForward(int,double*,int,int,void*);
static int RecursiveDoublingReverse(int,double*,int,int,void*);
#endif

int tridiagLURD(double **a,double **b,double **c,double **x,int n,int ns,void *r,void *m)
{
  int         d,i,j;
  const int   nvar = 4;
#ifndef serial
  MPIContext  *mpi      = (MPIContext*)    m;
  int         ierr = 0;
  int         rank,nproc;
  int         *proc;
  MPI_Comm    *comm;
  MPI_Request sndreq;
  MPI_Status  rcvsts;

  if (mpi) {
    rank  = mpi->rank;
    nproc = mpi->nproc;
    comm  = (MPI_Comm*) mpi->comm;
    proc  = mpi->proc;
    if (!proc) {
      fprintf(stderr,"Error in tridiagLU() on process %d: ",rank);
      fprintf(stderr,"aray \"proc\" is NULL.\n");
      return(-1);
    }
  } else {
    rank  = 0;
    nproc = 1;
    comm  = NULL;
    proc  = NULL;
  }
#endif

  /* Some allocations */
  double *bmm,*bpp,*cmm,*cpp,*xmm,*xpp,*xmp,*xnp;
  bmm = (double*) calloc (ns,sizeof(double));
  bpp = (double*) calloc (ns,sizeof(double));
  cmm = (double*) calloc (ns,sizeof(double));
  cpp = (double*) calloc (ns,sizeof(double));
  xmm = (double*) calloc (ns,sizeof(double));
  xpp = (double*) calloc (ns,sizeof(double));
  xmp = (double*) calloc (ns,sizeof(double));
  xnp = (double*) calloc (ns,sizeof(double));

  /* Step 1 -> send the last c to the next proc */
  for (d=0; d<ns; d++) { cpp[d] = 0.0; cmm[d] = c[d][n-1]; }
#ifndef serial
  if (rank+1 < nproc) MPI_Isend(cmm,ns,MPI_DOUBLE,proc[rank+1],0,*comm,&sndreq);
  if (rank-1 >= 0   ) MPI_Recv (cpp,ns,MPI_DOUBLE,proc[rank-1],0,*comm,&rcvsts);
#endif

  /* Sequential recursion */
  double *S = (double*) calloc (ns*nvar,sizeof(double));
  for (d = 0; d < ns; d++) {
    S[d*nvar+0] = 1.0; S[d*nvar+1] = 0.0; S[d*nvar+2] = 0.0; S[d*nvar+3] = 1.0;
    for (i = 0; i < n; i++) {
      double alpha,beta,s[4];
      if (b[d][i] == 0)  return(-1); /* singular system */
      alpha = b[d][i];
      beta  = -a[d][i] * (i==0 ? cpp[d] : c[d][i-1]);
      s[0] = alpha*S[d*nvar+0] + beta*S[d*nvar+2];
      s[1] = alpha*S[d*nvar+1] + beta*S[d*nvar+3];
      s[2] = S[d*nvar+0];
      s[3] = S[d*nvar+1];
      for (j=0; j<nvar; j++) S[d*nvar+j] = s[j];
    }
  }

  /* Full Recursive Doubling (Forward) */
#ifndef serial
  ierr = RecursiveDoublingForward(ns,S,rank,nproc,mpi);
  if (ierr) return(ierr);
#endif

  /* Combine step */
  for (d=0; d<ns; d++) { 
    b[d][n-1] = (S[d*nvar+0]+S[d*nvar+1]) / (S[d*nvar+2]+S[d*nvar+3]);
    bpp[d] = 1.0; bmm[d] = b[d][n-1];
  }
  free(S);
#ifndef serial
  if (rank+1 < nproc) MPI_Isend(bmm,ns,MPI_DOUBLE,proc[rank+1],1,*comm,&sndreq);
  if (rank-1 >= 0   ) MPI_Recv (bpp,ns,MPI_DOUBLE,proc[rank-1],1,*comm,&rcvsts);
#endif
  for (d = 0; d < ns; d++) {
    for (i = 0; i < n-1; i++) {
      double factor = a[d][i] / (i==0 ? bpp[d] : b[d][i-1]);
      b[d][i] -= factor * (i==0 ? cpp[d] : c[d][i-1]); 
    }
  }

  /* Forward Sweep - Sequential Recursion */
  double *L = (double*) calloc (ns*nvar,sizeof(double));
  for (d = 0; d < ns; d++) {
    L[d*nvar+0] = 1.0; L[d*nvar+1] = 0.0; L[d*nvar+2] = 0.0; L[d*nvar+3] = 1.0;
    for (i = 0; i < n; i++) {
      double alpha,beta,gamma,l[4];
      alpha = -a[d][i];
      beta  =  x[d][i]*(i==0 ? bpp[d] : b[d][i-1]);
      gamma =  (i==0 ? bpp[d] : b[d][i-1]);
      l[0] = alpha*L[d*nvar+0] + beta*L[d*nvar+2];
      l[1] = alpha*L[d*nvar+1] + beta*L[d*nvar+3];
      l[2] = gamma*L[d*nvar+2];
      l[3] = gamma*L[d*nvar+3];
      for (j=0; j<nvar; j++) L[d*nvar+j] = l[j];
    }
  }

  /* Forward Sweep - Full Recursive Doubling (Forward) */
#ifndef serial
  ierr = RecursiveDoublingForward(ns,L,rank,nproc,m);
  if (ierr) return(ierr);
#endif

  /* Forward Sweep - Combine step */
  for (d=0; d<ns; d++) { 
    x[d][n-1] = (L[d*nvar+0]+L[d*nvar+1]) / (L[d*nvar+2]+L[d*nvar+3]);
    xpp[d] = 0; xmm[d] = x[d][n-1]; 
  }
  free(L);
#ifndef serial
  if (rank+1 < nproc) MPI_Isend(xmm,ns,MPI_DOUBLE,proc[rank+1],2,*comm,&sndreq);
  if (rank-1 >= 0   ) MPI_Recv (xpp,ns,MPI_DOUBLE,proc[rank-1],2,*comm,&rcvsts);
#endif
  for (d = 0; d < ns; d++) {
    for (i = 0; i < n-1; i++) {
      double factor = a[d][i] / (i==0 ? bpp[d] : b[d][i-1]);
      x[d][i] -= factor * (i==0 ? xpp[d] : x[d][i-1]); 
    }
  }

  /* Backward Sweep - Sequential recursion*/
  double *U = (double*) calloc (ns*nvar,sizeof(double));
  for (d = 0; d < ns; d++) {
    U[d*nvar+0] = 1.0; U[d*nvar+1] = 0.0; U[d*nvar+2] = 0.0; U[d*nvar+3] = 1.0;
    for (i = n-1; i >= 0; i--) {
      double alpha,beta,gamma,u[4];
      alpha = -c[d][i];
      beta  =  x[d][i];
      gamma =  b[d][i];
      u[0] = alpha*U[d*nvar+0] + beta*U[d*nvar+2];
      u[1] = alpha*U[d*nvar+1] + beta*U[d*nvar+3];
      u[2] = gamma*U[d*nvar+2];
      u[3] = gamma*U[d*nvar+3];
      for (j=0; j<nvar; j++) U[d*nvar+j] = u[j];
    }
  }

  /* Backward Sweep - Full recursive doubling in reverse */
#ifndef serial
  ierr = RecursiveDoublingReverse(ns,U,rank,nproc,m);
  if (ierr) return(ierr);
#endif

  /* Backward Sweep - Combine step */
  for (d=0; d<ns; d++) { 
    x[d][0] = (U[d*nvar+0]+U[d*nvar+1]) / (U[d*nvar+2]+U[d*nvar+3]);
    xnp[d] = 0.0; xmp[d] = x[d][0];
  }
  free(U);
#ifndef serial
  if (rank-1 >= 0   ) MPI_Isend(xmp,ns,MPI_DOUBLE,proc[rank-1],3,*comm,&sndreq);
  if (rank+1 < nproc) MPI_Recv (xnp,ns,MPI_DOUBLE,proc[rank+1],3,*comm,&rcvsts);
#endif
  for (d = 0; d < ns; d++) {
    for (i = n-1; i > 0; i--) {
      if (b[d][i] == 0) return(-1); /* singular system */
      x[d][i] = (x[d][i] - c[d][i] * (i==(n-1) ? xnp[d] : x[d][i+1])) / b[d][i];
    }
  }

  /* Done! */
  free(bmm); free(bpp);
  free(cmm); free(cpp);
  free(xmm); free(xpp);
  free(xmp); free(xnp);
  return(0);
}

#ifndef serial
int RecursiveDoublingForward(int ns,double *S,int rank,int nproc,void *m)
{
  MPIContext *mpi  = (MPIContext*) m;
  MPI_Comm   *comm = (MPI_Comm*) mpi->comm;
  int        *proc = mpi->proc;

  int const nvar=4;
  int       d,n;
  double    *T = (double*) calloc (ns*nvar,sizeof(double));
  for (d = 0; 1<<d < nproc; d++) {
    MPI_Request sndreq;
    MPI_Status  rcvsts;
    for (n=0; n<ns; n++) { T[n*nvar+0] = T[n*nvar+3] = 1; T[n*nvar+1] = T[n*nvar+2] = 0;  }
    if (rank+(1<<d) < nproc) MPI_Isend(S,nvar*ns,MPI_DOUBLE,proc[rank+(1<<d)],d,*comm,&sndreq);
    if (rank-(1<<d) >= 0   ) MPI_Recv (T,nvar*ns,MPI_DOUBLE,proc[rank-(1<<d)],d,*comm,&rcvsts);
    for (n = 0; n < ns; n++) {
      double s[4];
      s[0] = S[n*nvar+0]*T[n*nvar+0] + S[n*nvar+1]*T[n*nvar+2];
      s[1] = S[n*nvar+0]*T[n*nvar+1] + S[n*nvar+1]*T[n*nvar+3];
      s[2] = S[n*nvar+2]*T[n*nvar+0] + S[n*nvar+3]*T[n*nvar+2];
      s[3] = S[n*nvar+2]*T[n*nvar+1] + S[n*nvar+3]*T[n*nvar+3];
      int j; for (j=0; j<nvar; j++) S[n*nvar+j] = s[j];
    }
  }
  free(T);
  return(0);
}
int RecursiveDoublingReverse(int ns,double *S,int rank,int nproc,void *m)
{
  MPIContext *mpi  = (MPIContext*) m;
  MPI_Comm   *comm = (MPI_Comm*) mpi->comm;
  int        *proc = mpi->proc;

  int const nvar=4;
  int       d,n;
  double    *T = (double*) calloc (ns*nvar,sizeof(double));
  for (d = 0; 1<<d < nproc; d++) {
    MPI_Request sndreq;
    MPI_Status  rcvsts;
    for (n=0; n<ns; n++) { T[n*nvar+0] = T[n*nvar+3] = 1; T[n*nvar+1] = T[n*nvar+2] = 0;  }
    if (rank-(1<<d) >= 0   ) MPI_Isend(S,nvar*ns,MPI_DOUBLE,proc[rank-(1<<d)],d,*comm,&sndreq);
    if (rank+(1<<d) < nproc) MPI_Recv (T,nvar*ns,MPI_DOUBLE,proc[rank+(1<<d)],d,*comm,&rcvsts);
    for (n = 0; n < ns; n++) {
      double s[4];
      s[0] = S[n*nvar+0]*T[n*nvar+0] + S[n*nvar+1]*T[n*nvar+2];
      s[1] = S[n*nvar+0]*T[n*nvar+1] + S[n*nvar+1]*T[n*nvar+3];
      s[2] = S[n*nvar+2]*T[n*nvar+0] + S[n*nvar+3]*T[n*nvar+2];
      s[3] = S[n*nvar+2]*T[n*nvar+1] + S[n*nvar+3]*T[n*nvar+3];
      int j; for (j=0; j<nvar; j++) S[n*nvar+j] = s[j];
    }
  }
  free(T);
  return(0);
}
#endif
