#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>

#ifndef serial
static int RecursiveDoublingForward(double*,int,int,void*);
static int RecursiveDoublingReverse(double*,int,int,void*);
#endif

int tridiagLURD(double *a,double *b,double *c,double *x,int n,int rank,int nproc,void *r,void *comnctr)
{
  int         i,ierr;
  const int   nvar = 4;
#ifndef serial
  MPI_Comm        *comm = (MPI_Comm*) comnctr;
  MPI_Request     sndreq;
  MPI_Status      rcvsts;
#endif

  /* Step 1 -> send the last c to the next proc */
  double cpp = 0; /* last c from previous process */
#ifndef serial
  if (rank+1 < nproc) MPI_Isend(&c[n-1],1,MPI_DOUBLE,rank+1,0,*comm,&sndreq);
  if (rank-1 >= 0   ) MPI_Recv (&cpp   ,1,MPI_DOUBLE,rank-1,0,*comm,&rcvsts);
#endif

  /* Sequential recursion */
  double S[4] = {1,0,0,1};
  for (i = 0; i < n; i++) {
    double alpha,beta,s[4];
    int j;
    alpha = b[i];
    beta  = -a[i] * (i==0 ? cpp : c[i-1]);
    s[0] = alpha*S[0] + beta*S[2];
    s[1] = alpha*S[1] + beta*S[3];
    s[2] = S[0];
    s[3] = S[1];
    for (j=0; j<nvar; j++) S[j] = s[j];
  }

  /* Full Recursive Doubling (Forward) */
#ifndef serial
  ierr = RecursiveDoublingForward(&S[0],rank,nproc,comm);
#endif

  /* Combine step */
  b[n-1] = (S[0]+S[1]) / (S[2]+S[3]);
  double bpp = 1; /* last b from previous process */
#ifndef serial
  if (rank+1 < nproc) MPI_Isend(&b[n-1],1,MPI_DOUBLE,rank+1,0,*comm,&sndreq);
  if (rank-1 >= 0   ) MPI_Recv (&bpp   ,1,MPI_DOUBLE,rank-1,0,*comm,&rcvsts);
#endif
  for (i = 0; i < n-1; i++) {
    double factor = a[i] / (i==0 ? bpp : b[i-1]);
    b[i] -= factor * (i==0 ? cpp : c[i-1]); 
  }

  /* Forward Sweep - Sequential Recursion */
  double L[4] = {1,0,0,1};
  for (i = 0; i < n; i++) {
    int j;
    double alpha,beta,gamma,l[4];
    alpha = -a[i];
    beta  =  x[i]*(i==0 ? bpp : b[i-1]);
    gamma =  (i==0 ? bpp : b[i-1]);
    l[0] = alpha*L[0] + beta*L[2];
    l[1] = alpha*L[1] + beta*L[3];
    l[2] = gamma*L[2];
    l[3] = gamma*L[3];
    for (j=0; j<nvar; j++) L[j] = l[j];
  }

  /* Forward Sweep - Full Recursive Doubling (Forward) */
#ifndef serial
  ierr = RecursiveDoublingForward(&L[0],rank,nproc,comm);
#endif

  /* Forward Sweep - Combine step */
  x[n-1] = (L[0]+L[1]) / (L[2]+L[3]);
  double xpp = 0; /* last b from previous process */
#ifndef serial
  if (rank+1 < nproc) MPI_Isend(&x[n-1],1,MPI_DOUBLE,rank+1,0,*comm,&sndreq);
  if (rank-1 >= 0   ) MPI_Recv (&xpp   ,1,MPI_DOUBLE,rank-1,0,*comm,&rcvsts);
#endif
  for (i = 0; i < n-1; i++) {
    double factor = a[i] / (i==0 ? bpp : b[i-1]);
    x[i] -= factor * (i==0 ? xpp : x[i-1]); 
  }

  /* Backward Sweep - Sequential recursion*/
  double U[4] = {1,0,0,1};
  for (i = n-1; i >= 0; i--) {
    int j;
    double alpha,beta,gamma,u[4];
    alpha = -c[i];
    beta  =  x[i];
    gamma =  b[i];
    u[0] = alpha*U[0] + beta*U[2];
    u[1] = alpha*U[1] + beta*U[3];
    u[2] = gamma*U[2];
    u[3] = gamma*U[3];
    for (j=0; j<nvar; j++) U[j] = u[j];
  }

  /* Backward Sweep - Full recursive doubling in reverse */
#ifndef serial
  ierr = RecursiveDoublingReverse(&U[0],rank,nproc,comm);
#endif

  /* Backward Sweep - Combine step */
  x[0] = (U[0]+U[1]) / (U[2]+U[3]);
  double xnp = 0; /* first x from the next process */
#ifndef serial
  if (rank-1 >= 0   ) MPI_Isend(&x[0],1,MPI_DOUBLE,rank-1,0,*comm,&sndreq);
  if (rank+1 < nproc) MPI_Recv (&xnp ,1,MPI_DOUBLE,rank+1,0,*comm,&rcvsts);
#endif
  for (i = n-1; i > 0; i--) x[i] = (x[i] - c[i] * (i==(n-1) ? xnp : x[i+1])) / b[i];

  /* Done! */
  return(0);
}

#ifndef serial
int RecursiveDoublingForward(double *S,int rank,int nproc,void *c)
{
  MPI_Comm  *comm = (MPI_Comm*) c;
  const int nvar=4;
  int       d;
  for (d = 0; 1<<d < nproc; d++) {
    MPI_Request sndreq;
    MPI_Status  rcvsts;
    double      s[4];
    int         j;
    double      *T = (double*) calloc (nvar,sizeof(double));
    T[0] = 1; T[1] = 0; T[2] = 0; T[3] = 1;
    if (rank+(1<<d) < nproc) MPI_Isend(S,nvar,MPI_DOUBLE,rank+(1<<d),d,*comm,&sndreq);
    if (rank-(1<<d) >= 0   ) MPI_Recv (T,nvar,MPI_DOUBLE,rank-(1<<d),d,*comm,&rcvsts);
    s[0] = S[0]*T[0] + S[1]*T[2];
    s[1] = S[0]*T[1] + S[1]*T[3];
    s[2] = S[2]*T[0] + S[3]*T[2];
    s[3] = S[2]*T[1] + S[3]*T[3];
    for (j=0; j<nvar; j++) S[j] = s[j];
    free(T);
  }
  return(0);
}
int RecursiveDoublingReverse(double *S,int rank,int nproc,void *c)
{
  MPI_Comm  *comm = (MPI_Comm*) c;
  const int nvar=4;
  int       d;
  for (d = 0; 1<<d < nproc; d++) {
    MPI_Request sndreq;
    MPI_Status  rcvsts;
    double      s[4];
    int         j;
    double      *T = (double*) calloc (nvar,sizeof(double));
    T[0] = 1; T[1] = 0; T[2] = 0; T[3] = 1;
    if (rank-(1<<d) >= 0   ) MPI_Isend(S,nvar,MPI_DOUBLE,rank-(1<<d),d,*comm,&sndreq);
    if (rank+(1<<d) < nproc) MPI_Recv (T,nvar,MPI_DOUBLE,rank+(1<<d),d,*comm,&rcvsts);
    s[0] = S[0]*T[0] + S[1]*T[2];
    s[1] = S[0]*T[1] + S[1]*T[3];
    s[2] = S[2]*T[0] + S[3]*T[2];
    s[3] = S[2]*T[1] + S[3]*T[3];
    for (j=0; j<nvar; j++) S[j] = s[j];
    free(T);
  }
  return(0);
}
#endif
