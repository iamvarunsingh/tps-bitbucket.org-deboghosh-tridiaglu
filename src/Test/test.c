#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>

static void   CopyArray       (double*,double*,int);
#ifdef serial
static int    main_serial     (int);
static double CalculateError  (double*,double*,double*,double*,double*,int);
#else
static int    main_mpi        (int,int,int,int);
static int    partition1D     (int,int,int,int*);
static double CalculateError  (double*,double*,double*,double*,double*,int,int,int);
#endif

int main(int argc, char *argv[])
{
  int ierr,N;
#ifdef serial
  printf("Enter N: ");
  scanf ("%d",&N);
  ierr = main_serial(N);
  if (ierr) fprintf(stderr,"main_mpi() returned with an error code of %d.\n",ierr);
#else
  int NRuns;
  int rank,nproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  if (!rank) {
    scanf ("%d %d",&N,&NRuns);
  }
  MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&NRuns,1,MPI_INT,0,MPI_COMM_WORLD);
  ierr = main_mpi(N,NRuns,rank,nproc);
  if (ierr) fprintf(stderr,"main_mpi() returned with an error code of %d on rank %d.\n",ierr,rank);
  MPI_Finalize();
#endif
  return(0);
}

#ifdef serial
int main_serial(int N)
{
  double *a1,*b1,*c1,*x;
  double *a2,*b2,*c2,*y;
  int     i,ierr=0;
  double  error;

  srand(time(NULL));

  printf("Testing serial tridiagLU() with N=%d\n",N);

  /* allocate arrays */
  a1 = (double*) calloc (N,sizeof(double));
  b1 = (double*) calloc (N,sizeof(double));
  c1 = (double*) calloc (N,sizeof(double));
  a2 = (double*) calloc (N,sizeof(double));
  b2 = (double*) calloc (N,sizeof(double));
  c2 = (double*) calloc (N,sizeof(double));
  x  = (double*) calloc (N,sizeof(double));
  y = (double*) calloc (N,sizeof(double));

  /* Test 1: [I]x = b => x = b */
  for (i = 0; i < N; i++) {
    a1[i] = 0.0;
    b1[i] = 1.0;
    c1[i] = 0.0;
    x[i]  = rand();
  }
  CopyArray(a1,a2,N);
  CopyArray(b1,b2,N);
  CopyArray(c1,c2,N);
  CopyArray(x ,y ,N);
  /* solve */  
  printf("Serial test 1 ([I]x = b => x = b):        \t");
  ierr = tridiagLU(a1,b1,c1,x,N,0,1,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N);
  printf("error=%E\n",error);

  /* Test 2: [U]x = b => x = [U]^(-1)b */
  for (i = 0; i < N; i++) {
    a1[i] = 0.0;
    b1[i] = 0.5;
    c1[i] = (i == N-1 ? 0 : 0.5);
    x[i]  = 1.0;
  }
  CopyArray(a1,a2,N);
  CopyArray(b1,b2,N);
  CopyArray(c1,c2,N);
  CopyArray(x ,y ,N);
  /* solve */  
  printf("Serial test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr  = tridiagLU(a1,b1,c1,x,N,0,1,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N);
  printf("error=%E\n",error);

  /* Test 3: [A]x = b => x = [A]^(-1)b */
  for (i = 0; i < N; i++) {
    a1[i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
    b1[i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
    c1[i] = (i == N-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
    x[i]  = ((double) rand()) / ((double) RAND_MAX);
  }
  CopyArray(a1,a2,N);
  CopyArray(b1,b2,N);
  CopyArray(c1,c2,N);
  CopyArray(x ,y ,N);
  /* solve */  
  printf("Serial test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = tridiagLU(a1,b1,c1,x,N,0,1,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N);
  printf("error=%E\n",error);


  /* deallocate arrays */
  free(a1);
  free(b1);
  free(c1);
  free(a2);
  free(b2);
  free(c2);
  free(x);
  free(y);
  return(0);
}
#else
int main_mpi(int N,int NRuns,int rank,int nproc)
{
  double *a1,*b1,*c1,*x;
  double *a2,*b2,*c2,*y;
  int     i,ierr=0,nlocal;
  double  error,total_error;

  srand(time(NULL));

  if (!rank) printf("Testing MPI tridiagLU() with N=%d on %d processes\n",N,nproc);
  MPI_Barrier(MPI_COMM_WORLD);

  /* find local size */
  ierr = partition1D(N,nproc,rank,&nlocal);
  printf("Rank: %10d,\tLocal Size: %10d\n",rank,nlocal);

  /* allocate arrays */
  a1 = (double*) calloc (nlocal,sizeof(double));
  b1 = (double*) calloc (nlocal,sizeof(double));
  c1 = (double*) calloc (nlocal,sizeof(double));
  a2 = (double*) calloc (nlocal,sizeof(double));
  b2 = (double*) calloc (nlocal,sizeof(double));
  c2 = (double*) calloc (nlocal,sizeof(double));
  x  = (double*) calloc (nlocal,sizeof(double));
  y  = (double*) calloc (nlocal,sizeof(double));
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 1: [I]x = b => x = b */
  for (i = 0; i < nlocal; i++) {
    a1[i] = 0.0;
    b1[i] = 1.0;
    c1[i] = 0.0;
    x[i]  = rand();
  }
  CopyArray(a1,a2,nlocal);
  CopyArray(b1,b2,nlocal);
  CopyArray(c1,c2,nlocal);
  CopyArray(x ,y ,nlocal);
  /* solve */  
  if (!rank)  printf("MPI test 1 ([I]x = b => x = b):        \t");
  ierr = tridiagLU(a1,b1,c1,x,nlocal,rank,nproc,NULL);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank)  printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 2: [U]x = b => x = [U]^(-1)b */
  for (i = 0; i < nlocal; i++) {
    a1[i] = 0.0;
    b1[i] = 0.5;
    if (rank == nproc-1) c1[i] = (i == nlocal-1 ? 0 : 0.5);
    else                 c1[i] = 0.5;
    x[i]  = 1.0;
  }
  CopyArray(a1,a2,nlocal);
  CopyArray(b1,b2,nlocal);
  CopyArray(c1,c2,nlocal);
  CopyArray(x ,y ,nlocal);
  /* solve */  
  if (!rank) printf("MPI test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr  = tridiagLU(a1,b1,c1,x,nlocal,rank,nproc,NULL);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 3: [A]x = b => x = [A]^(-1)b */
  for (i = 0; i < nlocal; i++) {
    if (!rank )           a1[i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
    else                  a1[i] = ((double) rand()) / ((double) RAND_MAX);
    b1[i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
    if (rank == nproc-1)  c1[i] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
    else                  c1[i] = ((double) rand()) / ((double) RAND_MAX);
    x[i]  = ((double) rand()) / ((double) RAND_MAX);
  }
  CopyArray(a1,a2,nlocal);
  CopyArray(b1,b2,nlocal);
  CopyArray(c1,c2,nlocal);
  CopyArray(x ,y ,nlocal);
  /* solve */  
  if (!rank) printf("MPI test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = tridiagLU(a1,b1,c1,x,nlocal,rank,nproc,NULL);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 4: Same as test 3, but for computational cost test */
  for (i = 0; i < nlocal; i++) {
    if (!rank )           a1[i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
    else                  a1[i] = ((double) rand()) / ((double) RAND_MAX);
    b1[i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
    if (rank == nproc-1)  c1[i] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
    else                  c1[i] = ((double) rand()) / ((double) RAND_MAX);
    x[i]  = ((double) rand()) / ((double) RAND_MAX);
  }
  /* solve */  
  if (!rank) printf("\nMPI test 4 (Speed test - %d Tridiagonal Solves):\n",NRuns);
  double runtimes[5] = {0.0,0.0,0.0,0.0,0.0};
  for (i = 0; i < NRuns; i++) {
    TridiagLUTime timing;
    ierr = tridiagLU(a1,b1,c1,x,nlocal,rank,nproc,&timing);
    runtimes[0] += timing.total_time;
    runtimes[1] += timing.stage1_time;
    runtimes[2] += timing.stage2_time;
    runtimes[3] += timing.stage3_time;
    runtimes[4] += timing.stage4_time;
  }
  MPI_Allreduce(MPI_IN_PLACE,&runtimes[0],5,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  if (!rank) printf("\t\tTotal  walltime = %E\n",runtimes[0]);
  if (!rank) printf("\t\tStage1 walltime = %E\n",runtimes[1]);
  if (!rank) printf("\t\tStage2 walltime = %E\n",runtimes[2]);
  if (!rank) printf("\t\tStage3 walltime = %E\n",runtimes[3]);
  if (!rank) printf("\t\tStage4 walltime = %E\n",runtimes[4]);
  MPI_Barrier(MPI_COMM_WORLD);

  /* deallocate arrays */
  free(a1);
  free(b1);
  free(c1);
  free(a2);
  free(b2);
  free(c2);
  free(x);
  free(y);
  return(0);
}
#endif

void CopyArray(double *x,double *y,int N)
{
  int i;
  for (i = 0; i < N; i++) y[i] = x[i];
  return;
}

#ifdef serial
double CalculateError(double *a,double *b,double *c,double *y,double *x,int N)
{
  double error = 0;
  int i;
  error = 0;
  for (i = 0; i < N; i++) {
    double val;
    if (i == 0)         val = y[i] - (b[i]*x[i]+c[i]*x[i+1]);
    else if (i == N-1)  val = y[i] - (a[i]*x[i-1]+b[i]*x[i]);
    else                val = y[i] - (a[i]*x[i-1]+b[i]*x[i]+c[i]*x[i+1]);
    error += val * val;
  }
  return(error);
}
#else
double CalculateError(double *a,double *b,double *c,double *y,double *x,int N,int rank,int nproc)
{
  double        error = 0;
  int           i,nreq;
  double        xp1, xm1; /* solution from neighboring processes */
  MPI_Status    *status;
  MPI_Request   *request;

  xp1 = 0;
  if (nproc > 1) {
    nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
    request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
    status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
    if (rank)                   MPI_Isend(&x[0],1,MPI_DOUBLE,rank-1,1738,MPI_COMM_WORLD,&request[0]);
    if (!rank)                  MPI_Irecv(&xp1 ,1,MPI_DOUBLE,rank+1,1738,MPI_COMM_WORLD,&request[0]);
    else if (rank != nproc-1)   MPI_Irecv(&xp1 ,1,MPI_DOUBLE,rank+1,1738,MPI_COMM_WORLD,&request[1]);
    MPI_Waitall(nreq,&request[0],&status[0]);
    free(request);
    free(status);
  }
  
  xm1 = 0;
  if (nproc > 1) {
    nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
    request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
    status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
    if (rank != nproc-1)  MPI_Isend(&x[N-1],1,MPI_DOUBLE,rank+1,1739,MPI_COMM_WORLD,&request[0]);
    if (rank == nproc-1 ) MPI_Irecv(&xm1   ,1,MPI_DOUBLE,rank-1,1739,MPI_COMM_WORLD,&request[0]);
    else if (rank)        MPI_Irecv(&xm1   ,1,MPI_DOUBLE,rank-1,1739,MPI_COMM_WORLD,&request[1]);
    MPI_Waitall(nreq,&request[0],&status[0]);
    free(request);
    free(status);
  }

  error = 0;
  for (i = 0; i < N; i++) {
    double val;
    if (i == 0)         val = y[i] - (a[i]*xm1   +b[i]*x[i]+c[i]*x[i+1]);
    else if (i == N-1)  val = y[i] - (a[i]*x[i-1]+b[i]*x[i]+c[i]*xp1   );
    else                val = y[i] - (a[i]*x[i-1]+b[i]*x[i]+c[i]*x[i+1]);
    error += val * val;
  }
  return(error);
}
#endif

#ifndef serial
int partition1D(int nglobal,int nproc,int rank,int* nlocal)
{
  if (nglobal%nproc == 0) *nlocal = nglobal/nproc;
  else {
    if (rank == nproc-1)  *nlocal = nglobal/nproc+nglobal%nproc;
    else                  *nlocal = nglobal/nproc;
  }
  return(0);
}
#endif
