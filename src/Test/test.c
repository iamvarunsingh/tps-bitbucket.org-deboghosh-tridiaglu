#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>

static void   CopyArray       (double**,double**,int,int);
#ifdef serial
static int    main_serial     (int,int);
static double CalculateError  (double**,double**,double**,double**,double**,int,int);
#else
static int    main_mpi        (int,int,int,int,int);
static int    partition1D     (int,int,int,int*);
static double CalculateError  (double**,double**,double**,double**,double**,int,int,int,int);
#endif

int main(int argc, char *argv[])
{
  int ierr,N,Ns,NRuns;
#ifdef serial
  scanf ("%d %d %d",&N,&Ns,&NRuns);
  ierr = main_serial(N,Ns);
  if (ierr) fprintf(stderr,"main_mpi() returned with an error code of %d.\n",ierr);
#else
  int rank,nproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  if (!rank) scanf ("%d %d %d",&N,&Ns,&NRuns);
  MPI_Bcast(&N ,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&NRuns,1,MPI_INT,0,MPI_COMM_WORLD);
  ierr = main_mpi(N,Ns,NRuns,rank,nproc);
  if (ierr) fprintf(stderr,"main_mpi() returned with an error code of %d on rank %d.\n",ierr,rank);
  MPI_Finalize();
#endif
  return(0);
}

#ifdef serial
int main_serial(int N,int Ns)
{
  double **a1,**b1,**c1,**x;
  double **a2,**b2,**c2,**y;
  int     d,i,ierr=0;
  double  error;

  srand(time(NULL));

  /* allocate arrays */
  a1 = (double**) calloc (Ns,sizeof(double*));
  b1 = (double**) calloc (Ns,sizeof(double*));
  c1 = (double**) calloc (Ns,sizeof(double*));
  a2 = (double**) calloc (Ns,sizeof(double*));
  b2 = (double**) calloc (Ns,sizeof(double*));
  c2 = (double**) calloc (Ns,sizeof(double*));
  x  = (double**) calloc (Ns,sizeof(double*));
  y  = (double**) calloc (Ns,sizeof(double*));
  for (d = 0; d < Ns; d++) {
    a1[d] = (double*) calloc (N,sizeof(double));
    b1[d] = (double*) calloc (N,sizeof(double));
    c1[d] = (double*) calloc (N,sizeof(double));
    a2[d] = (double*) calloc (N,sizeof(double));
    b2[d] = (double*) calloc (N,sizeof(double));
    c2[d] = (double*) calloc (N,sizeof(double));
    x[d]  = (double*) calloc (N,sizeof(double));
    y[d]  = (double*) calloc (N,sizeof(double));
  }

  printf("Testing serial tridiagLU() with N=%d, Ns=%d\n",N,Ns);

  /* Test 1: [I]x = b => x = b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 1.0;
      c1[d][i] = 0.0;
      x [d][i] = rand();
    }
  }
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);
  /* solve */  
  printf("TridiagLU Serial test 1 ([I]x = b => x = b):        \t");
  ierr = tridiagLU(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* Test 2: [U]x = b => x = [U]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 0.5;
      c1[d][i] = (i == N-1 ? 0 : 0.5);
      x [d][i] = 1.0;
    }
  }
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);
  /* solve */  
  printf("TridiagLU Serial test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = tridiagLU(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* Test 3: [A]x = b => x = [A]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      b1[d][i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
      c1[d][i] = (i == N-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      x [d][i] = ((double) rand()) / ((double) RAND_MAX);
    }
  }
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);
  /* solve */  
  printf("TridiagLU Serial test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = tridiagLU(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  printf("Testing serial tridiagLURD() with N=%d, Ns=%d\n",N,Ns);

  /* Test 1: [I]x = b => x = b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 1.0;
      c1[d][i] = 0.0;
      x [d][i] = rand();
    }
  }
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);
  /* solve */  
  printf("TridiagLURD Serial test 1 ([I]x = b => x = b):        \t");
  ierr = tridiagLURD(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* Test 2: [U]x = b => x = [U]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 2.0;
      c1[d][i] = (i == N-1 ? 0 : 0.5);
      x [d][i] = 1.0;
    }
  }
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);
  /* solve */  
  printf("TridiagLURD Serial test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = tridiagLURD(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* Test 3: [A]x = b => x = [A]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      b1[d][i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
      c1[d][i] = (i == N-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      x [d][i] = ((double) rand()) / ((double) RAND_MAX);
    }
  }
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);
  /* solve */  
  printf("TridiagLURD Serial test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = tridiagLURD(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* deallocate arrays */
  for (d = 0; d < Ns; d++) {
    free(a1[d]);
    free(b1[d]);
    free(c1[d]);
    free(a2[d]);
    free(b2[d]);
    free(c2[d]);
    free(x[d]);
    free(y[d]);
  }
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
int main_mpi(int N,int Ns,int NRuns,int rank,int nproc)
{
  double    **a1,**b1,**c1,**x;
  double    **a2,**b2,**c2,**y;
  int       i,d,ierr=0,nlocal;
  double    error,total_error;
  MPI_Comm  world;
  
  MPI_Comm_dup(MPI_COMM_WORLD,&world);
  srand(time(NULL));

  /* find local size */
  ierr = partition1D(N,nproc,rank,&nlocal);
  printf("Rank: %10d,\tLocal Size: %10d\n",rank,nlocal);
  MPI_Barrier(MPI_COMM_WORLD);

  /* allocate arrays */
  a1 = (double**) calloc (Ns,sizeof(double*));
  b1 = (double**) calloc (Ns,sizeof(double*));
  c1 = (double**) calloc (Ns,sizeof(double*));
  a2 = (double**) calloc (Ns,sizeof(double*));
  b2 = (double**) calloc (Ns,sizeof(double*));
  c2 = (double**) calloc (Ns,sizeof(double*));
  x  = (double**) calloc (Ns,sizeof(double*));
  y  = (double**) calloc (Ns,sizeof(double*));
  for (d = 0; d < Ns; d++) {
    a1[d] = (double*) calloc (nlocal,sizeof(double));
    b1[d] = (double*) calloc (nlocal,sizeof(double));
    c1[d] = (double*) calloc (nlocal,sizeof(double));
    a2[d] = (double*) calloc (nlocal,sizeof(double));
    b2[d] = (double*) calloc (nlocal,sizeof(double));
    c2[d] = (double*) calloc (nlocal,sizeof(double));
    x [d] = (double*) calloc (nlocal,sizeof(double));
    y [d] = (double*) calloc (nlocal,sizeof(double));
  }

  if (!rank) printf("Testing MPI tridiagLU() with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 1: [I]x = b => x = b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 1.0;
      c1[d][i] = 0.0;
      x [d][i] = rand();
    }
  }
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  /* solve */  
  if (!rank)  printf("MPI test 1 ([I]x = b => x = b):        \t");
  ierr = tridiagLU(a1,b1,c1,x,nlocal,Ns,NULL,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank)  printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 2: [U]x = b => x = [U]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 0.5;
      if (rank == nproc-1) c1[d][i] = (i == nlocal-1 ? 0 : 0.5);
      else                 c1[d][i] = 0.5;
      x[d][i]  = 1.0;
    }
  }
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  /* solve */  
  if (!rank) printf("MPI test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = tridiagLU(a1,b1,c1,x,nlocal,Ns,NULL,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 3: [A]x = b => x = [A]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      if (!rank )           a1[d][i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      else                  a1[d][i] = ((double) rand()) / ((double) RAND_MAX);
      b1[d][i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
      if (rank == nproc-1)  c1[d][i] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      else                  c1[d][i] = ((double) rand()) / ((double) RAND_MAX);
      x[d][i]  = ((double) rand()) / ((double) RAND_MAX);
    }
  }
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  /* solve */  
  if (!rank) printf("MPI test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = tridiagLU(a1,b1,c1,x,nlocal,Ns,NULL,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) printf("Testing MPI tridiagLURD() with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 1: [I]x = b => x = b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 1.0;
      c1[d][i] = 0.0;
      x [d][i] = rand();
    }
  }
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  /* solve */  
  if (!rank)  printf("MPI test 1 ([I]x = b => x = b):        \t");
  ierr = tridiagLURD(a1,b1,c1,x,nlocal,Ns,NULL,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank)  printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 2: [U]x = b => x = [U]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 0.5;
      if (rank == nproc-1) c1[d][i] = (i == nlocal-1 ? 0 : 0.5);
      else                 c1[d][i] = 0.5;
      x[d][i]  = 1.0;
    }
  }
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  /* solve */  
  if (!rank) printf("MPI test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = tridiagLURD(a1,b1,c1,x,nlocal,Ns,NULL,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Test 3: [A]x = b => x = [A]^(-1)b */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      if (!rank )           a1[d][i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      else                  a1[d][i] = ((double) rand()) / ((double) RAND_MAX);
      b1[d][i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
      if (rank == nproc-1)  c1[d][i] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      else                  c1[d][i] = ((double) rand()) / ((double) RAND_MAX);
      x[d][i]  = ((double) rand()) / ((double) RAND_MAX);
    }
  }
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  /* solve */  
  if (!rank) printf("MPI test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = tridiagLURD(a1,b1,c1,x,nlocal,Ns,NULL,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /**** Scalability Test for tridiagLU() ****/

  /* Test 4: Same as test 3, but for computational cost test */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      if (!rank )           a1[d][i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      else                  a1[d][i] = ((double) rand()) / ((double) RAND_MAX);
      b1[d][i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
      if (rank == nproc-1)  c1[d][i] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      else                  c1[d][i] = ((double) rand()) / ((double) RAND_MAX);
      x[d][i]  = ((double) rand()) / ((double) RAND_MAX);
    }
  }
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  /* solve */  
  if (!rank) printf("\nMPI test 4 (Speed test - %d Tridiagonal Solves):\n",NRuns);
  double runtimes[5] = {0.0,0.0,0.0,0.0,0.0};
  error = 0;
  for (i = 0; i < NRuns; i++) {
    TridiagLUTime timing;
    CopyArray(a2,a1,nlocal,Ns);
    CopyArray(b2,b1,nlocal,Ns);
    CopyArray(c2,c1,nlocal,Ns);
    CopyArray(y ,x ,nlocal,Ns);
    ierr         = tridiagLU(a1,b1,c1,x,nlocal,Ns,&timing,&world);
    double err   = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
    runtimes[0] += timing.total_time;
    runtimes[1] += timing.stage1_time;
    runtimes[2] += timing.stage2_time;
    runtimes[3] += timing.stage3_time;
    runtimes[4] += timing.stage4_time;
    if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
    error += err;
  }
  error /= NRuns;
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  MPI_Allreduce(MPI_IN_PLACE,&runtimes[0],5,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
  if (!rank) {
    printf("\t\tTotal  walltime = %E\n",runtimes[0]);
    printf("\t\tStage1 walltime = %E\n",runtimes[1]);
    printf("\t\tStage2 walltime = %E\n",runtimes[2]);
    printf("\t\tStage3 walltime = %E\n",runtimes[3]);
    printf("\t\tStage4 walltime = %E\n",runtimes[4]);
    printf("\t\tAverage error   = %E\n",total_error);
    FILE *out;
    out = fopen("walltimes.dat","w");
    fprintf(out,"%5d  %E  %E  %E  %E  %E\n",nproc,runtimes[0],
            runtimes[1],runtimes[2],runtimes[3],runtimes[4]);
    fclose(out);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* deallocate arrays */
  for (d = 0; d < Ns; d++) {
    free(a1[d]);
    free(b1[d]);
    free(c1[d]);
    free(a2[d]);
    free(b2[d]);
    free(c2[d]);
    free(x [d]);
    free(y [d]);
  }
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

void CopyArray(double **x,double **y,int N,int Ns)
{
  int i,d;
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      y[d][i] = x[d][i];
    }
  }
  return;
}

#ifdef serial
double CalculateError(double **a,double **b,double **c,double **y,double **x,
                      int N,int Ns)
{
  int i,d;
  double error = 0;
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      double val;
      if (i == 0)         val = y[d][i] - (b[d][i]*x[d][i]+c[d][i]*x[d][i+1]);
      else if (i == N-1)  val = y[d][i] - (a[d][i]*x[d][i-1]+b[d][i]*x[d][i]);
      else                val = y[d][i] - (a[d][i]*x[d][i-1]+b[d][i]*x[d][i]+c[d][i]*x[d][i+1]);
      error += val * val;
    }
  }
  return(error);
}
#else
double CalculateError(double **a,double **b,double **c,double **y,double **x,
                      int N,int Ns,int rank,int nproc)
{
  double        error = 0;
  int           i,d,nreq;
  double        xp1, xm1; /* solution from neighboring processes */
  MPI_Status    *status;
  MPI_Request   *request;

  for (d = 0; d < Ns; d++) {
    xp1 = 0;
    if (nproc > 1) {
      nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
      request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
      status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
      if (rank)                   MPI_Isend(&x[d][0],1,MPI_DOUBLE,rank-1,1738,MPI_COMM_WORLD,&request[0]);
      if (!rank)                  MPI_Irecv(&xp1    ,1,MPI_DOUBLE,rank+1,1738,MPI_COMM_WORLD,&request[0]);
      else if (rank != nproc-1)   MPI_Irecv(&xp1    ,1,MPI_DOUBLE,rank+1,1738,MPI_COMM_WORLD,&request[1]);
      MPI_Waitall(nreq,&request[0],&status[0]);
      free(request);
      free(status);
    }
  
    xm1 = 0;
    if (nproc > 1) {
      nreq = ((rank == 0 || rank == nproc-1) ? 1 : 2);
      request  = (MPI_Request*) calloc (nreq,sizeof(MPI_Request));
      status   = (MPI_Status*)  calloc (nreq,sizeof(MPI_Status));
      if (rank != nproc-1)  MPI_Isend(&x[d][N-1],1,MPI_DOUBLE,rank+1,1739,MPI_COMM_WORLD,&request[0]);
      if (rank == nproc-1 ) MPI_Irecv(&xm1      ,1,MPI_DOUBLE,rank-1,1739,MPI_COMM_WORLD,&request[0]);
      else if (rank)        MPI_Irecv(&xm1      ,1,MPI_DOUBLE,rank-1,1739,MPI_COMM_WORLD,&request[1]);
      MPI_Waitall(nreq,&request[0],&status[0]);
      free(request);
      free(status);
    }

    error = 0;
    for (i = 0; i < N; i++) {
      double val = 0;
      if (i == 0)    val += a[d][i]*xm1;
      else           val += a[d][i]*x[d][i-1];
      val += b[d][i]*x[d][i];
      if (i == N-1)  val += c[d][i]*xp1;
      else           val += c[d][i]*x[d][i+1];
      val = y[d][i] - val;
      error += val * val;
    }
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
