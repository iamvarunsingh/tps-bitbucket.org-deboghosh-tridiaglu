#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>

/* Function declarations */
static void   CopyArray       (double**,double**,int,int);
#ifdef serial
static int    main_serial     (int,int);
static int    test_serial     (int,int,int(*)(double**,double**,double**,double**,int,int,void*,void*));
static double CalculateError  (double**,double**,double**,double**,double**,int,int);
#else
static int    main_mpi        (int,int,int,int,int);
static int    test_mpi        (int,int,int,int,int,int,int(*)(double**,double**,double**,double**,int,int,void*,void*));
static int    partition1D     (int,int,int,int*);
static double CalculateError  (double**,double**,double**,double**,double**,int,int,int,int);
#endif

int main(int argc, char *argv[])
{
  int ierr,N,Ns,NRuns;

#ifdef serial

  /* If compiled in serial, run the serial tests                   */
  /* read in size of system, number of system and number of solves */
  FILE *in; in=fopen("input","r");
  fscanf (in,"%d %d %d",&N,&Ns,&NRuns);
  fclose(in);
  /* call the test function */
  ierr = main_serial(N,Ns);
  if (ierr) fprintf(stderr,"main_mpi() returned with an error code of %d.\n",ierr);

#else

  /* If compiled in parallel, run the MPI tests                    */
  int rank,nproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  
  /* read in size of system, number of system and number of solves */
  
  if (!rank) {
    FILE *in; in=fopen("input","r");
    fscanf (in,"%d %d %d",&N,&Ns,&NRuns);
    fclose(in);
  }
  /* Broadcast the input values to all the processes */
  MPI_Bcast(&N ,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&NRuns,1,MPI_INT,0,MPI_COMM_WORLD);

  /* call the test function */
  ierr = main_mpi(N,Ns,NRuns,rank,nproc);
  if (ierr) fprintf(stderr,"main_mpi() returned with an error code of %d on rank %d.\n",ierr,rank);

  MPI_Finalize();
#endif
  return(0);
}

#ifdef serial

/* 
  THIS FUNCTION CALLS THE TEST FUNCTION FOR THE DIFFERENT 
  TRIDIAGONAL SOLVERS
*/
int main_serial(int N,int Ns)
{
  int ierr = 0;

  printf("Testing serial tridiagLURD() with N=%d, Ns=%d\n",N,Ns);
  ierr = test_serial(N,Ns,&tridiagLURD); if(ierr) return(ierr);

  printf("Testing serial tridiagLU() with N=%d, Ns=%d\n",N,Ns);
  ierr = test_serial(N,Ns,&tridiagLU); if(ierr) return(ierr);

  /* Return */
  return(0);
}

/* 
    THIS FUNCTION TESTS THE SERIAL IMPLEMENTATION OF A
    TRIDIAGONAL SOLVER
*/
int test_serial(int N,int Ns,int (*LUSolver)(double**,double**,double**,double**,int,int,void*,void*))
{
  /* Variable declarations */
  double **a1;    /* sub-diagonal                               */
  double **b1;    /* diagonal                                   */
  double **c1;    /* super-diagonal                             */
  double **x;     /* right hand side, will contain the solution */ 

  /* 
    Since a,b,c and x are not preserved, declaring variables to
    store a copy of them to calculate error after the solve
  */
  double **a2,**b2,**c2,**y;

  /* Other variables */
  int     d,i,ierr=0;
  double  error;

  /* Initialize random number generator */
  srand(time(NULL));

  /* 
    Allocate arrays of dimension (Ns x N) 
    Ns -> number of systems
    N  -> size of each system
  */
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

  /* 
    TEST 1: Solution of an identity matrix with random
            right hand side
            [I]x = b => x = b 
  */

  /* 
    Set the values of the matrix elements and the 
    right hand side
  */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 1.0;
      c1[d][i] = 0.0;
      x [d][i] = rand();
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);
  
  /* Solve */  
  printf("TridiagLU Serial test 1 ([I]x = b => x = b):        \t");
  ierr = LUSolver(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* 
    TEST 2: Solution of an upper triangular matrix 
            [U]x = b  
  */

  /* 
    Set the values of the matrix elements and the 
    right hand side
  */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 0.5;
      c1[d][i] = (i == N-1 ? 0 : 0.5);
      x [d][i] = 1.0;
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);

  /* Solve */  
  printf("TridiagLU Serial test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* 
    TEST 3: Solution of a tridiagonal matrix with random
            entries and right hand side
            [A]x = b => x = b 
  */

  /* 
    Set the values of the matrix elements and the 
    right hand side
  */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      a1[d][i] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      b1[d][i] = 1.0 + ((double) rand()) / ((double) RAND_MAX);
      c1[d][i] = (i == N-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      x [d][i] = ((double) rand()) / ((double) RAND_MAX);
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArray(a1,a2,N,Ns);
  CopyArray(b1,b2,N,Ns);
  CopyArray(c1,c2,N,Ns);
  CopyArray(x ,y ,N,Ns);

  /* Solve */  
  printf("TridiagLU Serial test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,N,Ns,NULL,NULL);
  if (ierr == -1) printf("Error - system is singular\t");

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* 
    DEALLOCATE ALL ARRAYS
  */
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

  /* Return */
  return(0);
}

#else

/*
  THIS FUNCTION CALLS THE TEST FUNCTION FOR THE DIFFERENT 
  TRIDIAGONAL SOLVERS
*/
int main_mpi(int N,int Ns,int NRuns,int rank,int nproc)
{
  int ierr = 0;

  if (!rank) printf("Testing MPI tridiagLURD() with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  ierr = test_mpi(N,Ns,NRuns,rank,nproc,0,&tridiagLURD); if (ierr) return(ierr);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) printf("Testing MPI tridiagLU() with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  ierr = test_mpi(N,Ns,NRuns,rank,nproc,1,&tridiagLU); if (ierr) return(ierr);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Return */
  return(0);
}

/* 
    THIS FUNCTION TESTS THE PARALLEL IMPLEMENTATION OF A 
    TRIDIAGONAL SOLVER
*/
int test_mpi(int N,int Ns,int NRuns,int rank,int nproc, int flag,
             int(*LUSolver)(double**,double**,double**,double**,int,int,void*,void*))
{
  /* Variable declarations */
  double **a1;    /* sub-diagonal                               */
  double **b1;    /* diagonal                                   */
  double **c1;    /* super-diagonal                             */
  double **x;     /* right hand side, will contain the solution */ 

  /* 
    Since a,b,c and x are not preserved, declaring variables to
    store a copy of them to calculate error after the solve
  */
  double **a2,**b2,**c2,**y;

  /* Other variables */
  int       i,d,ierr=0,nlocal;
  double    error,total_error;
  MPI_Comm  world;
 
  /* Creating a duplicate communicator */
  MPI_Comm_dup(MPI_COMM_WORLD,&world);

  /* Initialize random number generator */
  srand(time(NULL));

  /* 
    Calculate local size on this process, given
    the total size N and number of processes 
    nproc
  */
  ierr = partition1D(N,nproc,rank,&nlocal);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Create MPI Context for the tridiagonal solver functions */
  MPIContext mpi;         /* Context              */
  mpi.rank = rank;        /* Rank of the process  */
  mpi.nproc = nproc;      /* Number of processes  */
  mpi.comm  = &world;     /* Communicator         */
  /* 
    Create the array containing the mapping of the actual rank
    with the rank that the tridiagonal solver wants
  */
  mpi.proc = (int*) calloc (nproc,sizeof(int));
  for (d=0; d<nproc; d++) mpi.proc[d] = d;

  /* 
    Allocate arrays of dimension (Ns x nlocal) 
    Ns      -> number of systems
    nlocal  -> local size of each system
  */
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

  /*
      TESTING THE LU SOLVER
  */
  MPI_Barrier(MPI_COMM_WORLD);

  /* 
    TEST 1: Solution of an identity matrix with random
            right hand side
            [I]x = b => x = b 
  */

  /* 
    Set the values of the matrix elements and the 
    right hand side
  */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 1.0;
      c1[d][i] = 0.0;
      x [d][i] = rand();
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);
  
  /* Solve */  
  if (!rank)  printf("MPI test 1 ([I]x = b => x = b):        \t");
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,NULL,&mpi);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank)  printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* 
    TEST 2: Solution of an upper triangular matrix 
            [U]x = b  
  */

  /* 
    Set the values of the matrix elements and the 
    right hand side
  */
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < nlocal; i++) {
      a1[d][i] = 0.0;
      b1[d][i] = 0.5;
      if (rank == nproc-1) c1[d][i] = (i == nlocal-1 ? 0 : 0.5);
      else                 c1[d][i] = 0.5;
      x[d][i]  = 1.0;
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);

  /* Solve */  
  if (!rank) printf("MPI test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,NULL,&mpi);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /* 
    TEST 3: Solution of a tridiagonal matrix with random
            entries and right hand side
            [A]x = b => x = b 
  */

  /* 
    Set the values of the matrix elements and the 
    right hand side
  */
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

  /*
    Copy the original values to calculate error later
  */
  CopyArray(a1,a2,nlocal,Ns);
  CopyArray(b1,b2,nlocal,Ns);
  CopyArray(c1,c2,nlocal,Ns);
  CopyArray(x ,y ,nlocal,Ns);

  /* Solve */  
  if (!rank) printf("MPI test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,NULL,&mpi);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            total_error = error;
  if (!rank) printf("error=%E\n",total_error);
  MPI_Barrier(MPI_COMM_WORLD);

  /*
      DONE TESTING THE LU SOLVER
  */


  /*
      TESTING WALLTIMES FOR NRuns NUMBER OF RUNS OF TRIDIAGLU()
      FOR SCALABILITY CHECK IF REQUIRED
  */

  if (flag) {

    /* 
      TEST 4: Same as TEST 3
      Set the values of the matrix elements and the 
      right hand side
    */
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

    /*
      Keep a copy of the original values 
    */
    CopyArray(a1,a2,nlocal,Ns);
    CopyArray(b1,b2,nlocal,Ns);
    CopyArray(c1,c2,nlocal,Ns);
    CopyArray(x ,y ,nlocal,Ns);

    if (!rank) 
      printf("\nMPI test 4 (Speed test - %d Tridiagonal Solves):\n",NRuns);
    double runtimes[5] = {0.0,0.0,0.0,0.0,0.0};
    error = 0;
    /* 
      Solve the systen NRuns times
    */  
    for (i = 0; i < NRuns; i++) {
      TridiagLUTime timing;
      /* Copy the original values */
      CopyArray(a2,a1,nlocal,Ns);
      CopyArray(b2,b1,nlocal,Ns);
      CopyArray(c2,c1,nlocal,Ns);
      CopyArray(y ,x ,nlocal,Ns);
      /* Solve the system */
      ierr         = tridiagLU(a1,b1,c1,x,nlocal,Ns,&timing,&mpi);
      /* Calculate errors */
      double err   = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
      /* Add the walltimes to the cumulative total */
      runtimes[0] += timing.total_time;
      runtimes[1] += timing.stage1_time;
      runtimes[2] += timing.stage2_time;
      runtimes[3] += timing.stage3_time;
      runtimes[4] += timing.stage4_time;
      if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
      error += err;
    }
  
    /* Calculate average error */
    error /= NRuns;
    if (nproc > 1)  MPI_Allreduce(&error,&total_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    else            total_error = error;

    /* Calculate maximum value of walltime across all processes */
    MPI_Allreduce(MPI_IN_PLACE,&runtimes[0],5,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

    /* Print results */
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
  }

  /*
      DONE TESTING TRIDIAGLU() WALLTIMES
  */

  /* 
    DEALLOCATE ALL ARRAYS
  */
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
  free(mpi.proc);

  /* Return */
  return(0);
}


#endif


/*
  Function to copy the values of one 2D array into another
*/
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

/* 
  Functions to calculate the error in the computed solution 
*/
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

/*
  Function to calculate the local number of points for each
  process, given process rank, global number of points and 
  number of processes
*/
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
