#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef serial
#include <mpi.h>
#endif
#include <tridiagLU.h>
#include <matops.h>

/* Maximum block size to test */
int MAX_BS = 0;

/* Function declarations */
static void   CopyArray       (double*,double*,int,int);
static void   CopyArraySimple (double*,double*,int);
#ifdef serial
static int    main_serial         (int,int);
static int    test_serial         (int,int,int(*)(double*,double*,double*,double*,int,int,void*,void*));
static int    test_block_serial   (int,int,int,int(*)(double*,double*,double*,double*,int,int,int,void*,void*));
static double CalculateError      (double*,double*,double*,double*,double*,int,int);
static double CalculateErrorBlock (double*,double*,double*,double*,double*,int,int,int);
#else
static int    main_mpi            (int,int,int,int,int,int);
static int    test_mpi            (int,int,int,int,int,int,int,int(*)(double*,double*,double*,double*,int,int,void*,void*),const char*);
static int    test_block_mpi      (int,int,int,int,int,int,int,int(*)(double*,double*,double*,double*,int,int,int,void*,void*),const char*);
static int    partition1D         (int,int,int,int*);
static double CalculateError      (double*,double*,double*,double*,double*,int,int,int,int);
static double CalculateErrorBlock (double*,double*,double*,double*,double*,int,int,int,int,int);
#endif

#ifdef with_scalapack
extern void Cblacs_pinfo();
extern void Cblacs_get();
extern void Cblacs_gridinit();
extern void Cblacs_gridexit();
extern void Cblacs_exit();
#endif

int main(int argc, char *argv[])
{
  int ierr,N,Ns,NRuns;

#ifdef serial

  /* If compiled in serial, run the serial tests                   */
  /* read in size of system, number of system and number of solves */
  FILE *in; in=fopen("input","r");
  ierr = fscanf (in,"%d %d %d",&N,&Ns,&NRuns);
  if (ierr != 3) {
    fprintf(stderr,"Invalid input file.\n");
    return(0);
  }
  fclose(in);
  /* call the test function */
  ierr = main_serial(N,Ns);
  if (ierr) fprintf(stderr,"main_serial() returned with an error code of %d.\n",ierr);

#else

  /* If compiled in parallel, run the MPI tests                    */
  int rank,nproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

#ifdef with_scalapack
  /* initialize BLACS */
  int rank_blacs,nproc_blacs,blacs_context;
  Cblacs_pinfo(&rank_blacs,&nproc_blacs);
  Cblacs_get(-1,0,&blacs_context);
  Cblacs_gridinit(&blacs_context,"R",1,nproc_blacs);
#else
  int blacs_context = -1;
#endif
  
  /* read in size of system, number of system and number of solves */
  
  if (!rank) {
    FILE *in; in=fopen("input","r");
    ierr = fscanf (in,"%d %d %d",&N,&Ns,&NRuns); 
    if (ierr != 3) {
      fprintf(stderr,"Invalid input file.\n");
      return(0);
    }
    fclose(in);
  }
  /* Broadcast the input values to all the processes */
  MPI_Bcast(&N ,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&NRuns,1,MPI_INT,0,MPI_COMM_WORLD);

  /* call the test function */
  ierr = main_mpi(N,Ns,NRuns,rank,nproc,blacs_context);
  if (ierr) fprintf(stderr,"main_mpi() returned with an error code of %d on rank %d.\n",ierr,rank);

#ifdef with_scalapack
  Cblacs_gridexit(blacs_context);
  Cblacs_exit(0);
#else
  MPI_Finalize();
#endif
#endif
  return(0);
}

#ifdef serial

/* 
  THIS FUNCTION CALLS THE TEST FUNCTION FOR THE DIFFERENT 
  TRIDIAGONAL SOLVERS (Serial)
*/
int main_serial(int N,int Ns)
{
  int ierr = 0;

  printf("Testing serial tridiagLUGS() with N=%d, Ns=%d\n",N,Ns);
  ierr = test_serial(N,Ns,&tridiagLUGS); if(ierr) return(ierr);

  printf("Testing serial tridiagIterJacobi() with N=%d, Ns=%d\n",N,Ns);
  ierr = test_serial(N,Ns,&tridiagIterJacobi); if(ierr) return(ierr);

  printf("Testing serial tridiagLU() with N=%d, Ns=%d\n",N,Ns);
  ierr = test_serial(N,Ns,&tridiagLU); if(ierr) return(ierr);

  int bs;
  for (bs=1; bs <= MAX_BS; bs++) {
    printf("Testing serial blocktridiagIterJacobi() with N=%d, Ns=%d, bs=%d\n",N,Ns,bs);
    ierr = test_block_serial(N,Ns,bs,&blocktridiagIterJacobi); if(ierr) return(ierr);
  }
  for (bs=1; bs <= MAX_BS; bs++) {
    printf("Testing serial blocktridiagLU() with N=%d, Ns=%d, bs=%d\n",N,Ns,bs);
    ierr = test_block_serial(N,Ns,bs,&blocktridiagLU); if(ierr) return(ierr);
  }

  /* Return */
  return(0);
}

/* 
    THIS FUNCTION TESTS THE SERIAL IMPLEMENTATION OF A
    TRIDIAGONAL SOLVER
*/
int test_serial(int N,int Ns,int (*LUSolver)(double*,double*,double*,double*,int,int,void*,void*))
{
  int     d,i,ierr=0;
  double  error;
  TridiagLU context;

  /* Initialize tridiagonal solver parameters */
  ierr = tridiagLUInit(&context,NULL); if (ierr) return(ierr);

  /* Variable declarations */
  double *a1;    /* sub-diagonal                               */
  double *b1;    /* diagonal                                   */
  double *c1;    /* super-diagonal                             */
  double *x;     /* right hand side, will contain the solution */ 

  /* 
    Since a,b,c and x are not preserved, declaring variables to
    store a copy of them to calculate error after the solve
  */
  double *a2,*b2,*c2,*y;

  /* Initialize random number generator */
  srand(time(NULL));

  /* 
    Allocate arrays of dimension (Ns x N) 
    Ns -> number of systems
    N  -> size of each system
  */
  a1 = (double*) calloc (N*Ns,sizeof(double));
  b1 = (double*) calloc (N*Ns,sizeof(double));
  c1 = (double*) calloc (N*Ns,sizeof(double));
  a2 = (double*) calloc (N*Ns,sizeof(double));
  b2 = (double*) calloc (N*Ns,sizeof(double));
  c2 = (double*) calloc (N*Ns,sizeof(double));
  x  = (double*) calloc (N*Ns,sizeof(double));
  y  = (double*) calloc (N*Ns,sizeof(double));

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
      a1[i*Ns+d] = 0.0;
      b1[i*Ns+d] = 1.0;
      c1[i*Ns+d] = 0.0;
      x [i*Ns+d] = ((double) rand()) / ((double) RAND_MAX);
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
  ierr = LUSolver(a1,b1,c1,x,N,Ns,&context,NULL);
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
      a1[i*Ns+d] = 0.0;
      b1[i*Ns+d] = 100.0;
      c1[i*Ns+d] = (i == N-1 ? 0 : 0.5);
      x [i*Ns+d] = 1.0;
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
  ierr = LUSolver(a1,b1,c1,x,N,Ns,&context,NULL);
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
      a1[i*Ns+d] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      b1[i*Ns+d] = 100.0*(1.0+((double) rand()) / ((double) RAND_MAX));
      c1[i*Ns+d] = (i == N-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      x [i*Ns+d] = ((double) rand()) / ((double) RAND_MAX);
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
  ierr = LUSolver(a1,b1,c1,x,N,Ns,&context,NULL);
  if (ierr == -1) printf("Error - system is singular\t");

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,N,Ns);
  printf("error=%E\n",error);

  /* 
    DEALLOCATE ALL ARRAYS
  */
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

/* 
    THIS FUNCTION TESTS THE SERIAL IMPLEMENTATION OF A
    BLOCK TRIDIAGONAL SOLVER
*/
int test_block_serial(int N,int Ns,int bs,int (*LUSolver)(double*,double*,double*,double*,int,int,int,void*,void*))
{
  int     d,i,j,k,ierr=0;
  double  error;
  TridiagLU context;

  /* Initialize tridiagonal solver parameters */
  ierr = tridiagLUInit(&context,NULL); if (ierr) return(ierr);

  /* Variable declarations */
  double *a1;    /* sub-diagonal                               */
  double *b1;    /* diagonal                                   */
  double *c1;    /* super-diagonal                             */
  double *x;     /* right hand side, will contain the solution */ 

  /* 
    Since a,b,c and x are not preserved, declaring variables to
    store a copy of them to calculate error after the solve
  */
  double *a2,*b2,*c2,*y;

  /* Initialize random number generator */
  srand(time(NULL));

  /* 
    Allocate arrays of dimension (Ns x N) 
    Ns -> number of systems
    N  -> size of each system
  */
  a1 = (double*) calloc (N*Ns*bs*bs,sizeof(double));
  b1 = (double*) calloc (N*Ns*bs*bs,sizeof(double));
  c1 = (double*) calloc (N*Ns*bs*bs,sizeof(double));
  a2 = (double*) calloc (N*Ns*bs*bs,sizeof(double));
  b2 = (double*) calloc (N*Ns*bs*bs,sizeof(double));
  c2 = (double*) calloc (N*Ns*bs*bs,sizeof(double));
  x  = (double*) calloc (N*Ns*bs   ,sizeof(double));
  y  = (double*) calloc (N*Ns*bs   ,sizeof(double));

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
      for (j=0; j<bs; j++) {
        for (k=0; k<bs; k++) {
          a1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
          if (j==k) b1[(i*Ns+d)*bs*bs+j*bs+k] = 1.0;
          else      b1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
          c1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
        }
        x [(i*Ns+d)*bs+j] = ((double) rand()) / ((double) RAND_MAX);
      }
    }
  }


  /*
    Copy the original values to calculate error later
  */
  CopyArraySimple(a2,a1,N*Ns*bs*bs);
  CopyArraySimple(b2,b1,N*Ns*bs*bs);
  CopyArraySimple(c2,c1,N*Ns*bs*bs);
  CopyArraySimple(y ,x ,N*Ns*bs   );
  
  /* Solve */  
  printf("Block TridiagLU Serial test 1 ([I]x = b => x = b):        \t");
  ierr = LUSolver(a1,b1,c1,x,N,Ns,bs,&context,NULL);
  if (ierr == -1) printf("Error - system is singular\t");

  /*
    Calculate Error
  */
  error = CalculateErrorBlock(a2,b2,c2,y,x,N,Ns,bs);
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
      for (j=0; j<bs; j++) {
        for (k=0; k<bs; k++) {
          a1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
          b1[(i*Ns+d)*bs*bs+j*bs+k] = 100.0 + ((double) rand()) / ((double) RAND_MAX);
          c1[(i*Ns+d)*bs*bs+j*bs+k] = (i == N-1 ? 0 : 0.5*(((double) rand()) / ((double) RAND_MAX)));
        }
        x [(i*Ns+d)*bs+j] = ((double) rand()) / ((double) RAND_MAX);
      }
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArraySimple(a2,a1,N*Ns*bs*bs);
  CopyArraySimple(b2,b1,N*Ns*bs*bs);
  CopyArraySimple(c2,c1,N*Ns*bs*bs);
  CopyArraySimple(y ,x ,N*Ns*bs   );

  /* Solve */  
  printf("Block TridiagLU Serial test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,N,Ns,bs,&context,NULL);
  if (ierr == -1) printf("Error - system is singular\t");

  /*
    Calculate Error
  */
  error = CalculateErrorBlock(a2,b2,c2,y,x,N,Ns,bs);
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
      for (j=0; j<bs; j++) {
        for (k=0; k<bs; k++) {
          a1[(i*Ns+d)*bs*bs+j*bs+k] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
          b1[(i*Ns+d)*bs*bs+j*bs+k] = 100.0*(1.0+((double) rand()) / ((double) RAND_MAX));
          c1[(i*Ns+d)*bs*bs+j*bs+k] = (i == N-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
        }
        x [(i*Ns+d)*bs+j] = ((double) rand()) / ((double) RAND_MAX);
      }
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArraySimple(a2,a1,N*Ns*bs*bs);
  CopyArraySimple(b2,b1,N*Ns*bs*bs);
  CopyArraySimple(c2,c1,N*Ns*bs*bs);
  CopyArraySimple(y ,x ,N*Ns*bs   );

  /* Solve */  
  printf("Block TridiagLU Serial test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,N,Ns,bs,&context,NULL);
  if (ierr == -1) printf("Error - system is singular\t");

  /*
    Calculate Error
  */
  error = CalculateErrorBlock(a2,b2,c2,y,x,N,Ns,bs);
  printf("error=%E\n",error);

  /* 
    DEALLOCATE ALL ARRAYS
  */
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
int main_mpi(int N,int Ns,int NRuns,int rank,int nproc,int blacs_context)
{
  int ierr = 0;

  if (!rank) printf("\nTesting MPI tridiagLUGS()       with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  ierr = test_mpi(N,Ns,NRuns,rank,nproc,0,blacs_context,&tridiagLUGS,"walltimes_tridiagLUGS.dat"); if (ierr) return(ierr);
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (!rank) printf("\nTesting MPI tridiagIterJacobi() with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  ierr = test_mpi(N,Ns,NRuns,rank,nproc,0,blacs_context,&tridiagIterJacobi,"walltimes_tridiagIterJac.dat"); if (ierr) return(ierr);
  MPI_Barrier(MPI_COMM_WORLD);

  if (!rank) printf("\nTesting MPI tridiagLU()         with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  ierr = test_mpi(N,Ns,NRuns,rank,nproc,1,blacs_context,&tridiagLU,"walltimes_tridiagLU.dat"); if (ierr) return(ierr);
  MPI_Barrier(MPI_COMM_WORLD);

#ifdef with_scalapack
  if (!rank) printf("\nTesting MPI tridiagScaLPK()     with N=%d, Ns=%d on %d processes\n",N,Ns,nproc);
  ierr = test_mpi(N,Ns,NRuns,rank,nproc,1,blacs_context,&tridiagScaLPK,"walltimes_tridiagScaLPK.dat"); if (ierr) return(ierr);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (!rank) printf("-----------------------------------------------------------------\n");
  MPI_Barrier(MPI_COMM_WORLD);

  int bs;
  for (bs=1; bs <= MAX_BS; bs++) {
    if (!rank) printf("\nTesting MPI blocktridiagIterJacobi() with N=%d, Ns=%d, bs=%d on %d processes\n",N,Ns,bs,nproc);
    ierr = test_block_mpi(N,Ns,bs,NRuns,rank,nproc,0,&blocktridiagIterJacobi,"walltimes_blocktridiagIterJac.dat"); if(ierr) return(ierr);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  for (bs=1; bs <= MAX_BS; bs++) {
    if (!rank) printf("\nTesting MPI blocktridiagLU()         with N=%d, Ns=%d, bs=%d on %d processes\n",N,Ns,bs,nproc);
    ierr = test_block_mpi(N,Ns,bs,NRuns,rank,nproc,1,&blocktridiagLU,"walltimes_blocktridiagLU.dat"); if(ierr) return(ierr);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  /* Return */
  return(0);
}

/* 
    THIS FUNCTION TESTS THE PARALLEL IMPLEMENTATION OF A 
    TRIDIAGONAL SOLVER
*/
int test_mpi(int N,int Ns,int NRuns,int rank,int nproc, int flag, int blacs_context,
             int(*LUSolver)(double*,double*,double*,double*,int,int,void*,void*),const char *filename)
{
  int       i,d,ierr=0,nlocal;
  double    error;
  MPI_Comm  world;
  TridiagLU context;
  /* Variable declarations */
  double *a1;    /* sub-diagonal                               */
  double *b1;    /* diagonal                                   */
  double *c1;    /* super-diagonal                             */
  double *x;     /* right hand side, will contain the solution */ 

  /* 
    Since a,b,c and x are not preserved, declaring variables to
    store a copy of them to calculate error after the solve
  */
  double *a2,*b2,*c2,*y;

  /* Creating a duplicate communicator */
  MPI_Comm_dup(MPI_COMM_WORLD,&world);

  /* Initialize tridiagonal solver parameters */
  ierr = tridiagLUInit(&context,&world); if (ierr) return(ierr);

#ifdef with_scalapack
  context.blacs_ctxt = blacs_context;
#endif

  /* Initialize random number generator */
  srand(time(NULL));

  /* 
    Calculate local size on this process, given
    the total size N and number of processes 
    nproc
  */
  ierr = partition1D(N,nproc,rank,&nlocal);
  MPI_Barrier(MPI_COMM_WORLD);

  /* 
    Allocate arrays of dimension (Ns x nlocal) 
    Ns      -> number of systems
    nlocal  -> local size of each system
  */
  a1 = (double*) calloc (Ns*nlocal,sizeof(double));
  b1 = (double*) calloc (Ns*nlocal,sizeof(double));
  c1 = (double*) calloc (Ns*nlocal,sizeof(double));
  a2 = (double*) calloc (Ns*nlocal,sizeof(double));
  b2 = (double*) calloc (Ns*nlocal,sizeof(double));
  c2 = (double*) calloc (Ns*nlocal,sizeof(double));
  x  = (double*) calloc (Ns*nlocal,sizeof(double));
  y  = (double*) calloc (Ns*nlocal,sizeof(double));


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
      a1[i*Ns+d] = 0.0;
      b1[i*Ns+d] = 1.0;
      c1[i*Ns+d] = 0.0;
      x [i*Ns+d] = rand();
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
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,&context,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (!rank)  printf("error=%E\n",error);
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
      a1[i*Ns+d] = 0.0;
      b1[i*Ns+d] = 400.0;
      if (rank == nproc-1) c1[i*Ns+d] = (i == nlocal-1 ? 0 : 0.5);
      else                 c1[i*Ns+d] = 0.5;
      x[i*Ns+d]  = 1.0;
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
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,&context,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (!rank) printf("error=%E\n",error);
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
      if (!rank )           a1[i*Ns+d] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
      else                  a1[i*Ns+d] = ((double) rand()) / ((double) RAND_MAX);
      b1[i*Ns+d] = 100.0*(1.0 + ((double) rand()) / ((double) RAND_MAX));
      if (rank == nproc-1)  c1[i*Ns+d] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
      else                  c1[i*Ns+d] = ((double) rand()) / ((double) RAND_MAX);
      x[i*Ns+d]  = ((double) rand()) / ((double) RAND_MAX);
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
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,&context,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
  if (!rank) printf("error=%E\n",error);
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

        if (!rank )               a1[i*Ns+d] = (i == 0 ? 0.0 : 0.3 );
        else                      a1[i*Ns+d] = 0.3;

        if (!rank)                b1[i*Ns+d] = (i == 0 ? 1.0 : 0.6 );
        else if (rank == nproc-1) b1[i*Ns+d] = (i == nlocal-1 ? 1.0 : 0.6 );
        else                      b1[i*Ns+d] = 0.6;

        if (rank == nproc-1)      c1[i*Ns+d] = (i == nlocal-1 ? 0 : 0.1 );
        else                      c1[i*Ns+d] = 0.1;

        x[i*Ns+d]  = ((double) rand()) / ((double) RAND_MAX);
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
      /* Copy the original values */
      CopyArray(a2,a1,nlocal,Ns);
      CopyArray(b2,b1,nlocal,Ns);
      CopyArray(c2,c1,nlocal,Ns);
      CopyArray(y ,x ,nlocal,Ns);
      /* Solve the system */
      ierr         = LUSolver(a1,b1,c1,x,nlocal,Ns,&context,&world);
      /* Calculate errors */
      double err   = CalculateError(a2,b2,c2,y,x,nlocal,Ns,rank,nproc);
      /* Add the walltimes to the cumulative total */
      runtimes[0] += context.total_time;
      runtimes[1] += context.stage1_time;
      runtimes[2] += context.stage2_time;
      runtimes[3] += context.stage3_time;
      runtimes[4] += context.stage4_time;
      if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
      error += err;
    }
  
    /* Calculate average error */
    error /= NRuns;

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
      printf("\t\tAverage error   = %E\n",error);
      FILE *out;
      out = fopen(filename,"w");
      fprintf(out,"%5d  %1.16E  %1.16E  %1.16E  %1.16E  %1.16E  %1.16E\n",nproc,runtimes[0],
              runtimes[1],runtimes[2],runtimes[3],runtimes[4],error);
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
  free(a1);
  free(b1);
  free(c1);
  free(a2);
  free(b2);
  free(c2);
  free(x);
  free(y);
  MPI_Comm_free(&world);

  /* Return */
  return(0);
}

/* 
    THIS FUNCTION TESTS THE PARALLEL IMPLEMENTATION OF A 
    BLOCK TRIDIAGONAL SOLVER
*/
int test_block_mpi(int N,int Ns,int bs,int NRuns,int rank,int nproc, int flag,
             int(*LUSolver)(double*,double*,double*,double*,int,int,int,void*,void*),const char *filename)
{
  int       i,j,k,d,ierr=0,nlocal;
  double    error;
  MPI_Comm  world;
  TridiagLU context;
  /* Variable declarations */
  double *a1;    /* sub-diagonal                               */
  double *b1;    /* diagonal                                   */
  double *c1;    /* super-diagonal                             */
  double *x;     /* right hand side, will contain the solution */ 

  /* 
    Since a,b,c and x are not preserved, declaring variables to
    store a copy of them to calculate error after the solve
  */
  double *a2,*b2,*c2,*y;

  /* Creating a duplicate communicator */
  MPI_Comm_dup(MPI_COMM_WORLD,&world);

  /* Initialize tridiagonal solver parameters */
  ierr = tridiagLUInit(&context,&world); if (ierr) return(ierr);

  /* Initialize random number generator */
  srand(time(NULL));

  /* 
    Calculate local size on this process, given
    the total size N and number of processes 
    nproc
  */
  ierr = partition1D(N,nproc,rank,&nlocal);
  MPI_Barrier(MPI_COMM_WORLD);

  /* 
    Allocate arrays of dimension (Ns x nlocal) 
    Ns      -> number of systems
    nlocal  -> local size of each system
  */
  a1 = (double*) calloc (Ns*nlocal*bs*bs,sizeof(double));
  b1 = (double*) calloc (Ns*nlocal*bs*bs,sizeof(double));
  c1 = (double*) calloc (Ns*nlocal*bs*bs,sizeof(double));
  a2 = (double*) calloc (Ns*nlocal*bs*bs,sizeof(double));
  b2 = (double*) calloc (Ns*nlocal*bs*bs,sizeof(double));
  c2 = (double*) calloc (Ns*nlocal*bs*bs,sizeof(double));
  x  = (double*) calloc (Ns*nlocal*bs   ,sizeof(double));
  y  = (double*) calloc (Ns*nlocal*bs   ,sizeof(double));


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
      for (j=0; j<bs; j++) {
        for (k=0; k<bs; k++) {
          a1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
          if (j==k) b1[(i*Ns+d)*bs*bs+j*bs+k] = 1.0;
          else      b1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
          c1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
        }
        x [(i*Ns+d)*bs+j] = rand();
      }
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArraySimple(a2,a1,nlocal*Ns*bs*bs);
  CopyArraySimple(b2,b1,nlocal*Ns*bs*bs);
  CopyArraySimple(c2,c1,nlocal*Ns*bs*bs);
  CopyArraySimple(y ,x ,nlocal*Ns*bs   );
  
  /* Solve */  
  if (!rank)  printf("Block MPI test 1 ([I]x = b => x = b):        \t");
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,bs,&context,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateErrorBlock(a2,b2,c2,y,x,nlocal,Ns,bs,rank,nproc);
  if (!rank)  printf("error=%E\n",error);
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
      for (j=0; j<bs; j++) {
        for (k=0; k<bs; k++) {
          a1[(i*Ns+d)*bs*bs+j*bs+k] = 0.0;
          if (j==k) b1[(i*Ns+d)*bs*bs+j*bs+k] = 400.0 + ((double) rand()) / ((double) RAND_MAX);
          else      b1[(i*Ns+d)*bs*bs+j*bs+k] = 100.0 + ((double) rand()) / ((double) RAND_MAX);
          if (rank == nproc-1) c1[(i*Ns+d)*bs*bs+j*bs+k] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
          else                 c1[(i*Ns+d)*bs*bs+j*bs+k] = ((double) rand()) / ((double) RAND_MAX);
        }
        x[(i*Ns+d)*bs+j]  = ((double) rand()) / ((double) RAND_MAX);
      }
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArraySimple(a2,a1,nlocal*Ns*bs*bs);
  CopyArraySimple(b2,b1,nlocal*Ns*bs*bs);
  CopyArraySimple(c2,c1,nlocal*Ns*bs*bs);
  CopyArraySimple(y ,x ,nlocal*Ns*bs   );

  /* Solve */  
  if (!rank) printf("Block MPI test 2 ([U]x = b => x = [U]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,bs,&context,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateErrorBlock(a2,b2,c2,y,x,nlocal,Ns,bs,rank,nproc);
  if (!rank) printf("error=%E\n",error);
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
      for (j=0; j<bs; j++) {
        for (k=0; k<bs; k++) {
          if (!rank )           a1[(i*Ns+d)*bs*bs+j*bs+k] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
          else                  a1[(i*Ns+d)*bs*bs+j*bs+k] = ((double) rand()) / ((double) RAND_MAX);
          if (j==k)             b1[(i*Ns+d)*bs*bs+j*bs+k] = 200.0*(1.0 + ((double) rand()) / ((double) RAND_MAX));
          else                  b1[(i*Ns+d)*bs*bs+j*bs+k] = 100.0*(1.0 + ((double) rand()) / ((double) RAND_MAX));
          if (rank == nproc-1)  c1[(i*Ns+d)*bs*bs+j*bs+k] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
          else                  c1[(i*Ns+d)*bs*bs+j*bs+k] = ((double) rand()) / ((double) RAND_MAX);
        }
        x[(i*Ns+d)*bs+j]  = ((double) rand()) / ((double) RAND_MAX);
      }
    }
  }

  /*
    Copy the original values to calculate error later
  */
  CopyArraySimple(a2,a1,nlocal*Ns*bs*bs);
  CopyArraySimple(b2,b1,nlocal*Ns*bs*bs);
  CopyArraySimple(c2,c1,nlocal*Ns*bs*bs);
  CopyArraySimple(y ,x ,nlocal*Ns*bs   );

  /* Solve */  
  if (!rank) printf("Block MPI test 3 ([A]x = b => x = [A]^(-1)b):\t");
  ierr = LUSolver(a1,b1,c1,x,nlocal,Ns,bs,&context,&world);
  if (ierr == -1) printf("Error - system is singular on process %d\t",rank);

  /*
    Calculate Error
  */
  error = CalculateErrorBlock(a2,b2,c2,y,x,nlocal,Ns,bs,rank,nproc);
  if (!rank) printf("error=%E\n",error);
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
        for (j=0; j<bs; j++) {
          for (k=0; k<bs; k++) {
            if (!rank )           a1[(i*Ns+d)*bs*bs+j*bs+k] = (i == 0 ? 0.0 : ((double) rand()) / ((double) RAND_MAX));
            else                  a1[(i*Ns+d)*bs*bs+j*bs+k] = ((double) rand()) / ((double) RAND_MAX);
            if (j==k)             b1[(i*Ns+d)*bs*bs+j*bs+k] = 200.0*(1.0 + ((double) rand()) / ((double) RAND_MAX));
            else                  b1[(i*Ns+d)*bs*bs+j*bs+k] = 100.0*(1.0 + ((double) rand()) / ((double) RAND_MAX));
            if (rank == nproc-1)  c1[(i*Ns+d)*bs*bs+j*bs+k] = (i == nlocal-1 ? 0 : ((double) rand()) / ((double) RAND_MAX));
            else                  c1[(i*Ns+d)*bs*bs+j*bs+k] = ((double) rand()) / ((double) RAND_MAX);
          }
          x[(i*Ns+d)*bs+j]  = ((double) rand()) / ((double) RAND_MAX);
        }
      }
    }

    /*
      Keep a copy of the original values 
    */
    CopyArraySimple(a2,a1,nlocal*Ns*bs*bs);
    CopyArraySimple(b2,b1,nlocal*Ns*bs*bs);
    CopyArraySimple(c2,c1,nlocal*Ns*bs*bs);
    CopyArraySimple(y ,x ,nlocal*Ns*bs   );

    if (!rank) 
      printf("\nBlock MPI test 4 (Speed test - %d Tridiagonal Solves):\n",NRuns);
    double runtimes[5] = {0.0,0.0,0.0,0.0,0.0};
    error = 0;
    /* 
      Solve the systen NRuns times
    */  
    for (i = 0; i < NRuns; i++) {
      /* Copy the original values */
      CopyArraySimple(a1,a2,nlocal*Ns*bs*bs);
      CopyArraySimple(b1,b2,nlocal*Ns*bs*bs);
      CopyArraySimple(c1,c2,nlocal*Ns*bs*bs);
      CopyArraySimple(x ,y ,nlocal*Ns*bs   );
      /* Solve the system */
      ierr         = LUSolver(a1,b1,c1,x,nlocal,Ns,bs,&context,&world);
      /* Calculate errors */
      double err   = CalculateErrorBlock(a2,b2,c2,y,x,nlocal,Ns,bs,rank,nproc);
      /* Add the walltimes to the cumulative total */
      runtimes[0] += context.total_time;
      runtimes[1] += context.stage1_time;
      runtimes[2] += context.stage2_time;
      runtimes[3] += context.stage3_time;
      runtimes[4] += context.stage4_time;
      if (ierr == -1) printf("Error - system is singular on process %d\t",rank);
      error += err;
    }
  
    /* Calculate average error */
    error /= NRuns;

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
      printf("\t\tAverage error   = %E\n",error);
      FILE *out;
      out = fopen(filename,"w");
      fprintf(out,"%5d  %1.16E  %1.16E  %1.16E  %1.16E  %1.16E  %1.16E\n",nproc,runtimes[0],
              runtimes[1],runtimes[2],runtimes[3],runtimes[4],error);
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
  free(a1);
  free(b1);
  free(c1);
  free(a2);
  free(b2);
  free(c2);
  free(x);
  free(y);
  MPI_Comm_free(&world);

  /* Return */
  return(0);
}

#endif


/*
  Function to copy the values of one array into another
*/
void CopyArraySimple(double *x,double *y,int N)
{
  int i;
  for (i = 0; i < N; i++) x[i] = y[i];
  return;
}

/*
  Function to copy the values of one 2D array into another
*/
void CopyArray(double *x,double *y,int N,int Ns)
{
  int i,d;
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      y[i*Ns+d] = x[i*Ns+d];
    }
  }
  return;
}

/* 
  Functions to calculate the error in the computed solution 
*/
#ifdef serial

double CalculateError(double *a,double *b,double *c,double *y,double *x,
                      int N,int Ns)
{
  int i,d;
  double error = 0;
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      double val;
      if (i == 0)         val = y[i*Ns+d] - (b[i*Ns+d]*x[i*Ns+d]+c[i*Ns+d]*x[(i+1)*Ns+d]);
      else if (i == N-1)  val = y[i*Ns+d] - (a[i*Ns+d]*x[(i-1)*Ns+d]+b[i*Ns+d]*x[i*Ns+d]);
      else                val = y[i*Ns+d] - (a[i*Ns+d]*x[(i-1)*Ns+d]+b[i*Ns+d]*x[i*Ns+d]+c[i*Ns+d]*x[(i+1)*Ns+d]);
      error += val * val;
    }
  }
  return(error);
}


double CalculateErrorBlock(double *a,double *b,double *c,double *y,double *x,
                           int N,int Ns,int bs)
{
  int i,d,j;
  double error = 0;
  for (d = 0; d < Ns; d++) {
    for (i = 0; i < N; i++) {
      double val[bs]; for (j=0; j<bs; j++) val[j] = y[(i*Ns+d)*bs+j];
      _MatVecMultiplySubtract_(val,b+(i*Ns+d)*bs*bs,x+(i*Ns+d)*bs,bs);
      if (i != 0)   _MatVecMultiplySubtract_(val,a+(i*Ns+d)*bs*bs,x+((i-1)*Ns+d)*bs,bs);
      if (i != N-1) _MatVecMultiplySubtract_(val,c+(i*Ns+d)*bs*bs,x+((i+1)*Ns+d)*bs,bs);
      for (j=0; j<bs; j++) error += val[j] * val[j];
    }
  }
  return(error);
}

#else

double CalculateError(double *a,double *b,double *c,double *y,double *x,
                      int N,int Ns,int rank,int nproc)
{
  double        error = 0, norm = 0;
  int           i,d;
  double        xp1, xm1; /* solution from neighboring processes */

  for (d = 0; d < Ns; d++) {
    xp1 = 0;
    if (nproc > 1) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (rank != nproc-1)  MPI_Irecv(&xp1,1,MPI_DOUBLE,rank+1,1738,MPI_COMM_WORLD,&request);
      if (rank)             MPI_Send(&x[d],1,MPI_DOUBLE,rank-1,1738,MPI_COMM_WORLD);
      MPI_Wait(&request,MPI_STATUS_IGNORE);
    }
  
    xm1 = 0;
    if (nproc > 1) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (rank)             MPI_Irecv(&xm1,1,MPI_DOUBLE,rank-1,1739,MPI_COMM_WORLD,&request);
      if (rank != nproc-1)  MPI_Send(&x[d+(N-1)*Ns],1,MPI_DOUBLE,rank+1,1739,MPI_COMM_WORLD);
      MPI_Wait(&request,MPI_STATUS_IGNORE);
    }

    error = 0;
    norm = 0;
    for (i = 0; i < N; i++) {
      double val = 0;
      if (i == 0)    val += a[i*Ns+d]*xm1;
      else           val += a[i*Ns+d]*x[(i-1)*Ns+d];
      val += b[i*Ns+d]*x[i*Ns+d];
      if (i == N-1)  val += c[i*Ns+d]*xp1;
      else           val += c[i*Ns+d]*x[(i+1)*Ns+d];
      val = y[i*Ns+d] - val;
      error += val * val;
      norm += y[i*Ns+d] * y[i*Ns+d];
    }
  }

  double global_error = 0, global_norm = 0;
  if (nproc > 1)  MPI_Allreduce(&error,&global_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            global_error = error;
  if (nproc > 1)  MPI_Allreduce(&norm,&global_norm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            global_norm = norm;
  if (global_norm != 0.0) global_error /= global_norm;

  return(global_error);
}


double CalculateErrorBlock(double *a,double *b,double *c,double *y,double *x,
                      int N,int Ns,int bs,int rank,int nproc)
{
  double        error = 0, norm = 0;
  int           i,j,d;
  double        xp1[bs], xm1[bs]; /* solution from neighboring processes */

  for (d = 0; d < Ns; d++) {
    for (i=0; i<bs; i++) xp1[i] = 0;
    if (nproc > 1) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (rank != nproc-1)  MPI_Irecv(&xp1,bs,MPI_DOUBLE,rank+1,1738,MPI_COMM_WORLD,&request);
      if (rank)             MPI_Send(&x[d*bs],bs,MPI_DOUBLE,rank-1,1738,MPI_COMM_WORLD);
      MPI_Wait(&request,MPI_STATUS_IGNORE);
    }
  
    for (i=0; i<bs; i++) xm1[i] = 0;
    if (nproc > 1) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (rank)             MPI_Irecv(&xm1,bs,MPI_DOUBLE,rank-1,1739,MPI_COMM_WORLD,&request);
      if (rank != nproc-1)  MPI_Send (&x[(d+(N-1)*Ns)*bs],bs,MPI_DOUBLE,rank+1,1739,MPI_COMM_WORLD);
      MPI_Wait(&request,MPI_STATUS_IGNORE);
    }

    error = 0;
    norm = 0;
    for (i = 0; i < N; i++) {
      double val[bs]; for (j=0; j<bs; j++) val[j] = y[(i*Ns+d)*bs+j];
      _MatVecMultiplySubtract_(val,b+(i*Ns+d)*bs*bs,x+(i*Ns+d)*bs,bs);
      if (i == 0)   _MatVecMultiplySubtract_(val,a+(i*Ns+d)*bs*bs,xm1,bs)
      else          _MatVecMultiplySubtract_(val,a+(i*Ns+d)*bs*bs,x+((i-1)*Ns+d)*bs,bs)
      if (i == N-1) _MatVecMultiplySubtract_(val,c+(i*Ns+d)*bs*bs,xp1,bs)
      else          _MatVecMultiplySubtract_(val,c+(i*Ns+d)*bs*bs,x+((i+1)*Ns+d)*bs,bs)
      for (j=0; j<bs; j++) error += val[j] * val[j];
      for (j=0; j<bs; j++) norm += y[(i*Ns+d)*bs+j] * y[(i*Ns+d)*bs+j];
    }
  }

  double global_error = 0, global_norm = 0;
  if (nproc > 1)  MPI_Allreduce(&error,&global_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            global_error = error;
  if (nproc > 1)  MPI_Allreduce(&norm,&global_norm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  else            global_norm = norm;
  if (global_norm != 0.0) global_error /= global_norm;

  return(global_error);
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
