#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int    main_serial();
static int    main_parallel();
static void   CopyArray       (double*,double*,int);
static double CalculateError  (double*,double*,double*,double*,double*,int);

int tridiagLU(double*,double*,double*,double*,int,int,int);

int main()
{
  int ierr;
  ierr = main_serial();
}


int main_serial()
{
  double *a1,*b1,*c1,*x;
  double *a2,*b2,*c2,*y;
  int     N,i,ierr=0;
  double  error,walltime;
  clock_t start,end;

  srand(time(NULL));

  printf("Enter N: ");
  scanf ("%d",&N);
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
  start = clock();
  ierr = tridiagLU(a1,b1,c1,x,N,0,1);
  end   = clock();
  walltime = ((double) (end-start)) / ((double) CLOCKS_PER_SEC);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N);
  printf("error=%E\twalltime=%E\n",error,walltime);

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
  start = clock();
  ierr  = tridiagLU(a1,b1,c1,x,N,0,1);
  end   = clock();
  walltime = ((double) (end-start)) / ((double) CLOCKS_PER_SEC);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N);
  printf("error=%E\twalltime=%E\n",error,walltime);

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
  start = clock();
  ierr = tridiagLU(a1,b1,c1,x,N,0,1);
  end   = clock();
  walltime = ((double) (end-start)) / ((double) CLOCKS_PER_SEC);
  if (ierr == -1) printf("Error - system is singular\t");
  /* calculate error */
  error = CalculateError(a2,b2,c2,y,x,N);
  printf("error=%E\twalltime=%E\n",error,walltime);


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

void CopyArray(double *x,double *y,int N)
{
  int i;
  for (i = 0; i < N; i++) y[i] = x[i];
  return;
}

double CalculateError(double *a,double *b,double *c,double *y,double *x,int N)
{
  int i;
  double error = 0;
  for (i = 0; i < N; i++) {
    double val;
    if (i == 0)         val = y[i] - (b[i]*x[i]+c[i]*x[i+1]);
    else if (i == N-1)  val = y[i] - (a[i]*x[i-1]+b[i]*x[i]);
    else                val = y[i] - (a[i]*x[i-1]+b[i]*x[i]+c[i]*x[i+1]);
    error += val * val;
  }
  return(error);
}

