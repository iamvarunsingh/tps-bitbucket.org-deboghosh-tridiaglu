#include <stdio.h>

#if !defined(INLINE)
# define INLINE inline
#endif

INLINE int MatrixZero              (double*,int);
INLINE int MatrixInvert            (double*,double*,int);
INLINE int MatrixMultiply          (double*,double*,double*,int);
INLINE int MatrixMultiplySubtract  (double*,double*,double*,int);
INLINE int MatVecMultiply          (double*,double*,double*,int);
INLINE int MatVecMultiplySubtract  (double*,double*,double*,int);

INLINE int MatrixZero(double *A, int N)
{
  int i;
  for (i=0; i<N*N; i++) A[i] = 0.0;
  return(0);
}

/* C = AB */
INLINE int MatrixMultiply(double *A, double *B, double *C, int N)
{
  int i,j,k;
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      C[i*N+j] = 0;
      for (k=0; k<N; k++) C[i*N+j] += (A[i*N+k] * B[k*N+j]);
    }
  }
  return(0);
}


/* C = C - AB */
INLINE int MatrixMultiplySubtract(double *C, double *A, double *B, int N)
{
  int i,j,k;
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      for (k=0; k<N; k++) C[i*N+j] -= (A[i*N+k] * B[k*N+j]);
    }
  }
  return(0);
}

/* y = Ax */
INLINE int MatVecMultiply(double *A, double *x, double *y, int N)
{
  int i,j;
  for (i=0; i<N; i++) {
    y[i] = 0;
    for (j=0; j<N; j++) y[i] += (A[i*N+j] * x[j]);
  }
  return(0);
}

/* y = y - Ax */
INLINE int MatVecMultiplySubtract(double *A, double *x, double *y, int N)
{
  int i,j;
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) y[i] -= (A[i*N+j] * x[j]);
  }
  return(0);
}

/* B =A^{-1}                */
/* Note: A is not preserved */
INLINE int MatrixInvert(double *A, double *B, int N) 
{
  int i,j,k;
  double factor, sum;
  
  /* set B as the identity matrix */
  for (i=0; i<N*N; i++) B[i] = 0.0;
  for (i=0; i<N  ; i++) B[i*N+i] = 1.0;

  /* LU Decomposition - Forward Sweep */
  for (i=0; i<N-1; i++) {
    if (A[i*N+i] == 0) {
      fprintf(stderr,"Error in MatrixInvert(): Matrix is singular.\n");
      return(0);
    }
    for (j=i+1; j<N; j++) {
      factor = A[j*N+i]/A[i*N+i];
      for (k=i+1; k<N; k++) A[j*N+k] -= (factor*A[i*N+k]);
      for (k=0  ; k<N; k++) B[j*N+k] -= (factor*B[i*N+k]);
    }
  }

  /* LU Decomposition - Backward Sweep */
  for (i=N-1; i>=0; i--) {
    for (k=0; k<N; k++) {
      sum = 0.0;
      for (j=i+1; j<N; j++) sum += (A[i*N+j]*B[j*N+k]);
      B[i*N+k] = (B[i*N+k] - sum) / A[i*N+i];
    }
  }

  /* Done - B contains A^{-1} now */

  return(0);
}
