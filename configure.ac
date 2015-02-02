AC_INIT([tridiagLU],[0.1],[ghosh@mcs.anl.gov])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])
echo -n AC_PACKAGE_VERSION>VERSION
AC_SUBST(PACKAGE_VERSION)
AC_MSG_NOTICE([tridiagLU AC_PACKAGE_VERSION])

AM_INIT_AUTOMAKE
AC_PREFIX_DEFAULT('$(top_srcdir)')

AC_ARG_WITH([mpi_dir],AC_HELP_STRING([--with-mpi-dir],[Specify path where mpicc is installed.]))
AC_ARG_ENABLE([scalapack],AS_HELP_STRING([--enable-scalapack],[Enable ScaLAPACK]))
AC_ARG_WITH([blas_dir],AC_HELP_STRING([--with-blas-dir],[Specify path where BLAS is installed.]))
AC_ARG_WITH([lapack_dir],AC_HELP_STRING([--with-lapack-dir],[Specify path where LAPACK is installed.]))
AC_ARG_WITH([scalapack_dir],AC_HELP_STRING([--with-scalapack-dir],[Specify path where ScaLAPACK is installed.]))

AC_PROG_CC
if test "$CC" = "gcc" ; then
  CFLAGS="$CFLAGS -std=c99"
  LIBS="$LIBS -lm"
  MPI_CC=mpicc
else 
  if test "$CC" = "xlc" ; then
    CFLAGS="$CFLAGS -qlanglvl=stdc99"
    MPI_CC=mpixlc
  fi
fi

if test "x$with_mpi_dir" != "x" ; then
  AC_MSG_CHECKING([for ${MPI_CC} in $with_mpi_dir/bin/])
  if test -e $with_mpi_dir/bin/${MPI_CC} ; then
    AC_MSG_RESULT([yes])
    MPICC=$with_mpi_dir/bin/${MPI_CC}
  else
    AC_MSG_RESULT([no])
  fi
fi
if test "x$MPICC" = "x" ; then
  AC_CHECK_PROG(MPICC,${MPI_CC},${MPI_CC})
fi

if test "x$MPICC" = "x" ; then
  AC_MSG_WARN([Cannot find mpicc. Will compile with gcc. Use --with-mpi-dir to specify the location of mpicc.])
  CFLAGS="$CFLAGS -Dserial"
else
  CC=$MPICC
fi

if test "x$enable_scalapack" = "xyes" ; then
  AC_MSG_NOTICE([Compiling with ScaLAPACK support.])
  CFLAGS="$CFLAGS -Dwith_scalapack"
  if test "x$with_blas_dir" != "x" ; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,$with_blas_dir/lib -L$with_blas_dir/lib"
  fi
  if test "x$with_lapack_dir" != "x" ; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,$with_lapack_dir/lib -L$with_lapack_dir/lib"
  fi
  if test "x$with_scalapack_dir" != "x" ; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,$with_scalapack_dir/lib -L$with_scalapack_dir/lib"
  fi
  LIBS="$LIBS -lscalapack -llapack -lblas -lgfortran"
fi

CFLAGS="$CFLAGS -I/opt/local/include -I$includedir"
LDFLAGS="$LDFLAGS -L/opt/local/lib"
LIBS="$LIBS -lm"

AC_PROG_RANLIB
AC_PROG_INSTALL
AC_CONFIG_FILES([
 Makefile
 src/Makefile
  src/TridiagLU/Makefile
  src/Test/Makefile
])

AC_OUTPUT
