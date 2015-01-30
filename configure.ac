AC_INIT([tridiagLU],[0.1],[ghosh@mcs.anl.gov])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])
echo -n AC_PACKAGE_VERSION>VERSION
AC_SUBST(PACKAGE_VERSION)
AC_MSG_NOTICE([tridiagLU AC_PACKAGE_VERSION])

AM_INIT_AUTOMAKE
AC_PREFIX_DEFAULT('$(top_srcdir)')

AC_ARG_WITH([mpidir],AC_HELP_STRING([--with-mpidir],[Specify path where mpicc is installed.]))
AC_ARG_ENABLE([scalapack],AS_HELP_STRING([--enable-scalapack],[Enable ScaLAPACK]))
AC_ARG_WITH([scalapackdir],AC_HELP_STRING([--with-scalapackdir],[Specify path where ScaLAPACK is installed.]))

if test "x$with_mpidir" != "x" ; then
  AC_MSG_CHECKING([for mpicc in $with_mpidir/bin/])
  if test -e $with_mpidir/bin/mpicc ; then
    AC_MSG_RESULT([yes])
    MPICC=$with_mpidir/bin/mpicc
  else
    AC_MSG_RESULT([no])
  fi
fi
if test "x$MPICC" = "x" ; then
  AC_CHECK_PROG(MPICC,mpicc,mpicc)
fi

AC_PROG_CC

if test "x$MPICC" = "x" ; then
  AC_MSG_WARN([Cannot find mpicc. Will compile with gcc. Use --with-mpidir to specify the location of mpicc.])
  CFLAGS="$CFLAGS -Dserial"
else
  CC=$MPICC
fi

if test "x$enable_scalapack" = "xyes" ; then
  AC_MSG_NOTICE([Compiling with ScaLAPACK support.])
  CFLAGS="$CFLAGS -Dwith_scalapack"
  if test "x$with_scalapackdir" != "x" ; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,$with_scalapackdir/lib -L$with_scalapackdir/lib"
  fi
  LIBS="$LIBS -lscalapack -lblas -llapack"
fi


CFLAGS="$CFLAGS -I/opt/local/include -I$includedir -std=c99"
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
