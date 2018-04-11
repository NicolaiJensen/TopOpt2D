PETSC_DIR=/home/nicolai/opt/petsc-3.7.4
PETSC_ARCH=linux-gnu-dbg32_01
CFLAGS = -I.
FFLAGS=
CPPFLAGS=-I.
FPPFLAGS=
LOCDIR=
EXAMPLESC=
EXAMPLESF=
MANSEC=
CLEANFILES=
NP=


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

topopt: main.o TopOpt.o LinearElasticity.o FiniteDiff.o MMA.o Filter.o PDEFilter.o MPIIO.o WriteVTS.o chkopts
	rm -rf topopt
	-${CLINKER} -o topopt main.o TopOpt.o LinearElasticity.o FiniteDiff.o MMA.o Filter.o PDEFilter.o MPIIO.o WriteVTS.o ${PETSC_SYS_LIB}
	${RM}  main.o TopOpt.o LinearElasticity.o FiniteDiff.o MMA.o Filter.o PDEFilter.o MPIIO.o WriteVTS.o
	rm -rf *.o

myclean:
	rm -rf topopt *.o output* binary* log* makevtu.pyc Restart*  *.vts

