#!/bin/bash

make clean-all
module purge

# gcc
#module load gcc/13.2.0
#module load mpich/gcc
#module load binutils/2.43-gcc14.2.0

# intel
LANG=en_US.utf8
export MKLROOT=/opt/intel/oneapi/mkl/latest
module load intel/2021.2.0
module load impi/2021.2.0
make intel_cpu_intelmpi
