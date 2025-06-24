#!/bin/bash

# Source file inside LAMMPS directory
source ~/act_conda.sh
module purge
LANG=en_US.utf8
export MKLROOT=/opt/intel/oneapi/mkl/latest
module load cmake3/3.24.3 
module load intel/2021.2.0
module load impi/2021.2.0

cd build
cmake ../cmake
cmake --build /scratch/chance/lammps/build
