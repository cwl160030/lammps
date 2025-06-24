#!/bin/bash

# Source file inside LAMMPS directory
source ~/act_conda.sh
module purge
LANG=en_US.utf8
export MKLROOT=/opt/intel/oneapi/mkl/latest
module load cmake3/3.24.3 
module load intel/2021.2.0
module load impi/2021.2.0
#mkdir build; cd build    # create and use a build directory
cd build # use if build exists
cmake -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_Fortran_COMPILER=ifx \
      -DPKG_EXTRA-COMMAND=on \
      -DPKG_EXTRA-FIX=on \
      -DPKG_EXTRA-MOLECULE=on \
      -DPKG_EXTRA-PAIR=on \
      -DPKG_INTEL=on \
      -DPKG_KSPACE=on \
      -DPKG_MOFF=on \
      -DPKG_MOLECULE=on \
      -DPKG_OPENMP=on \
      -DPKG_OPT=on \
      -DPKG_QMHUB=on \
      ../cmake
cmake --build build

# In build
# make
# make install

#module load gcc/13.2.0
#module load mpich/gcc
#module load cmake3/3.24.3 
#module load binutils/2.43-gcc14.2.0
#source ~/act_conda.sh
#
#mkdir build; cd build
#cmake -DPKG_MDI=yes -DPKG_MOLECULE=yes -DPKG_KSPACE=yes -DLAMMPS_MACHINE=mpi ../cmake
#cmake --build .

# cd lammps                # change to the LAMMPS distribution directory
# mkdir build; cd build    # create and use a build directory
# cmake ../cmake           # configuration reading CMake scripts from ../cmake
# cmake --build .          # compilation (or type "make")

#yes-AMOEBA
#yes-COLVARS
#yes-DIELECTRIC
#yes-DRUDE
#yes-ELECTRODE
#yes-EXTRA-COMMAND
#yes-EXTRA-FIX
#yes-EXTRA-MOLECULE
#yes-EXTRA-PAIR
#yes-FEP
#yes-KSPACE
#yes-MANIFOLD
#yes-MC
#yes-MISC
#yes-MOFF
#yes-MOLECULE
#yes-OPT
#yes-ORIENT
#yes-PHONON
#yes-QEQ
#yes-REPLICA
#yes-RIGID
#yes-SMTBQ
