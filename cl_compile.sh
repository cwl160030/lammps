#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=8
#  #SBATCH --cpus-per-task=1
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --job-name=lmp_comp

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
date
cmake -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_Fortran_COMPILER=ifx \
      -DPKG_AMOEBA=on \
      -DPKG_COLVARS=on \
      -DPKG_DRUDE=on \
      -DPKG_EXTRA-COMMAND=on \
      -DPKG_EXTRA-DUMP=on \
      -DPKG_EXTRA-FIX=on \
      -DPKG_EXTRA-MOLECULE=on \
      -DPKG_EXTRA-PAIR=on \
      -DPKG_FEP=on \
      -DPKG_INTEL=on \
      -DPKG_KSPACE=on \
      -DPKG_MANIFOLD=on \
      -DPKG_MISC=on \
      -DPKG_MOFF=on \
      -DPKG_MOLECULE=on \
      -DPKG_OPENMP=on \
      -DPKG_OPT=on \
      -DPKG_ORIENT=on \
      -DPKG_PHONON=on \
      -DPKG_REPLICA=on \
      -DPKG_RIGID=on \
      -DPKG_QMHUB=on \
      ../cmake
cmake --build .

date

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
