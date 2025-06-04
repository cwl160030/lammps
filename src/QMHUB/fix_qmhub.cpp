#include "fix_qmhub.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"

#include <cstdio>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixQmhub::FixQmhub(LAMMPS *lmp, int narg, char **arg) : 
    Fix(lmp, narg, arg)
{
  // fix ID all qmhub qm_q qm_spin
  if (narg < 5) utils::missing_cmd_args(FLERR, "fix qmhub", error);
  if (arg[1] != "all") error->all(FLERR, "fix qmhub error: group-ID must be 'all'");
  qm_r_chrg = utils::inumeric(FLERR, arg[3], false, lmp);
  qm_r_spin = utils::inumeric(FLERR, arg[4], false, lmp);

  if ((domain->xperiodic == 0) && (domain->yperiodic == 0) && (domain->zperiodic == 0)) is_pbc = 0;
  else if ((domain->xperiodic == 1) && (domain->yperiodic == 1) && (domain->zperiodic == 1)) is_pbc = 1;
  else error->all(FLERR, "fix qmhub error: cell must either be periodic in all directions or not periodic in all directions");

  int igroup_qm = group->find("QM");
  if (igroup_qm == -1) error->all(FLERR, "fix qmhub error: group 'QM' not defined");
  num_qm      = group->count(igroup_qm);
  groupbit_qm = group->bitmask[igroup_qm];

  int igroup_mm = group->find("MM");
  if (igroup_mm == -1) error->all(FLERR, "fix qmhub error: group 'MM' not defined");
  num_mm      = group->count(igroup_mm);
  groupbit_mm = group->bitmask[igroup_mm]; 
}

/* ---------------------------------------------------------------------- */

FixQmhub::~FixQmhub()
{
  // Empty destructor
}

/* ---------------------------------------------------------------------- */

int FixQmhub::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixQmhub::setup()
{
  // To do: check qmhub installation
   
  // get positions, charges, QM atom types, and cell vectors
  
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double  *q = atom->q;
  if (q == nullptr) error->all(FLERR, "fix qmhub error: atoms do not have 'q' attribute. Ensure atom style allows charges.");
  int *type = atom->type;  

  int num_qm_local = 0;
  int num_mm_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) num_qm_local++;
    if (atom->mask[i] & groupbit_mm) num_mm_local++;
  }
  
  double *qm_coord_local = nullptr;
  double *qm_chrgs_local = nullptr;
  double *qm_types_local = nullptr;
  double *mm_coord_local = nullptr;
  double *mm_chrgs_local = nullptr;
  
  memory->create(qm_coord_local, num_qm_local*3, "fix/qmhub:qm_coord_local");
  memory->create(qm_chrgs_local, num_qm_local  , "fix/qmhub:qm_chrgs_local");
  memory->create(qm_types_local, num_qm_local  , "fix/qmhub:qm_types_local");
  memory->create(mm_coord_local, num_mm_local*3, "fix/qmhub:mm_coord_local");
  memory->create(mm_chrgs_local, num_mm_local  , "fix/qmhub:mm_chrgs_local");

  int count_qm = 0;
  int count_mm = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) {
      for (int dim = 0; dim < 3; dim++){
        qm_coord_local[3*count_qm+dim] = x[i][dim];
      }
      qm_chrgs_local[count_qm] = q[i];
      qm_types_local[count_qm] = type[i];
      count_qm++;
    }
    if (atom->mask[i] & groupbit_mm) {
      for (int dim = 0; dim < 3; dim++){
        mm_coord_local[3*count_mm+dim] = x[i][dim];
      }
      mm_chrgs_local[count_mm] = q[i];
      count_mm++;
    }
  }

  double *qm_coord = nullptr;
  double *qm_chrgs = nullptr; 
  double *qm_types = nullptr; 
  double *mm_coord = nullptr;
  double *mm_chrgs = nullptr;

  int nprocs;
  MPI_Comm_Size(world, &nprocs);

  int *count_qm_all = nullptr;
  int *count_mm_all = nullptr;

  if (comm->me == 0) {
    memory->create(count_qm_all, nprocs, "fix/qmhub:count_qm_all");
    memory->create(count_mm_all, nprocs, "fix/qmhub:count_mm_all");
  }

  MPI_Gather(&count_qm, 1, MPI_INT, count_qm_all, 1, MPI_INT, 0, world);
  MPI_Gather(&count_mm, 1, MPI_INT, count_mm_all, 1, MPI_INT, 0, world);

  int *recv_qm_x = nullptr;
  int *disp_qm_x = nullptr;
  int *recv_qm_q = nullptr;
  int *recv_qm_t = nullptr;
  int *disp_qm_t = nullptr;
  int *disp_qm_q = nullptr;
  int *recv_mm_x = nullptr;
  int *disp_mm_x = nullptr;
  int *recv_mm_q = nullptr;
  int *disp_mm_q = nullptr;

  if (comm->me == 0) {
    memory->create(qm_coord, num_qm*3, "fix/qmhub:qm_coord");
    memory->create(qm_chrgs, num_qm  , "fix/qmhub:qm_chrgs");
    memory->create(mm_coord, num_mm*3, "fix/qmhub:mm_coord");
    memory->create(mm_chrgs, num_mm  , "fix/qmhub:mm_chrgs");

    memory->create(recv_qm_x, nprocs, "fix/qmhub:recv_qm_x");
    memory->create(disp_qm_x, nprocs, "fix/qmhub:disp_qm_x");
    memory->create(recv_qm_q, nprocs, "fix/qmhub:recv_qm_q");
    memory->create(disp_qm_q, nprocs, "fix/qmhub:disp_qm_q");
    memory->create(recv_qm_t, nprocs, "fix/qmhub:recv_qm_t");
    memory->create(disp_qm_t, nprocs, "fix/qmhub:disp_qm_t");
    memory->create(recv_mm_x, nprocs, "fix/qmhub:recv_mm_x");
    memory->create(disp_mm_x, nprocs, "fix/qmhub:disp_mm_x");
    memory->create(recv_mm_q, nprocs, "fix/qmhub:recv_mm_q");
    memory->create(disp_mm_q, nprocs, "fix/qmhub:disp_mm_q");

    for (int i = 0; i < nprocs; i++){
      recv_qm_x[i] = 3*count_qm_all[i];
      recv_qm_q[i] =   count_qm_all[i];
      recv_qm_t[i] =   count_qm_all[i];
      recv_mm_x[i] = 3*count_mm_all[i];
      recv_mm_q[i] =   count_mm_all[i];
    }

    disp_qm_x[0] = 0;
    disp_qm_q[0] = 0;
    disp_qm_t[0] = 0;
    disp_mm_x[0] = 0;
    disp_mm_q[0] = 0;

    for (int i = 1; i < nprocs; i++) { 
      disp_qm_x[i] = disp_qm_x[i-1] + recv_qm_x[i-1];
      disp_qm_q[i] = disp_qm_q[i-1] + recv_qm_q[i-1];
      disp_qm_t[i] = disp_qm_t[i-1] + recv_qm_t[i-1];
      disp_mm_x[i] = disp_mm_x[i-1] + recv_mm_x[i-1];
      disp_mm_q[i] = disp_mm_q[i-1] + recv_mm_q[i-1];
    }
  }

  MPI_Gatherv(qm_coord_local, num_qm_local*3, MPI_DOUBLE, qm_coord, recv_qm_x, disp_qm_x, MPI_DOUBLE, 0, world);
  MPI_Gatherv(qm_chrgs_local, num_qm_local  , MPI_DOUBLE, qm_chrgs, recv_qm_q, disp_qm_q, MPI_DOUBLE, 0, world);
  MPI_Gatherv(qm_types_local, num_qm_local  , MPI_DOUBLE, qm_types, recv_qm_t, disp_qm_t, MPI_DOUBLE, 0, world);
  MPI_Gatherv(mm_coord_local, num_mm_local*3, MPI_DOUBLE, mm_coord, recv_mm_x, disp_mm_x, MPI_DOUBLE, 0, world);
  MPI_Gatherv(mm_chrgs_local, num_mm_local  , MPI_DOUBLE, mm_chrgs, recv_mm_q, disp_mm_q, MPI_DOUBLE, 0, world);
  
  memory->destroy(qm_coord_local);
  memory->destroy(qm_chrgs_local);
  memory->destroy(qm_types_local);
  memory->destroy(mm_coord_local);
  memory->destroy(mm_chrgs_local);
  
  if (comm->me == 0) {
    memory->destroy(count_qm_all);
    memory->destroy(count_mm_all);

    memory->destroy(recv_qm_x);
    memory->destroy(disp_qm_x);
    memory->destroy(recv_qm_q);
    memory->destroy(disp_qm_q);
    memory->destroy(recv_qm_t);
    memory->destroy(disp_qm_t);
    memory->destroy(recv_mm_x);
    memory->destroy(disp_mm_x);
    memory->destroy(recv_mm_q);
    memory->destroy(disp_mm_q);
  }

  // write to FIFO qmmm.inp

  if (comm->me == 0) {
    //fifo
    system("mkfifo qmmm.inp"); 
    // will continue here!
    memory->destroy(qm_coord);
    memory->destroy(qm_chrgs);
    memory->destroy(qm_types);
    memory->destroy(mm_coord);
    memory->destroy(mm_chrgs);
  }
  // system call to qmhub 
}

/* ---------------------------------------------------------------------- */

void FixQmhub::post_integrate()
{
    // get positions and charges
    // write to FIFO qmmm.inp
    // system call to qmhub
}

/* ---------------------------------------------------------------------- */

void FixQmhub::post_force()
{
  // read gradients from FIFO qmmm.out
  // convert to forces with correct units
  // add forces
}
