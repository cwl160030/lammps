#include "fix_qmhub.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "update.h"

#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_qmhub[] =
  "fix qmhub command: https://doi.org/10.1063/5.0038120\n\n"
  "@Article{Pan2021,\n"
  " author = {Xiaoliang Pan, Kwangho Nam, Evgeny Epifanovsky, Andrew C. Simmonett, Edina Rosta, and Yihan Shao},\n"
  " title = {A simplified charge projection scheme for long-range electrostatics in ab initio QM/MM calculations},\n"
  " journal = {J. Chem. Phys.},\n"
  " year =    2021,\n"
  " volume =  154,\n"
  " number =  2, \n"
  " pages =   024115\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

// fix ID all qmhub qm_r_chrg qm_r_spin atomic_number1 atomic_number2 ...
FixQmhub::FixQmhub(LAMMPS *lmp, int narg, char **arg) : 
    Fix(lmp, narg, arg)
{
  // for compute_scalar()
  scalar_flag = 1;
  global_freq = 1;
  extscalar   = 1;

  int ntypes = atom->ntypes;
  if (narg < 5+ntypes) utils::missing_cmd_args(FLERR, "fix qmhub", error);
  if (strcmp(arg[1], "all") != 0) error->all(FLERR, "fix qmhub error: group-ID must be 'all'");
  qm_r_chrg = utils::inumeric(FLERR, arg[3], false, lmp);
  qm_r_spin = utils::inumeric(FLERR, arg[4], false, lmp);

  if ((domain->xperiodic == 0) && (domain->yperiodic == 0) && (domain->zperiodic == 0)) is_pbc = 0;
  else if ((domain->xperiodic == 1) && (domain->yperiodic == 1) && (domain->zperiodic == 1)) is_pbc = 1;
  else error->all(FLERR, "fix qmhub error: cell must either be periodic in all directions or not periodic in all directions");

  atomic_numbers = nullptr;
  memory->create(atomic_numbers, ntypes, "fix/qmhub:atomic_numbers");
  for (int i = 0; i < ntypes; i++) {
    atomic_numbers[i] = utils::inumeric(FLERR, arg[5+i], false, lmp);
  }

  int igroup_qm = group->find("QM");
  if (igroup_qm == -1) error->all(FLERR, "fix qmhub error: group 'QM' not defined");
  num_qm      = group->count(igroup_qm);
  groupbit_qm = group->bitmask[igroup_qm];

  int igroup_mm = group->find("MM");
  if (igroup_mm == -1) error->all(FLERR, "fix qmhub error: group 'MM' not defined");
  num_mm      = group->count(igroup_mm);
  groupbit_mm = group->bitmask[igroup_mm];

  E_SCF = 0.0; 
}

/* ---------------------------------------------------------------------- */

FixQmhub::~FixQmhub()
{
  memory->destroy(atomic_numbers);
}

/* ---------------------------------------------------------------------- */

void FixQmhub::post_constructor()
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_qmhub);
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

void FixQmhub::setup(int vflag)
{ 
  // Enforce units real
  if (strcmp(update->unit_style, "real") != 0) error->all(FLERR, "fix qmhub error: units must be 'real'");

  if ((comm->me==0) && screen) {
    fprintf(screen, "\n>>> fix qmhub: BEGIN QM/MM CALCULATION <<<\n");
    fprintf(screen, " Xiaoliang Pan and Yihan Shao\n");
    fprintf(screen, " QMHub: A universal QM/MM interface\n");
    fprintf(screen, " https://github.com/panxl/qmhub\n\n");
  }

  // make qmhub directory for QM engine calculations
  int mkret = mkdir("qmhub", 0777);  
  if ((mkret != 0) && (errno != EEXIST)) error->all(FLERR, "fix qmhub error: could not create or access directory ./qmhub/");

  post_integrate();
}

/* ---------------------------------------------------------------------- */

void FixQmhub::post_integrate()
{
  // get positions, charges, QM atom types, and cell vectors 
 
  double *qm_coord = nullptr;
  double *qm_chrgs = nullptr;
  int    *qm_types = nullptr;
  double *mm_coord = nullptr;
  double *mm_chrgs = nullptr;

  if (comm->me == 0) {
    memory->create(qm_coord, num_qm*3, "fix/qmhub:qm_coord");
    memory->create(qm_chrgs, num_qm  , "fix/qmhub:qm_chrgs");
    memory->create(qm_types, num_qm  , "fix/qmhub:qm_types");
    memory->create(mm_coord, num_mm*3, "fix/qmhub:mm_coord");
    memory->create(mm_chrgs, num_mm  , "fix/qmhub:mm_chrgs");
  }

  get_lmp_data(qm_coord, qm_chrgs, qm_types, mm_coord, mm_chrgs);  

  if (comm->me == 0) {   
    FILE *fp_qmmm_inp = fopen("./qmhub/qmmm.inp", "w");
    if (fp_qmmm_inp == nullptr) error->all(FLERR, "fix qmhub error: cannot open 'qmmm.inp'");
    
    fprintf(fp_qmmm_inp, "%d %d %d %d %d\n", num_qm, num_mm, qm_r_chrg, qm_r_spin, is_pbc);
    for (int i = 0; i < num_qm; i++) {
      fprintf(fp_qmmm_inp, "% .15E % .15E % .15E % .15E %d\n", qm_coord[3*i], qm_coord[3*i+1], qm_coord[3*i+2], qm_chrgs[i], atomic_numbers[qm_types[i]-1]);
    }
    for (int i = 0; i < num_mm; i++) {
      fprintf(fp_qmmm_inp, "% .15E % .15E % .15E % .15E\n", mm_coord[3*i], mm_coord[3*i+1], mm_coord[3*i+2], mm_chrgs[i]);
    }
    fprintf(fp_qmmm_inp, "% .15E % .15E % .15E\n", domain->h[0], 0.0         , 0.0         );
    fprintf(fp_qmmm_inp, "% .15E % .15E % .15E\n", domain->h[5], domain->h[1], 0.0         );
    fprintf(fp_qmmm_inp, "% .15E % .15E % .15E\n", domain->h[4], domain->h[3], domain->h[2]);
   
    fflush(fp_qmmm_inp); 
    fclose(fp_qmmm_inp);

    memory->destroy(qm_coord);
    memory->destroy(qm_chrgs);
    memory->destroy(qm_types);
    memory->destroy(mm_coord);
    memory->destroy(mm_chrgs);

    // system call to qmhub
    int callret = system("qmhub qmhub.ini --text ./qmhub/qmmm.inp --driver sander");
    if (callret != 0) error->all(FLERR, "fix qmhub error: qmhub execution failed");
  }
}

/* ---------------------------------------------------------------------- */

void FixQmhub::post_force(int vflag)
{
  // read gradients from qmmm.out
  double *qm_grad = nullptr;
  double *mm_grad = nullptr;
  if (comm->me == 0) {
    memory->create(qm_grad, 3*num_qm, "fix/qmhub:qm_grad");
    memory->create(mm_grad, 3*num_mm, "fix/qmhub:mm_grad");

    FILE *fp_qmmm_out = fopen("./qmhub/qmmm.out", "r");
    if (fp_qmmm_out == nullptr) error->all(FLERR, "fix qmhub error: cannot open 'qmmm.out'");

    fscanf(fp_qmmm_out, "%lf", &E_SCF);
    for (int i = 0; i < num_qm; i++) {
      fscanf(fp_qmmm_out, "%lf %lf %lf", &qm_grad[3*i], &qm_grad[3*i+1], &qm_grad[3*i+2]);
    }
    for (int i = 0; i < num_mm; i++) {
      fscanf(fp_qmmm_out, "%lf %lf %lf", &mm_grad[3*i], &mm_grad[3*i+1], &mm_grad[3*i+2]);
    }

    fclose(fp_qmmm_out);
  }

  // send global gradients to local
  int nlocal = atom->nlocal;
  double *qm_grad_local = nullptr;
  double *mm_grad_local = nullptr;
  
  int num_qm_local = 0;
  int num_mm_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) num_qm_local++;
    if (atom->mask[i] & groupbit_mm) num_mm_local++;
  }

  memory->create(qm_grad_local, 3*num_qm_local, "fix/qmmm:qm_grad_local");
  memory->create(mm_grad_local, 3*num_mm_local, "fix/qmmm:mm_grad_local");  

  int nprocs;
  MPI_Comm_size(world, &nprocs);

  int *count_qm_all = nullptr;
  int *count_mm_all = nullptr;

  if (comm->me == 0) {
    memory->create(count_qm_all, nprocs, "fix/qmhub:count_qm_all");
    memory->create(count_mm_all, nprocs, "fix/qmhub:count_mm_all");
  }

  MPI_Gather(&num_qm_local, 1, MPI_INT, count_qm_all, 1, MPI_INT, 0, world);
  MPI_Gather(&num_mm_local, 1, MPI_INT, count_mm_all, 1, MPI_INT, 0, world);

  int *send_qm = nullptr;
  int *send_mm = nullptr;
  int *disp_qm = nullptr;
  int *disp_mm = nullptr;

  if (comm->me == 0) {
    memory->create(send_qm, nprocs, "fix/qmhub:send_qm");
    memory->create(send_mm, nprocs, "fix/qmhub:send_mm");
    memory->create(disp_qm, nprocs, "fix/qmhub:disp_qm");
    memory->create(disp_mm, nprocs, "fix/qmhub:disp_mm");

    for (int i = 0; i < nprocs; i++) {
      send_qm[i] = 3*count_qm_all[i];
      send_mm[i] = 3*count_mm_all[i];
    }
    
    disp_qm[0] = 0;
    disp_mm[0] = 0;

    for (int i = 1; i < nprocs; i++) {
      disp_qm[i] = disp_qm[i-1] + send_qm[i-1];
      disp_mm[i] = disp_mm[i-1] + send_mm[i-1];
    }
  }

  MPI_Scatterv(qm_grad, send_qm, disp_qm, MPI_DOUBLE, qm_grad_local, 3*num_qm_local, MPI_DOUBLE, 0, world);
  MPI_Scatterv(mm_grad, send_mm, disp_mm, MPI_DOUBLE, mm_grad_local, 3*num_mm_local, MPI_DOUBLE, 0, world);

  // convert to forces (Ha/Bohr -> -kcal/mol/Angstrom) and add
  const double HABOHR_KCALMOLA = (627.5096080305927) / (0.529177210544);
  int count_qm = 0;
  int count_mm = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) {
      for (int dim = 0; dim < 3; dim++) {
        atom->f[i][dim] -= HABOHR_KCALMOLA * qm_grad_local[3*count_qm+dim];
      }
      count_qm++;
    }
    if (atom->mask[i] & groupbit_mm) {
      for (int dim = 0; dim < 3; dim++) {
        atom->f[i][dim] -= HABOHR_KCALMOLA * mm_grad_local[3*count_mm+dim];
      }
      count_mm++;
    }
  }

  if (comm->me == 0) {
    memory->destroy(qm_grad);
    memory->destroy(mm_grad);

    memory->destroy(count_qm_all);
    memory->destroy(count_mm_all);

    memory->destroy(send_qm);
    memory->destroy(send_mm);
    memory->destroy(disp_qm);
    memory->destroy(disp_mm);
  }
  memory->destroy(qm_grad_local);
  memory->destroy(mm_grad_local);
}

/* ---------------------------------------------------------------------- */

void FixQmhub::get_lmp_data(double *qm_coord, double *qm_chrgs, int *qm_types, double *mm_coord, double *mm_chrgs)
{
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
  int    *qm_types_local = nullptr;
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

  int nprocs;
  MPI_Comm_size(world, &nprocs);

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
  MPI_Gatherv(qm_types_local, num_qm_local  , MPI_INT   , qm_types, recv_qm_t, disp_qm_t, MPI_INT   , 0, world);
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
}

/* ---------------------------------------------------------------------- */

// Add SCF Energy (Ha) to thermo via thermo_style custom ... f_ID ...
double FixQmhub::compute_scalar()
{
  return E_SCF;
} 

/* ---------------------------------------------------------------------- */
