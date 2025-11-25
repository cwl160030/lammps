#include "fix_qmhub.h"

#include "atom.h"
#include "force.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "update.h"
#include "tokenizer.h"

#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>  // Remove later, only for std::cout -CL
#include <cstdlib>   // fabs
#include <algorithm> // find for vector
#include <array>
#include <iterator>  // begin and end

using namespace LAMMPS_NS;
using namespace FixConst;

/* atom, atom_vec : position, force, molecule index
 * neighbor* : neighbor lists
 * group
 * force
 */

/* Tasks
 * (0)   -   Current: Getting boundary atoms: setup_qm_link
 * (1)   -   Set angle, dihedral, pair, etc. QM-MM1 terms to 0: setup_qm_link
 * (2) Done  Charge conservation: setup_qm_link
 * (3) Done  Charge redistribution: setup_qm_link
 * (4)   -   Add hydrogen atom to QM calculation: post_integrate
 * (5)   -   Redistribute QM/MM forces: post_force
 */

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

void FixQmhub::set_atomic_numbers(int narg, char *qm_atom_index_filename, int *atomic_numbers) {
  FILE *fp;
  static constexpr int BUFLEN = 4096;
  char linebuf[BUFLEN];

  // Add sanity check that atoms even exist! -CL
  if (comm->me == 0) {
    fp = fopen(qm_atom_index_filename, "r");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open QM atom labels file for reading: {}", utils::getsyserror());
    utils::logmesg(lmp, "Reading QM atom labels from index file\n");
  }

  if (narg == 6) {
    // Try to read QM atom indices from file
    try {
      // Read line in fp
      int i=0;
      while (fgets(linebuf, BUFLEN, fp)) {
        // Tokenize values in linebuf
        ValueTokenizer values(linebuf);
        while (values.has_next()) {
          atomic_numbers[i] = values.next_int();
          i++;
        }
      }
      // Define values in header to use below
      // if (atom->ntypes != values.count()) { 
      //   error->one(FLERR, "Number of atom types != number of labels in the atomic_numbers file.", utils::getsyserror());
      // } 
    } catch (std::exception &e) {
      error->one(FLERR, e.what());
    }
    // Consider Bcast from QMMM package? -CL
    // num = atomic_numbers.size(); // cannot use since this is no longer a vector.
    // MPI_Bcast(&num, 1, MPI_LMP_BIGINT, 0, world);
    // MPI_Bcast((void *) atomic_numbers.data(), num, MPI_LMP_TAGINT, 0, world);
    // int ntypes = atom->ntypes;
    // Check ntypes is equal to number of tag values
  } else error->one(FLERR, "Incorrect number of arguments for command: read_atomic_numbers", utils::getsyserror());
  if (comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

// fix ID all qmhub qm_r_chrg qm_r_spin atomic_number1 atomic_number2 ...
FixQmhub::FixQmhub(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  // for compute_scalar()
  scalar_flag = 1;
  global_freq = 1;
  extscalar   = 1;

  int ntypes = atom->ntypes;
  atomic_numbers = nullptr;
  memory->create(atomic_numbers, ntypes, "fix/qmhub:atomic_numbers");

  if (strcmp(arg[1], "all") != 0) error->all(FLERR, "fix qmhub error: group-ID must be 'all'");
  qm_r_chrg = utils::inumeric(FLERR, arg[3], false, lmp);
  qm_r_spin = utils::inumeric(FLERR, arg[4], false, lmp);
  // Add keywords for charge conservation, charge balance, and H link atom
  qm_atom_index_filename = arg[5];
  // Setting QM atomic numbers for QCHEM input file
  set_atomic_numbers(narg, qm_atom_index_filename, atomic_numbers);

  // Working for debugging QM Atom labels -CL
  // for (int i=0; i <= sizeof(atomic_numbers) / sizeof(int); i++){
  //   printf("QM Atom: %d\n", atomic_numbers[i]);
  // }

  if ((domain->xperiodic == 0) && (domain->yperiodic == 0) && (domain->zperiodic == 0)) is_pbc = 0;
  else if ((domain->xperiodic == 1) && (domain->yperiodic == 1) && (domain->zperiodic == 1)) is_pbc = 1;
  else error->all(FLERR, "fix qmhub error: cell must either be periodic in all directions or not periodic in all directions");
  // for (int i = 0; i < ntypes; i++) {atomic_numbers[i] = utils::inumeric(FLERR, arg[5+i], false, lmp);}

  int igroup_qm = group->find("QM");
  if (igroup_qm == -1) error->all(FLERR, "fix qmhub error: group 'QM' not defined");
  num_qm      = group->count(igroup_qm);
  groupbit_qm = group->bitmask[igroup_qm];

  int igroup_mm = group->find("MM");
  if (igroup_mm == -1) error->all(FLERR, "fix qmhub error: group 'MM' not defined");
  num_mm      = group->count(igroup_mm);
  groupbit_mm = group->bitmask[igroup_mm];

  // Get boundary atoms
  int nlinkatoms = 0; // might need to be ptr for gather?

  // Setup QM-MM boundary terms
  setup_qm_link(nlinkatoms);
  
  // Count number of link atoms needed
  // Conserve Charge
  // Zero out MM1 charge
  // Zero out angle, dihedral, pair, etc.

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
  mask |= MIN_PRE_FORCE;
  mask |= MIN_POST_FORCE;
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
  // get positions, charges, QM atom types, and cell vectors. QMHub system call 
 
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
  // Need to account for link atoms and adjust gradient for QM-MM boundary atoms -CL
  // Tasks for link atom gradient
  // (1) Make qm vectors 3*num + nlink
  // (2) keep boundary indices
  // (3) ??? handle indexing issues
  // (4) adjust gradient

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
/* Setup minimization functions using already defined MD functions.  */

void FixQmhub::min_setup(int vflag)
{
  setup(vflag);
}

void FixQmhub::min_pre_force(int vflag)
{
  post_integrate();
}

void FixQmhub::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQmhub::get_lmp_data(double *qm_coord, double *qm_chrgs, int *qm_types, double *mm_coord, double *mm_chrgs)
{
  // Gather local data and save to full arrays. Full arrays are passed in and out and destroyed in post_integrate()
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
void FixQmhub::setup_qm_link(int nlinkatoms)
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
  
  int **bond_index = atom->bond_atom;
  int nlinkatoms_local = 0; 
  double *qm_chrgs_local = nullptr;
  double *mm_chrgs_local = nullptr;
  
  memory->create(qm_chrgs_local, num_qm_local  , "fix/qmhub:qm_chrgs_local");
  memory->create(mm_chrgs_local, num_mm_local  , "fix/qmhub:mm_chrgs_local");

 // Use qm atoms to get nlinkatoms_local
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) {
      for (int j=0; j < atom->num_bond[i]; j++) {
        if (atom->mask[bond_index[i][j]-1] & groupbit_mm) {
          nlinkatoms_local += 1;
        }
      } 
    }
  }

  int *qm_boundary_idx_local = nullptr;
  int *mm1_boundary_idx_local = nullptr;

  memory->create(qm_boundary_idx_local,  nlinkatoms_local, "fix/qmhub:qm_boundary_idx_local");
  memory->create(mm1_boundary_idx_local, nlinkatoms_local, "fix/qmhub:mm1_boundary_idx_local");

  printf("nlocal %d\n", nlocal);

  int count_qm = 0;
  int count_mm = 0;
  double cluster_q_local = 0.0;
  double cluster_q = 0.0;
  int count_boundary = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) {
      qm_chrgs_local[count_qm] = q[i];
      cluster_q_local += q[i]; // Get charges for QM cluster atoms
      printf("QM_cluster[%d] = %f\n", i, q[i]);
      count_qm++;
      // Start bond search
      // Indexing: i indexes 0, bond_index indexes 1
      for (int j=0; j < atom->num_bond[i]; j++) {
        if (atom->mask[bond_index[i][j]-1] & groupbit_mm) {
          printf("QM1 %d MM1 %d\n", i+1, bond_index[i][j]);
          cluster_q_local += q[bond_index[i][j]-1]; // Charges for MM1 atoms
          printf("MM_cluster[%d] = %f\n", bond_index[i][j]-1, q[bond_index[i][j]-1]);
          qm_boundary_idx_local[count_boundary] = i;
          printf("qm_boundary_idx_local[%d] %d\n", count_boundary, i);
          // need tag?
          // mm1_boundary_idx_local[count_boundary] = atom->tag[bond_index[i][j]-1];
          mm1_boundary_idx_local[count_boundary] = bond_index[i][j]-1;
          printf("mm1_boundary_idx_local[%d] %d\n", count_boundary, bond_index[i][j]-1);
          count_boundary++;
        }
      } // End bond search
    }
    if (atom->mask[i] & groupbit_mm) {
      mm_chrgs_local[count_mm] = q[i];
      count_mm++;
    }
  }

  printf("cluster_q_local sum = %f\n", cluster_q_local);
  int nprocs, root;
  MPI_Comm_size(world, &nprocs); // how many procs are in use
  // Sum nlinkatoms and cluster_q
  MPI_Allreduce(&nlinkatoms_local, &nlinkatoms, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&cluster_q_local, &cluster_q, 1, MPI_DOUBLE, MPI_SUM, world);

  // Get the MM QM charge difference
  double qmmm_delta_q = 0.0;
  qmmm_delta_q = cluster_q - (double) qm_r_chrg;
  printf("Cluster  Charge:   %f\n", cluster_q);
  printf("QM Int   Charge:   %d\n", qm_r_chrg);
  printf("Charge Difference: %f\n", qmmm_delta_q);

  int num_nba = count_mm - nlinkatoms; // N_MM - N_MM1 = N_nonboundary MM atoms
  double redis_charge_local = 0.0;
  double redis_charge = 0.0;
  printf("This system has %d QM link-atoms\n", nlinkatoms);

  // Case where qm_r_chrg = 0 and qmmm_delta_q = 0 but running QMMM (think UFF4MOFF)

  if (fabs(qmmm_delta_q) > 0 && num_nba > 0) { // Avoid divide by 0
    // Send redis_charge to local
    redis_charge = qmmm_delta_q / num_nba;
    printf("redis_charge %f\n", redis_charge);
    MPI_Scatter(&redis_charge, 1, MPI_DOUBLE, &redis_charge_local, 1, MPI_DOUBLE, 0, world);
    // Check if MM and not MM1 : not efficient -CL
    count_mm = 0;
  }

  // MPI_Gatherv(qm_chrgs_local, num_qm_local  , MPI_DOUBLE, qm_chrgs, recv_qm_q, disp_qm_q, MPI_DOUBLE, 0, world);
  // MPI_Gatherv(qm_types_local, num_qm_local  , MPI_INT   , qm_types, recv_qm_t, disp_qm_t, MPI_INT   , 0, world);
  // MPI_Gatherv(mm_chrgs_local, num_mm_local  , MPI_DOUBLE, mm_chrgs, recv_mm_q, disp_mm_q, MPI_DOUBLE, 0, world);

  count_qm = 0;
  count_mm = 0;
  // Change the charges for the atoms class
  for (int i = 0; i < nlocal; i++) {
    // Zero out QM charges
    if (atom->mask[i] & groupbit_qm) {
      atom->q[i] = 0;
      printf("q[%d] = %f\n", i, atom->q[i]);
    }
    // Zero adjust MM charges
    if (atom->mask[i] & groupbit_mm) {
      atom->q[i] += redis_charge; // mm_chrgs_local[count_mm];
      printf("q[%d] = %f\n", i, atom->q[i]);
      count_mm++;
    }
  }
  for (int j=0; j < nlinkatoms_local; j++) {
    atom->q[mm1_boundary_idx_local[j]] = 0;
    printf("qMM1[%d] = %f\n", mm1_boundary_idx_local[j], atom->q[mm1_boundary_idx_local[j]]);
  }

  memory->destroy(qm_chrgs_local);
  memory->destroy(mm_chrgs_local);
  memory->destroy(qm_boundary_idx_local);
  memory->destroy(mm1_boundary_idx_local);

  // To update charges globally, see how positions are updated after timestep

}


/* ---------------------------------------------------------------------- */

// Add SCF Energy (Ha) to thermo via thermo_style custom ... f_ID ...
double FixQmhub::compute_scalar()
{
  return E_SCF;
} 

/* ---------------------------------------------------------------------- */
