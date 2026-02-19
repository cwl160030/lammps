#include "fix_qmhub.h"

#include "atom.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "update.h"
#include "tokenizer.h"

#include "modify.h" // needed for adding compute energy? -CL

#include <cmath> // for power in distance

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

#include "pair_lj_cut.h" // testing pair interactions
using namespace LAMMPS_NS;
using namespace FixConst;

/* atom, atom_vec : position, force, molecule index
 * neighbor* : neighbor lists
 * group
 * force
 */

/* Tasks
 * (1) Part  Create charge, bond, angle, dihedral, pair functions to 0 out terms
 *            - Grow arrays instead of using dummy atoms.
 *            - Address deleting pairs and kspace solver warning
 * (2) Part  Charge conservation: setup_qm_link
 *            - Write QM-ref charges to qmmm.inp for charge projection
 *            - Should QM-ref charges receive charges from charge redistribution?
 * (3) Done  Charge redistribution: setup_qm_link
 *
 * (4) Done  Add hydrogen atom to QM calculation: post_integrate
 *
 * (5) ----  Redistribute QM/MM forces: post_force
 *            - Force error is too large!
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

// Need FixQmhub::init() to make neighbor request
void FixQmhub::init()
{
  // Remove neighbor if not needed
  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);
  int nlinkatoms = 0;
  int *qm_boundary_idx = nullptr;
  int *mm1_boundary_idx = nullptr;
  // double *mm1_boundary_charge = nullptr;
}

/* ---------------------------------------------------------------------- */

void FixQmhub::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

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
    // Check ntypes is equal to number of tag values
  } else error->one(FLERR, "Incorrect number of arguments for command: read_atomic_numbers", utils::getsyserror());
  if (comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

// fix ID all qmhub qm_r_chrg qm_r_spin qm_labels.txt
FixQmhub::FixQmhub(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  // for compute_scalar()
  scalar_flag = 1;
  global_freq = 1;
  extscalar   = 1;
  // may be needed for minimization/md
  energy_global_flag = 1;
  // thermo_energy = 1;

  int ntypes = atom->ntypes;
  atomic_numbers = nullptr;
  memory->create(atomic_numbers, ntypes, "fix/qmhub:atomic_numbers");

  if (strcmp(arg[1], "all") != 0) error->all(FLERR, "fix qmhub error: group-ID must be 'all'");
  qm_r_chrg = utils::inumeric(FLERR, arg[3], false, lmp);
  qm_r_spin = utils::inumeric(FLERR, arg[4], false, lmp);
  // Add keywords for charge conservation, charge balance, and H link atom
  qm_atom_index_filename = arg[5];
  // Setting QM atomic numbers for QC input file
  set_atomic_numbers(narg, qm_atom_index_filename, atomic_numbers);

  if ((domain->xperiodic == 0) && (domain->yperiodic == 0) && (domain->zperiodic == 0)) is_pbc = 0;
  else if ((domain->xperiodic == 1) && (domain->yperiodic == 1) && (domain->zperiodic == 1)) is_pbc = 1;
  else error->all(FLERR, "fix qmhub error: cell must either be periodic in all directions or not periodic in all directions");

  int igroup_qm = group->find("QM");
  if (igroup_qm == -1) error->all(FLERR, "fix qmhub error: group 'QM' not defined");
  num_qm      = group->count(igroup_qm);
  groupbit_qm = group->bitmask[igroup_qm];
  double total_qm_charge = group->charge(igroup_qm); // Total QM charge

  int igroup_mm = group->find("MM");
  if (igroup_mm == -1) error->all(FLERR, "fix qmhub error: group 'MM' not defined");
  num_mm      = group->count(igroup_mm);
  groupbit_mm = group->bitmask[igroup_mm];
  double total_mm_charge = group->charge(igroup_mm); // Total MM charge

  // Setup QM-MM boundary terms
  setup_qm_link(nlinkatoms);
  // Setup for QMMM is done.
  
  // Initialize SCF energy for thermo/min
  E_SCF = 0.0; 
}

/* ---------------------------------------------------------------------- */

FixQmhub::~FixQmhub()
{
  memory->destroy(atomic_numbers);
  memory->destroy(qm_boundary_idx);
  memory->destroy(mm1_boundary_idx);
  // memory->destroy(mm1_boundary_charge);
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
  // post_force(); // May change so that run 0 will run QC calculation
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


  // Check QMMM boundary connections

  double xqm, yqm, zqm;
  double delx, dely, delz;
  double linkdist = 1.09; // make readable variable later -CL
  int *tag = atom->tag;
  double **x = atom->x; 
  int linkatom_sym = 1;
  // idx arrays will be global index.

  if (comm->me == 0) {   
    FILE *fp_qmmm_inp = fopen("./qmhub/qmmm.inp", "w");
    if (fp_qmmm_inp == nullptr) error->all(FLERR, "fix qmhub error: cannot open 'qmmm.inp'");
    
    // change is_pbc to frame number (not that important since qmhub.py FIFO doesn't match TXT) -CL
    fprintf(fp_qmmm_inp, "%d %d %d %d %d\n", num_qm+nlinkatoms, num_mm-nlinkatoms, qm_r_chrg, qm_r_spin, is_pbc);
    for (int i = 0; i < num_qm; i++) {
      fprintf(fp_qmmm_inp, "% .15E % .15E % .15E % .15E %d\n", qm_coord[3*i], qm_coord[3*i+1], qm_coord[3*i+2], qm_chrgs[i], atomic_numbers[qm_types[i]-1]);
      printf("Writing  QM  %2d inp  = %8.4f %8.4f %8.4f %8.4f %2d\n", i, qm_coord[3*i], qm_coord[3*i+1], qm_coord[3*i+2], qm_chrgs[i], atomic_numbers[qm_types[i]-1]);
    }
    if (nlinkatoms > 0) {
      // Print X-link atom type and coord into QC input file
      for (int i = 0; i < nlinkatoms; i++) {
        // wrap this into a function to reuse for projecting the link force -CL
        xqm = x[tag[qm_boundary_idx[i]]-1][0]; 
        yqm = x[tag[qm_boundary_idx[i]]-1][1]; 
        zqm = x[tag[qm_boundary_idx[i]]-1][2]; 
        delx = xqm - x[tag[mm1_boundary_idx[i]]-1][0]; 
        dely = yqm - x[tag[mm1_boundary_idx[i]]-1][1]; 
        delz = zqm - x[tag[mm1_boundary_idx[i]]-1][2]; 
        xqm = xqm - linkdist * delx / sqrt(delx * delx + dely * dely + delz * delz);
        yqm = yqm - linkdist * dely / sqrt(delx * delx + dely * dely + delz * delz);
        zqm = zqm - linkdist * delz / sqrt(delx * delx + dely * dely + delz * delz);
        // later change 0.0 to FF charge from data file -CL
        fprintf(fp_qmmm_inp, "% .15E % .15E % .15E % .15E %d\n", xqm, yqm, zqm, 0.0, linkatom_sym);
        printf("Writing  LA  %2d inp  = %8.4f %8.4f %8.4f %8.4f %2d\n", i, xqm, yqm, zqm, 0.0, linkatom_sym);
      }
    }
    // Slow to loop through link atoms for each MM atom... -CL
    int counter = 0; // count MM atoms
    if (nlinkatoms > 0) {
      // i : index all atoms (global)
      for (int i = 0; i < num_mm+num_qm; i++) {
        // if i is MM
        if (atom->mask[i] & groupbit_mm) {
          // j : nlink atom index
          for (int j=0; j < nlinkatoms; j++) {
            // If global index i == global index MM1
            // skip it by incrementing the index
            if (i == mm1_boundary_idx[j]) {
              counter++;
            }
            // If atom is not MM1, write MM info
            else {
              fprintf(fp_qmmm_inp, "% .15E % .15E % .15E % .15E\n", 
                      mm_coord[3*counter], mm_coord[3*counter+1], 
                      mm_coord[3*counter+2], mm_chrgs[counter]);
              printf("Writing  MM  %2d inp  = %8.4f %8.4f %8.4f %8.4f\n", counter, mm_coord[3*counter], 
                     mm_coord[3*counter+1], mm_coord[3*counter+2], mm_chrgs[counter]);
              counter++;
            }
          }
        }
      }
    }
    else {
      for (int i = 0; i < num_mm; i++) {
        fprintf(fp_qmmm_inp, "% .15E % .15E % .15E % .15E\n", 
                mm_coord[3*i], mm_coord[3*i+1], 
                mm_coord[3*i+2], mm_chrgs[i]);
      }
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
  // Read gradients from qmmm.out
  // Need to account for link atoms and adjust gradient for QM-MM boundary atoms -CL
  double *qm_grad = nullptr;
  double *mm_grad = nullptr;
  double *link_grad = nullptr;
  if (comm->me == 0) {
    memory->create(qm_grad, 3*num_qm, "fix/qmhub:qm_grad");
    memory->create(mm_grad, 3*num_mm, "fix/qmhub:mm_grad");
    memory->create(link_grad, 3*nlinkatoms, "fix/qmhub:link_grad");

    FILE *fp_qmmm_out = fopen("./qmhub/qmmm.out", "r");
    if (fp_qmmm_out == nullptr) error->all(FLERR, "fix qmhub error: cannot open 'qmmm.out'");

    fscanf(fp_qmmm_out, "%lf", &E_SCF);
    // Read QM forces (always assume there are QM atoms)
    for (int i = 0; i < num_qm; i++) {
      fscanf(fp_qmmm_out, "%lf %lf %lf", &qm_grad[3*i], &qm_grad[3*i+1], &qm_grad[3*i+2]);
      printf("Reading QM  %2d grad = %15.8f %15.8f %15.8f\n", i, qm_grad[3*i],
             qm_grad[3*i+1], qm_grad[3*i+2]);
    }
    // Read link atom forces if present
    if (nlinkatoms > 0) {
      for (int i = 0; i < nlinkatoms; i++) {
        fscanf(fp_qmmm_out, "%lf %lf %lf", &link_grad[3*i], &link_grad[3*i+1], &link_grad[3*i+2]);
        printf("Reading LA  %2d grad = %15.8f %15.8f %15.8f\n", i, link_grad[3*i],
               link_grad[3*i+1], link_grad[3*i+2]);
      }
    }
    // If link atoms present, there will be MM atoms missing in qmmm.out (MM1 atoms)
    int counter = 0; // Count MM atoms, be able to skip link atoms
    if (nlinkatoms > 0) {
      // loop all atom global index
      for (int i = 0; i < num_mm+num_qm; i++) {
        // if global index is MM atom
        if (atom->mask[i] & groupbit_mm) {
          // loop link atoms (global)
          for (int j=0; j < nlinkatoms; j++) {
            // if MM1 atom, zero grad, will receive f_FF and f_link later
            if (i == mm1_boundary_idx[j]) {
              // The total f_MM1 = f_FF_MM1 + 0*f_QMMM_MM1 + f_link_MM1
              printf("Reading MM1 %2d grad = %15.8f %15.8f %15.8f\n", counter, 0.0, 0.0, 0.0);
              mm_grad[3*counter]   = 0.0;
              mm_grad[3*counter+1] = 0.0;
              mm_grad[3*counter+2] = 0.0;
              counter++;
            }
            else {
              // Read MM gradient
              fscanf(fp_qmmm_out, "%lf %lf %lf", &mm_grad[3*counter], 
                     &mm_grad[3*counter+1], &mm_grad[3*counter+2]);
              printf("Reading MM  %2d grad = %15.8f %15.8f %15.8f\n", counter, mm_grad[3*counter],
                     mm_grad[3*counter+1], mm_grad[3*counter+2]);
              counter++;
            }
          }
        }
      }
    }
    else {
      for (int i = 0; i < num_mm; i++) {
        fscanf(fp_qmmm_out, "%lf %lf %lf", &mm_grad[3*i], &mm_grad[3*i+1], &mm_grad[3*i+2]);
      }
    }

    fclose(fp_qmmm_out);
  }

  // send global gradients to local
  int nlocal = atom->nlocal;
  double *qm_grad_local = nullptr;
  double *mm_grad_local = nullptr;
  double *link_grad_proj = nullptr;
  
  int num_qm_local = 0;
  int num_mm_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) num_qm_local++;
    if (atom->mask[i] & groupbit_mm) num_mm_local++;
  }

  memory->create(qm_grad_local, 3*num_qm_local, "fix/qmmm:qm_grad_local");
  memory->create(mm_grad_local, 3*num_mm_local, "fix/qmmm:mm_grad_local");  
  // link atom grad updated every pair so only need array of 3 double
  memory->create(link_grad_proj, 3, "fix/qmmm:link_grad_proj");  

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

  const double HABOHR_KCALMOLA = (627.5094740631) / (0.529177210544);
  int count_qm = 0;
  int count_mm = 0;
  for (int i = 0; i < nlocal; i++) {
    // Add forces for all QM atoms
    if (atom->mask[i] & groupbit_qm) {
      for (int dim = 0; dim < 3; dim++) {
        atom->f[i][dim] -= HABOHR_KCALMOLA * qm_grad_local[3*count_qm+dim];
      }
      count_qm++;
      if (nlinkatoms > 0) {
        // calculate QM portion of link atom force
        for (int j=0; j < nlinkatoms; j++) {
          // i is local index. map gives local index, j is link atom index
          if (atom->tag[i]-1 == qm_boundary_idx[j]) {
            // QM atom is link and local! Do thing
            // j can index qm_boundary_idx and link_grad
            link_atom_force_method(qm_boundary_idx[j], mm1_boundary_idx[j], 
                    link_grad[3*j], link_grad[3*j+1], link_grad[3*j+2], link_grad_proj);
            printf("\n f_qm   = %15.8f %15.8f %15.8f\n", atom->f[i][0]/HABOHR_KCALMOLA, 
                    atom->f[i][1]/HABOHR_KCALMOLA, atom->f[i][2]/HABOHR_KCALMOLA);
            printf(" f_link = %15.8f %15.8f %15.8f\n", (-1)*link_grad[3*j+0], (-1)*link_grad[3*j+1], (-1)*link_grad[3*j+2]);
            printf(" f_proj = %15.8f %15.8f %15.8f\n", (-1)*link_grad_proj[3*j+0], (-1)*link_grad_proj[3*j+1], (-1)*link_grad_proj[3*j+2]);
            for (int dim=0; dim < 3; dim++) {
              // QM-MM link grad
              // Chain rule similar to Amber
              atom->f[i][dim] -= HABOHR_KCALMOLA *(link_grad[3*j+dim] - link_grad_proj[dim]);
            }
            printf("\n*f_qm   = f_qm + f_link - f_proj\n");
            printf("*f_qm   = %15.8f %15.8f %15.8f\n", atom->f[i][0]/HABOHR_KCALMOLA, 
                    atom->f[i][1]/HABOHR_KCALMOLA, atom->f[i][2]/HABOHR_KCALMOLA);
          }
        }
      }
    }
    // Add MM forces (MM1 QMMM forces are already set to 0)
    if (atom->mask[i] & groupbit_mm) {
      for (int dim = 0; dim < 3; dim++) {
        atom->f[i][dim] -= HABOHR_KCALMOLA * mm_grad_local[3*count_mm+dim];
      }
      if (nlinkatoms > 0) {
        // calculate MM portion of link atom force...
        for (int j=0; j < nlinkatoms; j++) {
          if (atom->tag[i]-1 == mm1_boundary_idx[j]) {
            link_atom_force_method(qm_boundary_idx[j], mm1_boundary_idx[j], 
                    link_grad[3*j], link_grad[3*j+1], link_grad[3*j+2], link_grad_proj);
            printf("\n f_mm   = %15.8f %15.8f %15.8f\n", atom->f[i][0]/HABOHR_KCALMOLA, 
                    atom->f[i][1]/HABOHR_KCALMOLA, atom->f[i][2]/HABOHR_KCALMOLA);
            printf(" f_link = %15.8f %15.8f %15.8f\n", (-1)*link_grad[3*j+0], (-1)*link_grad[3*j+1], (-1)*link_grad[3*j+2]);
            printf(" f_proj = %15.8f %15.8f %15.8f\n", (-1)*link_grad_proj[3*j+0], (-1)*link_grad_proj[3*j+1], (-1)*link_grad_proj[3*j+2]);
            for (int dim=0; dim < 3; dim++) {
              // MM-QM link grad
              // Chain rule similar to Amber
              atom->f[i][dim] -= HABOHR_KCALMOLA * link_grad_proj[dim];
            }
            printf("\n*f_mm   = f_mm + f_proj\n");
            printf("*f_mm   = %15.8f %15.8f %15.8f\n", atom->f[i][0]/HABOHR_KCALMOLA, 
                    atom->f[i][1]/HABOHR_KCALMOLA, atom->f[i][2]/HABOHR_KCALMOLA);
            printf("\n");
          }
        }
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
  memory->destroy(link_grad);
  memory->destroy(link_grad_proj);
}

/* ---------------------------------------------------------------------- */
/* Setup minimization functions using already defined MD functions.  */

void FixQmhub::min_setup(int vflag)
{
  // setup(vflag);
  printf("min_setup\n");
  post_force(vflag);
}

void FixQmhub::min_pre_force(int vflag)
{
  // This writes the qmmm.inp file before force calculation
  printf("min_pre_force\n");
  post_integrate(); // maybe rename? -CL 
}

void FixQmhub::min_post_force(int vflag)
{
  // This reads the qmmm.out file after force calculation
  printf("min_post_force\n");
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


void FixQmhub::set_qmmm_charges(int nlinkatoms, int count_nlink_local, double mm1_charges_local, int *qm_boundary_idx_local, int *mm1_boundary_idx_local)
{
  // Zero out QM and MM1 charges and redistribute this charge to non QMMM1 atoms.
  // The charge to be redistributed q_red = q_qm + q_mm1 - q_qm(int)
  int nlocal = atom->nlocal;
  int num_non_qmmm1_atoms = 0;
  double mm1_charges = 0.0;
  double qmmm_delta_q = 0.0;
  double redis_charge = 0.0; 

  int nprocs, root;
  MPI_Comm_size(world, &nprocs); // how many procs are in use
  MPI_Allreduce(&mm1_charges_local, &mm1_charges, 1, MPI_DOUBLE, MPI_SUM, world);

  qmmm_delta_q = total_qm_charge + mm1_charges - (double) qm_r_chrg;

  printf("Cluster  Charge:   %f\n", total_qm_charge); // g
  printf("MM1      Charge:   %f\n", mm1_charges); // g
  printf("QM Int   Charge:   %d\n", qm_r_chrg); // g
  printf("Charge Difference: %f\n", qmmm_delta_q); // pass
  printf("This system has %d QM link-atoms\n", nlinkatoms); // g

  num_non_qmmm1_atoms = num_mm - nlinkatoms;
  if (num_non_qmmm1_atoms > 0) redis_charge = qmmm_delta_q / num_non_qmmm1_atoms;

  // Change the charges for the atoms class
  for (int i = 0; i < nlocal; i++) {
    // Zero out QM charges
    if (atom->mask[i] & groupbit_qm) {
      atom->q[i] = 0;
    }
    // Adjust MM charges
    if (atom->mask[i] & groupbit_mm) {
      atom->q[i] += redis_charge;
    }
  }
  // Zero out MM1 charges
  for (int j=0; j < count_nlink_local; j++) {
    atom->q[mm1_boundary_idx_local[j]] = 0;
    printf("qMM1[%d] = %f\n", mm1_boundary_idx_local[j], atom->q[mm1_boundary_idx_local[j]]);
  }

  // Just for debugging
  for (int i = 0; i < nlocal; i++) {
    // Adjusted QM charges
    if (atom->mask[i] & groupbit_qm) {
      printf("Q[%d] = %f\n", i, atom->q[i]);
    }
    // Adjusted MM charges
    if (atom->mask[i] & groupbit_mm) {
      printf("Q[%d] = %f\n", i, atom->q[i]);
    }
  }
}


/* ---------------------------------------------------------------------- */
void FixQmhub::zero_qmmm_bonds()
{
  // We can avoid neighbors here because we are directly changing atoms 
  // (1) QM-QM bonds are zero
  int nlocal = atom->nlocal;
  int **bond_type = atom->bond_type;
  int nbondtypes = atom->nbondtypes;
  int nbonds = atom->nbonds;
  int *num_bond = atom->num_bond;

  // There is no check that nbondtypes[-1] is 0. This just 
  // prints bondtype and hopes you pay attention... -CL
  //
  // tag returns index from 1, bond_atom returns index from 1
  for (int i=0; i < nlocal; i++) {
    for (int j=0; j < num_bond[i]; j++) {
      // Only zero out QM-QM bonds
      if ((atom->mask[i] & groupbit_qm) && 
          (atom->mask[atom->bond_atom[i][j]-1] & groupbit_qm)) {
        printf("Change bond type for %2d %2d %2d -> %2d\n", 
            atom->tag[i], atom->bond_atom[i][j], bond_type[i][j], nbondtypes);
            atom->bond_type[i][j] = nbondtypes; // Set to dummy bond type
      }
    }
  }
}


/* ---------------------------------------------------------------------- */
void FixQmhub::zero_qmmm_angles(int num_qmmm_ratio_angle)
{
  // This zeros out QMMM angles
  // Amber style: Keep any angle with an MM atom
  // Gromacs style: Keep any angle with 1 or more MM atoms
  int nlocal = atom->nlocal;
  int **angle_type = atom->angle_type;
  int nangletypes = atom->nangletypes;
  int nangles = atom->nangles;
  int *num_angle = atom->num_angle;

  int count_qm_angle_atom;
  int **angle_atom1 = atom->angle_atom1;
  int **angle_atom2 = atom->angle_atom2;
  int **angle_atom3 = atom->angle_atom3;
  // Amber style QMMM angles (only delete QM-QM-QM)
  if (num_qmmm_ratio_angle == 3) {
    for (int i=0; i < nlocal; i++) {
      count_qm_angle_atom = 0;
      for (int j=0; j < num_angle[i]; j++) {
        if ((atom->mask[angle_atom1[i][j]-1] & groupbit_qm) &&
            (atom->mask[angle_atom2[i][j]-1] & groupbit_qm) && 
            (atom->mask[angle_atom3[i][j]-1] & groupbit_qm)) {
          printf("Change angle type for %2d %2d %2d %2d -> %2d\n",
          angle_atom1[i][j], angle_atom2[i][j], angle_atom3[i][j],
          angle_type[i][j], nangletypes); 
          // Set to dummy angle type
          atom->angle_type[i][j] = nangletypes;
        } 
      }
    }
  }
}

/* ---------------------------------------------------------------------- */
void FixQmhub::zero_qmmm_dihedrals(int num_qmmm_ratio_dihedral)
{
  // This zeros out QMMM dihedrals
  // Amber style: Keep any dihedral with an MM atom
  // Gromacs style: Keep any dihedral with 1 or more MM atoms
  int nlocal = atom->nlocal;
  int **dihedral_type = atom->dihedral_type;
  int ndihedraltypes = atom->ndihedraltypes;
  int ndihedrals = atom->ndihedrals;
  int *num_dihedral = atom->num_dihedral;

  int count_qm_dihedral_atom;
  int **dihedral_atom1 = atom->dihedral_atom1;
  int **dihedral_atom2 = atom->dihedral_atom2;
  int **dihedral_atom3 = atom->dihedral_atom3;
  int **dihedral_atom4 = atom->dihedral_atom4;
  // Amber style QMMM dihedrals (only delete QM-QM-QM)
  if (num_qmmm_ratio_dihedral == 4) {
    for (int i=0; i < nlocal; i++) {
      count_qm_dihedral_atom = 0;
      for (int j=0; j < num_dihedral[i]; j++) {
        if ((atom->mask[dihedral_atom1[i][j]-1] & groupbit_qm) &&
            (atom->mask[dihedral_atom2[i][j]-1] & groupbit_qm) && 
            (atom->mask[dihedral_atom3[i][j]-1] & groupbit_qm) && 
            (atom->mask[dihedral_atom4[i][j]-1] & groupbit_qm)) {
          printf("Change dihedral type for %2d %2d %2d %2d %2d -> %2d\n",
          dihedral_atom1[i][j], dihedral_atom2[i][j],
          dihedral_atom3[i][j], dihedral_atom4[i][j],
          dihedral_type[i][j], ndihedraltypes); 
          // Set to dummy dihedral type
          atom->dihedral_type[i][j] = ndihedraltypes;
        } 
      }
    }
  }
}

/* ---------------------------------------------------------------------- */
void FixQmhub::zero_qmmm_impropers()
{
  // This zeros out QMMM impropers if all atoms are in improper list.
  // This assumes we never cut a conjugated bond. While standard,
  // to avoid,there should be a warning if done.
  int nlocal = atom->nlocal;
  int **improper_type = atom->improper_type;
  int nimpropertypes = atom->nimpropertypes;
  int nimpropers = atom->nimpropers;
  int *num_improper = atom->num_improper;

  int count_qm_improper_atom;
  int **improper_atom1 = atom->improper_atom1;
  int **improper_atom2 = atom->improper_atom2;
  int **improper_atom3 = atom->improper_atom3;
  int **improper_atom4 = atom->improper_atom4;
  // Amber style QMMM impropers (only delete QM-QM-QM)
  for (int i=0; i < nlocal; i++) {
    count_qm_improper_atom = 0;
    for (int j=0; j < num_improper[i]; j++) {
      if ((atom->mask[improper_atom1[i][j]-1] & groupbit_qm) &&
          (atom->mask[improper_atom2[i][j]-1] & groupbit_qm) && 
          (atom->mask[improper_atom3[i][j]-1] & groupbit_qm) && 
          (atom->mask[improper_atom4[i][j]-1] & groupbit_qm)) {
        printf("Change improper type for %2d %2d %2d %2d %2d -> %2d\n",
        improper_atom1[i][j], improper_atom2[i][j],
        improper_atom3[i][j], improper_atom4[i][j],
        improper_type[i][j], nimpropertypes); 
        // Set to dummy improper type
        atom->improper_type[i][j] = nimpropertypes;
      } 
    }
  }
}

/* ---------------------------------------------------------------------- */
void FixQmhub::setup_qm_link(int nlinkatoms)
{
  int nlocal = atom->nlocal;
  int max_nlinkatoms = num_qm; // At most 1 link atom for a QM atom... 
  int count_nlink_local = 0;
  double mm1_charges_local = 0.0;

  int **bond_index = atom->bond_atom;
  int *qm_boundary_idx_local = nullptr;
  int *mm1_boundary_idx_local = nullptr;
  // double *mm1_boundary_charge_local = nullptr;
  memory->create(qm_boundary_idx_local,  max_nlinkatoms, "fix/qmhub:qm_boundary_idx_local");
  memory->create(mm1_boundary_idx_local, max_nlinkatoms, "fix/qmhub:mm1_boundary_idx_local");
  // memory->create(mm1_boundary_charge_local, max_nlinkatoms, "fix/qmhub:mm1_boundary_charge_local");

  // Get number of link atoms in local and add to local indexing arrays
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit_qm) {
      for (int j=0; j < atom->num_bond[i]; j++) {
        if (atom->mask[bond_index[i][j]-1] & groupbit_mm) {
          qm_boundary_idx_local[count_nlink_local] = i;
          mm1_boundary_idx_local[count_nlink_local] = bond_index[i][j]-1;
          // mm1_boundary_charge_local[count_nlink_local] = atom->q[i];
          mm1_charges_local += atom->q[bond_index[i][j]-1];
          count_nlink_local++;
        }
      } 
    }
  }

  // Leave nprocs and root here? better way to handle MPI variables? -CL
  int nprocs, root;
  MPI_Comm_size(world, &nprocs); // how many procs are in use
  // Sum nlinkatoms
  MPI_Allreduce(&count_nlink_local, &nlinkatoms, 1, MPI_INT, MPI_SUM, world);

  int *count_nlink_all = nullptr;
  if (comm->me == 0) {
    memory->create(count_nlink_all, nprocs, "fix/qmhub:count_nlink_all");
  }
  MPI_Gather(&count_nlink_local, 1, MPI_INT, count_nlink_all, 1, MPI_INT, 0, world);
  // MPI terminoology for this part: send local receive global
  //
  int *recv_nlink = nullptr;
  int *disp_nlink = nullptr;
  if (comm->me == 0) {
    memory->create(recv_nlink, nprocs, "fix/qmhub:recv_nlink");
    memory->create(disp_nlink, nprocs, "fix/qmhub:disp_nlink");
    for (int i = 0; i < nprocs; i++){
      recv_nlink[i] = count_nlink_all[i]; // number of elements recieved per core 
    }
    disp_nlink[0] = 0;
    for (int i = 1; i < nprocs; i++) {
      disp_nlink[i] = disp_nlink[i-1] + recv_nlink[i-1];
    }
  }

  // Probably better to gather -> make sorted shortlist -> bcast
  // gather may already sort so no need for buffer?

  memory->create(qm_boundary_idx,  max_nlinkatoms, "fix/qmhub:qm_boundary_idx");
  memory->create(mm1_boundary_idx, max_nlinkatoms, "fix/qmhub:mm1_boundary_idx");
  // memory->create(mm1_boundary_charge, max_nlinkatoms, "fix/qmhub:mm1_boundary_charge");

  // Buffers?
  // int *buf_qm_boundary_idx =  nullptr;
  // int *buf_mm1_boundary_idx = nullptr;
  // memory->create(buf_qm_boundary_idx,  nlinkatoms, "fix/buf_qmhub:qm_boundary_idx");
  // memory->create(buf_mm1_boundary_idx, nlinkatoms, "fix/buf_qmhub:mm1_boundary_idx");

  MPI_Gatherv(qm_boundary_idx_local, // buffer send
              count_nlink_local, // count send
              MPI_INT, // send datatype
              qm_boundary_idx, // buffer recv
              recv_nlink, // recv count
              disp_nlink, // disp index
              MPI_INT, // recv datatype
              0, // send this data to root 0 proc
              world); // world communicator

  MPI_Gatherv(mm1_boundary_idx_local, // buffer send
              count_nlink_local, // count send
              MPI_INT, // send datatype
              mm1_boundary_idx, // buffer recv
              recv_nlink, // recv count
              disp_nlink, // disp index
              MPI_INT, // recv datatype
              0, // send this data to root 0 proc
              world); // world communicator

  // MPI_Gatherv(mm1_boundary_charge_local, // buffer send
  //             count_nlink_local, // count send
  //             MPI_INT, // send datatype
  //             mm1_boundary_charge, // buffer recv
  //             recv_nlink, // recv count
  //             disp_nlink, // disp index
  //             MPI_INT, // recv datatype
  //             0, // send this data to root 0 proc
  //             world); // world communicator

  MPI_Bcast(qm_boundary_idx, // buffer
            max_nlinkatoms, // count ? (might be too big?)
            MPI_INT, // datatype
            0, // broadcast root from proc 0
            world); // world communicator

  MPI_Bcast(mm1_boundary_idx, // buffer
            max_nlinkatoms, // count ? (might be too big?)
            MPI_INT, // datatype
            0, // broadcast root from proc 0
            world); // world communicator

  // MPI_Bcast(mm1_boundary_charge, // buffer
  //           max_nlinkatoms, // count ? (might be too big?)
  //           MPI_INT, // datatype
  //           0, // broadcast root from proc 0
  //           world); // world communicator

  // if (comm->me == 0) {
  //   memory->destroy(count_qm_all);
  // }
  // Destroy things?


  if (nlinkatoms > 0) {
    // Later: add ability to choose Amber or GROMACS style for handling boundary -CL
    // Charge balancing and redistribution
    set_qmmm_charges(nlinkatoms, count_nlink_local, mm1_charges_local, qm_boundary_idx_local, mm1_boundary_idx_local);
  }
  // Always Zero out QM Pair Interactions (LJ/Coul/etc.)
  // for now, change input and later see neigh_modify!
  // *** THIS ONLY ZEROS OUT QM-QM PAIR-INTERACTION IN CELL 0
  //
  // Excluding pairwise interactions will not work correctly when also using a 
  // long-range solver via the kspace_style command. LAMMPS will give a warning 
  // to this effect. This is because the short-range pairwise interaction needs 
  // to subtract off a term from the total energy for pairs whose short-range 
  // interaction is excluded, to compensate for how the long-range solver 
  // treats the interaction. This is done correctly for pairwise interactions 
  // that are excluded (or weighted) via the special_bonds command. But it is 
  // not done for interactions that are excluded via these neigh_modify exclude 
  // options.
  // See: https://docs.lammps.org/neigh_modify.html
  //
  // zero_qmmm_pair_coeff();

  // Always Zero out QMMM bonds
  // Zero QMMM1 bonds
  zero_qmmm_bonds();

  // If link atoms exist, zero out angles, dihedrals, impropers
  if (nlinkatoms > 0) {
    // Zero QMMM1 angles
    if (atom->nangletypes > 0) {
    int num_qmmm_ratio_angle = 3; // Amber
    zero_qmmm_angles(num_qmmm_ratio_angle);
    }
    // Zero QMMM1 dihedrals
    printf("ndihedraltypes %d\n", atom->ndihedraltypes);
    if (atom->ndihedraltypes > 0) {
      int num_qmmm_ratio_dihedral = 4; // Amber
      zero_qmmm_dihedrals(num_qmmm_ratio_dihedral);
    }
    // Zero QMMM1 impropers
    printf("nimpropertypes %d\n", atom->nimpropertypes);
    if (atom->nimpropertypes > 0) {
      zero_qmmm_impropers();
    }
  }

  // Destroy things...

}

/* ---------------------------------------------------------------------- */
// Distribute QM-MM link atom boundary force
void FixQmhub::link_atom_force_method(int qm_idx, int mm1_idx, double link_gradx,
     double link_grady, double link_gradz, double *link_grad_proj)
{
  // qm_idx is qm boundary index (global),  mm1_idx is mm1 boundary index (global)
  // ASSUMPTION: Since QM-MM1 are bound, should always be in neighbor list,
  // so if one is on proc, other should at least be ghost...
  double x_lx, x_ly, x_lz;
  double x_qmx, x_qmy, x_qmz;
  double r_qmx, r_qmy, r_qmz;
  double r_qlx, r_qly, r_qlz;
  int *tag = atom->tag;
  double **x = atom->x; 
  double norm_ql, norm_qm, f_qlqm, dotprod;
  double unit_qmx, unit_qmy, unit_qmz;
  double linkdist = 1.09; // Fix hardcoding, pass or global -CL
  // QM position
  x_qmx = x[tag[qm_idx]-1][0];
  x_qmy = x[tag[qm_idx]-1][1];
  x_qmz = x[tag[qm_idx]-1][2];
  // R_qm
  r_qmx  = x_qmx - x[tag[mm1_idx]-1][0]; 
  r_qmy  = x_qmy - x[tag[mm1_idx]-1][1]; 
  r_qmz  = x_qmz - x[tag[mm1_idx]-1][2]; 
  // norm(R_qm)
  norm_qm = sqrt(r_qmx * r_qmx + r_qmy * r_qmy + r_qmz * r_qmz);
  // L position
  x_lx = x_qmx - linkdist * r_qmx / norm_qm;
  x_ly = x_qmy - linkdist * r_qmy / norm_qm;
  x_lz = x_qmz - linkdist * r_qmz / norm_qm;
  // R_ql
  r_qlx  = x_qmx - x_lx; 
  r_qly  = x_qmy - x_ly; 
  r_qlz  = x_qmz - x_lz; 
  // norm(R_ql)
  // Note: norm_ql = linkdist... change later -CL
  norm_ql = sqrt(r_qlx * r_qlx + r_qly * r_qly + r_qlz * r_qlz);
  // ql/qm ratio | dimensionless
  f_qlqm = norm_ql / norm_qm;
  // qm unit vector
  unit_qmx = r_qmx / norm_qm;
  unit_qmy = r_qmy / norm_qm;
  unit_qmz = r_qmz / norm_qm;
  // g_link cdot qm unit vector
  dotprod = link_gradx * unit_qmx + link_grady * unit_qmy + link_gradz * unit_qmz;
  // get projected gradient correction
  // Note link_grad is only term with units
  link_grad_proj[0] = f_qlqm * (link_gradx - dotprod * unit_qmx);
  link_grad_proj[1] = f_qlqm * (link_grady - dotprod * unit_qmx);
  link_grad_proj[2] = f_qlqm * (link_gradz - dotprod * unit_qmx);
}
/* ---------------------------------------------------------------------- */
// Add SCF Energy (Ha) to thermo via thermo_style custom ... f_ID ...
double FixQmhub::compute_scalar()
{
  return E_SCF * 627.5094740631;
} 

/* ---------------------------------------------------------------------- */

//
// auto req = neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);
// if (cutflag) req->set_cutoff(mycutneigh);

// neighbor->build_one(list);
// int **bondlist = list->bondlist;
// int nbondlist = list->nbondlist;

// per atom.cpp: int *num_bond, int **bond_type, tagint **bond_atom
// add_peratom("num_bond",&num_bond,INT,0);
// add_peratom_vary("bond_type",&bond_type,INT,&bond_per_atom,&num_bond);
// add_peratom_vary("bond_atom",&bond_atom,tagintsize,&bond_per_atom,&num_bond);

