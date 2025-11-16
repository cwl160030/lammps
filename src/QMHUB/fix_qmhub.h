#ifdef FIX_CLASS
// clang-format off
FixStyle(qmhub,FixQmhub);
// clang-format on
#else

#ifndef LMP_FIX_QMHUB_H
#define LMP_FIX_QMHUB_H

#include "fix.h"

namespace LAMMPS_NS {


class FixQmhub : public Fix {
 public:
  FixQmhub(class LAMMPS *, int, char **);
  ~FixQmhub() override;
  int setmask() override;
  void post_constructor() override;
  // Need to pass pointer instead of vector
  // std::vector<int> set_qm_atom_labels(int, char*, std::vector<int>);
  void set_atomic_numbers(int, char*, int*);

  void setup(int) override;		// send positions and charges to QMHub
  void post_integrate() override;	// send positions and charges to QMHub
  void post_force(int) override;	// receive forces from QMHub
  void min_setup(int) override;	
  void min_pre_force(int) override;	// is it necessary since setup calls post_integrate()?
  void min_post_force(int) override;	// receive forces from QMHub

  void get_lmp_data(double *qm_coord,   // get positions, charges, and QM types
                    double *qm_chrgs, 
                    int    *qm_types, 
                    double *mm_coord, 
                    double *mm_chrgs);

  void setup_qm_link(int nlinkatoms);   // Handle setup of QM-MM boundary
  // void setup_qm_link(double *qm_coord,   // Handle setup of QM-MM boundary
  //                   double *qm_chrgs, 
  //                   int    *qm_types, 
  //                   double *mm_coord, 
  //                   double *mm_chrgs);

  double compute_scalar() override;	// For printing to thermo via thermo_style

 protected:
  int num_qm;				// Number of QM atoms
  int num_mm;				// Number of MM atoms
  int qm_r_chrg;			// Charge of QM region
  int qm_r_spin;			// Spin mulciplicity of QM region
  int is_pbc;				// 0 -> no pbc, 1 -> pbc
  int *atomic_numbers;			/* list of atomic numbers of atoms
					   in the simulation; ordered so
					   atomic_numbers[i] corresponds to
					   atom type i */
  char *qm_atom_index_filename; // Pointer to name of file containing QM atomic numbers
  // std::vector<int> qm_atom_labels; // Vector of QM atomic numbers

  int groupbit_qm;			// Groupbit for region 'QM'
  int groupbit_mm;			// Groupbit for region 'MM'

  int nlinkatoms;

  double E_SCF;				// SCF Energy from QM package (Hartree)
};

} // namespace LAMMPS_NS

#endif
#endif
