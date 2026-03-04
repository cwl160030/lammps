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
  void init() override;
  void init_list(int, class NeighList *) override;
  int setmask() override;
  void post_constructor() override;
  void set_atomic_numbers(int, char*, int*);

  void setup(int) override;		    // send positions and charges to QMHub
  void setup_pre_force(int) override;	
  void post_integrate() override;	// send positions and charges to QMHub
  void pre_force(int) override;	    // R/W qmmm.inp qmmm.out and update E&F before run 0
  void qmmm_force();                // receive forces from QMHub
  void min_setup(int) override;	
  void min_setup_pre_force(int);	
  void min_pre_force(int) override;

  void get_lmp_data(double *qm_coord,   // get positions, charges, and QM types
                    double *qm_chrgs, 
                    int    *qm_types, 
                    double *mm_coord, 
                    double *mm_chrgs);

  void set_qmmm_charges(int nlinkatoms,
                        int count_nlink_local,
                        double mm1_charges_local, 
                        int *qm_boundary_idx_local,
                        int *mm1_boundary_idx_local);

  void zero_qmmm_pair_coeff();
  void zero_qmmm_bonds();
  void zero_qmmm_angles(int num_qmmm_ratio_angle);
  void zero_qmmm_dihedrals(int num_qmmm_ratio_dihedral);
  void zero_qmmm_impropers();

  void setup_qm_link(); // int nlinkatoms);   // Handle setup of QM-MM boundary

  void link_atom_force_method(int qm_idx,
                              int mm1_idx,
                              double link_gradx,
                              double link_grady,
                              double link_gradz,
                              double *link_grad_proj);

  double compute_scalar() override;	// For printing to thermo via thermo_style

  int nlinkatoms;

 // protected:
  int num_qm;				// Number of QM atoms
  int num_mm;				// Number of MM atoms
  double total_qm_charge;   // Summed charge of QM atoms using FF charges
  double total_mm_charge;   // Summed charge of MM atoms using FF charges
  int qm_r_chrg;			// Charge of QM region
  int qm_r_spin;			// Spin mulciplicity of QM region
  int is_pbc;				// 0 -> no pbc, 1 -> pbc
  int *atomic_numbers;      // List atomtype atom numbers for QMHub

  char *qm_atom_index_filename; // Pointer to name of file containing QM atomic numbers
  int *qm_boundary_idx;         // Array of indices for QM atoms bound to MM1 atoms
  int *mm1_boundary_idx;        // Array of indices mapping global MM1 index to 0:num_mm index
  int *mm1_boundary_mapped;     // Array of indices for MM1 atoms bound to QM atoms
  double *qm_qmmm_charge;       // Array of initial QM FF charges
  double *mm1_qmmm_charge;      // Array of initial MM1 FF charges
  // double *mm1_boundary_charge;  // Array of charges for MM1 atom needed for qmmm.inp/out

  int igroup_qm;			// Groupbit Int     for region 'QM'
  int igroup_mm;			// Groupbit Int     for region 'MM'
  int groupbit_qm;			// Groupbit bitMask for region 'QM'
  int groupbit_mm;			// Groupbit bitMask for region 'MM'

  double E_SCF;				// SCF Energy from QM package (kcal/mol)

  private:
  class NeighList *list;
};

} // namespace LAMMPS_NS

#endif
#endif
