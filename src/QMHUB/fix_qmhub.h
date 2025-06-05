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
  
  void setup() override;		// send positions and charges to QMHub
  void post_integrate() override;	//

  void post_force() override;		// receive forces from QMHub

 protected:
  int num_qm;				// Number of QM atoms
  int num_mm;				// Number of MM atoms
  int qm_r_chrg;			// Charge of QM region
  int qm_r_spin;			// Spin mulciplicity of QM region
  int is_pbc;				// 0 -> no, 1 -> yes

  int groupbit_qm;			// Groupbit for region 'QM'
  int groupbit_mm;			// Groupbit for region 'MM'
};

} // namespace LAMMPS_NS

#endif
#endif
