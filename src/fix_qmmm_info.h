#ifdef FIX_CLASS
// clang-format off
FixStyle(qmmm/info,FixQmmmInfo);
// clang-format on
#else

#ifndef LMP_FIX_QMMM_INFO_H
#define LMP_FIX_QMMM_INFO_H

#include "fix.h"

namespace LAMMPS_NS {

class FixQmmmInfo : public Fix {
 public:
  FixQmmmInfo(class LAMMPS *, int, char **);
  ~FixQmmmInfo() override;
  int setmask() override;
  void end_of_step() override;
};

} // namespace LAMMPS_NS

#endif
#endif
