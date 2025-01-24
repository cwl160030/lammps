#include "fix_qmmm_info.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "library.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixQmmmInfo::FixQmmmInfo(LAMMPS *lmp, int narg, char **arg) : 
    Fix(lmp, narg, arg)
{
  if (narg < 4) utils::missing_cmd_args(FLERR, "fix qmmm/info", error);
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0)
    error->all(FLERR, "Illegal fix qmmm/info nevery value: {}", nevery);
}

FixQmmmInfo::~FixQmmmInfo(){
  // empty destructor
}

int FixQmmmInfo::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

void FixQmmmInfo::end_of_step()
{
  // global properties (all processors)
  int natoms = atom->natoms;
  double *global_positions = nullptr;
  double *global_charges = nullptr;

  memory->create(global_positions, natoms*3, "fix/qmmm/info:global_positions");
  memory->create(global_charges, natoms, "fix/qmmm/info:global_charges");

  lammps_gather_atoms(lmp, "x", 1, 3, global_positions);
  lammps_gather_atoms(lmp, "q", 1, 1, global_charges);

  if ((comm->me == 0) && screen) {
    fmt::print(screen,"{:^6} {:^21} {:^21} {:^21} {:^21}\n", "ID", "Q", "X", "Y", "Z");
    for(int aID = 0; aID < natoms; aID++) {
      fmt::print(screen,"{:> 6} {:< 21} {:< 21} {:< 21} {:< 21}\n", aID, global_charges[aID], global_positions[aID*3], global_positions[aID*3+1], global_positions[aID*3+2]);
    }
  }
  memory->destroy(global_positions);
  memory->destroy(global_charges);
}
