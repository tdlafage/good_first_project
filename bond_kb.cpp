/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "bond_kb.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondKB::BondKB(LAMMPS *lmp) : Bond(lmp)
{
  reinitflag = 1;
}

/* ---------------------------------------------------------------------- */

BondKB::~BondKB()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(ep);
    memory->destroy(sigma);
  }
}

/* ---------------------------------------------------------------------- */

void BondKB::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,r2inv,r6inv,sigma2,sigma6;

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
 
    sigma2 = sigma[type]*sigma[type];
    sigma6 = sigma2*sigma2*sigma2;
    r2inv = 1.0/rsq;
    r6inv = r2inv*r2inv*r2inv;
    
        // force & energy

    if (rsq > 0.0) 
      {
	fbond = ep[type] * r6inv * r2inv * sigma6 * (156.0 * sigma6 * r6inv - 180.0 *sigma2*sigma2*r2inv*r2inv + 24.0);
      }
    else fbond = 0.0;

    if (eflag) {
      ebond = ep[type] * r6inv * sigma6 * (13.0 * r6inv * sigma6 - 18.0*sigma2*sigma2*r2inv*r2inv + 4.0);
	}

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondKB::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(ep,n+1,"bond:ep");
  memory->create(sigma,n+1,"bond:sigma");

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondKB::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  double ep_one = force->numeric(FLERR,arg[1]);
  double sigma_one = force->numeric(FLERR,arg[2]);


  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    ep[i] = ep_one;
    sigma[i] = sigma_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondKB::equilibrium_distance(int i)
{
  return sigma[i]*1.1224;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondKB::write_restart(FILE *fp)
{
  fwrite(&ep[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&sigma[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondKB::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&ep[1],sizeof(double),atom->nbondtypes,fp);
    fread(&sigma[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&ep[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&sigma[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondKB::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g\n",i,ep[i],sigma[i]);
}

/* ---------------------------------------------------------------------- */

double BondKB::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  double r2inv,r6inv,forcelj,philj,sigma2,sigma6;

  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  sigma2 = sigma[type]*sigma[type];
  sigma6 = sigma2*sigma2*sigma2;
  
  forcelj = ep[type] * r6inv * r2inv * sigma6 * (156.0 * sigma6 * r6inv - 180.0 *sigma2*sigma2*r2inv*r2inv + 24);;


  fforce = 0;
  if (rsq > 0.0) fforce = forcelj*r2inv;
  return ep[type] * r6inv * sigma6 * (13.0 * r6inv *sigma6 - 18.0*sigma2*sigma2*r2inv*r2inv + 4.0);

}

/* ----------------------------------------------------------------------
    Return ptr to internal members upon request.
------------------------------------------------------------------------ */
void *BondKB::extract( char *str, int &dim )
{
  dim = 1;
  if( strcmp(str,"epsilon")==0) return (void*) ep;
  if( strcmp(str,"sigma")==0) return (void*) sigma;
  return NULL;
}


