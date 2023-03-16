#include "gtest_mpi.hpp"

#include "operations.hpp"
#include "cg_solver.hpp"

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <limits>

// Forcing term
double f(double x, double y, double z)
{
  return 0.0;
}

// boundary condition at z=0
double g_0(double x, double y)
{
  return 0.0;
}

stencil3d laplace3d_stencil(int nx, int ny, int nz)
{
  if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
  stencil3d L;
  L.nx=nx; L.ny=ny; L.nz=nz;
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  L.value_c = 2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
  L.value_n = -1.0/(dy*dy);
  L.value_e = -1.0/(dx*dx);
  L.value_s = -1.0/(dy*dy);
  L.value_w = -1.0/(dx*dx);
  L.value_t = -1.0/(dz*dz);
  L.value_b = -1.0/(dz*dz);
  return L;
}


TEST(gc_solver, homogenous) {
  int nx = 16, ny = 16, nz = 16;

  // total number of unknowns
  int n=nx*ny*nz;

  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);

  // Laplace operator
  stencil3d L = laplace3d_stencil(nx,ny,nz);

  // solution vector: start with a 0 vector
  double *x = new double[n];
  init(n, x, 0.0);

  // right-hand side
  double *b = new double[n];
  init(n, b, 0.0);

  // initialize the rhs with f(x,y,z) in the interior of the domain
#pragma omp parallel for schedule(static)
  for (int k=0; k<nz; k++)
  {
    double z = k*dz;
    for (int j=0; j<ny; j++)
    {
      double y = j*dy;
      for (int i=0; i<nx; i++)
      {
        double x = i*dx;
        int idx = L.index_c(i,j,k);
        b[idx] = f(x,y,z);
      }
    }
  }
  // Dirichlet boundary conditions at z=0 (others are 0 in our case, initialized above)
  for (int j=0; j<ny; j++)
    for (int i=0; i<nx; i++)
    {
      b[L.index_c(i,j,0)] -= L.value_b*g_0(i*dx, j*dy);
    }

  // solve the linear system of equations using CG
  int numIter, maxIter=500;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  try {
  cg_solver(&L, n, x, b, tol, maxIter, &resNorm, &numIter);
  } catch(std::exception e)
  {
    std::cerr << "Caught an exception in cg_solve: " << e.what() << std::endl;
    exit(-1);
  }
  double *y = new double[n];
  init(n, y, 0.0);
  
  EXPECT_NEAR(x, y, n*std::numeric_limits<double>::epsilon())
  delete [] x;
  delete [] y;
  delete [] b;

}