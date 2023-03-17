#include "operations.hpp"
#include "cg_solver.hpp"
#include "timer.hpp"

#include <iostream>
#include <cmath>
#include <limits>

double f(double x, double y, double z)
{
  return z*sin(2*M_PI*x)*std::sin(M_PI*y) + 8*z*z*z;
}

// boundary condition at z=0
double g_0(double x, double y)
{
  return x*(1.0-x)*y*(1-y);
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

int main(int argc, char* argv[])
{
  {
  int nx, ny, nz;

  if      (argc==1) {nx=128;           ny=128;           nz=128;}
  else if (argc==2) {nx=atoi(argv[1]); ny=nx;            nz=nx;}
  else if (argc==4) {nx=atoi(argv[1]); ny=atoi(argv[2]); nz=atoi(argv[3]);}
  else {std::cerr << "Invalid number of arguments (should be 0, 1 or 3)"<<std::endl; exit(-1);}
  if (ny<0) ny=nx;
  if (nz<0) nz=nx;

  // total number of unknowns
  int n=nx*ny*nz;

  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);

  // Laplace operator
  stencil3d L = laplace3d_stencil(nx,ny,nz);

  // solution vector: start with a 0 vector
  double *x = new double[n];

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

  //loop over thread numbers
  for (int tn =4; tn<=48; tn+=2){
    Timer timer("CG solver for " + std::to_string(tn) + " threads");
    // solution vector: start with a 0 vector
    init(n, x, 0.0);
    try {
    cg_solver_threads(&L, n, x, b, tol, maxIter, &resNorm, &numIter, tn);
    } catch(std::exception e)
    {
        std::cerr << "Caught an exception in cg_solve: " << e.what() << std::endl;
        exit(-1);
    }
    
  }
  delete [] x;
  delete [] b;


  Timer::summarize();
  }



  // Loop over nx size
  {
  //int nx_arr[6] = {32, 64, 128, 256, 512, 1024};
  int nx_arr[0] = {};
  for(int nx: nx_arr){
    std::cout<< nx <<std::endl;
    int ny = nx;
    int nz = nx;

    // total number of unknowns
    int n=nx*ny*nz;

    double *x1 = new double[n];
    double *b1 = new double[n];

    double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);

    // Laplace operator
    stencil3d L = laplace3d_stencil(nx,ny,nz);

    // solution vector: start with a 0 vector
    init(n, x1, 0.0);

    // right-hand side
    init(n, b1, 0.0);

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
            b1[idx] = f(x,y,z);
        }
        }
    }

    // Dirichlet boundary conditions at z=0 (others are 0 in our case, initialized above)
    for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
        {
        b1[L.index_c(i,j,0)] -= L.value_b*g_0(i*dx, j*dy);
        }

    // solve the linear system of equations using CG
    int numIter, maxIter=500;
    double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

    //loop over thread numbers
    {
    Timer timer("CG solver for n = " + std::to_string(nx));
    try {
    cg_solver_threads(&L, n, x1, b1, tol, maxIter, &resNorm, &numIter, 32);
    } catch(std::exception e)
    {
        std::cerr << "Caught an exception in cg_solve: " << e.what() << std::endl;
        exit(-1);
    }     
    }
    delete [] x1;
    delete [] b1;
    }
    

    Timer::summarize();}

  return 0;
}