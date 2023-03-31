
#include "cg_solver.hpp"
#include "operations.hpp"
#include "timer.hpp"

#include <cmath>
#include <stdexcept>

#include <iostream>
#include <iomanip>

void cg_solver(stencil3d const* op, int n, double* x, double const* b,
        double tol, int maxIter,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }

  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];

  double alpha, beta, rho=1.0, rho_old=0.0;

  // r = op * x
  apply_stencil3d(op, x, r);

  // r = b - r;
  axpby(n, 1.0, b, -1.0, r);

  // p = q = 0
  init(n, p, 0.0);
  init(n, q, 0.0);

  // start CG iteration
  int iter = -1;
  while (true)
  {
    Timer timerA("total iteration");
    iter++;

    // rho = <r, r>
    {Timer timer("rho = <r, r>"); rho = dot(n, r, r);}

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
    }

    // check for convergence or failure
    if ((std::sqrt(rho) < tol) || (iter > maxIter))
    {
      break;
    }

    if (rho_old==0.0)
    {
      alpha = 0.0;
    }
    else
    {
      alpha = rho / rho_old;
    }
    // p = r + alpha * p
    {Timer timerB("p = r + alpha * p"); axpby(n, 1.0, r, alpha, p);}

    // q = op * p
    {Timer timerC("q = op * p"); apply_stencil3d(op, p, q);}

    // beta = <p,q>
    {Timer timerD("beta = <p,q>"); beta = dot(n, p, q);}

    alpha = rho / beta;

    // x = x + alpha * p
    {Timer timerE("x = x + alpha * p"); axpby(n, alpha, p, 1.0, x);}

    // r = r - alpha * q
    {Timer timerF("r = r - alpha * q"); axpby(n, -alpha, q, 1.0, r);}
    //twice_axpby(n , alpha , p , 1.0 , x , - alpha , q , 1.0 , r );

    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;

  // return number of iterations and achieved residual
  *resNorm = rho;
  *numIter = iter;
  return;
}

//Run the CG solver on threadnum threads. 
void cg_solver_threads(stencil3d const* op, int n, double* x, double const* b,
        double tol, int maxIter,
        double* resNorm, int* numIter, int threadnum,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }

  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];

  double alpha, beta, rho=1.0, rho_old=0.0;

  // r = op * x
  apply_stencil3d_threads(op, x, r, threadnum);

  // r = b - r;
  axpby_threads(n, 1.0, b, -1.0, r, threadnum);

  // p = q = 0
  init_threads(n, p, 0.0, threadnum);
  init_threads(n, q, 0.0, threadnum);

  // start CG iteration
  int iter = -1;
  while (true)
  { 
    iter++;

    // rho = <r, r>
    rho = dot_threads(n, r, r, threadnum);

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho << std::endl;
    }

    // check for convergence or failure
    if ((std::sqrt(rho) < tol) || (iter > maxIter))
    {
      break;
    }

    if (rho_old==0.0)
    {
      alpha = 0.0;
    }
    else
    {
      alpha = rho / rho_old;
    }
    // p = r + alpha * p
    axpby_threads(n, 1.0, r, alpha, p, threadnum);

    // q = op * p
    apply_stencil3d_threads(op, p, q, threadnum);

    // beta = <p,q>
    beta = dot_threads(n, p, q, threadnum);

    alpha = rho / beta;

    // x = x + alpha * p
    axpby_threads(n, alpha, p, 1.0, x, threadnum);

    // r = r - alpha * q
    axpby_threads(n, -alpha, q, 1.0, r, threadnum);

    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;

  // return number of iterations and achieved residual
  *resNorm = rho;
  *numIter = iter;
  return;
}
