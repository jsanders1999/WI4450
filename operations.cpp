#include "operations.hpp"
#include <omp.h>

// initialize a vector with a constant value, x[i] = value for 0<=i<n
void init(int n, double* x, double value)
{
  // A for loop that gives each of the n elements of array x the same value
  #pragma omp parallel for
  for(int i = 0; i < n; i++)
  {
    x[i] = value;
  }
  return;
}

void init_threads(int n, double* x, double value, int threadnum)
{
  // A for loop that gives each of the n elements of array x the same value
  #pragma omp parallel for num_threads(threadnum)
  for(int i = 0; i < n; i++)
  {
    x[i] = value;
  }
  return;
}

// scalar product: return sum_i x[i]*y[i] for 0<=i<n
double dot(int n, double const* x, double const* y)
{
  // A for loop that computes the inner product of n dimensional array x and n dimensional array y
  double res;
  #pragma omp parallel for reduction(+:res)
  for(int i = 0; i < n; i++)
  {
    res += x[i]*y[i];
  }
  return res;
}

double dot_threads(int n, double const* x, double const* y, int threadnum)
{
  // A for loop that computes the inner product of n dimensional array x and n dimensional array y
  double res;
  #pragma omp parallel for reduction(+:res) num_threads(threadnum)
  for(int i = 0; i < n; i++)
  {
    res += x[i]*y[i];
  }
  return res; 
}

// vector update: compute y[i] = a*x[i] + b*y[i] for 0<=i<n
void axpby(int n, double a, double const* x, double b, double* y)
{
  // A for loop that computes a*x+b*y elementwise and stores it in n dimensional array y
  #pragma omp parallel for 
  for(int i = 0; i < n; i++)
  {
    y[i] = a*x[i] + b*y[i];
  }
  return;
}

// combines two axpby vector updates into one, with only one for loop
void twice_axpby(int n, double a, double const* x, double b, double* y, double c, double const* z, double d, double* s)
{
  // A for loop that computes a*x+b*y elementwise and stores it in n dimensional array y
  #pragma omp parallel for 
  for(int i = 0; i < n; i++)
  {
    y[i] = a*x[i] + b*y[i];
    s[i] = c*z[i] + d*s[i];
  }
  return;
}

// vector update: compute y[i] = a*x[i] + b*y[i] for 0<=i<n
void axpby_threads(int n, double a, double const* x, double b, double* y, int threadnum)
{
  // A for loop that computes a*x+b*y elementwise and stores it in n dimensional array y
#pragma omp parallel for num_threads(threadnum)
  for(int i = 0; i < n; i++)
  {
    y[i] = a*x[i] + b*y[i];
  }
  return;
}

//! apply a 7-point stencil to a vector, v = op*x
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  // A for loop over the three dimensions that applies the stencil S to vector u and stores it in v
  // Possible speedup improvement: Use case switching instead of in statements
  #pragma omp parallel for collapse(3)
  for(int k = 0; k < S->nz; k++){
    for(int j = 0; j < S->ny; j++){
      for(int i = 0; i < S->nx; i++){
        // Add the center value of the stencil
        v[S->index_c(i,j,k)] = S->value_c*u[S->index_c(i,j,k)];
        // Add the east and/or west values of the stencil (x direction)
        v[S->index_c(i,j,k)] += ((i==0) ? 0 : S->value_e*u[S->index_e(i,j,k)]) + ((i==S->nx-1) ? 0 : S->value_w*u[S->index_w(i,j,k)]) ;
        // Add the north and/or south values of the stencil (y direction)
        v[S->index_c(i,j,k)] += ((j==0) ? 0 : S->value_n*u[S->index_n(i,j,k)]) + ((j==S->ny-1) ? 0 : S->value_s*u[S->index_s(i,j,k)]) ;
        // Add the top and/or bottom values of the stencil (z direction)
        v[S->index_c(i,j,k)] += ((k==0) ? 0 : S->value_t*u[S->index_t(i,j,k)]) + ((k==S->nz-1) ? 0 : S->value_b*u[S->index_b(i,j,k)]) ;

/*         // Add the east and/or west values of the stencil (x direction)
        if(i==0){// Boundary x=0
          v[S->index_c(i,j,k)] += S->value_e*u[S->index_e(i,j,k)];
        }
        else if(i==S->nx-1){// Boundary x=1
          v[S->index_c(i,j,k)] += S->value_w*u[S->index_w(i,j,k)];
        }
        else{// Interior 0<x<1
          v[S->index_c(i,j,k)] += S->value_e*u[S->index_e(i,j,k)] + S->value_w*u[S->index_w(i,j,k)];
        }

        // Add the north and/or south values of the stencil (y direction)
        if(j==0){// Boundary y=0
          v[S->index_c(i,j,k)] += S->value_n*u[S->index_n(i,j,k)];
        }
        else if(j==S->ny-1){ // Boundary y=1
          v[S->index_c(i,j,k)] += S->value_s*u[S->index_s(i,j,k)];
        }
        else{// Interior 0<y<1
          v[S->index_c(i,j,k)] += S->value_n*u[S->index_n(i,j,k)] + S->value_s*u[S->index_s(i,j,k)];
        }
        
        // Add the top and/or bottom values of the stencil (z direction)
        if(k==0){// Boundary z=0
          v[S->index_c(i,j,k)] += S->value_t*u[S->index_t(i,j,k)];
        }
        else if(k==S->nz-1){// Boundary z=1
          v[S->index_c(i,j,k)] += S->value_b*u[S->index_b(i,j,k)];
        }
        else{// Interior 0<z<1
          v[S->index_c(i,j,k)] += S->value_t*u[S->index_t(i,j,k)] + S->value_b*u[S->index_b(i,j,k)];
        } */

      }
    }
  }
  return;
}

//! apply a 7-point stencil to a vector, v = op*x
void apply_stencil3d_threads(stencil3d const* S,
        double const* u, double* v, int threadnum)
{
  // A for loop over the three dimensions that applies the stencil S to vector u and stores it in v
#pragma omp parallel for num_threads(threadnum)
  for(int k = 0; k < S->nz; k++){
    for(int j = 0; j < S->ny; j++){
      for(int i = 0; i < S->nx; i++){
        // Add the center value of the stencil
        v[S->index_c(i,j,k)] = S->value_c*u[S->index_c(i,j,k)];

        // Add the east and/or west values of the stencil (x direction)
        if(i==0){// Boundary x=0
          v[S->index_c(i,j,k)] += S->value_e*u[S->index_e(i,j,k)];
        }
        else if(i==S->nx-1){// Boundary x=1
          v[S->index_c(i,j,k)] += S->value_w*u[S->index_w(i,j,k)];
        }
        else{// Interior 0<x<1
          v[S->index_c(i,j,k)] += S->value_e*u[S->index_e(i,j,k)] + S->value_w*u[S->index_w(i,j,k)];
        }

        // Add the north and/or south values of the stencil (y direction)
        if(j==0){// Boundary y=0
          v[S->index_c(i,j,k)] += S->value_n*u[S->index_n(i,j,k)];
        }
        else if(j==S->ny-1){ // Boundary y=1
          v[S->index_c(i,j,k)] += S->value_s*u[S->index_s(i,j,k)];
        }
        else{// Interior 0<y<1
          v[S->index_c(i,j,k)] += S->value_n*u[S->index_n(i,j,k)] + S->value_s*u[S->index_s(i,j,k)];
        }
        
        // Add the top and/or bottom values of the stencil (z direction)
        if(k==0){// Boundary z=0
          v[S->index_c(i,j,k)] += S->value_t*u[S->index_t(i,j,k)];
        }
        else if(k==S->nz-1){// Boundary z=1
          v[S->index_c(i,j,k)] += S->value_b*u[S->index_b(i,j,k)];
        }
        else{// Interior 0<z<1
          v[S->index_c(i,j,k)] += S->value_t*u[S->index_t(i,j,k)] + S->value_b*u[S->index_b(i,j,k)];
        }

      }
    }
  }
  return;
}


