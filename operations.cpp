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
        v[S->index_c(i,j,k)] += ((i==0) ? 0 : S->value_w*u[S->index_w(i,j,k)]) + ((i==S->nx-1) ? 0 : S->value_e*u[S->index_e(i,j,k)]) ;
        // Add the north and/or south values of the stencil (y direction)
        v[S->index_c(i,j,k)] += ((j==0) ? 0 : S->value_s*u[S->index_s(i,j,k)]) + ((j==S->ny-1) ? 0 : S->value_n*u[S->index_n(i,j,k)]) ;
        // Add the top and/or bottom values of the stencil (z direction)
        v[S->index_c(i,j,k)] += ((k==0) ? 0 : S->value_b*u[S->index_b(i,j,k)]) + ((k==S->nz-1) ? 0 : S->value_t*u[S->index_t(i,j,k)]) ;
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
#pragma omp parallel for num_threads(threadnum) collapse(3)
  for(int k = 0; k < S->nz; k++){
    for(int j = 0; j < S->ny; j++){
      for(int i = 0; i < S->nx; i++){
        // Add the center value of the stencil
        v[S->index_c(i,j,k)] = S->value_c*u[S->index_c(i,j,k)];
        // Add the east and/or west values of the stencil (x direction)
        v[S->index_c(i,j,k)] += ((i==0) ? 0 : S->value_w*u[S->index_w(i,j,k)]) + ((i==S->nx-1) ? 0 : S->value_e*u[S->index_e(i,j,k)]) ;
        // Add the north and/or south values of the stencil (y direction)
        v[S->index_c(i,j,k)] += ((j==0) ? 0 : S->value_s*u[S->index_s(i,j,k)]) + ((j==S->ny-1) ? 0 : S->value_n*u[S->index_n(i,j,k)]) ;
        // Add the top and/or bottom values of the stencil (z direction)
        v[S->index_c(i,j,k)] += ((k==0) ? 0 : S->value_b*u[S->index_b(i,j,k)]) + ((k==S->nz-1) ? 0 : S->value_t*u[S->index_t(i,j,k)]) ;

      }
    }
  }
  return;
}

void apply_stencil3d_noif(stencil3d const* S,
        double const* u, double* v)
{
  // A for loop over the three dimensions that applies the stencil S to vector u and stores it in v

  //    0,    0,    0
  v[S->index_c(0,0,0)] = S->value_c*u[S->index_c(0,0,0)] + S->value_e*u[S->index_e(0,0,0)] + S->value_n*u[S->index_n(0,0,0)] + S->value_t*u[S->index_t(0,0,0)];
  //  ...,    0,    0
  
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,0,0)] = S->value_c*u[S->index_c(i,0,0)] + S->value_e*u[S->index_e(i,0,0)] + S->value_w*u[S->index_w(i,0,0)] + S->value_n*u[S->index_n(i,0,0)] + S->value_t*u[S->index_t(i,0,0)];
  }
  // nx-1,    0,    0
  v[S->index_c(S->nx-1,0,0)] = S->value_c*u[S->index_c(S->nx-1,0,0)] + S->value_w*u[S->index_w(S->nx-1,0,0)] + S->value_n*u[S->index_n(S->nx-1,0,0)] + S->value_t*u[S->index_t(S->nx-1,0,0)];

  //    0,  ...,    0
  #pragma omp parallel
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(0,j,0)] = S->value_c*u[S->index_c(0,j,0)] + S->value_e*u[S->index_e(0,j,0)] + S->value_n*u[S->index_n(0,j,0)] + S->value_s*u[S->index_s(0,j,0)] + S->value_t*u[S->index_t(0,j,0)];
  }
  //  ...,  ...,    0
  #pragma omp parallel for collapse(2)
  for(int j = 1; j < S->ny-1; j++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,j,0)] = S->value_c*u[S->index_c(i,j,0)] + S->value_e*u[S->index_e(i,j,0)] + S->value_w*u[S->index_w(i,j,0)] + S->value_n*u[S->index_n(i,j,0)] + S->value_s*u[S->index_s(i,j,0)] + S->value_t*u[S->index_t(i,j,0)];
    }
  }
  // nx-1,  ...,    0
  #pragma omp parallel for
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(S->nx-1,j,0)] = S->value_c*u[S->index_c(S->nx-1,j,0)] + S->value_w*u[S->index_w(S->nx-1,j,0)] + S->value_n*u[S->index_n(S->nx-1,j,0)]+ S->value_s*u[S->index_s(S->nx-1,j,0)] + S->value_t*u[S->index_t(S->nx-1,j,0)];
  }

  //    0, ny-1,    0
  v[S->index_c(0,S->ny-1,0)] = S->value_c*u[S->index_c(0,S->ny-1,0)] + S->value_e*u[S->index_e(0,S->ny-1,0)] + S->value_s*u[S->index_s(0,S->ny-1,0)] + S->value_t*u[S->index_t(0,S->ny-1,0)];
  //  ..., ny-1,    0
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,S->ny-1,0)] = S->value_c*u[S->index_c(i,S->ny-1,0)] + S->value_e*u[S->index_e(i,S->ny-1,0)] + S->value_w*u[S->index_w(i,S->ny-1,0)] + S->value_s*u[S->index_s(i,S->ny-1,0)] + S->value_t*u[S->index_t(i,S->ny-1,0)];
  }
  // nx-1, ny-1,    0
  v[S->index_c(S->nx-1,S->ny-1,0)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,0)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,0)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,0)] + S->value_t*u[S->index_t(S->nx-1,S->ny-1,0)];


  
  //    0,    0,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(0,0,k)] = S->value_c*u[S->index_c(0,0,k)] + S->value_e*u[S->index_e(0,0,k)] + S->value_n*u[S->index_n(0,0,k)] + S->value_t*u[S->index_t(0,0,k)] + S->value_b*u[S->index_b(0,0,k)];
  }
  //  ...,    0,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,0,k)] = S->value_c*u[S->index_c(i,0,k)] + S->value_e*u[S->index_e(i,0,k)] + S->value_w*u[S->index_w(i,0,k)] + S->value_n*u[S->index_n(i,0,k)] + S->value_t*u[S->index_t(i,0,k)] + S->value_b*u[S->index_b(i,0,k)];
    }
  }
  // nx-1,    0,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(S->nx-1,0,k)] = S->value_c*u[S->index_c(S->nx-1,0,k)] + S->value_w*u[S->index_w(S->nx-1,0,k)] + S->value_n*u[S->index_n(S->nx-1,0,k)] + S->value_t*u[S->index_t(S->nx-1,0,k)] + S->value_b*u[S->index_b(S->nx-1,0,k)];
  }


  //    0,  ...,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int j = 1; j < S->ny-1; j++){
      v[S->index_c(0,j,k)] = S->value_c*u[S->index_c(0,j,k)] + S->value_e*u[S->index_e(0,j,k)] + S->value_n*u[S->index_n(0,j,k)] + S->value_s*u[S->index_s(0,j,k)] + S->value_t*u[S->index_t(0,j,k)] + S->value_b*u[S->index_b(0,j,k)] ;
    }
  }
  //  ...,  ...,  ...
  #pragma omp parallel for collapse(3)
  for(int k = 1; k < S->nz-1; k++){
    for(int j = 1; j < S->ny-1; j++){
      for(int i = 1; i < S->nx-1; i++){
        v[S->index_c(i,j,k)] = S->value_c*u[S->index_c(i,j,k)] + S->value_e*u[S->index_e(i,j,k)] + S->value_w*u[S->index_w(i,j,k)] + S->value_n*u[S->index_n(i,j,k)] + S->value_s*u[S->index_s(i,j,k)] + S->value_t*u[S->index_t(i,j,k)] + S->value_b*u[S->index_b(i,j,k)];
      }
    }
  }
  // nx-1,  ...,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int j = 1; j < S->ny-1; j++){
      v[S->index_c(S->nx-1,j,k)] = S->value_c*u[S->index_c(S->nx-1,j,k)] + S->value_w*u[S->index_w(S->nx-1,j,k)] + S->value_n*u[S->index_n(S->nx-1,j,k)]+ S->value_s*u[S->index_s(S->nx-1,j,k)] + S->value_t*u[S->index_t(S->nx-1,j,k)] + S->value_b*u[S->index_b(S->nx-1,j,k)];
    }
  }
    
  //    0, ny-1,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(0,S->ny-1,k)] = S->value_c*u[S->index_c(0,S->ny-1,k)] + S->value_e*u[S->index_e(0,S->ny-1,k)] + S->value_s*u[S->index_s(0,S->ny-1,k)] + S->value_t*u[S->index_t(0,S->ny-1,k)] + S->value_b*u[S->index_b(0,S->ny-1,k)];
  }  
  //  ..., ny-1,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,S->ny-1,k)] = S->value_c*u[S->index_c(i,S->ny-1,k)] + S->value_e*u[S->index_e(i,S->ny-1,k)] + S->value_w*u[S->index_w(i,S->ny-1,k)] + S->value_s*u[S->index_s(i,S->ny-1,k)] + S->value_t*u[S->index_t(i,S->ny-1,k)] + S->value_b*u[S->index_b(i,S->ny-1,k)] ;
    }
  }
  // nx-1, ny-1,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(S->nx-1,S->ny-1,k)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,k)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,k)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,k)] + S->value_t*u[S->index_t(S->nx-1,S->ny-1,k)] + S->value_b*u[S->index_b(S->nx-1,S->ny-1,k)];
  }
  

  //    0,    0, nz-1
  v[S->index_c(0,0,S->nz-1)] = S->value_c*u[S->index_c(0,0,S->nz-1)] + S->value_e*u[S->index_e(0,0,S->nz-1)] + S->value_n*u[S->index_n(0,0,S->nz-1)] + S->value_b*u[S->index_b(0,0,S->nz-1)];
  //  ...,    0, nz-1
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,0,S->nz-1)] = S->value_c*u[S->index_c(i,0,S->nz-1)] + S->value_e*u[S->index_e(i,0,S->nz-1)] + S->value_w*u[S->index_w(i,0,S->nz-1)] + S->value_n*u[S->index_n(i,0,S->nz-1)] + S->value_b*u[S->index_b(i,0,S->nz-1)];
  }
  // nx-1,    0, nz-1
  v[S->index_c(S->nx-1,0,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,0,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,0,S->nz-1)] + S->value_n*u[S->index_n(S->nx-1,0,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,0,S->nz-1)];

  //    0,  ..., nz-1
  #pragma omp parallel for
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(0,j,S->nz-1)] = S->value_c*u[S->index_c(0,j,S->nz-1)] + S->value_e*u[S->index_e(0,j,S->nz-1)] + S->value_n*u[S->index_n(0,j,S->nz-1)] + S->value_s*u[S->index_s(0,j,S->nz-1)] + S->value_b*u[S->index_b(0,j,S->nz-1)];
  }  
    //  ...,  ..., nz-1
    #pragma omp parallel for collapse(2)
  for(int j = 1; j < S->ny-1; j++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,j,S->nz-1)] = S->value_c*u[S->index_c(i,j,S->nz-1)] + S->value_e*u[S->index_e(i,j,S->nz-1)] + S->value_w*u[S->index_w(i,j,S->nz-1)] + S->value_n*u[S->index_n(i,j,S->nz-1)] + S->value_s*u[S->index_s(i,j,S->nz-1)] + S->value_b*u[S->index_b(i,j,S->nz-1)];
    }
  }
    // nx-1,  ..., nz-1
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(S->nx-1,j,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,j,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,j,S->nz-1)] + S->value_n*u[S->index_n(S->nx-1,j,S->nz-1)]+ S->value_s*u[S->index_s(S->nx-1,j,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,j,S->nz-1)];
  }

  //    0, ny-1, nz-1
  v[S->index_c(0,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(0,S->ny-1,S->nz-1)] + S->value_e*u[S->index_e(0,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(0,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(0,S->ny-1,S->nz-1)];
  //  ..., ny-1, nz-1
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(i,S->ny-1,S->nz-1)] + S->value_e*u[S->index_e(i,S->ny-1,S->nz-1)] + S->value_w*u[S->index_w(i,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(i,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(i,S->ny-1,S->nz-1)];
  }
  // nx-1, ny-1, nz-1
   v[S->index_c(S->nx-1,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,S->ny-1,S->nz-1)];

  return;
}


void apply_stencil3d_noif_block(stencil3d const* S,
        double const* u, double* v, int blockx, int blocky)
{
  // A for loop over the three dimensions that applies the stencil S to vector u and stores it in v

  //    0,    0,    0
  v[S->index_c(0,0,0)] = S->value_c*u[S->index_c(0,0,0)] + S->value_e*u[S->index_e(0,0,0)] + S->value_n*u[S->index_n(0,0,0)] + S->value_t*u[S->index_t(0,0,0)];
  //  ...,    0,    0
  
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,0,0)] = S->value_c*u[S->index_c(i,0,0)] + S->value_e*u[S->index_e(i,0,0)] + S->value_w*u[S->index_w(i,0,0)] + S->value_n*u[S->index_n(i,0,0)] + S->value_t*u[S->index_t(i,0,0)];
  }
  // nx-1,    0,    0
  v[S->index_c(S->nx-1,0,0)] = S->value_c*u[S->index_c(S->nx-1,0,0)] + S->value_w*u[S->index_w(S->nx-1,0,0)] + S->value_n*u[S->index_n(S->nx-1,0,0)] + S->value_t*u[S->index_t(S->nx-1,0,0)];

  //    0,  ...,    0
  #pragma omp parallel
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(0,j,0)] = S->value_c*u[S->index_c(0,j,0)] + S->value_e*u[S->index_e(0,j,0)] + S->value_n*u[S->index_n(0,j,0)] + S->value_s*u[S->index_s(0,j,0)] + S->value_t*u[S->index_t(0,j,0)];
  }
  //  ...,  ...,    0
  #pragma omp parallel for collapse(2)
  for(int j = 1; j < S->ny-1; j++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,j,0)] = S->value_c*u[S->index_c(i,j,0)] + S->value_e*u[S->index_e(i,j,0)] + S->value_w*u[S->index_w(i,j,0)] + S->value_n*u[S->index_n(i,j,0)] + S->value_s*u[S->index_s(i,j,0)] + S->value_t*u[S->index_t(i,j,0)];
    }
  }
  // nx-1,  ...,    0
  #pragma omp parallel for
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(S->nx-1,j,0)] = S->value_c*u[S->index_c(S->nx-1,j,0)] + S->value_w*u[S->index_w(S->nx-1,j,0)] + S->value_n*u[S->index_n(S->nx-1,j,0)]+ S->value_s*u[S->index_s(S->nx-1,j,0)] + S->value_t*u[S->index_t(S->nx-1,j,0)];
  }

  //    0, ny-1,    0
  v[S->index_c(0,S->ny-1,0)] = S->value_c*u[S->index_c(0,S->ny-1,0)] + S->value_e*u[S->index_e(0,S->ny-1,0)] + S->value_s*u[S->index_s(0,S->ny-1,0)] + S->value_t*u[S->index_t(0,S->ny-1,0)];
  //  ..., ny-1,    0
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,S->ny-1,0)] = S->value_c*u[S->index_c(i,S->ny-1,0)] + S->value_e*u[S->index_e(i,S->ny-1,0)] + S->value_w*u[S->index_w(i,S->ny-1,0)] + S->value_s*u[S->index_s(i,S->ny-1,0)] + S->value_t*u[S->index_t(i,S->ny-1,0)];
  }
  // nx-1, ny-1,    0
  v[S->index_c(S->nx-1,S->ny-1,0)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,0)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,0)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,0)] + S->value_t*u[S->index_t(S->nx-1,S->ny-1,0)];


  
  //    0,    0,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(0,0,k)] = S->value_c*u[S->index_c(0,0,k)] + S->value_e*u[S->index_e(0,0,k)] + S->value_n*u[S->index_n(0,0,k)] + S->value_t*u[S->index_t(0,0,k)] + S->value_b*u[S->index_b(0,0,k)];
  }
  //  ...,    0,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,0,k)] = S->value_c*u[S->index_c(i,0,k)] + S->value_e*u[S->index_e(i,0,k)] + S->value_w*u[S->index_w(i,0,k)] + S->value_n*u[S->index_n(i,0,k)] + S->value_t*u[S->index_t(i,0,k)] + S->value_b*u[S->index_b(i,0,k)];
    }
  }
  // nx-1,    0,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(S->nx-1,0,k)] = S->value_c*u[S->index_c(S->nx-1,0,k)] + S->value_w*u[S->index_w(S->nx-1,0,k)] + S->value_n*u[S->index_n(S->nx-1,0,k)] + S->value_t*u[S->index_t(S->nx-1,0,k)] + S->value_b*u[S->index_b(S->nx-1,0,k)];
  }


  //    0,  ...,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int j = 1; j < S->ny-1; j++){
      v[S->index_c(0,j,k)] = S->value_c*u[S->index_c(0,j,k)] + S->value_e*u[S->index_e(0,j,k)] + S->value_n*u[S->index_n(0,j,k)] + S->value_s*u[S->index_s(0,j,k)] + S->value_t*u[S->index_t(0,j,k)] + S->value_b*u[S->index_b(0,j,k)] ;
    }
  }
  //  ...,  ...,  ...
  #pragma omp parallel for collapse(3)
  for(int k = 1; k < S->nz-1; k++){
    for(jb = 1; jb < S->ny-1; jb+=blocky){
      for(ib = 1; ib < S->nx-1; ib+=blockx){
        for(int j = jb; j < jb+blocky && j< S->ny-1 ; j++){
          for(int i = ib; i < ib+blockx && i< S->nx-1 ; i++){
            v[S->index_c(i,j,k)] = S->value_c*u[S->index_c(i,j,k)] + S->value_e*u[S->index_e(i,j,k)] + S->value_w*u[S->index_w(i,j,k)] + S->value_n*u[S->index_n(i,j,k)] + S->value_s*u[S->index_s(i,j,k)] + S->value_t*u[S->index_t(i,j,k)] + S->value_b*u[S->index_b(i,j,k)];
          }
        }
      }
    }
  }
  // nx-1,  ...,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int j = 1; j < S->ny-1; j++){
      v[S->index_c(S->nx-1,j,k)] = S->value_c*u[S->index_c(S->nx-1,j,k)] + S->value_w*u[S->index_w(S->nx-1,j,k)] + S->value_n*u[S->index_n(S->nx-1,j,k)]+ S->value_s*u[S->index_s(S->nx-1,j,k)] + S->value_t*u[S->index_t(S->nx-1,j,k)] + S->value_b*u[S->index_b(S->nx-1,j,k)];
    }
  }
    
  //    0, ny-1,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(0,S->ny-1,k)] = S->value_c*u[S->index_c(0,S->ny-1,k)] + S->value_e*u[S->index_e(0,S->ny-1,k)] + S->value_s*u[S->index_s(0,S->ny-1,k)] + S->value_t*u[S->index_t(0,S->ny-1,k)] + S->value_b*u[S->index_b(0,S->ny-1,k)];
  }  
  //  ..., ny-1,  ...
  #pragma omp parallel for collapse(2)
  for(int k = 1; k < S->nz-1; k++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,S->ny-1,k)] = S->value_c*u[S->index_c(i,S->ny-1,k)] + S->value_e*u[S->index_e(i,S->ny-1,k)] + S->value_w*u[S->index_w(i,S->ny-1,k)] + S->value_s*u[S->index_s(i,S->ny-1,k)] + S->value_t*u[S->index_t(i,S->ny-1,k)] + S->value_b*u[S->index_b(i,S->ny-1,k)] ;
    }
  }
  // nx-1, ny-1,  ...
  #pragma omp parallel for
  for(int k = 1; k < S->nz-1; k++){
    v[S->index_c(S->nx-1,S->ny-1,k)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,k)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,k)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,k)] + S->value_t*u[S->index_t(S->nx-1,S->ny-1,k)] + S->value_b*u[S->index_b(S->nx-1,S->ny-1,k)];
  }
  

  //    0,    0, nz-1
  v[S->index_c(0,0,S->nz-1)] = S->value_c*u[S->index_c(0,0,S->nz-1)] + S->value_e*u[S->index_e(0,0,S->nz-1)] + S->value_n*u[S->index_n(0,0,S->nz-1)] + S->value_b*u[S->index_b(0,0,S->nz-1)];
  //  ...,    0, nz-1
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,0,S->nz-1)] = S->value_c*u[S->index_c(i,0,S->nz-1)] + S->value_e*u[S->index_e(i,0,S->nz-1)] + S->value_w*u[S->index_w(i,0,S->nz-1)] + S->value_n*u[S->index_n(i,0,S->nz-1)] + S->value_b*u[S->index_b(i,0,S->nz-1)];
  }
  // nx-1,    0, nz-1
  v[S->index_c(S->nx-1,0,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,0,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,0,S->nz-1)] + S->value_n*u[S->index_n(S->nx-1,0,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,0,S->nz-1)];

  //    0,  ..., nz-1
  #pragma omp parallel for
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(0,j,S->nz-1)] = S->value_c*u[S->index_c(0,j,S->nz-1)] + S->value_e*u[S->index_e(0,j,S->nz-1)] + S->value_n*u[S->index_n(0,j,S->nz-1)] + S->value_s*u[S->index_s(0,j,S->nz-1)] + S->value_b*u[S->index_b(0,j,S->nz-1)];
  }  
    //  ...,  ..., nz-1
    #pragma omp parallel for collapse(2)
  for(int j = 1; j < S->ny-1; j++){
    for(int i = 1; i < S->nx-1; i++){
      v[S->index_c(i,j,S->nz-1)] = S->value_c*u[S->index_c(i,j,S->nz-1)] + S->value_e*u[S->index_e(i,j,S->nz-1)] + S->value_w*u[S->index_w(i,j,S->nz-1)] + S->value_n*u[S->index_n(i,j,S->nz-1)] + S->value_s*u[S->index_s(i,j,S->nz-1)] + S->value_b*u[S->index_b(i,j,S->nz-1)];
    }
  }
    // nx-1,  ..., nz-1
  for(int j = 1; j < S->ny-1; j++){
    v[S->index_c(S->nx-1,j,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,j,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,j,S->nz-1)] + S->value_n*u[S->index_n(S->nx-1,j,S->nz-1)]+ S->value_s*u[S->index_s(S->nx-1,j,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,j,S->nz-1)];
  }

  //    0, ny-1, nz-1
  v[S->index_c(0,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(0,S->ny-1,S->nz-1)] + S->value_e*u[S->index_e(0,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(0,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(0,S->ny-1,S->nz-1)];
  //  ..., ny-1, nz-1
  #pragma omp parallel for
  for(int i = 1; i < S->nx-1; i++){
    v[S->index_c(i,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(i,S->ny-1,S->nz-1)] + S->value_e*u[S->index_e(i,S->ny-1,S->nz-1)] + S->value_w*u[S->index_w(i,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(i,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(i,S->ny-1,S->nz-1)];
  }
  // nx-1, ny-1, nz-1
   v[S->index_c(S->nx-1,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,S->ny-1,S->nz-1)];

  return;
}


