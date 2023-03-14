#include "operations.hpp"
#include <omp.h>

// initialize a vector with a constant value, x[i] = value for 0<=i<n
void init(int n, double* x, double value)
{
  // [...]
  for(int i = 0; i < n; i++)
  {
    x[i] = value;
  }
  return;
}

// scalar product: return sum_i x[i]*y[i] for 0<=i<n
double dot(int n, double const* x, double const* y)
{
  // [...]
  double res;
  for(int i = 0; i < n; i++)
  {
    res += x[i]*y[i];
  }
  return res;
}

// vector update: compute y[i] = a*x[i] + b*y[i] for 0<=i<n
void axpby(int n, double a, double const* x, double b, double* y)
{
  // [...]
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
  // [...]
  // something with boundary conditions?
  for(int i = 0; i < S->nx; i++){
    for(int j = 0; i < S->ny; j++){
      for(int k = 0; i < S->nz; k++){
        v[S->index_c(i,j,k)] = S->value_c*u[S->index_c(i,j,k)];

        if(i==0){
          v[S->index_c(i,j,k)] += S->value_e*u[S->index_e(i,j,k)];
        }
        if(i==S->nx-1){
          v[S->index_c(i,j,k)] += S->value_w*u[S->index_w(i,j,k)];
        }
        else{
          v[S->index_c(i,j,k)] += S->value_e*u[S->index_e(i,j,k)] + S->value_w*u[S->index_w(i,j,k)];
        }

        if(j==0){
          v[S->index_c(i,j,k)] += S->value_n*u[S->index_n(i,j,k)];
        }
        if(j==S->ny-1){
          v[S->index_c(i,j,k)] += S->value_s*u[S->index_s(i,j,k)];
        }
        else{
          v[S->index_c(i,j,k)] += S->value_n*u[S->index_n(i,j,k)] + S->value_s*u[S->index_s(i,j,k)];
        }

        if(j==0){
          v[S->index_c(i,j,k)] += S->value_t*u[S->index_t(i,j,k)];
        }
        if(j==S->ny-1){
          v[S->index_c(i,j,k)] += S->value_b*u[S->index_b(i,j,k)];
        }
        else{
          v[S->index_c(i,j,k)] += S->value_t*u[S->index_t(i,j,k)] + S->value_b*u[S->index_b(i,j,k)];
        }

      }
    }
  }
  return;
}

