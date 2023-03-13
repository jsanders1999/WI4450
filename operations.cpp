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
  return x;
}

// scalar product: return sum_i x[i]*y[i] for 0<=i<n
double dot(int n, double const* x, double const* y)
{
  // [...]
  double res;
  for(int i = 0; i < n; i++)
  {
    res = x[i]*y[i];
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
  return y[i];
}

//! apply a 7-point stencil to a vector, v = op*x
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  // [...]
  v = value_c*u[S->index_c] + value_n*u[S->index_n] + value_e*u[S->index_e] + value_s*u[S->index_s] + value_w*u[S->index_w] + value_b*u[S->index_b] + value_t*u[S->index_t];
  return v;
}

