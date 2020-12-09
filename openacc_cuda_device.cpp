#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#pragma acc routine seq
extern "C" float saxpy_dev(float, float, float);

#pragma acc routine seq
extern "C" float foo(float, float);

inline
__device__ __half __hmul(const half a, const half b);

int main(int argc, char **argv)
{
  float *x, *y, tmp;
  float a = 0.16f;
  float b = 0.16f;
  float c = 0.16f; 
  fprintf(stdout, "c = %f\n", c);
  
  c = foo(a, b);
  fprintf(stdout, "c = %f\n", c);
  
  int n = 1<<20, i;

  x = (float*)malloc(n*sizeof(float));
  y = (float*)malloc(n*sizeof(float));

  #pragma acc data create(x[0:n]) copyout(y[0:n])
  {
    #pragma acc kernels
    {
      for( i = 0; i < n; i++)
      {
        x[i] = 1.0f;
        y[i] = 0.0f;
      }
    }
      
#pragma acc parallel loop
    for( i = 0; i < n; i++ )
    {
      y[i] = saxpy_dev(2.0, x[i], y[i]);
    }
  }

  fprintf(stdout, "y[0] = %f\n",y[0]);
  return 0;
}
