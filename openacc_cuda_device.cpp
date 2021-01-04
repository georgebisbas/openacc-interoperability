#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#pragma acc routine seq
extern "C" float saxpy_dev(float, float, float);

#pragma acc routine seq
extern "C" __half saxpy_dev_half(__half, __half, __half);

int main(int argc, char **argv)
{
  float *x, *y; 
  __half *xf, *yf;
 
  int n = 1<<15, i;
  fprintf(stdout, "[n] = %d\n", n);

  x = (float*)malloc(n*sizeof(float));
  y = (float*)malloc(n*sizeof(float));
  xf = (__half*)malloc(n*sizeof(__half));
  yf = (__half*)malloc(n*sizeof(__half));

  #pragma acc data create(x[0:n]) copyout(y[0:n]) create(xf[0:n]) copyout(yf[0:n])
  {
#pragma acc parallel loop default(present)
    for( i = 0; i < n; i++)
    {
      x[i] = 1.0f;
      y[i] = 0.0f;
    }

#pragma acc parallel loop default(present)
    for( i = 0; i < n; i++)
    {
      xf[i] = __half(1.0f);
      yf[i] = __half(0.0f);
    }
      
#pragma acc parallel loop default(present)
    for( i = 0; i < n; i++ )
    {
      y[i] = saxpy_dev(2.0, x[i], y[i]);
    }

#pragma acc parallel loop default(present)
    for( i = 0; i < n; i++ )
    {
      yf[i] = saxpy_dev_half(__float2half(2.0f), xf[i], yf[i]);
    }
  }

  fprintf(stdout, "y[0] = %f\n",y[0]);
  fprintf(stdout, "yf[0] = %f\n",__half2float(yf[0]));

  return 0;
}
