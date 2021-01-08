#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

int main(int argc, char **argv)
{
  float *x, *y; 
  __half *xf, *yf;
 
  int n = 1<<28, i;
  fprintf(stdout, "[n] = %d\n", n);

  x = (float*)malloc(n*sizeof(float));
  y = (float*)malloc(n*sizeof(float));
  xf = (__half*)malloc(n*sizeof(__half));
  yf = (__half*)malloc(n*sizeof(__half));

  #pragma acc data create(x[0:n]) copyout(y[0:n]) create(xf[0:n]) copyout(yf[0:n])
  {
// Initialize floats
#pragma acc parallel loop default(present)
    for( i = 0; i < n; i++)
    {
      x[i] = 1.1f;
      y[i] = 0.5f;
    }

//Initialize halfs
#pragma acc parallel loop default(present)
    for( i = 0; i < n; i++)
    {
      xf[i] = __half(1.1f);
      yf[i] = __half(0.5f);
    }

//Compute floats
#pragma acc parallel loop default(present)
    for( i = 1; i < n-1; i++ )
    {
      y[i] = 2.0 * x[i-1]*x[i+1] + y[i-1] + y[i+1];
    }

//Compute halfs
#pragma acc parallel loop default(present)
    for( i = 1; i < n-1; i++ )
    {
      yf[i] = 2.0 * xf[i-1]*xf[i+1] + yf[i-1] + yf[i+1];
    }
  }

//Prinf floats - half as float
for( i = 0; i < 2; i++ )
   {
   fprintf(stdout, "float: y[0] = %f - float(half): yf[0] = %f\n ", y[0], __half2float(yf[0]) );
  //fprintf(stdout, "yf[0] = %f\n",__half2float(yf[0]));
    }
  return 0;
}
