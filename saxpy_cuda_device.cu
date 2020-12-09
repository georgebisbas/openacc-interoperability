#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C"
__device__ 
float saxpy_dev(float a, float x, float y)
{
  return a * x + y;
}

extern "C"
__device__
float foo(float in, float multiplier)
{
    __half in_half = __float2half(in);
    __half multiplier_half = __float2half(multiplier);
    __half out_half =  __hmul(in_half, multiplier_half);
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);   
    return __half2float(out_half);
}
