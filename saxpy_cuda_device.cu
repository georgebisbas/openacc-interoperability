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
__half saxpy_dev_half(__half a, __half x, __half y)
{
  return a * x + y;
}
