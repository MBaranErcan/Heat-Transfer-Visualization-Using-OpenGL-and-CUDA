#ifndef __KERNEL__ // TODO: change NAME
#define __KERNEL__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "config.h"

// Select the device to run on
// __global__ void cudaSelectDevice(int device);

__host__ void selectDevice(int device);


__global__ void copy_heaters_kernel(float* iptr, const float* cptr);
__global__ void update_kernel(float* outSrc, const float* inSrc);



#endif