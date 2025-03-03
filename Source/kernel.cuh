#ifndef __KERNEL__ // TODO: change NAME
#define __KERNEL__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "config.h"

// Select the device to run on
// __global__ void cudaSelectDevice(int device);

cudaTextureObject_t texConstSrc, texIn, texOut;


__global__ void update_kernel(float* dst, bool dstOut);




#endif