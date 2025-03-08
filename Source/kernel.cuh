#pragma once

#include <iostream>
#include <windows.h>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>


#include "handle_error.h"

extern const float PI;

struct DataBlock {
	float* dev_inSrc;
	float* dev_outSrc;
	float* dev_constSrc;

	cudaArray* cudaArray;			// For texture binding
	cudaGraphicsResource_t resource;// OpenGL-CUDA interop resource
	GLuint textureID;				// OpenGL texture ID

	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

// TODO add implementation maybe, for both cold and hot regions
struct Heater {
	int x, y;
	float temp;
	float radius;
};

__host__ void init_anim(DataBlock* data, int size);
__host__ void selectDevice(int device);
__host__ void animate(DataBlock* data, dim3 threads, dim3 blocks, float SPEED, int DIM);
__host__ void initConstantRegion(DataBlock* data, float min, float max, int DIM, int heater_x, int heater_y, int radius);
__host__ void anim_exit(DataBlock* data);

__global__ void copy_heaters_kernel(float* iptr, const float* cptr, int DIM);
__global__ void update_anim_kernel(float* dev_inSrc, float* dev_outSrc, float SPEED, int DIM);