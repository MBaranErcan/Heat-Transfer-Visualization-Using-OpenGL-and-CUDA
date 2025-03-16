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
	float* dev_heaterIndices;	// Buffer to store heater indices
	float* dev_heaterTempVals;	// Buffer to store heater temp values

	cudaArray_t cudaArray;
	cudaGraphicsResource_t resource;// OpenGL-CUDA interop resource
	GLuint textureID;				// OpenGL texture ID TODO: not needed

	cudaEvent_t start, stop;
	float totalTime;
	float elapsedTime;
	float ms;
	float frames;
};

// TODO add implementation maybe, for both cold and hot regions
struct Heater {
	float temp;
	int x, y;
	float radius;
};


__host__ void selectDevice(int device);
__host__ void init_anim(DataBlock* data, int DIM, const float initTemp, dim3 blocks, dim3 threads);
__host__ void add_heater(DataBlock* data, Heater* heater, int DIM, dim3 blocks, dim3 threads);
__host__ void animate(DataBlock* data, dim3 blocks, dim3 threads, float SPEED, int DIM);
__host__ void anim_exit(DataBlock* data);


__global__ void init_buffer_kernel(float* ptr, float temp, int DIM);
__global__ void addHeaterKernel(float* heaterIndices, float* heaterTempVals, int DIM, int heater_x, int heater_y, float heater_temp, int radius);
__global__ void copy_heaters_kernel(float* iptr, const float* cptr, const float* vptr, int DIM);
__global__ void update_anim_kernel(float* dev_inSrc, float* dev_outSrc, float SPEED, int DIM);