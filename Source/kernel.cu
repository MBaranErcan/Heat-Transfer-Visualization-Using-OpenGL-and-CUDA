#include "kernel.cuh" // TODO: change NAME

// Constant Cells (Heaters)
__global__ void copy_heaters_kernel(float* iptr, const float* cptr) {
	// cptr -> constant pointer, iptr -> input pointer
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (cptr[offset] != 0) { // If marked as heater
		iptr[offset] = cptr[offset];
	}

}

// Update function
__global__ void update_kernel(float *outSrc, const float *inSrc) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Find neighbors
	int left = (x > 0) ? offset - 1 : offset;
	int right = (x < blockDim.x * gridDim.x - 1) ? offset + 1 : offset;
	int top = (y > 0) ? offset - (blockDim.x * gridDim.x) : offset;
	int bottom = (y < blockDim.y * gridDim.y - 1) ? (offset + (blockDim.x * gridDim.x)) : offset;

	// new value = old value + speed * (neighbors - 4 * old value)
	outSrc[offset] =	inSrc[offset] + SPEED * (inSrc[top] +
						inSrc[left] + inSrc[right] + inSrc[bottom] - (4 * inSrc[offset]));
}


// Select CUDA device
__host__ void selectDevice(int device) {

	// List all devices
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	// Print all devices
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
	}

	// Set the device
	cudaSetDevice(device);

	// Print device info
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	printf("Selected device: %s\n", deviceProp.name);
	printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("Total global memory: %lu\n", deviceProp.totalGlobalMem);
	printf("Shared memory per block: %lu\n", deviceProp.sharedMemPerBlock);
	printf("Registers per block: %d\n", deviceProp.regsPerBlock);
	printf("Warp size: %d\n", deviceProp.warpSize);
	printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Max threads dimensions: %d, %d, %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("Max grid size: %d, %d, %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Clock rate: %d\n", deviceProp.clockRate);
	printf("Total constant memory: %lu\n", deviceProp.totalConstMem);
	printf("Device Overlap: %s\n", deviceProp.deviceOverlap ? "Supported" : "Not available");
	printf("Integrated: %s\n", deviceProp.integrated ? "Yes" : "No");
}