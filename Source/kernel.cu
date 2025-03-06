#include "Kernel.cuh" // TODO: change NAME


// Select CUDA device
__host__ void selectDevice(int device) {

	// List all devices
	int deviceCount;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

	// Print all devices
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
	}

	// Set the device
	HANDLE_ERROR(cudaSetDevice(device));

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


// Initialize memory on device
__host__ void init_anim(DataBlock* data, int N) {

	size_t dataSize = N * sizeof(float);

	// Allocate memory on device
	HANDLE_ERROR(cudaMalloc((void**)&data->dev_inSrc, dataSize));
	HANDLE_ERROR(cudaMalloc((void**)&data->dev_outSrc, dataSize));
	HANDLE_ERROR(cudaMalloc((void**)&data->dev_constSrc, dataSize));


	// Create events
	HANDLE_ERROR(cudaEventCreate(&data->start));
	HANDLE_ERROR(cudaEventCreate(&data->stop));

	// Initialize time
	data->totalTime = 0;
	data->frames = 0;

	// Set empty. Heat sources will be set in initConstantRegion().
	HANDLE_ERROR(cudaMemset(data->dev_inSrc, 0, dataSize));
}


// Instantiate the heater region
__host__ void initConstantRegion(DataBlock* data, float min, float max, int DIM, int heater_x, int heater_y, int radius) {

	float* constBuffer = (float*)malloc(DIM * DIM * sizeof(float));

	for (int i = 0; i < DIM * DIM; i++) {
		int x = i % DIM;
		int y = i / DIM;
		int dist = sqrtf((x - heater_x) * (x - heater_x) + (y - heater_y) * (y - heater_y));
		// If distance to heater center is less than radius, set to max
		constBuffer[i] = (dist <= radius) ? 1.0f : 0.0f;
	}

	// Copy to device
	HANDLE_ERROR(cudaMemcpy(data->dev_constSrc, constBuffer, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice));
	free(constBuffer);
}


// Constant Cells (Heaters)
__global__ void copy_heaters_kernel(float* iptr, const float* cptr, int DIM) { 
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * DIM;

    if (x < DIM && y < DIM && cptr[offset] != 0) { // Check boundaries too TODO
		iptr[offset] = cptr[offset];
	}
}


// Update function
__global__ void update_anim_kernel(float* dev_inSrc, float* dev_outSrc, float SPEED, int DIM) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * DIM;

    if (x >= DIM || y >= DIM) return; // Boundary check

    int top    = (y > 0) ? offset - DIM : offset;
    int left   = (x > 0) ? offset - 1 : offset;
    int right  = (x < DIM - 1) ? offset + 1 : offset;
    int bottom = (y < DIM - 1) ? offset + DIM : offset;

    dev_outSrc[offset] =	dev_inSrc[offset] + SPEED * (dev_inSrc[top] + dev_inSrc[left] +
							dev_inSrc[right] + dev_inSrc[bottom] - (4.0f * dev_inSrc[offset])
    );
}


// Animate function
__host__ void animate(DataBlock* data, dim3 threads, dim3 blocks, float SPEED, int DIM) {
    bool wayOut = true;
    HANDLE_ERROR(cudaEventRecord(data->start, 0));

	// 1. Map the OpenGL resource before launching kernels
	HANDLE_ERROR(cudaGraphicsMapResources(1, &data->resource, 0));

	// 2. Get CUDA device pointer
    float* devPtr;
	size_t size;
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, data->resource));

	for (int i = 0; i < 30; i++) {
		float* in = wayOut ? data->dev_inSrc : data->dev_outSrc;
		float* out = wayOut ? data->dev_outSrc : data->dev_inSrc;

		copy_heaters_kernel<<<blocks, threads>>>(in, data->dev_constSrc, DIM);
		cudaDeviceSynchronize(); // TODO check if needed
		update_anim_kernel<<<blocks, threads>>>(in, out, SPEED, DIM);

		wayOut = !wayOut;
	}

	// 3. Copy final output to PBO
	float* finalOut = wayOut ? data->dev_outSrc : data->dev_inSrc;
	HANDLE_ERROR(cudaMemcpy(devPtr, finalOut, DIM * DIM * sizeof(float), cudaMemcpyDeviceToDevice));

	// 4. Unmap the OpenGL resource After kernel execution
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &data->resource, 0));
}


__host__ void anim_exit(DataBlock* data) {
	HANDLE_ERROR(cudaFree(data->dev_inSrc));
	HANDLE_ERROR(cudaFree(data->dev_outSrc));
	HANDLE_ERROR(cudaFree(data->dev_constSrc));

	HANDLE_ERROR(cudaEventRecord(data->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(data->stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, data->start, data->stop));
	printf("Elapsed time: %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(data->start));
	HANDLE_ERROR(cudaEventDestroy(data->stop));
}
