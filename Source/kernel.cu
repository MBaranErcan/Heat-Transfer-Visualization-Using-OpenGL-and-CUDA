#include "Kernel.cuh"


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

// Init buffer with a specific value
__global__ void init_buffer_kernel(float* ptr, float temp, int DIM) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * DIM;

	if (x < DIM && y < DIM) {
		ptr[offset] = temp;
	}
}


// Initialize memory on device
__host__ void init_anim(DataBlock* data, int DIM, const float initTemp, dim3 blocks, dim3 threads) {

	size_t dataSize = DIM * DIM * sizeof(float);

	// Allocate memory on device
	HANDLE_ERROR(cudaMalloc((void**)&data->dev_inSrc, dataSize));
	HANDLE_ERROR(cudaMalloc((void**)&data->dev_outSrc, dataSize));
	HANDLE_ERROR(cudaMalloc((void**)&data->dev_heaterIndices, dataSize));
	HANDLE_ERROR(cudaMalloc((void**)&data->dev_heaterTempVals, dataSize));


	// Create events
	HANDLE_ERROR(cudaEventCreate(&data->start));
	HANDLE_ERROR(cudaEventCreate(&data->stop));

	// Initialize time
	data->totalTime = 0;
	data->elapsedTime = 0;
	data->frames = 0;
	data->ms = 0;

	// Set initial value
	init_buffer_kernel<<<blocks, threads>>>(data->dev_inSrc, initTemp, DIM);	
	
	// Set 0
	HANDLE_ERROR(cudaMemset(data->dev_heaterIndices, 0 , dataSize));
	HANDLE_ERROR(cudaMemset(data->dev_heaterTempVals, 0 , dataSize));
}



__global__ void add_heater_kernel(DataBlock* data, int DIM, Heater* heater)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * DIM;


	if (x < DIM && y < DIM) {
		int dist = sqrtf((x - heater->x) * (x - heater->x) + (y - heater->y) * (y - heater->y));
		// If distance to heater center is less than radius
		if (dist <= heater->radius) {
			data->dev_heaterIndices[offset] = 1.0f; // Set it as a heater
			data->dev_heaterTempVals[offset] = heater->temp; // Set its temp
			printf("Heater with temp: %f added to x: %d, y:%d", heater->temp, x, y);
		}
	}
}



__host__ void add_heater(DataBlock* data, Heater* heater, int DIM, dim3 blocks, dim3 threads) {
    addHeaterKernel<<<blocks, threads>>>(data->dev_heaterIndices, data->dev_heaterTempVals, DIM, heater->x, heater->y, heater->temp, heater->radius);
    HANDLE_ERROR(cudaDeviceSynchronize());
}


__global__ void addHeaterKernel(float* heaterIndices, float* heaterTempVals, int DIM, int heater_x, int heater_y, float heater_temp, int radius) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * DIM;

	if (x < DIM && y < DIM) {
		int dist = sqrtf((x - heater_x) * (x - heater_x) + (y - heater_y) * (y - heater_y));
		// If distance to heater center is less than radius
		if (dist <= radius) {
			heaterIndices[offset] = 1.0f; // Set it as a heater
			heaterTempVals[offset] = heater_temp; // Set its temp
		}
	}
}


// Constant Cells (Heaters)
__global__ void copy_heaters_kernel(float* iptr, const float* cptr, const float* vptr, int DIM) { 
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * DIM;

    if (x < DIM && y < DIM && cptr[offset] != 0) {	// If it is marked as a heater,
		iptr[offset] = vptr[offset];				// set to its constant heater temp.
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

    dev_outSrc[offset] =	dev_inSrc[offset] + (SPEED/4) * (dev_inSrc[top] + dev_inSrc[left] +
							dev_inSrc[right] + dev_inSrc[bottom] - (4.0f * dev_inSrc[offset])
    );
}


// Animate function
__host__ void animate(DataBlock* data, dim3 blocks, dim3 threads, float SPEED, int DIM) {
    bool wayOut = true;
    HANDLE_ERROR(cudaEventRecord(data->start, 0));

	SPEED = SPEED < 0.0 ? 0.0 : SPEED;
	SPEED = SPEED > 1.0 ? 1.0 : SPEED;

    
	for (int i = 0; i < 90; i++) {
		float* in = wayOut ? data->dev_inSrc : data->dev_outSrc;
		float* out = wayOut ? data->dev_outSrc : data->dev_inSrc;

		copy_heaters_kernel<<<blocks, threads>>>(in, data->dev_heaterIndices, data->dev_heaterTempVals, DIM);
		update_anim_kernel<<<blocks, threads>>>(in, out, SPEED, DIM);

		wayOut = !wayOut;
	}

	float* finalOut = wayOut ? data->dev_outSrc : data->dev_inSrc;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &data->resource, 0));

	HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&data->cudaArray, data->resource, 0, 0));

	HANDLE_ERROR(cudaMemcpyToArray(data->cudaArray, 0, 0, finalOut, DIM * DIM * sizeof(float), cudaMemcpyDeviceToDevice));

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &data->resource, 0));

	// Update time and frame
	HANDLE_ERROR(cudaEventRecord(data->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(data->stop));
	HANDLE_ERROR(cudaEventElapsedTime(&data->ms, data->start, data->stop));
	data->elapsedTime	+= data->ms; // For elapsed time per frame (FPS)
	data->totalTime		+= data->ms; // For total time app running
	data->frames++;
}


__host__ void anim_exit(DataBlock* data) {
	HANDLE_ERROR(cudaFree(data->dev_inSrc));
	HANDLE_ERROR(cudaFree(data->dev_outSrc));
	HANDLE_ERROR(cudaFree(data->dev_heaterIndices));
	HANDLE_ERROR(cudaFree(data->dev_heaterTempVals));

	HANDLE_ERROR(cudaEventRecord(data->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(data->stop));
	HANDLE_ERROR(cudaEventElapsedTime(&data->elapsedTime, data->start, data->stop));
	data->totalTime += data->elapsedTime;
	printf("Total Elapsed Time: %3.1f ms\n", data->totalTime);

	HANDLE_ERROR(cudaEventDestroy(data->start));
	HANDLE_ERROR(cudaEventDestroy(data->stop));
}
