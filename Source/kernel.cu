#include "kernel.cuh" // TODO: change NAME

cudaTextureObject_t texConstSrc, texIn, texOut; // TODO: add constatnt source, like a heater.

// Update using texture memory
__global__ void update_kernel(float *dst, bool dstOut) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Find neighbors
	int left = (x > 0) ? offset - 1 : offset;
	int right = (x < blockDim.x * gridDim.x - 1) ? offset + 1 : offset;
	int top = (y > 0) ? offset - (blockDim.x * gridDim.x) : offset;
	int bottom = (y < blockDim.y * gridDim.y - 1) ? (offset + (blockDim.x * gridDim.x)) : offset;

	float t, l, c, r, b;
	if (dstOut) {
		t = tex2D // TODO: tex2D kullanýmdan kalkmýþ
		l = tex2D();
		c = tex2D();
		r = tex2D();
		b = tex2D();
	}
	else {
		t = tex2D();
		l = tex2D();
		c = tex2D();
		r = tex2D();
		b = tex2D();

	}
	dst[offset] = c + SPEED * (t + l + r + b - (4*c));
}