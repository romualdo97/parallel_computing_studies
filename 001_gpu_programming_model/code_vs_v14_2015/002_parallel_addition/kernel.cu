
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 512
#define THREADS_PER_BLOCK 216

// define kernel for realize parallel addition in GPU using CUDA blocks
__global__ void parallel_add_with_blocks_kernel(int *dev_c, int *dev_a, int *dev_b)
{
	// use cuda block identifiers for compute vectorial addition
	dev_c[blockIdx.x] = dev_a[blockIdx.x] + dev_b[blockIdx.x];
}

// define kernel for realize parallel addition in GPU using CUDA threads
__global__ void parallel_add_with_threads_kernel(int *dev_c, int *dev_a, int *dev_b)
{
	// use cuda thread idintifiers for compute vectorial addition
	dev_c[threadIdx.x] = dev_a[threadIdx.x] + dev_b[threadIdx.x];
}

// define kernel for realize parallel addition in GPU using CUDA threads and blocks simultaneously
__global__ void parallel_add_threads_blocks_kernel(int *dev_c, int *dev_a, int *dev_b)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	dev_c[index] = dev_a[index] + dev_b[index];
}

// declare helper function for assign ints to an array of ints
void assign_ints(int*, unsigned int);

int main()
{
	int *a, *b, *c;					// declare host memory for arrays a, b, c
	int *dev_a, *dev_b, *dev_c;		// declare device copies of a, b, c
	int size = N * sizeof(int);		// calculate memory size needed

	// allocate device memory for a, b, c
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	// allocate host memory for a, b, c
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	// assign values to host a, b
	assign_ints(a, N);
	assign_ints(b, N);

	// asign values to device a, b
	cudaMemcpy(dev_a, a, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// launch kernel for parallel addition using "N blocks" and "one thread per block"
	parallel_add_with_blocks_kernel<<<N, 1>>>(dev_c, dev_a, dev_b);

	// launch kernel for parallel addition using "one block" and "N threads per block"
	parallel_add_with_threads_kernel<<<1, N>>>(dev_c, dev_a, dev_b);

	// launch N parallel kernels for compute vectorial addition
	// using "N/THREADS_PER_BLOCK" blocks and "THREADS_PER_BLOCK" threads
	parallel_add_threads_blocks_kernel<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

	// copy result from device to host memory
	cudaMemcpy(c, dev_c, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// de-allocate host and device memory
	free(a); free(b); free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
    return 0;
}

// set random ints to an array arr of magnitude size
void assign_ints(int *arr, unsigned int size)
{
	for (int i = 0; i < size; i++)
	{
		arr[i] = i;
	}
}
