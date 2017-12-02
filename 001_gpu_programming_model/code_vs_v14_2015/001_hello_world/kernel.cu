
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/*
CUDA C keyword __global__ indicates that a function
	- runs on the device
	- and is called from host code
*/

// define kernel for add two integers
__global__ void add_kernel(int *dev_c, int const *dev_a, int const *dev_b)
{
	// assign to the value in address 'c' the addition of values in addresses 'a' and 'b'
	*dev_c = *dev_a + *dev_b;
}

int main()
{
	int a, b, c;					// host copies of a, b, c
	int *dev_a, *dev_b, *dev_c;		// device copies of a, b, c
	int size = sizeof(int);			// we need space for an integer

	// allocates device memory for a, b, c
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	// assign values to host a, b
	a = 2;
	b = 3;

	// assign values to device a, b
	cudaMemcpy((void*)dev_a, &a, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// launch kernel for addition
	add_kernel<<<1, 1>>>(dev_c, dev_a, dev_b);

	// copy result from device to host copy of c
	cudaMemcpy(&c, dev_c, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// log results
	printf("c: %d\n", c);

	// de-allocates device memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
    return 0;
}

