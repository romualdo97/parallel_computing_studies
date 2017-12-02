
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void dot_kernel(int *d_c, int *d_a, int *d_b)
{
	// terminology: A block of threads shares memory called... shared memory
	// extremely fast, on-chip memory (user-managed memory)
	// and is declared with the CUDA __shared__ keyword
	// each thread computes a pairwaise product


}

int main()
{

    return 0;
}