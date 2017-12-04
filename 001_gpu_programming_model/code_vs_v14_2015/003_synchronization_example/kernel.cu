
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <iostream>

#define N 32
#define THREADS_PER_BLOCK 16

// use N threads One block
__global__ void dot_kernel(int *d_c, int *d_a, int *d_b)
{
	// terminology: A block of threads shares memory called... shared memory
	// extremely fast, on-chip memory (user-managed memory)
	// and is declared with the CUDA __shared__ keyword
	// each thread computes a pairwaise product
	__shared__ int temp[N];
	temp[threadIdx.x] = d_a[threadIdx.x] * d_b[threadIdx.x];

	// * NEED THREADS TO SYNCHRONIZE HERE *
	// No thread can advance until all threads
	// have reached this point in the code
	// for avoid read-before-writing hazard
	__syncthreads(); // threads are synchronized within a block

	// thread 0 sum the pairwaise producs
	if (0 == threadIdx.x)
	{
		int sum = 0;
		for (int i = 0; i < N; i++)
		{
			sum += temp[i];
		}
		*d_c = sum;
		*d_c = 3;
	}
}

// use N/THREADS_PER_BLOCK blocks and N threads
// read this for undestand cuda capabilities https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
// study more about achieved occupancy https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm
__global__ void multiblock_dot_kernel(int *r, int *a, int *b)
{
	__shared__ int temp[THREADS_PER_BLOCK];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	temp[index] = a[index] * b[index];

	__syncthreads();

	if (0 == threadIdx.x)
	{
		int sum = 0;
		for (int i = 0; i < THREADS_PER_BLOCK; i++)
		{
			// no problem here beacase threads were synchronized
			sum += temp[index];
		}

		// ADD EACH BLOCK RESULT TO VALUE AT ADDRESS r
		// 1. read value at addres r							(read)
		// 2. add sum to value									(modify)
		// 3. write result to address r							(write)
		//*r  += sum; race condition no atomic operation
		atomicAdd(r, sum); // "read-modify-write" uninterrumpible when atomic
	}
}

void set_ints(int *arr, int const SIZE);

void print_vectors(char *name, int *arr, int const SIZE);

int main()
{
	int *a, *b, c = 0;				// host variables
	int *d_a, *d_b, *d_c;			// device variables
	int size = N * sizeof(int);		// for memory allocation

	// allocate host memory
	a = (int*)malloc(size);
	b = (int*)malloc(size);

	// init host memory
	set_ints(a, N);
	set_ints(b, N);

	// tell cuda what variables are for device (GPU) memory
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, sizeof(int));

	// init device memory
	cudaMemcpy(d_a, a, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// launch kernel
	//dot_kernel<<<1, N>>>(d_c, d_a, d_b);
	multiblock_dot_kernel<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_c, d_a, d_b);

	// copy result from device to host
	cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	print_vectors("a", a, N);
	print_vectors("b", b, N);
	std::cout << "\nResult: " << c << "\n\n";
	system("pause");

	free(a); free(b);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
    return 0;
}

void print_vectors(char *name, int *arr, int const SIZE)
{
	std::cout << name << ": ";
	for (int i = 0; i < N; i++)
	{
		std::cout << " " << arr[i];
	}
	std::cout << "\n\n";
}

void set_ints(int *arr, int const SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		arr[i] = 3;
	}
}