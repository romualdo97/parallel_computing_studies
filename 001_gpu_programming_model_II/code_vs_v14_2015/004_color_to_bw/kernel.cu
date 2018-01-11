// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Green, and Blue is in it.
//The 'A' stands for Alpha and is used for transparency; it will be
//ignored in this homework.

//Each channel Red, Blue, Green, and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "kernel.cuh"
#include <stdio.h>


__global__ void rgba_to_greyscale_kernel(uchar4 const * const rgbaImage,
										unsigned char * const greyImage,
										int numCols)
{
	//TODO
	//Fill in the kernel to convert from color to greyscale
	//the mapping from components of a uchar4 to RGBA is:
	// .x -> R ; .y -> G ; .z -> B ; .w -> A
	//
	//The output (greyImage) at each pixel should be the result of
	//applying the formula: output = .299f * R + .587f * G + .114f * B;
	//Note: We will be ignoring the alpha channel for this conversion

	//First create a mapping from the 2D block and grid locations
	//to an absolute 2D location in the image, then use that to
	//calculate a 1D offset

	// http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
	// full global thread ID in X and Y dimensions
    int col = threadIdx.x + blockIdx.x * blockDim.x; // x
	int row = threadIdx.y + blockIdx.y * blockDim.y; // y

	// ACCESSING MATRICES IN LINEAR MEMORY
	// • we cannot not use two-dimensional indices (e.g. A[row][column])
	// to access matrices
	// • We will need to know how the matrix is laid out in memory and
	// then compute the distance from the beginning of the matrix
	// • C uses row-major order --- rows are stored one after the
	// other in memory, i.e.row 0 then row 1 etc.
	int index = numCols * row + col; // W * row + col
	uchar4 const * texel = (rgbaImage + index);
	*(greyImage + index) = texel->x * .299f + texel->y * .587f + texel->z * .114f;
}

void your_rgba_to_greyscale(uchar4 const * const h_rgbaImage, uchar4 * const d_rgbaImage,
	unsigned char * const d_greyImage, size_t numRows, size_t numCols)
{
	std::cout << "\nLAUNCHING KERNEL:" << std::endl;


	int const BLOCKS = 32;
	int const X_THREADS_PER_BLOCK = numCols / BLOCKS;
	int const Y_THREADS_PER_BLOCK = numRows / BLOCKS;

	std::cout << "\t// remember max number of threads per block is 1024 in total, no in each dimension" << std::endl;
	std::cout << "\t- Num of blocks: " << BLOCKS << "x" << BLOCKS << std::endl;
	std::cout << "\t- Threads per block in X: " << X_THREADS_PER_BLOCK << std::endl;
	std::cout << "\t- Threads per block in Y: " << Y_THREADS_PER_BLOCK << std::endl;
	std::cout << "\t- Total threads per block: " << X_THREADS_PER_BLOCK * Y_THREADS_PER_BLOCK << std::endl;

	//You must fill in the correct sizes for the blockSize and gridSize
	//currently only one block with one thread is being launched
	const dim3 blockSize(X_THREADS_PER_BLOCK, Y_THREADS_PER_BLOCK, 1);		// How many threads per block?
	const dim3 gridSize(BLOCKS, BLOCKS, 1);											// How many blocks? (grid of blocks)
	rgba_to_greyscale_kernel <<<gridSize, blockSize>>> (d_rgbaImage, d_greyImage, numCols);

	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "\t- rgba_to_greyscale_kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	std::cout << "\t- Kernel launched succesfully" << std::endl;
	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/*
int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}*/
